#!/usr/bin/perl
use strict;
use warnings;

use FindBin qw($Bin);
use lib $Bin;

use File::Basename qw(basename);
use Getopt::Long   qw(GetOptions);
use Linux::Inotify2;
use Mojo::IOLoop;
use Tagger;

my $watch_dir = $ENV{WATCH_DIR}       // '.';
my $endpoint  = $ENV{TAGGER_ENDPOINT} // 'http://localhost:9100/analyse';
my $workers   = 5;
my $help;

GetOptions(
    'dir=s'      => \$watch_dir,
    'endpoint=s' => \$endpoint,
    'workers=i'  => \$workers,
    'help'       => \$help,
) or die usage();

if ($help) {
    print usage();
    exit 0;
}

$watch_dir =~ s{/+$}{};
die "Directory does not exist: $watch_dir\n" unless -d $watch_dir;

sub usage {
    return <<'END';
Usage: watcher.pl [options]

Watches a directory and its immediate subdirectories for AVIF file changes and
tags them via the imagetagger API, writing the results back to the file as XMP
metadata using ExifTool.

Options:
  --dir=DIR         Directory to watch for .avif changes
                    (default: $WATCH_DIR env var, or current directory)
  --endpoint=URL    Imagetagger API endpoint
                    (default: $TAGGER_ENDPOINT env var, or http://localhost:9100/analyse)
  --workers=N       Number of concurrent upload workers (default: 5)
  --help            Show this help

END
}

my $tagger = Tagger->new(
    endpoint   => $endpoint,
    workers    => $workers,
    on_failure => sub {
        my ($path) = @_;
        if (open(my $fh, '>>', 'imagetagger-failures.txt')) {
            print $fh "$path\n";
            close($fh);
        } else {
            warn "Cannot open imagetagger-failures.txt: $!\n";
        }
    },
);

my $shutting_down    = 0;
my $shutdown_started = 0;

my $inotify = Linux::Inotify2->new()
    or die "Cannot initialise inotify: $!\n";

$inotify->blocking(0);

my %watches;
my %enqueue_timers;
my $inotify_fh = $inotify->fh;
my $reactor    = Mojo::IOLoop->singleton->reactor;

sub is_hidden_dir {
    my ($path) = @_;
    return basename($path) =~ /^\./;
}

sub unwatch_dir {
    my ($dir) = @_;
    delete $watches{$dir};
    print "Stopped watching $dir\n";
}

sub schedule_enqueue {
    my ($path) = @_;

    return if $shutting_down;
    return unless $path =~ /\.AVIF\z/i;

    if (my $id = delete $enqueue_timers{$path}) {
        Mojo::IOLoop->remove($id);
    }

    $enqueue_timers{$path} = Mojo::IOLoop->timer(5 => sub {
        delete $enqueue_timers{$path};

        return if $shutting_down;
        return unless -f $path;

        $tagger->enqueue($path);
    });
}

sub watch_dir {
    my ($dir, $depth) = @_;

    return if $watches{$dir};
    return if $depth > 1;
    return if $depth > 0 && is_hidden_dir($dir);

    my $watch = $inotify->watch(
        $dir,
        IN_CREATE | IN_CLOSE_WRITE | IN_MOVED_TO | IN_DELETE_SELF | IN_MOVE_SELF | IN_IGNORED,
        sub {
            return if $shutting_down;

            my ($event) = @_;

            if ($event->IN_IGNORED || $event->IN_DELETE_SELF || $event->IN_MOVE_SELF) {
                unwatch_dir($dir);
                return;
            }

            my $path = $event->fullname;

            if ($event->IN_ISDIR) {
                return if is_hidden_dir($path);

                if ($depth == 0 && ($event->IN_CREATE || $event->IN_MOVED_TO)) {
                    watch_dir($path, 1);
                }

                return;
            }

            if ($event->IN_CLOSE_WRITE || $event->IN_MOVED_TO) {
                schedule_enqueue($path);
            }
        },
    );

    if (!$watch) {
        warn "Cannot watch '$dir': $!\n";
        return;
    }

    $watches{$dir} = $watch;
    print "Watching $dir\n";
}

sub refresh_one_level_watches {
    return if $shutting_down;

    watch_dir($watch_dir, 0) unless $watches{$watch_dir};

    for my $dir (keys %watches) {
        next if $dir eq $watch_dir;
        unwatch_dir($dir) unless -d $dir;
    }

    opendir(my $dh, $watch_dir) or do {
        warn "Cannot open '$watch_dir': $!\n";
        return;
    };

    while (my $entry = readdir($dh)) {
        next if $entry eq '.' || $entry eq '..';

        my $path = "$watch_dir/$entry";
        next unless -d $path;
        next if is_hidden_dir($path);

        watch_dir($path, 1);
    }

    closedir($dh);
}

sub begin_shutdown {
    return if $shutdown_started;
    $shutdown_started = 1;

    $reactor->remove($inotify_fh);

    for my $id (values %enqueue_timers) {
        Mojo::IOLoop->remove($id);
    }
    %enqueue_timers = ();

    eval { $tagger->cancel(); 1 }
        or warn "[error] cancel failed: $@\n";

    if (my $p = $tagger->{drain_promise}) {
        $p->finally(sub {
            Mojo::IOLoop->stop if Mojo::IOLoop->is_running;
        });
    } else {
        Mojo::IOLoop->stop if Mojo::IOLoop->is_running;
    }
}

$SIG{INT} = $SIG{TERM} = sub {
    if ($shutting_down) {
        warn "\nSecond interrupt, forcing exit.\n";
        exit 1;
    }

    $shutting_down = 1;
    print "\nInterrupt received, cancelling queued and in-flight jobs...\n";

    if (Mojo::IOLoop->is_running) {
        Mojo::IOLoop->next_tick(\&begin_shutdown);
    } else {
        begin_shutdown();
        exit 0;
    }
};

$inotify->on_overflow(sub {
    warn "[warn] inotify queue overflow, refreshing watches only...\n";
    refresh_one_level_watches();
});

watch_dir($watch_dir, 0);

opendir(my $dh, $watch_dir) or die "Cannot open '$watch_dir': $!\n";
while (my $entry = readdir($dh)) {
    next if $entry eq '.' || $entry eq '..';

    my $subdir = "$watch_dir/$entry";
    next unless -d $subdir;
    next if is_hidden_dir($subdir);

    watch_dir($subdir, 1);
}
closedir($dh);

$reactor->io($inotify_fh => sub {
    eval { $inotify->poll };
    warn "[error] inotify poll failed: $@\n" if $@;
})->watch($inotify_fh, 1, 0);

printf "Started %d workers\n", $workers;
print  "Endpoint: $endpoint\n";

Mojo::IOLoop->start unless Mojo::IOLoop->is_running;
