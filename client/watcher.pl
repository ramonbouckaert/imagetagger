#!/usr/bin/perl
use strict;
use warnings;

use FindBin qw($Bin);
use lib $Bin;

use File::Basename qw(basename);
use Getopt::Long   qw(GetOptions);
use Linux::Inotify2;
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
    endpoint => $endpoint,
    workers  => $workers,
);

my $shutting_down = 0;

$SIG{INT} = $SIG{TERM} = sub {
    if ($shutting_down) {
        warn "\nSecond interrupt, forcing exit.\n";
        exit 1;
    }
    $shutting_down = 1;
    print "\nInterrupt received, cancelling queued and in-flight jobs...\n";
    eval { $tagger->cancel(); 1 }
        or warn "[error] cancel failed: $@\n";
    exit 0;
};

my $inotify = Linux::Inotify2->new()
    or die "Cannot initialise inotify: $!\n";

my %watches;

sub is_hidden_dir {
    my ($path) = @_;
    return basename($path) =~ /^\./;
}

sub unwatch_dir {
    my ($dir) = @_;
    delete $watches{$dir};
    print "Stopped watching $dir\n";
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
                $tagger->enqueue($path);
            }
        },
    );

    if (!$watch) {
        warn "Cannot watch '$dir': $!\n";
        return;
    }

    $watches{$dir} = {
        watch => $watch,
        depth => $depth,
    };

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

printf "Started %d workers\n", $workers;
print  "Endpoint: $endpoint\n";

while (!$shutting_down) {
    $inotify->poll;
}
