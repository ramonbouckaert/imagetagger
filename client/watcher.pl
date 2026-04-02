#!/usr/bin/perl
use strict;
use warnings;

use FindBin qw($Bin);
use lib $Bin;

use Getopt::Long  qw(GetOptions);
use Linux::Inotify2;
use Tagger;

# ── Configuration ──────────────────────────────────────────────────────────────

my $watch_dir = $ENV{WATCH_DIR}       // '.';
my $endpoint  = $ENV{TAGGER_ENDPOINT} // 'http://localhost:9100/analyse';
my $workers   = 32;
my $help;

GetOptions(
    'dir=s'      => \$watch_dir,
    'endpoint=s' => \$endpoint,
    'workers=i'  => \$workers,
    'help'       => \$help,
) or die usage();

if ($help) { print usage(); exit 0; }

$watch_dir =~ s{/+$}{};
die "Directory does not exist: $watch_dir\n" unless -d $watch_dir;

sub usage {
    return <<END;
Usage: $0 [options]

Watches a directory for AVIF file changes and tags them via the imagetagger API,
writing the results back to the file as XMP metadata using ExifTool.

Options:
  --dir=DIR         Directory to watch for .avif changes
                    (default: \$WATCH_DIR env var, or current directory)
  --endpoint=URL    Imagetagger API endpoint
                    (default: \$TAGGER_ENDPOINT env var, or http://localhost:9100/analyse)
  --workers=N       Number of concurrent upload workers (default: 32)
  --help            Show this help

END
}

# ── Worker pool ────────────────────────────────────────────────────────────────

my $tagger = Tagger->new(endpoint => $endpoint, workers => $workers);

$SIG{INT} = $SIG{TERM} = sub {
    print "\nShutting down (draining queue)...\n";
    $tagger->drain();
    exit 0;
};

# ── Inotify watcher ────────────────────────────────────────────────────────────

my $inotify = Linux::Inotify2->new()
    or die "Cannot initialise inotify: $!\n";

sub watch_dir {
    my ($dir) = @_;
    $inotify->watch(
        $dir,
        IN_CLOSE_WRITE | IN_MOVED_TO,
        sub {
            my ($event) = @_;
            $tagger->enqueue($event->fullname) unless $event->IN_ISDIR;
        },
    ) or warn "Cannot watch '$dir': $!\n";
    print "Watching $dir\n";
}

$inotify->watch(
    $watch_dir,
    IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE,
    sub {
        my ($event) = @_;
        if ($event->IN_ISDIR) {
            watch_dir($event->fullname);
        } else {
            $tagger->enqueue($event->fullname);
        }
    },
) or die "Cannot watch '$watch_dir': $!\n";

opendir(my $dh, $watch_dir) or die "Cannot open '$watch_dir': $!\n";
while (my $entry = readdir($dh)) {
    next if $entry =~ /^\./;
    my $subdir = "$watch_dir/$entry";
    watch_dir($subdir) if -d $subdir;
}
closedir($dh);

printf "Started %d workers\n", $workers;
print  "Endpoint: $endpoint\n";

1 while $inotify->read;
