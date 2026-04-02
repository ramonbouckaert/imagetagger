#!/usr/bin/perl
use strict;
use warnings;

use FindBin qw($Bin);
use lib $Bin;

use Getopt::Long qw(GetOptions);
use Tagger;

# ── Configuration ──────────────────────────────────────────────────────────────

my $endpoint = $ENV{TAGGER_ENDPOINT} // 'http://localhost:9100/analyse';
my $workers  = 5;
my $help;

GetOptions(
    'endpoint=s' => \$endpoint,
    'workers=i'  => \$workers,
    'help'       => \$help,
) or die usage();

if ($help) { print usage(); exit 0; }

my $dir = shift @ARGV or die usage();
$dir =~ s{/+$}{};
die "Directory does not exist: $dir\n" unless -d $dir;

sub usage {
    return <<END;
Usage: $0 [options] DIR

Queues every AVIF file in DIR (and immediate subdirectories) for tagging via
the imagetagger API, writing results back as XMP metadata using ExifTool.

Options:
  --endpoint=URL    Imagetagger API endpoint
                    (default: \$TAGGER_ENDPOINT env var, or http://localhost:9100/analyse)
  --workers=N       Number of concurrent upload workers (default: 5)
  --help            Show this help

END
}

# ── Enqueue files ──────────────────────────────────────────────────────────────

my $tagger = Tagger->new(endpoint => $endpoint, workers => $workers);

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

sub enqueue_dir {
    my ($d) = @_;
    return if $shutting_down;
    for my $file (glob("$d/*.avif"), glob("$d/*.AVIF")) {
        last if $shutting_down;
        $tagger->enqueue($file);
    }
}

enqueue_dir($dir);

opendir(my $dh, $dir) or die "Cannot open '$dir': $!\n";
while (my $entry = readdir($dh)) {
    last if $shutting_down;
    next if $entry =~ /^\./;
    my $subdir = "$dir/$entry";
    enqueue_dir($subdir) if -d $subdir;
}
closedir($dh);

if (!$shutting_down) {
    printf "Queued files, waiting for %d workers to finish...\n", $workers;
    $tagger->drain();
    print "Done.\n";
}
