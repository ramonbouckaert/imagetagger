#!/usr/bin/perl
use strict;
use warnings;

use FindBin qw($Bin);
use lib $Bin;

use Getopt::Long qw(GetOptions);
use Tagger;

# ── Configuration ──────────────────────────────────────────────────────────────

my $endpoint = $ENV{TAGGER_ENDPOINT} // 'http://localhost:9100/analyse';
my $workers  = 32;
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
  --workers=N       Number of concurrent upload workers (default: 32)
  --help            Show this help

END
}

# ── Enqueue files ──────────────────────────────────────────────────────────────

my $tagger = Tagger->new(endpoint => $endpoint, workers => $workers);

sub enqueue_dir {
    my ($d) = @_;
    for my $file (glob("$d/*.avif"), glob("$d/*.AVIF")) {
        $tagger->enqueue($file);
    }
}

enqueue_dir($dir);

opendir(my $dh, $dir) or die "Cannot open '$dir': $!\n";
while (my $entry = readdir($dh)) {
    next if $entry =~ /^\./;
    my $subdir = "$dir/$entry";
    enqueue_dir($subdir) if -d $subdir;
}
closedir($dh);

printf "Queued files — waiting for %d workers to finish...\n", $workers;
$tagger->drain();
print "Done.\n";
