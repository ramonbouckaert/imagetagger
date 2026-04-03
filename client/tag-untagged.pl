#!/usr/bin/perl
use strict;
use warnings;

use FindBin qw($Bin);
use lib $Bin;

use Getopt::Long qw(GetOptions);
use Image::ExifTool;
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

@ARGV or die usage();

sub usage {
    return <<END;
Usage: $0 [options] PATTERN [PATTERN ...]

Like tag.pl, but skips any AVIF file that already has XMP keywords.
Quote patterns to let Perl expand them rather than the shell.

Examples:
  $0 /photos/*.avif
  $0 '/photos/**/*.avif'
  $0 /photos/2024/*.avif /photos/2025/*.avif

Options:
  --endpoint=URL    Imagetagger API endpoint
                    (default: \$TAGGER_ENDPOINT env var, or http://localhost:9100/analyse)
  --workers=N       Number of concurrent upload workers (default: 5)
  --help            Show this help

END
}

# ── Helpers ────────────────────────────────────────────────────────────────────

my $exif = Image::ExifTool->new();

sub has_keywords {
    my ($path) = @_;
    my $info    = $exif->ImageInfo($path, 'XMP:Subject');
    my $subject = $info->{'Subject'};
    return 0 unless defined $subject;
    return ref $subject eq 'ARRAY' ? scalar(@$subject) > 0 : 1;
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

for my $pattern (@ARGV) {
    last if $shutting_down;
    for my $file (glob($pattern)) {
        last if $shutting_down;
        if (has_keywords($file)) {
            print "[skip] $file (already tagged)\n";
            next;
        }
        $tagger->enqueue($file);
    }
}

if (!$shutting_down) {
    printf "Queued files, waiting for %d workers to finish...\n", $workers;
    $tagger->drain();
    print "Done.\n";
}
