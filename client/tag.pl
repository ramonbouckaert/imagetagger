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

@ARGV or die usage();

sub usage {
    return <<END;
Usage: $0 [options] PATTERN [PATTERN ...]

Tags AVIF files matching the given glob pattern(s), writing results back
as XMP metadata using ExifTool. Quote patterns to let Perl expand them
rather than the shell (useful for recursive globs).

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

# ── Enqueue files ──────────────────────────────────────────────────────────────

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
        $tagger->enqueue($file);
    }
}

if (!$shutting_down) {
    printf "Queued files, waiting for %d workers to finish...\n", $workers;
    $tagger->drain();
    print "Done.\n";
}
