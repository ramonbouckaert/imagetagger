#!/usr/bin/perl
use strict;
use warnings;

use threads;
use Thread::Queue;
use threads::shared;

use File::Basename qw(basename);
use Getopt::Long   qw(GetOptions);
use Linux::Inotify2;
use LWP::UserAgent;
use HTTP::Request::Common qw(POST);
use JSON                  qw(decode_json);
use Image::ExifTool;

# ── Configuration ──────────────────────────────────────────────────────────────

my $watch_dir = $ENV{WATCH_DIR}        // '.';
my $endpoint  = $ENV{TAGGER_ENDPOINT}  // 'http://localhost:9100/analyse';
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

# ── Queue and debounce state ───────────────────────────────────────────────────

my $queue = Thread::Queue->new();

# Shared across threads. Lock before read-modify-write.
my %last_written : shared;
my $DEBOUNCE_SECS = 5;

# ── Worker logic ───────────────────────────────────────────────────────────────
# Each worker thread creates its own LWP::UserAgent (not thread-safe to share).

sub process_file {
    my ($path, $ua) = @_;

    # Atomically check and claim the file to prevent two workers racing on the
    # same path (e.g. rapid successive events, or ExifTool's rewrite triggering
    # a new event that was already queued before the debounce stamp was set).
    {
        lock(%last_written);
        if (exists $last_written{$path} && (time() - $last_written{$path}) < $DEBOUNCE_SECS) {
            print "[skip] $path (debounce)\n";
            return;
        }
        $last_written{$path} = time();   # reserve; prevents other workers racing
    }

    print "[upload] $path\n";

    my $response = $ua->request(
        POST(
            $endpoint,
            Content_Type => 'multipart/form-data',
            Content      => [
                image => [$path, basename($path), 'Content-Type' => 'image/avif'],
            ],
        )
    );

    unless ($response->is_success) {
        warn "[error] Upload failed for $path: " . $response->status_line . "\n";
        warn "        " . $response->decoded_content . "\n" if $response->decoded_content;
        lock(%last_written);
        delete $last_written{$path};
        return;
    }

    my $data = eval { decode_json($response->decoded_content) };
    if ($@) {
        warn "[error] Could not parse JSON response for $path: $@\n";
        lock(%last_written);
        delete $last_written{$path};
        return;
    }

    my $description = $data->{description} // '';
    my @tags        = @{ $data->{tags}     // [] };

    printf "[result] description: %s\n", $description || '(none)';
    printf "[result] tags (%d): %s\n", scalar(@tags), join(', ', @tags);

    # ── Write XMP metadata ────────────────────────────────────────────────────

    my $exif = Image::ExifTool->new();

    # Read existing subject tags so we only add ones not already present.
    my $info     = $exif->ImageInfo($path, 'XMP:Subject');
    my $existing = $info->{'Subject'} // [];
    $existing    = [$existing] unless ref $existing eq 'ARRAY';
    my %existing_lc = map { lc($_) => 1 } @$existing;

    my @new_tags = grep { !$existing_lc{lc($_)} } @tags;

    printf "[result] %d new tag(s) to add (skipping %d already present)\n",
        scalar(@new_tags), scalar(@tags) - scalar(@new_tags);

    $exif->SetNewValue('XMP:Description', $description);
    for my $tag (@new_tags) {
        $exif->SetNewValue('XMP:Subject', $tag, { AddValue => 1 });
    }

    my ($ok, $err) = $exif->WriteInfo($path);

    if ($ok) {
        print "[xmp] written to $path\n";
    } else {
        warn "[error] ExifTool failed to write $path: $err\n";
        lock(%last_written);
        delete $last_written{$path};
    }
}

sub worker {
    my $ua = LWP::UserAgent->new(timeout => 300);
    while (defined(my $path = $queue->dequeue())) {
        process_file($path, $ua);
    }
}

# ── Thread pool ────────────────────────────────────────────────────────────────

my @pool = map { threads->create(\&worker) } 1..$workers;

$SIG{INT} = $SIG{TERM} = sub {
    print "\nShutting down (draining queue)...\n";
    $queue->end();
    $_->join() for @pool;
    exit 0;
};

# ── Inotify watcher ────────────────────────────────────────────────────────────

my $inotify = Linux::Inotify2->new()
    or die "Cannot initialise inotify: $!\n";

sub enqueue_file {
    my ($path) = @_;
    return unless $path =~ /\.avif$/i;
    $queue->enqueue($path);
    printf "[queued] %s (pending: %d)\n", $path, $queue->pending();
}

sub watch_dir {
    my ($dir) = @_;
    $inotify->watch(
        $dir,
        IN_CLOSE_WRITE | IN_MOVED_TO,
        sub {
            my ($event) = @_;
            enqueue_file($event->fullname) unless $event->IN_ISDIR;
        },
    ) or warn "Cannot watch '$dir': $!\n";
    print "Watching $dir\n";
}

# Watch the root directory for files, and for new immediate subdirectories.
$inotify->watch(
    $watch_dir,
    IN_CLOSE_WRITE | IN_MOVED_TO | IN_CREATE,
    sub {
        my ($event) = @_;
        if ($event->IN_ISDIR) {
            watch_dir($event->fullname);
        } else {
            enqueue_file($event->fullname);
        }
    },
) or die "Cannot watch '$watch_dir': $!\n";

# Watch all immediate subdirectories that already exist.
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
