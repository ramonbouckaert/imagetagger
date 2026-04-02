package Tagger;
use strict;
use warnings;

use threads;
use Thread::Queue;
use threads::shared;

use File::Basename        qw(basename);
use LWP::UserAgent;
use HTTP::Request::Common qw(POST);
use JSON                  qw(decode_json);
use Image::ExifTool;

$| = 1;  # unbuffer stdout so journald sees lines as they are printed

my $DEBOUNCE_SECS = 5;

# Shared across all worker threads.
our %last_written : shared;

sub new {
    my ($class, %args) = @_;
    my $self = bless {
        endpoint => $args{endpoint} // 'http://localhost:9100/analyse',
        queue    => Thread::Queue->new(),
        pool     => [],
    }, $class;

    my $n = $args{workers} // 32;
    push @{ $self->{pool} }, threads->create(sub { $self->_worker() }) for 1..$n;
    threads->create(sub { $self->_cleaner() })->detach();

    return $self;
}

# Enqueue a file path. Silently ignores non-AVIF paths and debounced files.
sub enqueue {
    my ($self, $path) = @_;
    return unless $path =~ /\.avif$/i;
    {
        lock(%last_written);
        if (exists $last_written{$path} && (time() - $last_written{$path}) < $DEBOUNCE_SECS) {
            print "[skip] $path (debounce)\n";
            return;
        }
        $last_written{$path} = time();
    }
    $self->{queue}->enqueue($path);
    printf "[queued] %s (pending: %d)\n", $path, $self->{queue}->pending();
}

# Signal end-of-input, drain the queue, and join all workers.
# Call this when no more files will be enqueued.
sub drain {
    my ($self) = @_;
    $self->{queue}->end();
    $_->join() for @{ $self->{pool} };
}

# ── Private ────────────────────────────────────────────────────────────────────

sub _worker {
    my ($self) = @_;
    my $ua = LWP::UserAgent->new(timeout => 300, keep_alive => 0);
    while (defined(my $path = $self->{queue}->dequeue())) {
        $self->_process_file($path, $ua);
    }
}

sub _cleaner {
    while (1) {
        sleep($DEBOUNCE_SECS * 2);
        lock(%last_written);
        my $cutoff = time() - $DEBOUNCE_SECS;
        delete $last_written{$_}
            for grep { $last_written{$_} < $cutoff } keys %last_written;
    }
}

sub _process_file {
    my ($self, $path, $ua) = @_;

    print "[upload] $path\n";

    my $response = $ua->request(
        POST(
            $self->{endpoint},
            Content_Type => 'multipart/form-data',
            Content      => [
                image => [$path, basename($path), 'Content-Type' => 'image/avif'],
            ],
        )
    );

    unless ($response->is_success) {
        warn "[error] Upload failed for $path: " . $response->status_line . "\n";
        warn "        " . $response->decoded_content . "\n" if $response->decoded_content;
        lock(%last_written); delete $last_written{$path};
        return;
    }

    my $data = eval { decode_json($response->decoded_content) };
    if ($@) {
        warn "[error] Could not parse JSON response for $path: $@\n";
        lock(%last_written); delete $last_written{$path};
        return;
    }

    my $description = $data->{description} // '';
    my @tags        = @{ $data->{tags}     // [] };

    printf "[result] description: %s\n", $description || '(none)';
    printf "[result] tags (%d): %s\n", scalar(@tags), join(', ', @tags);

    my $exif     = Image::ExifTool->new();
    my $info     = $exif->ImageInfo($path, 'XMP:Subject');
    my $existing = $info->{'Subject'} // [];
    $existing    = [$existing] unless ref $existing eq 'ARRAY';
    my %existing_lc = map { lc($_) => 1 } @$existing;

    my @new_tags = grep { !$existing_lc{lc($_)} } @tags;

    printf "[result] %d new tag(s) to add (skipping %d already present)\n",
        scalar(@new_tags), scalar(@tags) - scalar(@new_tags);

    $exif->SetNewValue('XMP:Description', $description, { Lang => 'en' });
    for my $tag (@new_tags) {
        $exif->SetNewValue('XMP:Subject', $tag, { AddValue => 1 });
    }

    my ($ok, $err) = $exif->WriteInfo($path);
    if ($ok) {
        print "[xmp] written to $path\n";
    } else {
        warn "[error] ExifTool failed to write $path: $err\n";
        lock(%last_written); delete $last_written{$path};
    }
}

1;
