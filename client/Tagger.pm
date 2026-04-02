package Tagger;
use strict;
use warnings;

use threads;
use Thread::Queue;
use threads::shared qw(shared_clone);

use File::Basename        qw(basename);
use LWP::UserAgent;
use HTTP::Request::Common qw(POST);
use JSON                  qw(decode_json);
use Image::ExifTool;

$| = 1;
binmode STDOUT, ':encoding(UTF-8)';
binmode STDERR, ':encoding(UTF-8)';

my $DEBOUNCE_SECS = 5;

our %last_written : shared;
our %inflight     : shared;

sub new {
    my ($class, %args) = @_;
    my $self = bless {
        endpoint => $args{endpoint} // 'http://localhost:9100/analyse',
        queue    => Thread::Queue->new(),
        pool     => [],
        state    => shared_clone({
            cancel       => 0,
            stop_cleaner => 0,
        }),
    }, $class;

    my $n = $args{workers} // 32;
    push @{ $self->{pool} }, threads->create(sub { $self->_worker() }) for 1..$n;
    $self->{cleaner} = threads->create(sub { $self->_cleaner() });

    return $self;
}

sub enqueue {
    my ($self, $path) = @_;
    return unless $path =~ /\.avif$/i;

    {
        lock(%{ $self->{state} });
        return if $self->{state}->{cancel};
    }

    {
        lock(%last_written);
        lock(%inflight);

        return if $inflight{$path};

        if (exists $last_written{$path} && (time() - $last_written{$path}) < $DEBOUNCE_SECS) {
            print "[skip] $path (debounce)\n";
            return;
        }
        $last_written{$path} = time();
        $inflight{$path}     = 1;
    }
    $self->{queue}->enqueue($path);
    printf "[queued] %s (pending: %d)\n", $path, ($self->{queue}->pending() // 0);
}

sub drain {
    my ($self) = @_;
    $self->{queue}->end();
    $_->join() for @{ $self->{pool} };

    {
        lock(%{ $self->{state} });
        $self->{state}->{stop_cleaner} = 1;
    }
    $self->{cleaner}->join() if $self->{cleaner};
}

sub cancel {
    my ($self) = @_;

    {
        lock(%{ $self->{state} });
        return if $self->{state}->{cancel};
        $self->{state}->{cancel}       = 1;
        $self->{state}->{stop_cleaner} = 1;
    }

    my @dropped;
    {
        lock($self->{queue});
        my $pending = $self->{queue}->pending() // 0;
        @dropped = $pending ? $self->{queue}->extract(0, $pending) : ();
        $self->{queue}->end();
    }

    if (@dropped) {
        lock(%last_written);
        lock(%inflight);
        delete $last_written{$_} for @dropped;
        delete $inflight{$_}     for @dropped;
    }

    printf "[cancel] dropped %d queued job(s)\n", scalar(@dropped);

    $_->kill('USR1') for grep { $_->is_running() } @{ $self->{pool} };

    $_->join() for @{ $self->{pool} };
    $self->{cleaner}->join() if $self->{cleaner};
}

sub _worker {
    my ($self) = @_;

    my $ua = LWP::UserAgent->new(
        timeout    => 5,
        keep_alive => 0,
    );

    local $SIG{USR1} = sub { };   # harmless when idle

    while (1) {
        {
            lock(%{ $self->{state} });
            last if $self->{state}->{cancel};
        }

        my $path = $self->{queue}->dequeue_timed(0.5);
        next unless defined $path;

        my $ok = eval {
            $self->_process_file($path, $ua);
            1;
        };

        if (!$ok) {
            my $e = $@ // 'unknown error';
            if ($e =~ /__TAGGER_CANCEL__/) {
                print "[cancel] worker stopping\n";
                last;
            }
            warn "[error] worker died on $path: $e\n";
        }

        {
            lock(%{ $self->{state} });
            last if $self->{state}->{cancel};
        }
    }
}

sub _cleaner {
    my ($self) = @_;

    while (1) {
        sleep($DEBOUNCE_SECS * 2);

        {
            lock(%{ $self->{state} });
            last if $self->{state}->{stop_cleaner};
        }

        lock(%last_written);
        lock(%inflight);

        my $cutoff = time() - $DEBOUNCE_SECS;
        delete $last_written{$_}
            for grep { !$inflight{$_} && $last_written{$_} < $cutoff } keys %last_written;
    }
}

sub _process_file {
    my ($self, $path, $ua) = @_;

    my $phase = 'request';
    my $cancel_after_write = 0;

    local $SIG{USR1} = sub {
        if ($phase eq 'writing') {
            $cancel_after_write = 1;
            return;
        }
        die "__TAGGER_CANCEL__\n";
    };

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

    {
        lock(%{ $self->{state} });
        if ($self->{state}->{cancel}) {
            print "[cancel] abandoning $path after request\n";
            _clear_inflight($path);
            return;
        }
    }

    unless ($response->is_success) {
        warn "[error] Upload failed for $path: " . $response->status_line . "\n";
        warn "        " . $response->decoded_content . "\n" if $response->decoded_content;
        _clear_inflight($path);
        return;
    }

    my $data = eval { decode_json($response->decoded_content) };
    if ($@) {
        warn "[error] Could not parse JSON response for $path: $@\n";
        _clear_inflight($path);
        return;
    }

    my $description = $data->{description} // '';
    my @tags        = @{ $data->{tags} // [] };

    printf "[result] description: %s\n", $description || '(none)';
    printf "[result] tags (%d): %s\n", scalar(@tags), join(', ', @tags);

    {
        lock(%{ $self->{state} });
        if ($self->{state}->{cancel}) {
            print "[cancel] skipping metadata write for $path\n";
            _clear_inflight($path);
            return;
        }
    }

    my ($atime, $mtime) = (stat($path))[8, 9];

    my $exif = Image::ExifTool->new();
    $exif->Options(Lang => 'en', Preserve => 1);

    my $info     = $exif->ImageInfo($path, 'XMP:Subject');
    my $existing = $info->{'Subject'} // [];
    $existing    = [$existing] unless ref $existing eq 'ARRAY';
    my %existing_lc = map { lc($_) => 1 } @$existing;

    my @new_tags = grep { !$existing_lc{lc($_)} } @tags;

    printf "[result] %d new tag(s) to add (skipping %d already present)\n",
        scalar(@new_tags), scalar(@tags) - scalar(@new_tags);

    $exif->SetNewValue('XMP:Description', $description, { Lang => 'en' })
        if length $description;

    for my $tag (@new_tags) {
        $exif->SetNewValue('XMP:Subject', $tag, { AddValue => 1 });
    }

    $phase = 'writing';
    my ($ok, $err) = $exif->WriteInfo($path);
    utime($atime, $mtime, $path) if $ok;
    if ($ok) {
        print "[xmp] written to $path\n";
    } else {
        warn "[error] ExifTool failed to write $path: $err\n";
    }

    _clear_inflight($path);

    die "__TAGGER_CANCEL__\n" if $cancel_after_write;
}

sub _clear_inflight {
    my ($path) = @_;
    lock(%inflight);
    delete $inflight{$path};
}

1;
