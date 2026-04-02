package Tagger;
use strict;
use warnings;

use File::Basename qw(basename);
use JSON           qw(decode_json);
use Image::ExifTool;
use Mojo::IOLoop;
use Mojo::Promise;
use Mojo::UserAgent;

$| = 1;
binmode STDOUT, ':encoding(UTF-8)';
binmode STDERR, ':encoding(UTF-8)';

my $DEBOUNCE_SECS = 5;

sub new {
    my ($class, %args) = @_;

    my $max_inflight = $args{workers} // 32;
    my $timeout      = $args{timeout} // 5;

    my $ua = Mojo::UserAgent->new;
    $ua->connect_timeout($timeout);
    $ua->request_timeout($timeout);
    $ua->inactivity_timeout($timeout);
    $ua->max_connections($max_inflight);

    my $self = bless {
        endpoint      => $args{endpoint} // 'http://localhost:9100/analyse',
        max_inflight  => $max_inflight,
        ua            => $ua,
        pending       => [],
        active        => 0,
        cancel        => 0,
        draining      => 0,
        drain_promise => undef,
        inflight      => {},
        last_written  => {},
    }, $class;

    return $self;
}

sub enqueue {
    my ($self, $path) = @_;
    return unless $path =~ /\.avif$/i;
    return if $self->{cancel};

    $self->_prune_last_written;

    return if $self->{inflight}{$path};

    if (exists $self->{last_written}{$path}
        && (time() - $self->{last_written}{$path}) < $DEBOUNCE_SECS) {
        print "[skip] $path (debounce)\n";
        return;
    }

    $self->{last_written}{$path} = time();
    $self->{inflight}{$path}     = 1;

    push @{ $self->{pending} }, $path;
    printf "[queued] %s (pending: %d)\n", $path, scalar @{ $self->{pending} };

    $self->_pump;
}

sub drain {
    my ($self) = @_;

    $self->{draining} = 1;
    $self->_pump;

    return if !$self->{active} && !@{ $self->{pending} };

    $self->{drain_promise} ||= Mojo::Promise->new;
    $self->{drain_promise}->wait;
}

sub cancel {
    my ($self) = @_;
    return if $self->{cancel};

    $self->{cancel} = 1;

    my @dropped = @{ delete $self->{pending} || [] };
    delete $self->{inflight}{$_}     for @dropped;
    delete $self->{last_written}{$_} for @dropped;

    printf "[cancel] dropped %d queued job(s)\n", scalar(@dropped);

    if (!$self->{active}) {
        $self->_resolve_drain;
        return;
    }

    $self->{drain_promise} ||= Mojo::Promise->new;
    $self->{drain_promise}->wait;
}

sub _pump {
    my ($self) = @_;

    while (
        !$self->{cancel}
        && $self->{active} < $self->{max_inflight}
        && @{ $self->{pending} }
    ) {
        my $path = shift @{ $self->{pending} };
        $self->{active}++;

        $self->_process_file_p($path)
            ->catch(sub {
                my ($err) = @_;
                $err = 'unknown error' unless defined $err;
                chomp $err;
                warn "[error] worker died on $path: $err\n";
            })
            ->finally(sub {
                $self->{active}--;
                $self->_clear_inflight($path);
                $self->_pump;
                $self->_resolve_drain_if_idle;
            });
    }

    $self->_resolve_drain_if_idle;
}

sub _process_file_p {
    my ($self, $path) = @_;

    print "[upload] $path\n";

    return $self->{ua}->post_p(
        $self->{endpoint} => form => {
            image => {
                file           => $path,
                filename       => basename($path),
                'Content-Type' => 'image/avif',
            },
        }
    )->then(sub {
        my ($tx) = @_;

        if ($self->{cancel}) {
            print "[cancel] abandoning $path after request\n";
            return;
        }

        my $res = $tx->result;

        unless ($res->is_success) {
            my $status = defined $res->code ? $res->code : '(no status)';
            my $msg    = $res->message // '';
            warn "[error] Upload failed for $path: $status $msg\n";

            my $body = $res->body;
            warn "        $body\n" if defined $body && length $body;
            return;
        }

        my $data = eval { decode_json($res->body) };
        if ($@) {
            warn "[error] Could not parse JSON response for $path: $@\n";
            return;
        }

        my $description = $data->{description} // '';
        my @tags        = @{ $data->{tags} // [] };

        printf "[result] description: %s\n", $description || '(none)';
        printf "[result] tags (%d): %s\n", scalar(@tags), join(', ', @tags);

        if ($self->{cancel}) {
            print "[cancel] skipping metadata write for $path\n";
            return;
        }

        return $self->_write_metadata_p($path, $description, \@tags);
    });
}

sub _write_metadata_p {
    my ($self, $path, $description, $tags) = @_;

    return Mojo::IOLoop->subprocess->run_p(sub {
        return _write_metadata_sync($path, $description, $tags);
    })->then(sub {
        my ($result) = @_;

        my $ok             = $result->{ok}             // 0;
        my $err            = $result->{err}            // '';
        my $new_tags       = $result->{new_tags}       // 0;
        my $existing_count = $result->{existing_count} // 0;

        printf "[result] %d new tag(s) to add (skipping %d already present)\n",
            $new_tags, $existing_count;

        if ($ok) {
            print "[xmp] written to $path\n";
        } else {
            warn "[error] ExifTool failed to write $path: $err\n";
        }

        return;
    });
}

sub _write_metadata_sync {
    my ($path, $description, $tags) = @_;

    my ($atime, $mtime) = (stat($path))[8, 9];

    my $exif = Image::ExifTool->new();
    $exif->Options(Lang => 'en', Preserve => 1);

    my $info     = $exif->ImageInfo($path, 'XMP:Subject');
    my $existing = $info->{'Subject'} // [];
    $existing    = [$existing] unless ref $existing eq 'ARRAY';
    my %existing_lc = map { lc($_) => 1 } @$existing;

    my @new_tags = grep { !$existing_lc{lc($_)} } @$tags;

    $exif->SetNewValue('XMP:Description', $description, { Lang => 'en' })
        if length $description;

    for my $tag (@new_tags) {
        $exif->SetNewValue('XMP:Subject', $tag, { AddValue => 1 });
    }

    my ($ok, $err) = $exif->WriteInfo($path);
    utime($atime, $mtime, $path) if $ok;

    return {
        ok             => $ok ? 1 : 0,
        err            => $err // '',
        new_tags       => scalar(@new_tags),
        existing_count => scalar(@$tags) - scalar(@new_tags),
    };
}

sub _clear_inflight {
    my ($self, $path) = @_;
    delete $self->{inflight}{$path};
    $self->_prune_last_written;
}

sub _prune_last_written {
    my ($self) = @_;

    my $cutoff = time() - $DEBOUNCE_SECS;
    delete $self->{last_written}{$_}
        for grep {
            !$self->{inflight}{$_}
                && $self->{last_written}{$_} < $cutoff
        } keys %{ $self->{last_written} };
}

sub _resolve_drain_if_idle {
    my ($self) = @_;
    return unless $self->{draining} || $self->{cancel};
    return if $self->{active};
    return if @{ $self->{pending} };
    $self->_resolve_drain;
}

sub _resolve_drain {
    my ($self) = @_;

    $self->{draining} = 0;

    if (my $p = delete $self->{drain_promise}) {
        $p->resolve;
    }
}

1;
