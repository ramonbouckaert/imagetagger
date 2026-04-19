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

    my $max_inflight       = $args{workers}            // 5;
    my $connect_timeout    = $args{connect_timeout}    // 5;
    my $inactivity_timeout = $args{inactivity_timeout} // 300;
    my $request_timeout    = $args{request_timeout}    // 0;
    my $max_retries        = $args{max_retries}        // 6;
    my $on_failure         = $args{on_failure};

    my $ua = Mojo::UserAgent->new;
    $ua->connect_timeout($connect_timeout);
    $ua->inactivity_timeout($inactivity_timeout);
    $ua->request_timeout($request_timeout);
    $ua->max_connections($max_inflight);

    my $self = bless {
        endpoint         => $args{endpoint} // 'http://localhost:9100/analyse',
        max_inflight     => $max_inflight,
        max_retries      => $max_retries,
        on_failure       => $on_failure,
        ua               => $ua,
        pending          => [],
        active           => 0,
        cancel           => 0,
        draining         => 0,
        drain_promise    => undef,
        inflight         => {},
        last_written     => {},
        tx_by_path       => {},
        conn_id_by_path  => {},
        retry_wait_by_path => {},
        cancel_requested => {},
    }, $class;

    return $self;
}

sub _is_avif {
    my ($path) = @_;
    open(my $fh, '<:raw', $path) or return 0;
    my $header;
    my $n = read($fh, $header, 12);
    close($fh);
    return 0 unless defined $n && $n == 12;
    return substr($header, 4, 4) eq 'ftyp'
        && substr($header, 8, 4) =~ /\Aavi[fsc]\z/;
}

sub enqueue {
    my ($self, $path) = @_;
    return if $self->{cancel};
    return unless -f $path;
    unless (_is_avif($path)) {
        print "[skip] $path (not an AVIF)\n";
        return;
    }

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
    delete $self->{inflight}{$_}         for @dropped;
    delete $self->{last_written}{$_}     for @dropped;
    delete $self->{cancel_requested}{$_} for @dropped;

    my @active_http = keys %{ $self->{tx_by_path} };
    for my $path (@active_http) {
        $self->{cancel_requested}{$path} = 1;

        my $id = delete $self->{conn_id_by_path}{$path};
        Mojo::IOLoop->remove($id) if defined $id;
    }

    my @retry_waits = keys %{ $self->{retry_wait_by_path} };
    for my $path (@retry_waits) {
        $self->{cancel_requested}{$path} = 1;

        my $wait = delete $self->{retry_wait_by_path}{$path} || {};
        Mojo::IOLoop->remove($wait->{id}) if defined $wait->{id};
        $wait->{resolve}->() if $wait->{resolve};
    }

    printf "[cancel] dropped %d queued job(s), aborting %d active HTTP request(s), cancelling %d retry wait(s)\n",
        scalar(@dropped), scalar(@active_http), scalar(@retry_waits);

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
                warn "[error] request pipeline failed for $path: $err\n";
                $self->{on_failure}->($path) if $self->{on_failure};
            })
            ->finally(sub {
                $self->{active}--;
                $self->_clear_request_state($path);
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

    return $self->_request_with_retry_p($path, 0)
        ->then(sub {
            my ($res) = @_;

            return unless $res;

            if ($self->{cancel_requested}{$path} || $self->{cancel}) {
                print "[cancel] abandoning $path after request\n";
                return;
            }

            my $data = eval { decode_json($res->body) };
            if ($@) {
                die "Could not parse JSON response: $@";
            }

            my $description = $data->{description} // '';
            my @tags        = @{ $data->{tags} // [] };

            printf "[result] description: %s\n", $description || '(none)';
            printf "[result] tags (%d): %s\n", scalar(@tags), join(', ', @tags);

            if ($self->{cancel_requested}{$path} || $self->{cancel}) {
                print "[cancel] skipping metadata write for $path\n";
                return;
            }

            return $self->_write_metadata_p($path, $description, \@tags);
        });
}

sub _request_with_retry_p {
    my ($self, $path, $attempt) = @_;

    return Mojo::Promise->resolve
        ->then(sub {
            return if $self->{cancel_requested}{$path} || $self->{cancel};
            return $self->_start_request_p($path);
        })
        ->catch(sub {
            my ($err) = @_;

            return if $self->{cancel_requested}{$path} || $self->{cancel};

            die $err unless $self->_should_retry_error($err, $attempt);

            my $delay          = 3 ** $attempt;
            my $next_attempt   = $attempt + 2;
            my $total_attempts = $self->{max_retries} + 1;

            warn "[retry] request failed for $path, retrying in ${delay}s (attempt $next_attempt/$total_attempts)\n";

            return $self->_wait_before_retry_p($path, $delay)
                ->then(sub {
                    return if $self->{cancel_requested}{$path} || $self->{cancel};
                    return $self->_request_with_retry_p($path, $attempt + 1);
                });
        });
}

sub _start_request_p {
    my ($self, $path) = @_;

    my $tx = $self->{ua}->build_tx(
        POST => $self->{endpoint} => form => {
            image => {
                file           => $path,
                filename       => basename($path),
                'Content-Type' => 'image/avif',
            },
        }
    );

    $self->{tx_by_path}{$path} = $tx;

    $tx->on(connection => sub {
        my ($tx, $id) = @_;
        $self->{conn_id_by_path}{$path} = $id;

        if ($self->{cancel_requested}{$path}) {
            delete $self->{conn_id_by_path}{$path};
            Mojo::IOLoop->remove($id);
        }
    });

    return $self->{ua}->start_p($tx)
        ->then(sub {
            my ($tx) = @_;

            return if $self->{cancel_requested}{$path} || $self->{cancel};

            my $res = eval { $tx->result };
            if (!$res) {
                my $err = $@ || 'unknown transport error';
                die $err;
            }

            return $res if $res->is_success;

            my $code = $res->code;
            my $msg  = $res->message // '';
            my $body = $res->body;

            if (
                defined $code
                && (
                    $res->is_server_error
                    || $code == 408
                    || $code == 425
                    || $code == 429
                )
            ) {
                warn "[warn] Upload failed for $path: $code $msg\n";
                warn "       $body\n" if defined $body && length $body;
                die "__RETRY_HTTP__:$code:$msg";
            }

            my $status = defined $code ? $code : '(no status)';
            warn "[error] Upload failed for $path: $status $msg\n";
            warn "        $body\n" if defined $body && length $body;
            return;
        })
        ->catch(sub {
            my ($err) = @_;

            if ($self->{cancel_requested}{$path} || $self->{cancel}) {
                print "[cancel] request cancelled for $path\n";
                return;
            }

            die $err;
        })
        ->finally(sub {
            delete $self->{conn_id_by_path}{$path};
            delete $self->{tx_by_path}{$path};
        });
}

sub _should_retry_error {
    my ($self, $err, $attempt) = @_;

    return 0 if $attempt >= $self->{max_retries};
    return 0 if $self->{cancel};
    return 0 unless defined $err && length $err;

    return 1 if $err =~ /^__RETRY_HTTP__:/;

    return 1 if $err =~ /Request timeout/i;
    return 1 if $err =~ /Inactivity timeout/i;
    return 1 if $err =~ /Connect timeout/i;
    return 1 if $err =~ /Connection refused/i;
    return 1 if $err =~ /Connection reset/i;
    return 1 if $err =~ /Premature connection close/i;
    return 1 if $err =~ /Broken pipe/i;
    return 1 if $err =~ /Network is unreachable/i;
    return 1 if $err =~ /Temporary failure in name resolution/i;
    return 1 if $err =~ /Name or service not known/i;

    return 0;
}

sub _wait_before_retry_p {
    my ($self, $path, $delay) = @_;

    my $p     = Mojo::Promise->new;
    my $state = { done => 0 };

    my $resolve = sub {
        return if $state->{done}++;
        $p->resolve;
    };

    my $id = Mojo::IOLoop->timer($delay => sub {
        delete $self->{retry_wait_by_path}{$path};
        $resolve->();
    });

    $self->{retry_wait_by_path}{$path} = {
        id      => $id,
        resolve => $resolve,
    };

    return $p;
}

sub _write_metadata_p {
    my ($self, $path, $description, $tags, $attempt) = @_;
    $attempt //= 0;

    return Mojo::IOLoop->subprocess->run_p(sub {
        return _write_metadata_sync($path, $description, $tags);
    })->then(sub {
        my ($result) = @_;

        my $ok             = $result->{ok}             // 0;
        my $err            = $result->{err}            // '';
        my $new_tags       = $result->{new_tags}       // 0;
        my $existing_count = $result->{existing_count} // 0;

        if ($ok) {
            printf "[result] %d new tag(s) to add (skipping %d already present)\n",
                $new_tags, $existing_count;
            print "[xmp] written to $path\n";
            return;
        }

        return if $self->{cancel_requested}{$path} || $self->{cancel};

        if ($attempt < $self->{max_retries}) {
            my $delay        = 3 ** $attempt;
            my $next_attempt = $attempt + 2;
            my $total        = $self->{max_retries} + 1;
            warn "[retry] ExifTool write failed for $path ($err), retrying in ${delay}s (attempt $next_attempt/$total)\n";
            return $self->_wait_before_retry_p($path, $delay)
                ->then(sub {
                    return if $self->{cancel_requested}{$path} || $self->{cancel};
                    return $self->_write_metadata_p($path, $description, $tags, $attempt + 1);
                });
        }

        warn "[error] ExifTool failed to write $path: $err\n";
        $self->{on_failure}->($path) if $self->{on_failure};
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

    if (length $description) {
        $exif->SetNewValue('XMP-dc:Description-x-default', $description);
        $exif->SetNewValue('XMP-dc:Description-en', $description);
    }

    for my $tag (@new_tags) {
        $exif->SetNewValue('XMP:Subject', $tag, { AddValue => 1 });
    }

    my $status  = $exif->WriteInfo($path);
    my $err     = $exif->GetValue('Error')   // '';
    my $warning = $exif->GetValue('Warning') // '';

    utime($atime, $mtime, $path) if $status;

    return {
        ok             => ($status == 1 || $status == 2) ? 1 : 0,
        err            => $err // '',
        new_tags       => scalar(@new_tags),
        existing_count => scalar(@$tags) - scalar(@new_tags),
    };
}

sub _clear_request_state {
    my ($self, $path) = @_;

    delete $self->{tx_by_path}{$path};
    delete $self->{conn_id_by_path}{$path};

    if (my $wait = delete $self->{retry_wait_by_path}{$path}) {
        Mojo::IOLoop->remove($wait->{id}) if defined $wait->{id};
        $wait->{resolve}->() if $wait->{resolve};
    }

    delete $self->{cancel_requested}{$path};
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
