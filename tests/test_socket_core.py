import os
import socket
import threading
import time
import pickle
import pytest

import doma.core as core
import doma.utils as utils


def test_get_socket_returns_unix_socket_and_path(monkeypatch, tmp_path):
    tmp_sock_path = os.path.join(tmp_path, "doma_test.sock")
    monkeypatch.setattr(core, "SOCKET_PATH", tmp_sock_path, raising=True)

    s, addr = core.get_socket()
    try:
        assert s.family == socket.AF_UNIX
        assert s.type == socket.SOCK_STREAM
        assert addr == tmp_sock_path
    finally:
        s.close()


def test_socket_timeout_sets_and_restores_timeout():
    class DummyConn:
        def __init__(self):
            self.calls = []

        def settimeout(self, value):
            self.calls.append(value)

    dummy = DummyConn()
    with core.socket_timeout(dummy, 1.23):
        pass
    assert dummy.calls == [1.23, None]


def test_send_and_recv_socket_data_roundtrip():
    s1, s2 = socket.socketpair()
    try:
        payload = core.SocketData(signal=core.Signal.GREETING)
        core.send_socket_data(s1, payload, timeout=0.5)
        received = core.recv_socket_data(s2, timeout=0.5)
        assert received.signal == core.Signal.GREETING
        assert received.config is None
    finally:
        s1.close()
        s2.close()


def test_recv_socket_data_handles_chunked_frames():
    s1, s2 = socket.socketpair()
    try:
        data = core.SocketData(signal=core.Signal.GREETING, config=None)
        data_bytes = pickle.dumps(data)
        # Send in two chunks, EOS only at the end
        s1.sendall(data_bytes[:10])
        time.sleep(0.01)
        s1.sendall(data_bytes[10:] + core.EOS)

        received = core.recv_socket_data(s2, timeout=0.5)
        assert received.signal == core.Signal.GREETING
        assert received.config is None
    finally:
        s1.close()
        s2.close()


def test_recv_socket_data_times_out_when_no_data():
    s1, s2 = socket.socketpair()
    try:
        # Do not send anything from s1; s2 should time out waiting for data
        with pytest.raises(socket.timeout):
            _ = core.recv_socket_data(s2, timeout=0.05)
    finally:
        s1.close()
        s2.close()


def test_exchange_socket_data_end_to_end(monkeypatch, tmp_path):
    server_address = os.path.join(tmp_path, "doma_test.sock")
    server_ready = threading.Event()

    def server():
        srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            if os.path.exists(server_address):
                os.remove(server_address)
            srv.bind(server_address)
            srv.listen(1)
            server_ready.set()
            conn, _ = srv.accept()
            with conn:
                msg = core.recv_socket_data(conn, timeout=0.5)
                assert msg.signal == core.Signal.GREETING
                core.send_socket_data(conn, core.SocketData(signal=core.Signal.GREETING), timeout=0.5)
        finally:
            srv.close()

    t = threading.Thread(target=server, daemon=True)
    t.start()
    server_ready.wait(timeout=1.0)

    def tmp_get_socket():
        return socket.socket(socket.AF_UNIX, socket.SOCK_STREAM), server_address

    monkeypatch.setattr(utils, "get_socket", tmp_get_socket, raising=True)

    result = utils.exchange_socket_data(core.SocketData(signal=core.Signal.GREETING), timeout=0.5)
    assert result.signal == core.Signal.GREETING

    t.join(timeout=1.0)

