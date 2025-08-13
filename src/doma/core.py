from contextlib import contextmanager
from enum import Enum
import os
import pickle
import socket
import sys
from typing import Callable, Optional

from pydantic import BaseModel, ConfigDict

from doma.configs import ControllerConfig

class Signal(Enum):
    START = 0
    STOP = 1
    RESTART = 2
    SHUTDOWN = 3
    GREETING = 4

class SocketData(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    signal: Signal
    config: Optional[ControllerConfig] = None
    error: Optional[Exception] = None

SOCKET_PATH = "/tmp/doma/doma.sock"
SOCKET_TIMEOUT = 5

def get_socket():
    server_address = SOCKET_PATH
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    return s, server_address

@contextmanager
def socket_timeout(conn:socket.socket, timeout:Optional[float]=None):
    if timeout:
        conn.settimeout(timeout)
    yield
    if timeout:
        conn.settimeout(None)

EOS = b"END_OF_SOCKET_DATA"

def recv_socket_data(conn:socket.socket, timeout:Optional[float]=None) -> SocketData:
    with socket_timeout(conn, timeout):
        bytes_buffer = b""
        while True:
            data = conn.recv(1024)
            bytes_buffer += data
            if EOS in bytes_buffer:
                break
        bytes_buffer = bytes_buffer.split(EOS)[0]
        data = pickle.loads(bytes_buffer)
    return data

def send_socket_data(conn:socket.socket, data:SocketData, timeout:Optional[float]=None):
    with socket_timeout(conn, timeout):
        conn.send(pickle.dumps(data) + EOS)

def daemonize(
    func: Callable, stdin="/dev/null", stdout="/dev/null", stderr="/dev/null"
):
    try:
        pid = os.fork()
        if pid > 0:
            return
    except OSError as e:
        sys.stderr.write("fork #1 failed: (%d) %s\n" % (e.errno, e.strerror))
        sys.exit(1)

    os.chdir("/")
    os.umask(0)
    os.setsid()

    try:
        pid = os.fork()
        if pid > 0:
            sys.exit(0)
    except OSError as e:
        sys.stderr.write("fork #2 failed: (%d) %s\n" % (e.errno, e.strerror))
        sys.exit(1)

    sys.stdout.flush()
    sys.stderr.flush()
    si = open(stdin, "r")
    so = open(stdout, "w")
    se = open(stderr, "w")
    os.dup2(si.fileno(), sys.stdin.fileno())
    os.dup2(so.fileno(), sys.stdout.fileno())
    os.dup2(se.fileno(), sys.stderr.fileno())

    func()
