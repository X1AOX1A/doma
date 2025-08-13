import builtins
import os
import sys

import pytest

import doma.core as core


class DummyStream:
    def __init__(self, fd: int):
        self._fd = fd
        self.flushed = False

    def fileno(self):
        return self._fd

    def flush(self):
        self.flushed = True


class DummyFile:
    def __init__(self, fd: int):
        self._fd = fd

    def fileno(self):
        return self._fd


def test_daemonize_success_redirects_and_runs(monkeypatch):
    calls = {"fork": 0, "chdir": 0, "umask": [], "setsid": 0, "open": [], "dup2": []}

    def fake_fork():
        calls["fork"] += 1
        # Simulate child process both times so sys.exit is never called
        return 0

    def fake_setsid():
        calls["setsid"] += 1

    def fake_chdir(path):
        calls["chdir"] += 1
        assert path == "/"

    def fake_umask(mask):
        calls["umask"].append(mask)

    def fake_open(path, mode="r", *args, **kwargs):
        calls["open"].append((path, mode, args, kwargs))
        if path == "in":
            return DummyFile(100)
        if path == "out":
            return DummyFile(101)
        if path == "err":
            return DummyFile(102)
        return DummyFile(200)

    def fake_dup2(src, dest):
        calls["dup2"].append((src, dest))

    # Patch OS-level calls
    monkeypatch.setattr(os, "fork", fake_fork, raising=True)
    monkeypatch.setattr(os, "setsid", fake_setsid, raising=True)
    monkeypatch.setattr(os, "chdir", fake_chdir, raising=True)
    monkeypatch.setattr(os, "umask", fake_umask, raising=True)
    monkeypatch.setattr(os, "dup2", fake_dup2, raising=True)

    # Patch open
    monkeypatch.setattr(builtins, "open", fake_open, raising=True)

    # Provide controllable stdio streams
    stdin = DummyStream(10)
    stdout = DummyStream(20)
    stderr = DummyStream(30)
    monkeypatch.setattr(sys, "stdin", stdin, raising=False)
    monkeypatch.setattr(sys, "stdout", stdout, raising=False)
    monkeypatch.setattr(sys, "stderr", stderr, raising=False)

    ran = {"ok": False}

    def target():
        ran["ok"] = True

    # Use custom paths to assert open modes/targets
    core.daemonize(target, stdin="in", stdout="out", stderr="err")

    # Validations
    assert calls["fork"] == 2
    assert calls["setsid"] == 1
    assert calls["chdir"] == 1
    assert calls["umask"] == [0]

    # open called for each stream with expected modes
    assert ("in", "r", (), {}) in calls["open"]
    assert ("out", "w", (), {}) in calls["open"]
    assert ("err", "w", (), {}) in calls["open"]

    # dup2 should map opened fds to stdio fds
    assert (100, 10) in calls["dup2"]  # stdin
    assert (101, 20) in calls["dup2"]  # stdout
    assert (102, 30) in calls["dup2"]  # stderr

    # stdout/stderr were flushed and target executed
    assert stdout.flushed is True
    assert stderr.flushed is True
    assert ran["ok"] is True


def test_daemonize_first_fork_failure_exits_and_writes_stderr(monkeypatch, capsys):
    def boom_fork():
        raise OSError(12, "fail")

    monkeypatch.setattr(os, "fork", boom_fork, raising=True)

    with pytest.raises(SystemExit) as ei:
        core.daemonize(lambda: None)

    assert ei.value.code == 1
    captured = capsys.readouterr()
    assert "fork #1 failed: (12) fail" in captured.err


def test_daemonize_second_fork_failure_exits_and_writes_stderr(monkeypatch, capsys):
    calls = {"fork_calls": 0}

    def fork_then_fail():
        calls["fork_calls"] += 1
        if calls["fork_calls"] == 1:
            return 0  # child from first fork
        raise OSError(34, "boom")

    # Pre-second-fork setup calls must succeed
    monkeypatch.setattr(os, "fork", fork_then_fail, raising=True)
    monkeypatch.setattr(os, "setsid", lambda: None, raising=True)
    monkeypatch.setattr(os, "chdir", lambda p: None, raising=True)
    monkeypatch.setattr(os, "umask", lambda m: None, raising=True)

    with pytest.raises(SystemExit) as ei:
        core.daemonize(lambda: None)

    assert ei.value.code == 1
    captured = capsys.readouterr()
    assert "fork #2 failed: (34) boom" in captured.err


def test_daemonize_redirects_stdout_stderr_to_real_files(monkeypatch, tmp_path):
    out_path = tmp_path / "daemon_out.txt"
    err_path = tmp_path / "daemon_err.bin"

    # Ensure we always execute child branch and avoid exiting
    monkeypatch.setattr(os, "fork", lambda: 0, raising=True)
    monkeypatch.setattr(os, "setsid", lambda: None, raising=True)
    monkeypatch.setattr(os, "chdir", lambda p: None, raising=True)
    monkeypatch.setattr(os, "umask", lambda m: None, raising=True)

    # Provide stdout/stderr objects with real fds that won't interfere with pytest's
    pre_out = open(tmp_path / "pre_out.txt", "w")
    pre_err = open(tmp_path / "pre_err.txt", "w")
    pre_in = open(tmp_path / "pre_in.txt", "w")
    monkeypatch.setattr(sys, "stdout", pre_out, raising=False)
    monkeypatch.setattr(sys, "stderr", pre_err, raising=False)
    monkeypatch.setattr(sys, "stdin", pre_in, raising=False)
    def target():
        print("hello-out")
        print("hello-err", file=sys.stderr)
        sys.stdout.flush()
        sys.stderr.flush()

    core.daemonize(target, stdin="/dev/null", stdout=str(out_path), stderr=str(err_path))

    # Close our pre streams to flush everything
    pre_out.close()
    pre_err.close()

    # Validate contents in the real files we redirected to
    assert out_path.exists()
    assert err_path.exists()

    assert "hello-out" in out_path.read_text()
    assert "hello-err" in err_path.read_text()


