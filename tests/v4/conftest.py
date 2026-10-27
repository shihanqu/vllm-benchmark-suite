import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def fake_gpu(tmp_path, monkeypatch):
    """Put a fake nvidia-smi on PATH and hide NVML so telemetry uses it."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    script = bin_dir / "nvidia-smi"
    script.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{HERE / "fake_nvidia_smi.py"}" "$@"\n')
    script.chmod(0o755)
    state = tmp_path / "gpu_state.json"
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setenv("MOCK_GPU_STATE", str(state))
    monkeypatch.setitem(sys.modules, "pynvml", None)  # force the nvidia-smi fallback
    return state


def _start_mock(state_path, *extra):
    port = _free_port()
    env = dict(os.environ, MOCK_GPU_STATE=str(state_path))
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "mock_server.py"), "--port", str(port), "--speed", "40", *extra],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    url = f"http://127.0.0.1:{port}"
    for _ in range(100):
        try:
            if requests.get(f"{url}/v1/models", timeout=1).status_code == 200:
                return proc, url
        except Exception:
            time.sleep(0.1)
    proc.kill()
    raise RuntimeError("mock server did not start")


@pytest.fixture
def mock_server(fake_gpu):
    proc, url = _start_mock(fake_gpu)
    yield url
    proc.kill()
    proc.wait()


@pytest.fixture
def mock_server_factory(fake_gpu):
    procs = []

    def start(*extra):
        proc, url = _start_mock(fake_gpu, *extra)
        procs.append(proc)
        return proc, url

    yield start
    for p in procs:
        p.kill()
        p.wait()
