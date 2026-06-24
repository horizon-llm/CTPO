"""
Minimal Firejail-based sandbox HTTP API.
Dependencies:
    pip install "fastapi[all]" uvicorn
System prerequisites:
    * firejail installed (sudo apt install firejail)
    * a non-login user called `sandboxer` (sudo useradd -r -M -s /usr/sbin/nologin sandboxer)
    * a Firejail profile at /etc/firejail/sandbox.profile such as:
        # /etc/firejail/sandbox.profile
        env none                    # start with a clean env
        env keep PATH
        env keep LANG
        env keep PYTHONIOENCODING
        net none                    # disable networking
        private                     # private filesystem rooted at --private dir
        rlimit as 512M              # memory cap
        rlimit cpu 3                # cpu-time cap
        rlimit nproc 50             # process count cap
        caps.drop all               # drop all capabilities
        seccomp
        whitelist /usr/bin/python3
        include /etc/firejail/whitelist-common.inc

Run the service with:
    uvicorn sandbox_api:app --host 0.0.0.0 --port 8000 --workers 4

The API is compatible with the `RunCodeRequest` / `RunResult` schema used by
ByteIntl Seed-Sandbox. A simple curl test:
    curl -X POST http://127.0.0.1:8000/faas/sandbox/ \
         -H 'Content-Type: application/json' \
         -d '{"code":"print(2+2)","language":"python","compile_timeout":1,"run_timeout":3}'
"""

import asyncio
import os
import shutil
import tempfile
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ---------------- Pydantic models ----------------

class RunStatus(str, Enum):
    """Execution outcome."""
    success = "success"
    timeout = "timeout"
    runtime_error = "runtime_error"


class RunCodeRequest(BaseModel):
    """Incoming JSON body from the client."""

    code: str
    stdin: str = ""
    language: str = "python"
    compile_timeout: float = 1.0  # kept for sdk compatibility, unused here
    run_timeout: float = 3.0


class RunResult(BaseModel):
    """JSON response back to the client."""

    status: RunStatus
    run_result: dict
    created_at: datetime


# ---------------- Core runner ----------------

SANDBOX_BACKEND = os.getenv("SANDBOX_BACKEND", "firejail").strip().lower()


def _build_firejail_cmd(workdir: Path, timeout: float) -> List[str]:
    return [
        "firejail",
        "--quiet",
        "--profile=/etc/firejail/default.profile",
        f"--private={workdir}",
        "--net=none",              # disable network
        "--rlimit-as=2048m",       # 2GB max RAM per sandbox
        f"--rlimit-cpu={int(timeout) + 2}",
        "--rlimit-nproc=256",
        "--",
        "python3",
        "main.py",
    ]


def _build_nsjail_cmd(workdir: Path, timeout: float) -> List[str]:
    # In restricted Docker environments, net namespace creation often fails.
    # We disable netns clone so nsjail can still enforce uid/rlimit/time isolation.
    limit = max(1, int(timeout) + 1)
    return [
        "nsjail",
        "--quiet",
        "--mode",
        "o",
        "--cwd",
        str(workdir),
        "--time_limit",
        str(limit),
        "--max_cpus",
        "1",
        "--rlimit_as",
        "2147483648",  # 2GB
        "--rlimit_cpu",
        str(limit),
        "--rlimit_nproc",
        "256",
        "--disable_clone_newnet",
        "--",
        "python3",
        "main.py",
    ]


def _build_bwrap_cmd(workdir: Path, timeout: float) -> List[str]:
    # bubblewrap does namespace/filesystem isolation.
    # Network namespace isolation can fail in restricted Docker, so it is optional.
    unshare_net = os.getenv("BWRAP_UNSHARE_NET", "0") == "1"
    cmd = [
        "bwrap",
        "--die-with-parent",
        "--new-session",
        "--unshare-user",
        "--unshare-ipc",
        "--unshare-uts",
        "--dev",
        "/dev",
        "--tmpfs",
        "/tmp",
        "--ro-bind",
        "/usr",
        "/usr",
    ]
    for bind_path in ["/bin", "/lib", "/lib64", "/etc"]:
        if Path(bind_path).exists():
            cmd.extend(["--ro-bind", bind_path, bind_path])
    # Reuse host /proc in restricted containers where mounting proc is forbidden.
    if Path("/proc").exists():
        cmd.extend(["--ro-bind", "/proc", "/proc"])
    if unshare_net:
        cmd.append("--unshare-net")
    else:
        cmd.append("--share-net")
    cmd.extend(
        [
            "--bind",
            str(workdir),
            "/workspace",
            "--chdir",
            "/workspace",
            "--setenv",
            "HOME",
            "/tmp",
            "--setenv",
            "PATH",
            "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--",
            "python3",
            "main.py",
        ]
    )
    return cmd


def _build_subprocess_cmd(workdir: Path, timeout: float) -> List[str]:
    """No isolation: just python3 main.py. For HPC where apt/firejail are unavailable.
    SECURITY: Malicious code can use full process privileges (FS, network, fork bomb, etc.).
    Use only inside a container or batch job so the blast radius is the container/job, not the host.
    Uses sys.executable so the sandbox inherits the same conda/venv Python (and its packages)
    as the uvicorn server process, instead of falling back to the bare system python3."""
    import sys
    return [sys.executable, "main.py"]


def _build_backend_cmd(workdir: Path, timeout: float) -> List[str]:
    if SANDBOX_BACKEND == "firejail":
        return _build_firejail_cmd(workdir, timeout)
    if SANDBOX_BACKEND == "nsjail":
        return _build_nsjail_cmd(workdir, timeout)
    if SANDBOX_BACKEND == "bwrap":
        return _build_bwrap_cmd(workdir, timeout)
    if SANDBOX_BACKEND == "subprocess":
        return _build_subprocess_cmd(workdir, timeout)
    raise RuntimeError(
        f"Unsupported SANDBOX_BACKEND='{SANDBOX_BACKEND}'. Use 'firejail', 'nsjail', 'bwrap', or 'subprocess'."
    )


def _sandbox_workdir_root() -> str:
    """Prefer /dev/shm for speed; on HPC (e.g. NCSA Delta) it may be missing, use system tmp."""
    return "/dev/shm" if Path("/dev/shm").is_dir() else tempfile.gettempdir()


async def _run_in_sandbox(code: str, timeout: float, stdin_data: str = "") -> dict:
    """Execute *code* inside an isolated backend and return stdout/stderr."""

    # 1) Write user code to a temp directory (/dev/shm if available, else system tmp)
    workdir = Path(tempfile.mkdtemp(prefix="sb_", dir=_sandbox_workdir_root()))
    src = workdir / "main.py"
    src.write_text(code)

    # 2) Build backend command line
    try:
        cmd = _build_backend_cmd(workdir, timeout)
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        return {
            "status": RunStatus.runtime_error,
            "stdout": "",
            "stderr": f"Sandbox backend config error: {e}\n",
        }

    # 3) Strip environment to stay below Firejail's MAX_ENVS=256 limit
    whitelist = ("PATH", "LANG", "LC_ALL", "PYTHONIOENCODING", "TERM", "LD_LIBRARY_PATH", "PYTHONPATH")
    clean_env = {k: os.environ[k] for k in whitelist if k in os.environ}

    # 4) Launch subprocess under asyncio, enforce wall-clock timeout
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=workdir,
            env=clean_env,
        )
    except Exception as e:
        shutil.rmtree(workdir, ignore_errors=True)
        return {
            "status": RunStatus.runtime_error,
            "stdout": "",
            "stderr": f"Failed to start sandbox backend '{SANDBOX_BACKEND}': {e}\n",
        }

    try:
        input_bytes = (stdin_data + "\n").encode() if len(stdin_data) > 0 else None
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(input=input_bytes),
            timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        shutil.rmtree(workdir, ignore_errors=True)
        return {
            "status": RunStatus.timeout,
            "stdout": "",
            "stderr": "Timeout\n",
        }

    status = RunStatus.success if proc.returncode == 0 else RunStatus.runtime_error

    # 5) Clean up tmpfs directory
    shutil.rmtree(workdir, ignore_errors=True)

    return {
        "status": status,
        "stdout": stdout.decode(),
        "stderr": stderr.decode(),
    }


# ---------------- FastAPI wiring ----------------

app = FastAPI()
POOL = asyncio.Semaphore(46)  # 20 per worker * 8 workers = 160 total concurrent sandboxes


@app.post("/faas/sandbox/", response_model=RunResult)
async def run_code(req: RunCodeRequest):
    """HTTP endpoint: compatible with the Seed-Sandbox client SDK."""

    if req.language != "python":
        raise HTTPException(400, "Only Python is supported in this minimal demo.")

    async with POOL:
        result = await _run_in_sandbox(req.code, req.run_timeout, req.stdin)

    return RunResult(status=result["status"], run_result=result, created_at=datetime.utcnow())
