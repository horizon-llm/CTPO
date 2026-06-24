# Local sandbox (agentic 自带)

本目录为 agentic 自带的 code execution sandbox，与 `LocalSandboxTool` / `sandbox_smoke_test.py` 配套使用，**无需依赖 simpletir 仓库**。

## 依赖

**完整环境（训练 + sandbox）**，按顺序执行即可：

```bash
# Python 包
python3 -m pip install -U wandb
pip install math-verify
pip install word2number
# Sandbox API 必需（否则 uvicorn 起不来）
pip install "fastapi[all]" uvicorn

# 系统包
apt-get update
apt-get install -y tmux
apt-get install -y firejail
which firejail
firejail --version
```

可选后端：`nsjail`、`bubblewrap`（见下）。

## 启动

在 **agentic 项目根目录** 可执行：

```bash
./start_sandbox.sh
```

或在本目录下手动启动：

```bash
cd agentic/sandbox
SANDBOX_BACKEND=firejail nohup python3 -m uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4 > sandbox_firejail.log 2>&1 &
sleep 2
tail -n 120 sandbox_firejail.log
```

使用 nsjail：

```bash
cd agentic/sandbox
SANDBOX_BACKEND=nsjail nohup python3 -m uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4 &
```

使用 bubblewrap：

```bash
cd agentic/sandbox
SANDBOX_BACKEND=bwrap nohup python3 -m uvicorn sandbox_api:app --host 127.0.0.1 --port 12345 --workers 4 &
```

## 测试

在 agentic 根目录：

```bash
LOCAL_SANDBOX_URL="http://127.0.0.1:12345/faas/sandbox/" python3 sandbox_smoke_test.py
```

或 curl：

```bash
curl -X POST http://127.0.0.1:12345/faas/sandbox/ -H 'Content-Type: application/json' \
  -d '{"code":"print(1+1)","language":"python","compile_timeout":1.0,"run_timeout":3.0}'
```

## 说明

- 默认后端：`SANDBOX_BACKEND=firejail`，可通过环境变量切换为 `nsjail` / `bwrap`。
- **无 firejail 时**（如 NCSA Delta）：脚本会自动用 `subprocess` 后端，无进程/网络隔离，**仅适合在容器或 batch job 里跑**，崩也只崩容器/任务，不波及宿主机。
- API 与 ByteIntl Seed-Sandbox 的 `RunCodeRequest` / `RunResult` 兼容，`ctpo_la.sh` 等脚本中的 `LOCAL_SANDBOX_URL` 指向 `http://127.0.0.1:12345/faas/sandbox/` 即可。
