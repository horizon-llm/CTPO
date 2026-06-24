# Sandbox 安全检测说明

## 当前架构

- **LocalSandboxTool**（训练/rollout 侧）：在把代码发给 sandbox API 之前做**正则 + 配置**检查，配置里 `block_dangerous_code: true`、`safety_mode: "strict"`。
- **sandbox_api.py**（HTTP 服务侧）：**不做**代码内容过滤，只做进程隔离（firejail/nsjail/bwrap）或裸跑（subprocess）。  
  训练时所有请求都经过 LocalSandboxTool，所以实际生效的是「Client 检测 + 服务端隔离」。

## 做得好的地方

1. **两层防护**：Client 拦截明显危险模式，Server 用 firejail 做隔离（有 firejail 时）。
2. **配置清晰**：`local_sandbox_tool_config.yaml` 里 `block_dangerous_code: true`、`safety_mode: "strict"`、`strict_max_timeout: 45`，默认就是严格模式。
3. **relaxed 覆盖**：rm -rf、shutil 删/移、os 删/重命名、pathlib 删/重命名、subprocess、os.system、常见网络 import、curl/wget/nc、`while True`、pip/apt install。
4. **strict 额外**：eval/exec/__import__、pickle/marshal.loads、ctypes/cffi、open(...,'w'等)、pathlib 写、multiprocessing/threading、os.environ。
5. **超时与长度**：strict 下 timeout 被 cap 到 45s，全局 timeout 上限 300；代码长度 20000 字符。
6. **语言白名单**：只允许 `allowed_languages`（默认 python）。

## 不足与可绕过点

1. **正则可被绕过**  
   - 例如：`getattr(os,'system')('rm -rf /')`、`exec("import os\nos.system('...')")`、用 `chr()` 拼出 `exec` 等，当前正则不会命中。  
   - 因此：**真正的安全边界是 firejail（或 nsjail/bwrap）**，Client 检测是「尽量拦住明显恶意」，不能替代隔离。

2. **服务端无过滤**  
   - 若有人直接调 `http://host:12345/faas/sandbox/` 且未经过 LocalSandboxTool，会执行任意代码。  
   - 当前用法（localhost、仅训练用）风险可控；若将来端口暴露，建议在 sandbox_api 里加一层与 client 同源的最小 blocklist（defense in depth）。

3. **漏掉的常见危险 API**  
   - `os.popen`：可执行 shell 命令，建议加入 relaxed 拦截。

4. **open 写文件**  
   - 当前只匹配 `open(..., 'w'|'a'|...)` 字面量；`mode='w'; open(path, mode)` 不匹配。要完全防写文件很难，strict 已明显提高门槛。

## 建议（已做 / 可选）

- **已做**：在 `LocalSandboxTool` 的 relaxed 里增加对 `os.popen` 的拦截（见下）。  
- **可选**：若将来 sandbox 端口对外，在 `sandbox_api.py` 的 `run_code` 里对 `req.code` 做一遍与 client 相同的 relaxed 规则，作为第二道防线。

---

*Review 日期：按代码库当前版本。*
