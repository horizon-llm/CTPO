#!/usr/bin/env bash
set -euo pipefail
set -x

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_PATH="$PROJECT_DIR/examples/sglang_multiturn/config"
TOOL_CONFIG_PATH="${TOOL_CONFIG_PATH:-$PROJECT_DIR/examples/sglang_multiturn/config/tool_config/local_sandbox_tool_config.yaml}"

# Local sandbox endpoint used by LocalSandboxTool
export LOCAL_SANDBOX_URL="${LOCAL_SANDBOX_URL:-http://127.0.0.1:12345/faas/sandbox/}"
# Force vLLM engine mode to avoid V1/V0 env mismatch.
export VLLM_USE_V1=1
# Use SimpleTIR-style fallback tool-call extraction (code-block -> code_interpreter).
export VERL_HERMES_FALLBACK="${VERL_HERMES_FALLBACK:-1}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"
export WANDB_MODE="${WANDB_MODE:-online}"
export PIP_DISABLE_PIP_VERSION_CHECK=1
python3 -m pip install -q -U wandb
python3 -m pip install -q math-verify word2number
python3 -m pip install -q "fastapi[all]" uvicorn
if [[ "${WANDB_MODE}" != "offline" && -n "${WANDB_API_KEY}" ]]; then
  python3 -c "import wandb; wandb.login(key='${WANDB_API_KEY}', relogin=True)"
fi

# deepscaler, rstar2, dapo
TRAIN_FILES="${TRAIN_FILES:-[\"${PROJECT_DIR}/dataset/deepscaler/train.parquet\"]}"
VAL_FILES="${VAL_FILES:-[\"${PROJECT_DIR}/dataset/aime/aime25.parquet\",\"${PROJECT_DIR}/dataset/competition_benchmarks/aime26.parquet\"]}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-32}"
PPO_MICRO_BATCH_SIZE="${PPO_MICRO_BATCH_SIZE:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_TURNS="${MAX_TURNS:-5}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-16000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"
ROLLOUT_GPU_UTIL="${ROLLOUT_GPU_UTIL:-0.6}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_UPDATE_BUCKET_MB="${ROLLOUT_UPDATE_BUCKET_MB:-4096}"
NGPUS="${NGPUS:-8}"
NNODES="${NNODES:-1}"
TEST_FREQ="${TEST_FREQ:-20}"
TOTAL_STEPS="${TOTAL_STEPS:-305}"
SAVE_FREQ="${SAVE_FREQ:-20}"
N_VAL="${N_VAL:-32}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
ENABLE_THINKING="${ENABLE_THINKING:-False}"
ROLLOUT_IS_LEVEL="${ROLLOUT_IS_LEVEL:-sequence}"
ROLLOUT_IS_THRESHOLD="${ROLLOUT_IS_THRESHOLD:-5.0}"
ROLLOUT_IS_BATCH_NORM="${ROLLOUT_IS_BATCH_NORM:-False}"
ROLLOUT_RS="${ROLLOUT_RS:-}"
ROLLOUT_RS_THRESHOLD="${ROLLOUT_RS_THRESHOLD:-5}"
MASK_VOID_TURNS="${MASK_VOID_TURNS:-False}"
CTPO_RATIO_MODE="${CTPO_RATIO_MODE:-cumprod}" # geomean | cumprod
# CTPO-LA log-space clip base values.  All three scale with t^LA_POWER.
# Effective ratio window at position t:
#   [exp(-LA_CLIP_LOW · t^p),  exp(LA_CLIP_HIGH · t^p)]
# Dual-clip threshold: exp(LA_CLIP_C · t^p)
# At t=1: window ≈ [0.97, 1.05], t=100: ≈ [0.74, 1.65] (la_power=0.5)
LA_CLIP_LOW="${LA_CLIP_LOW:-0.025}"
LA_CLIP_HIGH="${LA_CLIP_HIGH:-0.05}"
LA_CLIP_C="${LA_CLIP_C:-0.05}"
# Position exponent: 0.5=sqrt (cumprod), -0.5=inv-sqrt (geomean), 0=uniform
CLIP_RATIO_LA_POWER="${CLIP_RATIO_LA_POWER:-0.5}"
PROJECT_NAME="${PROJECT_NAME:-TIR}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-14B}"
MODEL_NAME_FOR_EXP="${MODEL_PATH##*/}"
TRAIN_DATASET_TAG="$(
  echo "${TRAIN_FILES}" \
  | sed -E "s#${PROJECT_DIR}/dataset/##g; s#\\.parquet##g" \
  | tr -d "[]'\" " \
  | tr '/' '-' \
  | tr ',' '+'
)"
TRAIN_DATASET_TAG="${TRAIN_DATASET_TAG:-unknown_dataset}"
if [[ "${CTPO_RATIO_MODE}" == "cumprod" ]]; then
  CTPO_LOSS_MODE="cum-token-cumprod-la"
else
  CTPO_LOSS_MODE="cum-token-geomean-la"
fi

# Build a short tag describing the rollout correction mode for the experiment name.
if [[ -n "${ROLLOUT_RS}" ]]; then
  ROLLOUT_CORR_TAG="rs${ROLLOUT_RS}_th${ROLLOUT_RS_THRESHOLD}"
else
  ROLLOUT_CORR_TAG="is${ROLLOUT_IS_LEVEL}_th${ROLLOUT_IS_THRESHOLD}"
fi

EXP_NAME="${EXP_NAME:-ctpola_${CTPO_RATIO_MODE}_${MODEL_NAME_FOR_EXP}_${TRAIN_DATASET_TAG}_bs${TRAIN_BATCH_SIZE}_mbs${PPO_MINI_BATCH_SIZE}_n${ROLLOUT_N}_p${MAX_PROMPT_LENGTH}_r${MAX_RESPONSE_LENGTH}_cll${LA_CLIP_LOW}_clh${LA_CLIP_HIGH}_clc${LA_CLIP_C}_lap${CLIP_RATIO_LA_POWER}_${ROLLOUT_CORR_TAG}_maskvoid${MASK_VOID_TURNS}_save${SAVE_FREQ}}"

THINKING_ARGS=()
if [[ "${MODEL_PATH}" == *"Qwen3"* ]]; then
  THINKING_ARGS+=(+data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING}")
fi

# Build rollout correction args: either hard rejection sampling or soft IS weights (mutually exclusive).
ROLLOUT_CORR_ARGS=()
if [[ -n "${ROLLOUT_RS}" ]]; then
  ROLLOUT_CORR_ARGS+=(
    algorithm.rollout_correction.rollout_rs="${ROLLOUT_RS}"
    algorithm.rollout_correction.rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD}"
  )
  echo "[rollout_corr] mode=hard-reject  rollout_rs=${ROLLOUT_RS}  threshold=${ROLLOUT_RS_THRESHOLD}  mask_void_turns=${MASK_VOID_TURNS}"
else
  ROLLOUT_CORR_ARGS+=(
    algorithm.rollout_correction.rollout_is="${ROLLOUT_IS_LEVEL}"
    algorithm.rollout_correction.rollout_is_threshold="${ROLLOUT_IS_THRESHOLD}"
    algorithm.rollout_correction.rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORM}"
  )
  echo "[rollout_corr] mode=soft-IS  rollout_is=${ROLLOUT_IS_LEVEL}  threshold=${ROLLOUT_IS_THRESHOLD}  batch_norm=${ROLLOUT_IS_BATCH_NORM}  mask_void_turns=${MASK_VOID_TURNS}"
fi

LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
CKPT_ROOT_DIR="${CKPT_ROOT_DIR:-$PROJECT_DIR/checkpoints}"
mkdir -p "${LOG_DIR}"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}_${RUN_TS}.log}"
echo "[log] writing to ${LOG_FILE}"
SANDBOX_PREFLIGHT="${SANDBOX_PREFLIGHT:-1}"
if [[ "${SANDBOX_PREFLIGHT}" == "1" ]]; then
  echo "[sandbox] detecting isolation backend (firejail > bwrap > subprocess)..."
  if command -v apt-get &>/dev/null; then
    apt-get update -qq && apt-get install -y -qq firejail bubblewrap &>/dev/null || true
  fi
  if command -v firejail &>/dev/null && firejail --quiet --net=none -- python3 -c "print('ok')" &>/dev/null 2>&1; then
    echo "[sandbox] backend=firejail"
  elif command -v bwrap &>/dev/null && bwrap --ro-bind /usr /usr --dev /dev --proc /proc --tmpfs /tmp -- python3 -c "print('ok')" &>/dev/null 2>&1; then
    export SANDBOX_BACKEND="${SANDBOX_BACKEND:-bwrap}"
    echo "[sandbox] backend=bwrap (firejail unavailable/broken)"
  else
    export SANDBOX_BACKEND="${SANDBOX_BACKEND:-subprocess}"
    echo "[sandbox] WARN: backend=subprocess (no isolation; firejail and bwrap both unavailable)"
  fi

  echo "[sandbox] stopping old sandbox..."
  pkill -f "uvicorn sandbox_api:app" 2>/dev/null || true
  sleep 1

  echo "[sandbox] starting sandbox..."
  _SANDBOX_LOG="${PROJECT_DIR}/sandbox/sandbox_firejail.log"
  cd "${PROJECT_DIR}/sandbox"
  SANDBOX_BACKEND="${SANDBOX_BACKEND:-firejail}" nohup python3 -m uvicorn sandbox_api:app \
    --host 127.0.0.1 --port 12345 --workers 4 > "${_SANDBOX_LOG}" 2>&1 &
  _SANDBOX_PID=$!
  echo "[sandbox] pid=${_SANDBOX_PID}  log=${_SANDBOX_LOG}"
  cd "${PROJECT_DIR}"

  echo "[sandbox] waiting 120s for uvicorn workers to start..."
  sleep 120

  echo "[sandbox] warming up firejail (first request initializes worker state)..."
  python3 -c "
import urllib.request, json
payload = json.dumps({'code':'print(1)','language':'python','compile_timeout':1.0,'run_timeout':10.0}).encode()
req = urllib.request.Request('${LOCAL_SANDBOX_URL}', data=payload, headers={'Content-Type':'application/json'}, method='POST')
try:
    urllib.request.urlopen(req, timeout=15)
except Exception as e:
    print(f'[warmup] {e}')
" 2>/dev/null || true
  sleep 2

  echo "[sandbox] running smoke test (all 12 must pass)..."
  if ! LOCAL_SANDBOX_URL="${LOCAL_SANDBOX_URL}" python3 "$PROJECT_DIR/sandbox_smoke_test.py" \
      --url "${LOCAL_SANDBOX_URL}" --timeout 20; then
    echo "[sandbox] ERROR: smoke test did not pass all 12 cases — aborting training."
    exit 1
  fi
  echo "[sandbox] all smoke tests passed, proceeding to training."
fi

python3 -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='gsm8k_multiturn_grpo' \
  algorithm.adv_estimator=grpo \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  "${THINKING_ARGS[@]}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.policy_loss.loss_mode="${CTPO_LOSS_MODE}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.actor.clip_ratio_la_power="${CLIP_RATIO_LA_POWER}" \
  actor_rollout_ref.actor.la_clip_low="${LA_CLIP_LOW}" \
  actor_rollout_ref.actor.la_clip_high="${LA_CLIP_HIGH}" \
  actor_rollout_ref.actor.la_clip_c="${LA_CLIP_C}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="${ROLLOUT_UPDATE_BUCKET_MB}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_UTIL}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.rollout.val_kwargs.n="${N_VAL}" \
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
  actor_rollout_ref.rollout.calculate_log_probs=True \
  actor_rollout_ref.rollout.multi_turn.enable=True \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.max_user_turns="${MAX_TURNS}" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="${TOOL_CONFIG_PATH}" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  "${ROLLOUT_CORR_ARGS[@]}" \
  +actor_rollout_ref.actor.mask_void_turns="${MASK_VOID_TURNS}" \
  trainer.critic_warmup=0 \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXP_NAME}" \
  trainer.default_local_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXP_NAME}" \
  trainer.n_gpus_per_node="${NGPUS}" \
  trainer.nnodes="${NNODES}" \
  trainer.save_freq="${SAVE_FREQ}" \
  trainer.test_freq="${TEST_FREQ}" \
  trainer.total_training_steps="${TOTAL_STEPS}" \
  trainer.val_before_train=False \
  trainer.logger='["console","wandb"]' \
  "$@" 2>&1 | tee "${LOG_FILE}"
