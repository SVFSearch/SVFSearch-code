#!/usr/bin/env bash
# run_all.sh — 顺序跑多个模型：部署 vllm → 跑 agent/workflow → 关闭 vllm → 下一个
#
# 用法：
#   bash run_benchmark.sh                  # 直接运行
#   DRY_RUN=1 bash run_benchmark.sh        # 只打印命令，不实际执行
#   RUNNER=workflow bash run_benchmark.sh  # 使用 run_workflow.py 代替 run_agent.py

set -euo pipefail

ROOT_DIR="./"
cd "${ROOT_DIR}"

# ── 全局配置 ──────────────────────────────────────────────────────────────────
PYTHON_BIN="${PYTHON_BIN:-python}"
VLLM_BIN="${VLLM_BIN:-vllm}"

# run_agent.py 或 run_workflow.py
RUNNER="${RUNNER:-agent}"   # "agent" | "workflow"

# 输入数据
INPUT_PATH="${INPUT_PATH:-query2QA.jsonl}"

# ANN 服务地址（提前启动好，不在本脚本里管理）
TEXT_ANN_URL="${TEXT_ANN_URL:-http://127.0.0.1:8004/text_ann}"
IMG_ANN_URL="${IMG_ANN_URL:-http://127.0.0.1:8001/img_ann}"
MULTIMODAL_ANN_URL="${MULTIMODAL_ANN_URL:-http://127.0.0.1:8003/multimodal_ann}"
BM25_ANN_URL="${BM25_ANN_URL:-http://127.0.0.1:8005/bm25_ann}"

# vllm 服务参数
VLLM_HOST="${VLLM_HOST:-0.0.0.0}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"   # 秒：等待 vllm 就绪的最长时间
VLLM_SHUTDOWN_TIMEOUT="${VLLM_SHUTDOWN_TIMEOUT:-30}"   # 秒：等待 vllm 优雅退出的最长时间

# agent 参数（可被每个模型条目覆盖）
ANN_TOPK="${ANN_TOPK:-5}"
ANN_TIMEOUT="${ANN_TIMEOUT:-200}"
MAX_PLAN_ROUNDS="${MAX_PLAN_ROUNDS:-6}"
MAX_EVIDENCE_ITEMS="${MAX_EVIDENCE_ITEMS:-6}"
ANSWER_MAX_ATTEMPTS="${ANSWER_MAX_ATTEMPTS:-6}"
LLM_TIMEOUT="${LLM_TIMEOUT:-5000}"
LLM_MAX_TOKENS="${LLM_MAX_TOKENS:-32768}"
LLM_TEMPERATURE="${LLM_TEMPERATURE:-0.2}"
DEBUG="${DEBUG:-1}"

DRY_RUN="${DRY_RUN:-0}"

# ── 模型列表 ──────────────────────────────────────────────────────────────────
# 格式（空格分隔，每行一个模型）：
#   CUDA_DEVICES  MODEL_PATH  TENSOR_PARALLEL  GPU_MEM_UTIL  OUTPUT_TAG
#
# OUTPUT_TAG 用于命名输出目录，不填则自动取模型路径最后一段。
# 数组每5个元素为一组。

MODELS=(
  # CUDA_DEVICES          MODEL_PATH                     TP   MEM   OUTPUT_TAG
  "0,1,2,3"               "Qwen3.5-0.8B"                 4    0.75   "Qwen3.5-0.8B"
  "0,1,2,3"               "Qwen3.5-2B"                   4    0.75   "Qwen3.5-2B"
  "0,1,2,3"               "Qwen3.5-4B"                   4    0.75   "Qwen3.5-4B"
  "0,1,2,3"               "Qwen3.5-9B"                   4    0.75   "Qwen3.5-98"
  "0,1,2,3"               "Qwen3.5-27B"                  4    0.75   "Qwen3.5-27B"
)
# 每组字段数
_STRIDE=5

# ── 工具函数 ──────────────────────────────────────────────────────────────────

log()  { echo "[$(date '+%H:%M:%S')] $*"; }
info() { log "[INFO]  $*"; }
warn() { log "[WARN]  $*"; }
err()  { log "[ERROR] $*" >&2; }

run() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] $*"
  else
    "$@"
  fi
}

# 等待 vllm HTTP 服务就绪
wait_for_vllm() {
  local url="http://${VLLM_HOST}:${VLLM_PORT}/health"
  local deadline=$(( $(date +%s) + VLLM_STARTUP_TIMEOUT ))
  info "等待 vllm 就绪: ${url} (最多 ${VLLM_STARTUP_TIMEOUT}s)"
  while true; do
    if curl -sf "${url}" -o /dev/null 2>/dev/null; then
      info "vllm 已就绪"
      return 0
    fi
    if (( $(date +%s) >= deadline )); then
      err "vllm 在 ${VLLM_STARTUP_TIMEOUT}s 内未就绪，放弃"
      return 1
    fi
    sleep 5
  done
}

# 关闭 vllm：先 kill 端口主进程，再清理目标 GPU 上所有残留进程
# 用法: kill_vllm <cuda_devices>   e.g. kill_vllm "4,5,6,7"
kill_vllm() {
  local cuda_devices="${1:-}"

  # ── Step 1: kill 监听端口的主进程 ──────────────────────────────────────────
  local pids
  pids=$(ss -tlnp "sport = :${VLLM_PORT}" 2>/dev/null \
         | grep -oP 'pid=\K[0-9]+' || true)
  if [[ -z "${pids}" ]]; then
    pids=$(lsof -ti tcp:"${VLLM_PORT}" 2>/dev/null || true)
  fi

  if [[ -n "${pids}" ]]; then
    info "关闭 vllm 主进程 PID=${pids} …"
    run kill -TERM ${pids} 2>/dev/null || true
    local t=0
    while kill -0 ${pids} 2>/dev/null; do
      sleep 2; t=$(( t + 2 ))
      if (( t >= VLLM_SHUTDOWN_TIMEOUT )); then
        warn "主进程未在 ${VLLM_SHUTDOWN_TIMEOUT}s 内退出，发送 SIGKILL"
        run kill -KILL ${pids} 2>/dev/null || true
        break
      fi
    done
  else
    warn "端口 ${VLLM_PORT} 上未发现主进程，跳过端口 kill"
  fi

  # ── Step 2: 等待目标 GPU 上的残留进程（Worker 子进程）彻底退出 ─────────────
  if [[ -n "${cuda_devices}" ]]; then
    _wait_gpu_free "${cuda_devices}"
  else
    sleep 3
  fi

  info "vllm 已关闭，GPU 显存已释放"
}

# 轮询 nvidia-smi，等待指定 GPU 上所有进程退出；超时后强制 SIGKILL
# 用法: _wait_gpu_free "4,5,6,7"
_wait_gpu_free() {
  local cuda_devices="$1"
  local gpu_wait_timeout=60   # 最多等 60s 让 Worker 自行退出
  local deadline=$(( $(date +%s) + gpu_wait_timeout ))

  info "等待 GPU [${cuda_devices}] 上残留进程退出 (最多 ${gpu_wait_timeout}s) …"

  while true; do
    local gpu_pids
    gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader \
               -i "${cuda_devices}" 2>/dev/null \
               | tr -d ' \r' | grep -v '^$' || true)

    if [[ -z "${gpu_pids}" ]]; then
      info "GPU [${cuda_devices}] 已无残留进程"
      return 0
    fi

    if (( $(date +%s) >= deadline )); then
      warn "GPU [${cuda_devices}] 仍有进程 (${gpu_pids// /,})，超时强制 SIGKILL"
      # shellcheck disable=SC2086
      run kill -KILL ${gpu_pids} 2>/dev/null || true
      sleep 3
      return 0
    fi

    info "GPU [${cuda_devices}] 仍有进程 (${gpu_pids// /,})，等待退出 …"
    sleep 3
  done
}

# ── 单模型执行函数 ────────────────────────────────────────────────────────────

run_one_model() {
  local cuda_devices="$1"
  local model_path="$2"
  local tensor_parallel="$3"
  local gpu_mem_util="$4"
  local output_tag="$5"

  # 输出目录
  local out_dir="${ROOT_DIR}/outputs/${RUNNER}_${output_tag}"
  local output_path="${out_dir}/predictions.jsonl"
  local answer_path="${out_dir}/answer_sheet.jsonl"
  local stats_path="${out_dir}/stats.json"
  local log_path="${out_dir}/run.log"

  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
  info "模型     : ${model_path}"
  info "GPUs     : ${cuda_devices}  (TP=${tensor_parallel})"
  info "输出目录 : ${out_dir}"
  info "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

  mkdir -p "${out_dir}"

  # ── 1. 启动 vllm ─────────────────────────────────────────────────────────
  local vllm_log="${out_dir}/vllm_serve.log"
  info "启动 vllm: ${model_path}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY_RUN] CUDA_VISIBLE_DEVICES=${cuda_devices} ${VLLM_BIN} serve ${model_path} \\"
    echo "            --host ${VLLM_HOST} --port ${VLLM_PORT} \\"
    echo "            --tensor-parallel-size ${tensor_parallel} \\"
    echo "            --gpu-memory-utilization ${gpu_mem_util}"
  else
    CUDA_VISIBLE_DEVICES="${cuda_devices}" \
    "${VLLM_BIN}" serve "${model_path}" \
      --host "${VLLM_HOST}" \
      --port "${VLLM_PORT}" \
      --tensor-parallel-size "${tensor_parallel}" \
      --gpu-memory-utilization "${gpu_mem_util}" \
      >"${vllm_log}" 2>&1 &
    local vllm_pid=$!
    info "vllm PID=${vllm_pid}，日志: ${vllm_log}"

    # 等待就绪，失败则清理并跳过
    if ! wait_for_vllm; then
      err "vllm 启动失败，跳过模型 ${output_tag}"
      kill_vllm "${cuda_devices}" || true
      return 1
    fi
  fi

  # ── 2. 跑 agent / workflow ───────────────────────────────────────────
  local runner_script
  if [[ "${RUNNER}" == "workflow" ]]; then
    runner_script="run_workflow.py"
  else
    runner_script="run_agent.py"
  fi

  local llm_base_url="http://127.0.0.1:${VLLM_PORT}/v1"

  local bench_cmd=(
    "${PYTHON_BIN}" "${runner_script}"
    "--input"               "${INPUT_PATH}"
    "--output"              "${output_path}"
    "--answer-sheet"        "${answer_path}"
    "--stats"               "${stats_path}"
    "--log-file"            "${log_path}"
    "--llm-model"           "${model_path}"
    "--llm-base-url"        "${llm_base_url}"
    "--llm-api-key"         "EMPTY"
    "--llm-timeout"         "${LLM_TIMEOUT}"
    "--max-tokens"          "${LLM_MAX_TOKENS}"
    "--temperature"         "${LLM_TEMPERATURE}"
    "--text-ann-url"        "${TEXT_ANN_URL}"
    "--img-ann-url"         "${IMG_ANN_URL}"
    "--ann-timeout"         "${ANN_TIMEOUT}"
    "--ann-topk"            "${ANN_TOPK}"
    "--max-evidence-items"  "${MAX_EVIDENCE_ITEMS}"
    "--answer-max-attempts" "${ANSWER_MAX_ATTEMPTS}"
  )

  # run_agent.py 特有参数
  if [[ "${RUNNER}" != "workflow" ]]; then
    bench_cmd+=(
      "--multimodal-ann-url" "${MULTIMODAL_ANN_URL}"
      "--max-plan-rounds"    "${MAX_PLAN_ROUNDS}"
    )
  fi

  if [[ "${DEBUG}" == "1" ]]; then
    bench_cmd+=("--debug")
  fi

  info "运行 ${runner_script} …"
  local console_log="${out_dir}/console_$(date +%Y%m%d_%H%M%S).log"

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] '
    printf '%q ' "${bench_cmd[@]}"
    printf '\n'
    local bench_ok=0
  else
    if "${bench_cmd[@]}" 2>&1 | tee "${console_log}"; then
      local bench_ok=0
    else
      local bench_ok=$?
      err "${runner_script} 以退出码 ${bench_ok} 结束，继续下一个模型"
    fi
  fi

  # ── 3. 关闭 vllm ─────────────────────────────────────────────────────────
  info "关闭 vllm 服务 …"
  kill_vllm "${cuda_devices}" || true

  return "${bench_ok}"
}

# ── 主流程 ────────────────────────────────────────────────────────────────────

# 先确保端口上没有残留的 vllm 进程（启动前清理，不知道用了哪些 GPU，只按端口 kill）
if lsof -ti tcp:"${VLLM_PORT}" >/dev/null 2>&1; then
  warn "端口 ${VLLM_PORT} 已被占用，先关闭 …"
  kill_vllm "" || true
fi

total_models=$(( ${#MODELS[@]} / _STRIDE ))
info "共 ${total_models} 个模型，runner=${RUNNER}"
info "输入数据: ${INPUT_PATH}"
echo ""

pass=0
fail=0
failed_tags=()

for (( i=0; i < ${#MODELS[@]}; i += _STRIDE )); do
  _cuda="${MODELS[i]}"
  _path="${MODELS[i+1]}"
  _tp="${MODELS[i+2]}"
  _mem="${MODELS[i+3]}"
  _tag="${MODELS[i+4]}"

  # 若 tag 为空，取路径最后一段
  if [[ -z "${_tag}" ]]; then
    _tag="$(basename "${_path}")"
  fi

  model_num=$(( i / _STRIDE + 1 ))
  info "========== 模型 ${model_num}/${total_models}: ${_tag} =========="

  if run_one_model "${_cuda}" "${_path}" "${_tp}" "${_mem}" "${_tag}"; then
    pass=$(( pass + 1 ))
    info "✓ ${_tag} 完成"
  else
    fail=$(( fail + 1 ))
    failed_tags+=("${_tag}")
    warn "✗ ${_tag} 失败"
  fi
  echo ""
done

# ── 汇总 ─────────────────────────────────────────────────────────────────────
echo "============================================================"
info "全部完成: ${pass} 成功 / ${fail} 失败 / ${total_models} 总计"
if (( fail > 0 )); then
  warn "失败的模型: ${failed_tags[*]}"
fi
echo "输出目录: ${ROOT_DIR}/outputs/"
echo "============================================================"
