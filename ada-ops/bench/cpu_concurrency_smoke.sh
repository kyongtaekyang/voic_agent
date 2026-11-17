#!/usr/bin/env bash
# 7단계 측정 도구(concurrency.py)를 GPU 없이 시험한다: 가짜 LLM·가짜 Moshi 로 두 Agent 를 띄우고 세션 1·3개를 붙인다.
# 수치는 CPU 와 가짜 모델의 값이므로 KPI 가 아니다. 도구가 끝까지 돌고 busy/오류를 바르게 세는지만 본다.
# gpu-wait 가 실행 중이면 잠시 멈췄다가 끝나면 다시 시작한다 (같은 포트를 쓰는 gpu-mode realtime 과 겹치지 않게).
# 자기가 띄운 PID 만 종료한다.
set -u
OPS="$(cd "$(dirname "$0")/.." && pwd)"
ROOT="$(dirname "$OPS")"
KO="$ROOT/korean_agent"; EN="$ROOT/english-agent"
LOG="$OPS/bench/out/cpu_smoke"; mkdir -p "$LOG"
LEVELS=${LEVELS:-1,3}; TURNS=${TURNS:-2}
for p in 7101 7102 7201 8001 8766 8999; do
  if ss -ltn "sport = :$p" | grep -q LISTEN; then echo "port $p busy, abort"; exit 1; fi
done
if [ -n "$(cat "$OPS/run/mode" 2>/dev/null)" ]; then echo "gpu-mode $(cat "$OPS/run/mode") is up, abort"; exit 1; fi
WAIT_MODE=""
if "$OPS/gpu-wait" status 2>/dev/null | grep -q '^감시 중'; then
  WAIT_MODE=$("$OPS/gpu-wait" status | sed -n 's/.*gpu-wait run \([a-z0-9-]*\).*/\1/p' | head -1)
  "$OPS/gpu-wait" stop
fi
export CUDA_VISIBLE_DEVICES=""
PIDS=()
cleanup() {
  kill "${PIDS[@]}" 2>/dev/null; sleep 2; kill -9 "${PIDS[@]}" 2>/dev/null
  rm -f "$LOG/en_services.yaml"
  if [ -n "$WAIT_MODE" ]; then "$OPS/gpu-wait" start "$WAIT_MODE"; fi
}
trap cleanup EXIT

# 한국어: 가짜 LLM(8001) + Realtime 서버(8766, Supertonic CPU) + 화면(7201, 운영 모드 확인 끔)
(cd "$KO" && exec .venv/bin/python bench/mock_llm_server.py --port 8001) > "$LOG/ko_mock_llm.log" 2>&1 & PIDS+=($!)
(cd "$KO" && exec .venv/bin/python -m realtime_server.main --tts supertonic --supertonic_tts_lang ko) > "$LOG/ko_server.log" 2>&1 & PIDS+=($!)
(cd "$KO" && KVA_WEB_IGNORE_MODE=1 exec .venv/bin/python -m web.server) > "$LOG/ko_web.log" 2>&1 & PIDS+=($!)
# 영어: 가짜 Moshi(8999) + Gateway(7102) + 화면(7101). 설정 사본에서 Moshi 주소와 운영 모드 확인만 바꾼다
sed -e 's#ws://127.0.0.1:8998/api/chat#ws://127.0.0.1:8999/api/chat#' -e 's#require_mode: true#require_mode: false#' \
  "$EN/config/services.yaml" > "$LOG/en_services.yaml"
(cd "$EN" && exec .venv/bin/python bench/mock_moshi.py --port 8999) > "$LOG/en_mock_moshi.log" 2>&1 & PIDS+=($!)
(cd "$EN" && EVA_CONFIG="$LOG/en_services.yaml" exec .venv-core/bin/python -m gateway.server) > "$LOG/en_gateway.log" 2>&1 & PIDS+=($!)
(cd "$EN" && EVA_CONFIG="$LOG/en_services.yaml" exec .venv-core/bin/python -m web.server) > "$LOG/en_web.log" 2>&1 & PIDS+=($!)

ready=0
for i in $(seq 1 240); do
  ok=1
  curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/v1/usage | grep -q 200 || ok=0
  for p in 7201 7101 7102 8999; do ss -ltn "sport = :$p" | grep -q LISTEN || ok=0; done
  if [ $ok = 1 ]; then ready=1; break; fi
  for pid in "${PIDS[@]}"; do kill -0 "$pid" 2>/dev/null || { echo "a service exited early (pid $pid)"; break 2; }; done
  sleep 1
done
echo "services_ready=$ready after ${i}s"
rc=1
if [ $ready = 1 ]; then
  "$EN/.venv-core/bin/python" "$OPS/bench/concurrency.py" --levels "$LEVELS" --turns "$TURNS" \
    --out "$LOG/concurrency_cpu_mock.json" "$@"
  rc=$?
fi
for f in ko_server ko_web en_gateway en_web; do echo "== $f (errors)"; grep -iE 'error|traceback|exception' "$LOG/$f.log" | tail -5; done
exit $rc
