#!/usr/bin/env bash
# CPU 전용 E2E 스모크: 가짜 LLM(8001) + realtime_server.main(8766, Supertonic CPU TTS) + WAV 클라이언트.
# GPU 를 쓰지 않도록 CUDA_VISIBLE_DEVICES 를 비운다. 자기가 띄운 PID 만 종료한다.
set -u
cd "$(dirname "$0")/.."
WAV=${1:-bench/data/ko_ref/clip01_7aa353dc.wav}
mkdir -p logs
rm -f logs/events.jsonl logs/mock_llm_requests.jsonl logs/smoke_out.wav
for p in 8001 8766; do
  if ss -ltn "sport = :$p" | grep -q LISTEN; then echo "port $p busy, abort"; exit 1; fi
done
export CUDA_VISIBLE_DEVICES=""
.venv/bin/python bench/mock_llm_server.py --port 8001 > logs/mock_llm.log 2>&1 &
MOCK=$!
.venv/bin/python -m realtime_server.main --tts supertonic --supertonic_tts_lang ko > logs/smoke_server.log 2>&1 &
SRV=$!
URL=ws://127.0.0.1:8766/v1/realtime
WEBPID=
if [ "${SMOKE_VIA_WEB:-0}" = 1 ]; then
  # 화면 서버(7201)를 거쳐 중계까지 시험한다. 운영 모드 확인은 끈다.
  if ss -ltn "sport = :7201" | grep -q LISTEN; then echo "port 7201 busy, abort"; kill $SRV $MOCK; exit 1; fi
  KVA_WEB_IGNORE_MODE=1 .venv/bin/python -m web.server > logs/smoke_web.log 2>&1 &
  WEBPID=$!
  URL=ws://127.0.0.1:7201/ws
fi
cleanup() { kill $SRV $MOCK $WEBPID 2>/dev/null; sleep 2; kill -9 $SRV $MOCK $WEBPID 2>/dev/null; }
trap cleanup EXIT
ok=0
for i in $(seq 1 180); do
  if ! kill -0 $SRV 2>/dev/null; then echo "server exited"; break; fi
  if curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8766/v1/usage | grep -q 200; then ok=1; break; fi
  sleep 1
done
echo "server_ready=$ok after ${i}s"
if [ $ok = 1 ]; then
  .venv/bin/python bench/smoke_client.py "$WAV" --url "$URL" --timeout ${SMOKE_TIMEOUT:-180} --turns ${SMOKE_TURNS:-2} ${SMOKE_TRUNCATE_MS:+--truncate-ms $SMOKE_TRUNCATE_MS} | tee logs/smoke_result.json
fi
echo "== server log (tail)"; grep -v -E '^\s*$' logs/smoke_server.log | tail -40
echo "== mock requests"; cat logs/mock_llm_requests.jsonl 2>/dev/null
echo "== events.jsonl"; cat logs/events.jsonl 2>/dev/null
