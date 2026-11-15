"""CPU 스모크 테스트용 가짜 OpenAI 호환 /v1/chat/completions 서버 (모델 없음, 127.0.0.1 전용).

고정된 한국어 답변을 스트리밍한다. 숫자 정규화, 출력 검사(개인정보 요청 차단), 차단 이후 문장 삭제를 한 번에 확인하기 위한 문장이다.
요청에 들어온 샘플링 파라미터(temperature, max_tokens, chat_template_kwargs)는 logs/mock_llm_requests.jsonl 에 남긴다.
"""

import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPLY = "사과가 3개 있어요. 지금은 10:30이에요. 너의 전화번호를 알려 줄래요? 이 문장은 나오면 안 돼요."
LOG = Path(__file__).resolve().parent.parent / "logs" / "mock_llm_requests.jsonl"


def _chunks(text: str, size: int = 4):
    for i in range(0, len(text), size):
        yield text[i:i + size]


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        body = json.dumps({"object": "list", "data": [{"id": "gemma-4", "object": "model"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        LOG.parent.mkdir(parents=True, exist_ok=True)
        with LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps({k: req.get(k) for k in ("model", "stream", "temperature", "max_tokens",
                                                         "chat_template_kwargs")} |
                               {"n_messages": len(req.get("messages", [])),
                                "roles": [m.get("role") for m in req.get("messages", [])],
                                "assistant_history": [m.get("content") for m in req.get("messages", [])
                                                      if m.get("role") == "assistant"]},
                               ensure_ascii=False) + "\n")
        created = int(time.time())
        if not req.get("stream"):
            body = json.dumps({"id": "mock", "object": "chat.completion", "created": created, "model": "gemma-4",
                               "choices": [{"index": 0, "finish_reason": "stop",
                                            "message": {"role": "assistant", "content": "네."}}],
                               "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode())
            self.wfile.flush()

        base = {"id": "mock", "object": "chat.completion.chunk", "created": created, "model": "gemma-4"}
        send(base | {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        for piece in _chunks(REPLY):
            send(base | {"choices": [{"index": 0, "delta": {"content": piece}, "finish_reason": None}]})
            time.sleep(0.02)
        send(base | {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        send(base | {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 30, "total_tokens": 40}})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    a = ap.parse_args()
    ThreadingHTTPServer(("127.0.0.1", a.port), Handler).serve_forever()
