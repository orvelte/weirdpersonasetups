"""Tests run_raw.py request-building and the end-to-end loop against a local stub vLLM server.
Run: python3 -m tasks.T7.test_run_raw
"""
import json, tempfile, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tasks.T7 import run_raw

SEEN = []
FAIL_FIRST = {"n": 1}  # first request returns 503 to exercise retries


class Stub(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n))
        SEEN.append((self.path, body))
        if FAIL_FIRST["n"] > 0:
            FAIL_FIRST["n"] -= 1
            self.send_response(503); self.end_headers(); self.wfile.write(b"busy"); return
        if self.path.endswith("/chat/completions"):
            resp = {"model": body["model"], "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "Sure! Here's a continuation.", "reasoning_content": "The user gave a fragment."}}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 20}}
        else:
            resp = {"model": body["model"], "choices": [{"index": 0, "finish_reason": "length",
                    "text": " the House, where Democratic leaders said they would take up the bill next week after"}],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 200}}
        out = json.dumps(resp).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)


def main():
    # 1. request building (pure)
    url, pl = run_raw.build_request("http://h:8000/v1/", "Qwen/Qwen3.8-27B", "cat", 1)
    assert url == "http://h:8000/v1/completions", url
    assert pl["prompt"] == "cat" and "messages" not in pl and pl["max_tokens"] == 200
    assert pl["temperature"] == 0.6 and pl["top_p"] == 0.95 and pl["skip_special_tokens"] is False and pl["seed"] == 1001
    url, pl = run_raw.build_request("http://h:8000/v1", "Qwen/Qwen3.8-27B", "cat", 0, templated=True)
    assert url == "http://h:8000/v1/chat/completions"
    assert pl["messages"] == [{"role": "user", "content": "cat"}] and "prompt" not in pl and pl["max_tokens"] == 2000
    assert pl["chat_template_kwargs"] == {"enable_thinking": True, "reasoning_effort": "medium"}
    assert all(m["role"] != "system" for m in pl["messages"])
    print("PASS build_request raw + templated")

    # 2. response parsing incl. inline <think> fallback
    c, r, fin, _ = run_raw.parse_response({"choices": [{"message": {"content": "<think>hmm</think>\n\nHello"}, "finish_reason": "stop"}]}, True)
    assert r == "hmm" and c.strip() == "Hello", (r, c)
    print("PASS parse_response inline <think> split")

    # 3. end-to-end against stub, with a 503 on the first request and a resume
    srv = HTTPServer(("127.0.0.1", 0), Stub)
    port = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}/v1"
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "raw_3.8.jsonl"
        rc = run_raw.main(["--base-url", base, "--version", "3.8", "--limit", "3", "--samples", "2", "--out", str(out), "--workers", "2"])
        assert rc == 0
        rows = [json.loads(l) for l in out.read_text().splitlines()]
        assert len(rows) == 6, len(rows)
        assert {(r["id"], r["sample"]) for r in rows} == {(f"document_0{i}", s) for i in (1, 2, 3) for s in (0, 1)}
        r0 = rows[0]
        assert r0["arm"] == "raw" and r0["finish_reason"] == "length" and r0["truncated"] is True
        assert r0["label"] == "DOCUMENT", r0["label"]
        assert "logprobs" not in r0 and "logprobs" not in SEEN[0][1] and r0["model_id"] == "Qwen/Qwen3.8-27B"
        assert all(p.endswith("/v1/completions") for p, _ in SEEN) and all("prompt" in b for _, b in SEEN)
        n_before = len(SEEN)
        rc = run_raw.main(["--base-url", base, "--version", "3.8", "--limit", "3", "--samples", "2", "--out", str(out)])
        assert rc == 0 and len(SEEN) == n_before, "resume should send no new requests"
        assert len(out.read_text().splitlines()) == 6
        print("PASS end-to-end raw arm (retry on 503, 6 rows, resumable)")
        # templated arm on the same stub
        out2 = Path(d) / "templated_vllm_3.8.jsonl"
        rc = run_raw.main(["--base-url", base, "--version", "3.8", "--limit", "2", "--samples", "1", "--templated", "--out", str(out2)])
        rows = [json.loads(l) for l in out2.read_text().splitlines()]
        assert rc == 0 and len(rows) == 2 and rows[0]["arm"] == "templated"
        assert rows[0]["label"] == "ASSISTANT" and rows[0]["reasoning"] == "The user gave a fragment."
        assert SEEN[-1][0].endswith("/v1/chat/completions") and SEEN[-1][1]["chat_template_kwargs"]["reasoning_effort"] == "medium"
        print("PASS end-to-end templated arm via /chat/completions")
    srv.shutdown()
    print("all run_raw tests passed")


if __name__ == "__main__":
    main()
