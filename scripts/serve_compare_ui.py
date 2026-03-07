#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from leanstack.compare_runner import build_comparison_payload, check_remote_status, ensure_vllm_ready


WEB_ROOT = REPO_ROOT / "web" / "compare-ui"


class CompareHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, directory: str | None = None, **kwargs):
        super().__init__(*args, directory=str(WEB_ROOT), **kwargs)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/status":
            self._handle_status()
            return
        if parsed.path in ("/", "/index.html", "/styles.css", "/app.js"):
            super().do_GET()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/api/start-vllm":
            self._handle_start_vllm()
            return
        if parsed.path == "/api/compare":
            self._handle_compare()
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not Found")

    def log_message(self, format: str, *args) -> None:
        return

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8"))

    def _send_json(self, payload: dict[str, object], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_status(self) -> None:
        try:
            self._send_json({"ok": True, "status": check_remote_status()})
        except Exception as exc:  # pragma: no cover - best-effort UI helper
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_start_vllm(self) -> None:
        try:
            payload = ensure_vllm_ready()
            self._send_json({"ok": True, "result": payload})
        except Exception as exc:  # pragma: no cover - best-effort UI helper
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def _handle_compare(self) -> None:
        try:
            body = self._read_json_body()
            prompt = str(body.get("prompt") or "").strip()
            profile = str(body.get("profile") or "decode_64_256")
            max_new_tokens = body.get("max_new_tokens")
            if not prompt:
                raise ValueError("prompt must not be empty")
            if max_new_tokens is not None:
                max_new_tokens = int(max_new_tokens)
            payload = build_comparison_payload(
                prompt=prompt,
                profile=profile,
                max_new_tokens=max_new_tokens,
            )
            self._send_json({"ok": True, "result": payload})
        except RuntimeError as exc:
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.CONFLICT)
        except Exception as exc:  # pragma: no cover - best-effort UI helper
            self._send_json({"ok": False, "error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve the local leanstack comparison UI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), CompareHandler)
    print(f"leanstack compare UI: http://{args.host}:{args.port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
