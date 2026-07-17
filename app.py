import json
import os
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

from predict import predict_from_canvas


INDEX_PATH = Path(__file__).with_name("index.html")


class Response:
    def __init__(self, status_code, body, headers=None):
        self.status_code = status_code
        self._body = body if isinstance(body, bytes) else str(body).encode("utf-8")
        self.headers = headers or {}

    def get_json(self):
        return json.loads(self._body.decode("utf-8"))

    def get_data(self):
        return self._body


class SimpleTestClient:
    def __init__(self, app):
        self.app = app

    def get(self, path):
        return self.app.handle_request("GET", path)

    def post(self, path, payload=None):
        body = json.dumps(payload).encode("utf-8") if payload is not None else b""
        return self.app.handle_request("POST", path, body=body)


class DigitRecognitionWebApp:
    def __init__(self):
        self.index_html = INDEX_PATH.read_text(encoding="utf-8") if INDEX_PATH.exists() else "<h1>Not found</h1>"

    def test_client(self):
        return SimpleTestClient(self)

    def handle_request(self, method, path, body=b"", headers=None):
        parsed = urlparse(path)
        if method == "GET" and parsed.path in {"/", "/index.html"}:
            return Response(200, self.index_html, {"Content-Type": "text/html; charset=utf-8"})

        if method == "GET" and parsed.path == "/health":
            return Response(200, '{"status": "ok"}', {"Content-Type": "application/json"})

        if method == "POST" and parsed.path == "/predict":
            try:
                payload = json.loads(body.decode("utf-8")) if body else {}
            except json.JSONDecodeError:
                return Response(400, '{"error": "invalid json"}', {"Content-Type": "application/json"})

            grid = payload.get("grid")
            if not isinstance(grid, list) or not grid:
                return Response(400, '{"error": "grid is required"}', {"Content-Type": "application/json"})

            try:
                canvas = normalize_grid(grid)
            except ValueError as exc:
                return Response(400, json.dumps({"error": str(exc)}), {"Content-Type": "application/json"})

            result = predict_from_canvas(canvas)
            return Response(200, json.dumps(result), {"Content-Type": "application/json"})

        return Response(404, "Not Found", {"Content-Type": "text/plain; charset=utf-8"})

    def run(self, host="127.0.0.1", port=8000):
        server = HTTPServer((host, port), RequestHandler)
        server.web_app = self
        print(f"Serving digit recognizer at http://{host}:{port}")
        server.serve_forever()


class RequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._dispatch()

    def do_POST(self):
        self._dispatch()

    def _dispatch(self):
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        response = self.server.web_app.handle_request(self.command, self.path, body=body)
        self.send_response(response.status_code)
        for key, value in response.headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(response.get_data())

    def log_message(self, format, *args):
        return


def normalize_grid(grid):
    if len(grid) != 28:
        raise ValueError("grid must have 28 rows")

    arr = np.zeros((28, 28), dtype=np.float32)
    for index, row in enumerate(grid):
        if not isinstance(row, list) or len(row) != 28:
            raise ValueError("each row must contain 28 values")
        arr[index] = np.asarray(row, dtype=np.float32)

    return arr


def create_app():
    return DigitRecognitionWebApp()


def main():
    port = int(os.environ.get("PORT", "8000"))
    app = create_app()
    app.run(port=port)


if __name__ == "__main__":
    main()
