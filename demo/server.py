"""Stdlib-only demo dashboard server.

Fronts the already-deployed heart-disease stack for a demo video: serves the
single-page dashboard, forwards predictions to the live API, reports cluster
status via kubectl, and runs a background traffic generator that replays
real patient records (plus occasional invalid ones) against /predict.

The live cluster (API at http://localhost) is treated as read-only
infrastructure: this server only ever calls /predict and kubectl get.
"""
import csv
import json
import random
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = 8888
REPO_ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML_PATH = Path(__file__).resolve().parent / "index.html"
ARCHITECTURE_PNG_PATH = REPO_ROOT / "docs" / "figures" / "architecture" / "architecture.png"
CSV_PATH = REPO_ROOT / "data" / "processed" / "heart_cleveland_clean.csv"

# ml/data/preprocess.py: TARGET_COL = "target" (verified against the CSV
# header). Hardcoded here rather than imported, per the demo's stdlib-only,
# zero-coupling-to-ml/ constraint.
TARGET_COL = "target"

API_BASE_URL = "http://localhost"
PREDICT_URL = f"{API_BASE_URL}/predict"

# The 13 raw feature columns the model expects, in api/sample_request.json
# order. Every field but oldpeak is cast to int; oldpeak is cast to float.
FEATURE_FIELDS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]
INVALID_PAYLOAD = {"age": 63, "cp": 9}
POOL_SIZE = 20


def _cast_feature_row(row: dict) -> dict:
    """Cast a raw CSV row (all strings) to the API's expected JSON types."""
    payload = {}
    for field in FEATURE_FIELDS:
        raw_value = row[field]
        if field == "oldpeak":
            payload[field] = float(raw_value)
        else:
            payload[field] = int(float(raw_value))
    return payload


def _load_payload_pool() -> list:
    """Load the processed dataset and sample a pool of valid payloads."""
    with open(CSV_PATH, newline="", encoding="utf-8") as csv_file:
        rows = [row for row in csv.DictReader(csv_file)]
    sample_size = min(POOL_SIZE, len(rows))
    sampled = random.sample(rows, sample_size)
    return [_cast_feature_row(row) for row in sampled]


def _post_json(url: str, payload: dict, timeout: float = 10.0):
    """POST JSON to url; return (status_code, parsed_body_or_none)."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, method="POST", headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read()
            status = response.status
    except urllib.error.HTTPError as exc:
        body = exc.read()
        status = exc.code
    try:
        parsed = json.loads(body) if body else None
    except json.JSONDecodeError:
        parsed = None
    return status, parsed


class TrafficGenerator:
    """Background thread that replays dataset payloads against /predict."""

    def __init__(self):
        self._lock = threading.Lock()
        self._pool = []
        self._running = False
        self._thread = None
        self._sent = 0
        self._errors_sent = 0
        self._last_label = ""

    def _ensure_pool(self):
        if not self._pool:
            self._pool = _load_payload_pool()

    def start(self):
        with self._lock:
            if self._running:
                return
            self._ensure_pool()
            self._running = True
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()

    def stop(self):
        with self._lock:
            self._running = False

    def status(self) -> dict:
        with self._lock:
            return {
                "running": self._running,
                "sent": self._sent,
                "errors_sent": self._errors_sent,
                "last_label": self._last_label,
            }

    def _is_running(self) -> bool:
        with self._lock:
            return self._running

    def _run(self):
        while self._is_running():
            time.sleep(random.uniform(0.3, 1.5))
            send_invalid = random.random() < 0.05
            payload = INVALID_PAYLOAD if send_invalid else random.choice(self._pool)
            try:
                status, body = _post_json(PREDICT_URL, payload)
            except Exception as exc:  # noqa: BLE001 - never let the thread die
                print(f"[traffic] request failed: {exc}", file=sys.stderr)
                continue
            if status == 200 and isinstance(body, dict) and "risk_label" in body:
                label = body["risk_label"]
            else:
                label = f"invalid ({status})"
            with self._lock:
                self._sent += 1
                if send_invalid:
                    self._errors_sent += 1
                self._last_label = label


traffic_generator = TrafficGenerator()


def _json_bytes(obj) -> bytes:
    return json.dumps(obj).encode("utf-8")


class DemoHandler(BaseHTTPRequestHandler):
    server_version = "HeartDiseaseDemo/1.0"

    def _send_json(self, status: int, obj):
        body = _json_bytes(obj)
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str):
        if not path.is_file():
            self._send_json(404, {"error": f"{path.name} not found"})
            return
        body = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        return json.loads(raw) if raw else {}

    def do_GET(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._send_file(INDEX_HTML_PATH, "text/html; charset=utf-8")
        elif path == "/architecture.png":
            self._send_file(ARCHITECTURE_PNG_PATH, "image/png")
        elif path == "/api/cluster":
            self._handle_cluster()
        elif path == "/api/traffic":
            self._send_json(200, traffic_generator.status())
        else:
            self._send_json(404, {"error": "not found"})

    def do_POST(self):  # noqa: N802 - BaseHTTPRequestHandler API
        path = self.path.split("?", 1)[0]
        if path == "/api/predict":
            self._handle_predict()
        elif path == "/api/traffic":
            self._handle_traffic_action()
        else:
            self._send_json(404, {"error": "not found"})

    def _handle_predict(self):
        length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(length) if length else b""
        request = urllib.request.Request(
            PREDICT_URL,
            data=raw_body,
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                status = response.status
                body = response.read()
        except urllib.error.HTTPError as exc:
            status = exc.code
            body = exc.read()
        except urllib.error.URLError as exc:
            self._send_json(502, {"error": f"upstream unreachable: {exc.reason}"})
            return
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_cluster(self):
        try:
            cmd = [
                "kubectl", "get", "pods,deployments,services",
                "-n", "heart-disease", "-o", "json",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as exc:
            self._send_json(503, {"error": str(exc)})
            return
        if result.returncode != 0:
            self._send_json(503, {"error": result.stderr.strip() or "kubectl failed"})
            return
        body = result.stdout.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_traffic_action(self):
        try:
            payload = self._read_json_body()
        except json.JSONDecodeError:
            self._send_json(400, {"error": "invalid JSON body"})
            return
        action = payload.get("action")
        if action == "start":
            traffic_generator.start()
        elif action == "stop":
            traffic_generator.stop()
        else:
            self._send_json(400, {"error": "action must be 'start' or 'stop'"})
            return
        self._send_json(200, traffic_generator.status())

    def log_message(self, format_str, *args):  # noqa: A002 - BaseHTTPRequestHandler API
        sys.stderr.write(f"[demo] {self.address_string()} {format_str % args}\n")


def main():
    server = ThreadingHTTPServer(("0.0.0.0", PORT), DemoHandler)
    print(f"Demo dashboard on http://localhost:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
