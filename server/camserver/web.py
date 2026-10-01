"""HTTP API, bound to the WireGuard IP.

  GET  /            control page (stream + sliders + status)
  GET  /stream      multipart MJPEG
  GET  /snap        latest frame as JPEG
  GET  /state       current framing as JSON
  GET  /status      motion, live tracks, recent tracks, upload queue
  GET  /debug.jpg   latest frame with the ROI and live track boxes drawn in
  POST /set?tilt=-15&pan=0&zoom=100   pan/tilt in degrees, zoom 100-400
"""
import json
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from . import camera, uploader
from . import config as C

PAGE = (Path(__file__).parent / "page.html").read_bytes()
ctx = {}  # set by __main__: pipeline, motion


def status() -> dict:
    pipe, motion = ctx["pipeline"], ctx["motion"]
    return dict(motion=round(motion.level, 4), detecting=motion.active(), live_tracks=len(pipe.tracker.tracks),
                stats=pipe.stats, recent=pipe.recent, upload=uploader.status,
                classifier=bool(pipe.clf), roi=C.ROI)


def debug_jpeg() -> bytes | None:
    jpeg, _ = camera.latest_frame(timeout=5)
    if jpeg is None:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    h, w = img.shape[:2]
    if C.ROI:
        cv2.polylines(img, [np.array([(x * w, y * h) for x, y in C.ROI], np.int32)], True, (0, 220, 255), 2)
    for tr in list(ctx["pipeline"].tracker.tracks):
        x1, y1, x2, y2 = (int(v) for v in tr.path[-1][1])
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f"#{tr.id} {tr.travel / w:.0%}", (x1, max(15, y1 - 6)), 0, 0.6, (0, 255, 0), 2)
    return cv2.imencode(".jpg", img)[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, ctype, body):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self.send(200, "text/html; charset=utf-8", PAGE)
        elif path == "/state":
            self.send(200, "application/json", json.dumps(camera.state).encode())
        elif path == "/status":
            self.send(200, "application/json", json.dumps(status()).encode())
        elif path in ("/snap", "/debug.jpg"):
            frame = camera.latest_frame(timeout=5)[0] if path == "/snap" else debug_jpeg()
            if frame:
                self.send(200, "image/jpeg", frame)
            else:
                self.send(503, "text/plain", b"No frame from camera yet")
        elif path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=f")
            self.end_headers()
            seen = -1
            try:
                while True:
                    # Wait for a frame newer than the last one sent, so no notification is missed.
                    frame, seen = camera.latest_frame(seen, timeout=5)
                    if frame is None:
                        continue
                    # Written outside the lock so a slow client never stalls capture.
                    self.wfile.write(b"--f\r\nContent-Type: image/jpeg\r\n"
                                     b"Content-Length: %d\r\n\r\n" % len(frame))
                    self.wfile.write(frame + b"\r\n")
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send(404, "text/plain", b"Not found")

    def do_POST(self):
        url = urlparse(self.path)
        if url.path != "/set":
            return self.send(404, "text/plain", b"Not found")
        q = {k: v[0] for k, v in parse_qs(url.query).items() if k in camera.CTRLS}
        try:
            camera.apply(q)
        except ValueError:
            return self.send(400, "text/plain", b"Values must be numbers")
        self.send(200, "application/json", json.dumps(camera.state).encode())
