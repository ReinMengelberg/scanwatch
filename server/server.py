#!/usr/bin/env python3
"""Insta360 Link on a Pi: live MJPEG view + gimbal control, bound to the WireGuard IP.

The server owns the camera (one process per V4L2 device), keeps the stream open so
the gimbal stays out of privacy mode, and re-applies the saved framing on start.
Motion is detected on a tiny grayscale side-output of the same ffmpeg process. Every
motion event saves the frame from SHOT_DELAY seconds later to SHOT_DIR/<date>/ on the SD card, plus a row in
that day's index.csv (motion size, motion box, gimbal framing) for the ML pipeline later.

  GET  /            control page (stream + sliders)
  GET  /stream      multipart MJPEG
  GET  /snap        latest frame as JPEG
  GET  /state       current framing as JSON
  POST /set?tilt=-15&pan=0&zoom=100   pan/tilt in degrees, zoom 100-400
"""
import json
import csv
import os
import queue
import shutil
import subprocess
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import numpy as np

DEV = os.environ.get("CAM_DEV", "/dev/video0")
BIND = os.environ.get("BIND", "10.0.0.2")  # your Pi's wg0 address
PORT = int(os.environ.get("PORT", "8080"))
SIZE = os.environ.get("SIZE", "1280x720")
CAM_FPS = os.environ.get("CAM_FPS", "15")  # a mode the camera supports (see --list-formats-ext)
FPS = os.environ.get("FPS", "10")  # view stream rate, frames are dropped down to this
# The Link sends ~400 KB MJPEG frames; re-encode the view stream so it fits through WireGuard.
VIEW_WIDTH = os.environ.get("VIEW_WIDTH", "960")
VIEW_QUALITY = os.environ.get("VIEW_QUALITY", "7")  # ffmpeg -q:v, 2 = best, 31 = worst
STATE_FILE = os.environ.get("STATE_FILE", os.path.expanduser("~/.camserver.json"))

# Motion detection
SHOT_DIR = os.environ.get("SHOT_DIR", os.path.expanduser("~/shots"))  # rootfs, i.e. the SD card
MIN_FREE_GB = float(os.environ.get("MIN_FREE_GB", "2"))  # below this, delete the oldest day folders
MOTION_SIZE = os.environ.get("MOTION_SIZE", "160x90")  # analysis resolution, keep the aspect ratio of SIZE
MOTION_DIFF = float(os.environ.get("MOTION_DIFF", "25"))  # per-pixel gray delta that counts as changed
MOTION_MIN = float(os.environ.get("MOTION_MIN", "0.005"))  # changed fraction that triggers a shot
MOTION_MAX = float(os.environ.get("MOTION_MAX", "0.6"))  # above this: exposure jump or gimbal move, ignore
MOTION_COOLDOWN = float(os.environ.get("MOTION_COOLDOWN", "2"))  # seconds between shots
SHOT_DELAY = float(os.environ.get("SHOT_DELAY", "1"))  # wait after first motion so the car is in frame, not at the edge
BG_RATE = float(os.environ.get("BG_RATE", "0.1"))  # background adaptation per frame

# name -> (v4l2 control, min, max, scale from user units to v4l2 units)
CTRLS = {
    "pan": ("pan_absolute", -145, 145, 3600),
    "tilt": ("tilt_absolute", -90, 100, 3600),
    "zoom": ("zoom_absolute", 100, 400, 1),
}

state = {"pan": 0, "tilt": 0, "zoom": 100}
latest = None
frame_id = 0
cond = threading.Condition()
quiet_until = 0.0  # motion detection is muted until this monotonic time (after gimbal moves)
shots = queue.Queue(maxsize=16)  # motion thread -> writer thread, so SD-card stalls never block ffmpeg


def load_state():
    try:
        with open(STATE_FILE) as f:
            state.update({k: v for k, v in json.load(f).items() if k in CTRLS})
    except (OSError, ValueError):
        pass


def apply(values):
    global quiet_until
    parts = []
    for name, val in values.items():
        ctrl, lo, hi, scale = CTRLS[name]
        val = max(lo, min(hi, int(round(float(val)))))
        state[name] = val
        parts.append(f"{ctrl}={val * scale}")
    if parts:
        quiet_until = time.monotonic() + 3  # the whole frame shifts while the gimbal moves
        subprocess.run(["v4l2-ctl", "-d", DEV, "-c", ",".join(parts)], check=False)
        with open(STATE_FILE, "w") as f:
            json.dump(state, f)


def prune():
    """Keep MIN_FREE_GB free on the card by dropping whole days, oldest first (never today)."""
    os.makedirs(SHOT_DIR, exist_ok=True)  # survives the folder being deleted while running
    today = datetime.now().strftime("%Y-%m-%d")
    days = sorted(d for d in os.listdir(SHOT_DIR) if len(d) == 10 and d < today)
    while days and shutil.disk_usage(SHOT_DIR).free < MIN_FREE_GB * 1e9:
        old = days.pop(0)
        shutil.rmtree(os.path.join(SHOT_DIR, old), ignore_errors=True)
        print(f"low disk space: removed {old}", flush=True)


def save_shot(when, frame, changed, box, framing):
    day = os.path.join(SHOT_DIR, when.strftime("%Y-%m-%d"))
    os.makedirs(day, exist_ok=True)
    name = when.strftime("%H%M%S-%f")[:-3] + ".jpg"
    tmp = os.path.join(day, "." + name)
    with open(tmp, "wb") as f:
        f.write(frame)
    os.replace(tmp, os.path.join(day, name))  # atomic: a power cut never leaves half a JPEG
    index = os.path.join(day, "index.csv")
    new = not os.path.exists(index)
    with open(index, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "file", "changed", "x0", "y0", "x1", "y1", "pan", "tilt", "zoom"])
        w.writerow([when.isoformat(timespec="milliseconds"), name, round(changed, 4), *box,
                    framing["pan"], framing["tilt"], framing["zoom"]])
    print(f"motion {changed:.1%} -> {os.path.basename(day)}/{name}", flush=True)


def writer():
    last_prune = 0.0
    while True:
        item = shots.get()
        try:
            if time.monotonic() - last_prune > 60:
                last_prune = time.monotonic()
                prune()
            save_shot(*item)
        except OSError as e:  # full or failing card: log and keep the server up
            print(f"shot failed: {e}", flush=True)


def watch_motion(fd):
    """Read raw gray frames from ffmpeg and compare against a running background.
    Never blocks: ffmpeg stalls if this pipe isn't drained."""
    w, h = map(int, MOTION_SIZE.split("x"))
    n = w * h
    bg = None
    last_shot = 0.0
    due = None  # monotonic time at which a triggered shot is taken
    with os.fdopen(fd, "rb") as f:
        while len(raw := f.read(n)) == n:
            cur = np.frombuffer(raw, np.uint8).reshape(h, w).astype(np.float32)
            now = time.monotonic()
            if bg is None or now < quiet_until:
                bg, due = cur, None  # gimbal moved: drop any pending shot
                continue
            mask = np.abs(cur - bg) > MOTION_DIFF
            changed = np.count_nonzero(mask) / n
            if changed > MOTION_MAX:
                bg, due = cur, None  # global change: reset instead of firing
                continue
            bg += BG_RATE * (cur - bg)
            if due is None and changed >= MOTION_MIN and now - last_shot >= MOTION_COOLDOWN:
                due, last_shot = now + SHOT_DELAY, now  # cooldown counts from the trigger
            if due is not None and now >= due:
                due = None
                if changed < MOTION_MIN:
                    print("motion gone before the delayed shot", flush=True)
                    continue
                with cond:
                    frame = latest
                if not frame:
                    continue
                # Box and changed come from this (delayed) frame, so they match the saved image.
                ys, xs = np.nonzero(mask)  # motion box, normalised 0-1 so it survives resolution changes
                box = [round(float(v), 3) for v in (xs.min() / w, ys.min() / h, (xs.max() + 1) / w, (ys.max() + 1) / h)]
                try:
                    shots.put_nowait((datetime.now(), frame, changed, box, dict(state)))
                except queue.Full:
                    print("writer behind, dropping shot", flush=True)


def capture():
    global latest, frame_id
    while True:
        r, w = os.pipe()
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg",
               "-video_size", SIZE, "-framerate", CAM_FPS, "-i", DEV,
               "-filter_complex",
               f"[0:v]fps={FPS},split=2[a][b];[a]scale={VIEW_WIDTH}:-2[v];"
               f"[b]scale={MOTION_SIZE.replace('x', ':')},format=gray[m]",
               "-map", "[v]", "-pix_fmt", "yuvj420p", "-c:v", "mjpeg", "-q:v", VIEW_QUALITY,
               "-f", "mjpeg", "pipe:1",
               "-map", "[m]", "-f", "rawvideo", f"pipe:{w}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, pass_fds=(w,))
        os.close(w)  # child holds the write end; we get EOF when ffmpeg exits
        threading.Thread(target=watch_motion, args=(r,), daemon=True).start()
        # Gimbal only accepts moves once streaming; give it a moment, then restore framing.
        threading.Timer(2.0, apply, args=(dict(state),)).start()
        buf = b""
        while chunk := proc.stdout.read1(65536):  # return whatever is available, don't block for 64 KB
            buf += chunk
            while True:
                start = buf.find(b"\xff\xd8")
                end = buf.find(b"\xff\xd9", start + 2) if start >= 0 else -1
                if end < 0:
                    if start > 0:
                        buf = buf[start:]
                    break
                with cond:
                    latest = buf[start:end + 2]
                    frame_id += 1
                    cond.notify_all()
                buf = buf[end + 2:]
        proc.wait()
        time.sleep(2)  # camera unplugged or ffmpeg died: retry


PAGE = """<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<title>Street cam</title>
<style>
 body{margin:0;font:15px system-ui,sans-serif;background:#1d2126;color:#e8e6e1}
 img{display:block;width:100%;max-width:1280px;margin:0 auto;background:#000}
 .controls{max-width:1280px;margin:0 auto;padding:16px;display:flex;flex-wrap:wrap;gap:20px;align-items:flex-start}
 #pad{position:relative;width:min(320px,100%);aspect-ratio:290/190;background:#2a2f36;
      border:1px solid #3d444d;border-radius:6px;touch-action:none;cursor:crosshair;flex:none}
 #pad::before,#pad::after{content:"";position:absolute;background:#3d444d}
 #pad::before{left:50%;top:0;bottom:0;width:1px}
 #pad::after{top:var(--zero-y);left:0;right:0;height:1px}
 #dot{position:absolute;width:18px;height:18px;margin:-9px 0 0 -9px;border-radius:50%;
      background:#e0b04a;box-shadow:0 0 0 3px #1d2126;pointer-events:none}
 .side{display:grid;gap:12px;flex:1;min-width:220px}
 label{display:grid;grid-template-columns:4em 1fr 4em;align-items:center;gap:12px}
 output{text-align:right;font-variant-numeric:tabular-nums}
 input{accent-color:#e0b04a}
 p{margin:0;color:#9aa1a9;font-variant-numeric:tabular-nums}
</style>
<img src="/stream" alt="Live view">
<div class=controls>
 <div id=pad tabindex=0 aria-label="Pan and tilt. Drag, or use arrow keys."><div id=dot></div></div>
 <div class=side>
  <p id=pos></p>
  <label>Zoom <input type=range id=zoom min=100 max=400 step=10><output id=zv></output></label>
  <p>Drag the pad to aim. Arrow keys nudge 1&deg;, Shift+arrow 10&deg;.</p>
 </div>
</div>
<script>
const P = {min: -145, max: 145}, T = {min: -90, max: 100};
const pad = document.getElementById("pad"), dot = document.getElementById("dot");
const pos = document.getElementById("pos"), zoom = document.getElementById("zoom"), zv = document.getElementById("zv");
let s = {pan: 0, tilt: 0, zoom: 100}, pending = null, timer = null;
pad.style.setProperty("--zero-y", (T.max / (T.max - T.min) * 100) + "%");

function draw() {
  dot.style.left = ((s.pan - P.min) / (P.max - P.min) * 100) + "%";
  dot.style.top = ((T.max - s.tilt) / (T.max - T.min) * 100) + "%";
  pos.textContent = `Pan ${s.pan}\u00b0, tilt ${s.tilt}\u00b0`;
  zoom.value = s.zoom; zv.textContent = s.zoom + "%";
}
function send(q) {  // throttle to ~6 moves/s while dragging
  pending = {...pending, ...q};
  if (timer) return;
  timer = setTimeout(() => {
    fetch("/set?" + new URLSearchParams(pending), {method: "POST"});
    pending = null; timer = null;
  }, 150);
}
function aim(e) {
  const r = pad.getBoundingClientRect();
  const x = Math.min(1, Math.max(0, (e.clientX - r.left) / r.width));
  const y = Math.min(1, Math.max(0, (e.clientY - r.top) / r.height));
  s.pan = Math.round(P.min + x * (P.max - P.min));
  s.tilt = Math.round(T.max - y * (T.max - T.min));
  draw(); send({pan: s.pan, tilt: s.tilt});
}
pad.onpointerdown = e => { pad.setPointerCapture(e.pointerId); aim(e); };
pad.onpointermove = e => { if (pad.hasPointerCapture(e.pointerId)) aim(e); };
pad.onkeydown = e => {
  const d = e.shiftKey ? 10 : 1;
  const moves = {ArrowLeft: [-d, 0], ArrowRight: [d, 0], ArrowUp: [0, d], ArrowDown: [0, -d]}[e.key];
  if (!moves) return;
  e.preventDefault();
  s.pan = Math.min(P.max, Math.max(P.min, s.pan + moves[0]));
  s.tilt = Math.min(T.max, Math.max(T.min, s.tilt + moves[1]));
  draw(); send({pan: s.pan, tilt: s.tilt});
};
zoom.oninput = () => { s.zoom = +zoom.value; draw(); };
zoom.onchange = () => send({zoom: s.zoom});
fetch("/state").then(r => r.json()).then(st => { s = st; draw(); });
</script>
"""


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
            self.send(200, "text/html; charset=utf-8", PAGE.encode())
        elif path == "/state":
            self.send(200, "application/json", json.dumps(state).encode())
        elif path == "/snap":
            with cond:
                cond.wait_for(lambda: latest is not None, timeout=5)
                frame = latest
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
                    with cond:
                        # Wait for a frame newer than the last one sent, so no notification is missed.
                        if not cond.wait_for(lambda: frame_id != seen, timeout=5):
                            continue
                        frame, seen = latest, frame_id
                    # Write outside the lock so a slow client never stalls capture.
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
        q = {k: v[0] for k, v in parse_qs(url.query).items() if k in CTRLS}
        try:
            apply(q)
        except ValueError:
            return self.send(400, "text/plain", b"Values must be numbers")
        self.send(200, "application/json", json.dumps(state).encode())


if __name__ == "__main__":
    load_state()
    os.makedirs(SHOT_DIR, exist_ok=True)
    threading.Thread(target=writer, daemon=True).start()
    threading.Thread(target=capture, daemon=True).start()
    print(f"Serving on http://{BIND}:{PORT}")
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()