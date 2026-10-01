"""Camera ownership: one ffmpeg process per V4L2 device, plus gimbal control.

ffmpeg emits two streams: the re-encoded MJPEG view (also what the detector decodes) and a
tiny grayscale side-output for the motion gate, so motion never needs a JPEG decode.
"""
import json
import os
import subprocess
import threading
import time

import numpy as np

from . import config as C

# name -> (v4l2 control, min, max, scale from user units to v4l2 units)
CTRLS = {
    "pan": ("pan_absolute", -145, 145, 3600),
    "tilt": ("tilt_absolute", -90, 100, 3600),
    "zoom": ("zoom_absolute", 100, 400, 1),
}

state = {"pan": 0, "tilt": 0, "zoom": 100}
latest = None  # newest view JPEG
frame_id = 0
cond = threading.Condition()
quiet_until = 0.0  # motion/tracking muted until this monotonic time (after gimbal moves)


def load_state():
    try:
        with open(C.STATE_FILE) as f:
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
        subprocess.run(["v4l2-ctl", "-d", C.CAM_DEV, "-c", ",".join(parts)], check=False)
        with open(C.STATE_FILE, "w") as f:
            json.dump(state, f)


def latest_frame(after: int = -1, timeout: float = 5):
    """(jpeg, frame_id) of a frame newer than `after`, or (None, after) on timeout."""
    with cond:
        if not cond.wait_for(lambda: latest is not None and frame_id != after, timeout=timeout):
            return None, after
        return latest, frame_id


def _read_gray(fd, on_gray):
    """Drain ffmpeg's gray side-output. Never blocks on anything else: ffmpeg stalls if this pipe isn't drained."""
    w, h = map(int, C.MOTION_SIZE.split("x"))
    n = w * h
    with os.fdopen(fd, "rb") as f:
        while len(raw := f.read(n)) == n:
            on_gray(np.frombuffer(raw, np.uint8).reshape(h, w), time.monotonic() < quiet_until)


def capture(on_gray):
    """Run forever: (re)start ffmpeg, publish view JPEGs, feed gray frames to on_gray(gray, gimbal_moving)."""
    global latest, frame_id
    while True:
        r, w = os.pipe()
        cmd = ["ffmpeg", "-loglevel", "error", "-f", "v4l2", "-input_format", "mjpeg",
               "-video_size", C.SIZE, "-framerate", C.CAM_FPS, "-i", C.CAM_DEV,
               "-filter_complex",
               f"[0:v]fps={C.FPS},split=2[a][b];[a]scale={C.VIEW_WIDTH}:-2[v];"
               f"[b]scale={C.MOTION_SIZE.replace('x', ':')},format=gray[m]",
               "-map", "[v]", "-pix_fmt", "yuvj420p", "-c:v", "mjpeg", "-q:v", C.VIEW_QUALITY,
               "-f", "mjpeg", "pipe:1",
               "-map", "[m]", "-f", "rawvideo", f"pipe:{w}"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, pass_fds=(w,))
        os.close(w)  # child holds the write end; we get EOF when ffmpeg exits
        threading.Thread(target=_read_gray, args=(r, on_gray), daemon=True).start()
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
