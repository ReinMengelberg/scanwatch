"""All settings, from server/.env. Real environment variables win (systemd, shell)."""
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path):
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        v = re.split(r"\s+#", v, maxsplit=1)[0]  # trailing comment
        os.environ.setdefault(k.strip(), v.strip().strip("'\""))


_load_dotenv(ROOT / ".env")
env = os.environ.get


def _path(v: str) -> str:
    """~ expanded; relative paths are relative to server/ (so models/... works from any cwd)."""
    if not v:
        return v
    p = Path(os.path.expanduser(v))
    return str(p if p.is_absolute() else ROOT / p)


def _roi(v: str) -> list[tuple[float, float]]:
    """'x,y;x,y;...' normalised 0-1 -> [(x, y)]. Empty = whole frame."""
    return [tuple(float(c) for c in p.split(",")) for p in v.split(";") if p.strip()] if v else []


# Camera + web
CAM_DEV = env("CAM_DEV", "/dev/video0")
BIND = env("BIND", "127.0.0.1")  # localhost only; reach it through an SSH tunnel
PORT = int(env("PORT", "8080"))
SIZE = env("SIZE", "1280x720")
CAM_FPS = env("CAM_FPS", "15")  # a mode the camera supports (v4l2-ctl --list-formats-ext)
FPS = env("FPS", "10")  # processing/view rate, frames are dropped down to this
# The Link sends ~400 KB MJPEG frames; re-encode so the view fits through the SSH tunnel.
# 960 wide = 960x540, the frame size the models are trained on. Don't change one without the other.
VIEW_WIDTH = env("VIEW_WIDTH", "960")
VIEW_QUALITY = env("VIEW_QUALITY", "7")  # ffmpeg -q:v, 2 = best, 31 = worst
STATE_FILE = _path(env("STATE_FILE", "~/.camserver.json"))

# Motion gate: only decides when to run the detector, never what gets stored
MOTION_SIZE = env("MOTION_SIZE", "160x90")  # analysis resolution, same aspect ratio as SIZE
MOTION_DIFF = float(env("MOTION_DIFF", "25"))  # per-pixel gray delta that counts as changed
MOTION_MIN = float(env("MOTION_MIN", "0.005"))  # changed fraction of the ROI that wakes the detector
MOTION_MAX = float(env("MOTION_MAX", "0.6"))  # above this: exposure jump or gimbal move, ignore
MOTION_HOLD = float(env("MOTION_HOLD", "3"))  # keep detecting this long after the last motion
BG_RATE = float(env("BG_RATE", "0.1"))  # background adaptation per frame
ROI = _roi(env("ROI", ""))  # road polygon; motion and tracks only count inside it

# Detection + tracking
DET_MODEL = _path(env("DET_MODEL", "models/yolo11n_416_ncnn_model"))
DET_IMGSZ = int(env("DET_IMGSZ", "416"))
DET_CONF = float(env("DET_CONF", "0.25"))
MIN_BOX = int(env("MIN_BOX", "40"))  # px, shorter side; must match training (crop.MIN_BOX)
# Separate pass before a full frame is stored: bigger input, low confidence, persons + all vehicle
# types, so parked cars the tracker ignores still get their plates blurred. Ship a 640 export for this.
BLUR_MODEL = _path(env("BLUR_MODEL", "models/yolo11n_640_ncnn_model"))
BLUR_IMGSZ = int(env("BLUR_IMGSZ", "640"))
BLUR_CONF = float(env("BLUR_CONF", "0.05"))
TRACK_IOU = float(env("TRACK_IOU", "0.2"))  # box overlap that continues a track
TRACK_LOST = float(env("TRACK_LOST", "1.5"))  # seconds unseen before a track ends
MIN_TRAVEL = float(env("MIN_TRAVEL", "0.15"))  # fraction of frame width a track must move (parked cars don't)
CROPS_PER_TRACK = int(env("CROPS_PER_TRACK", "3"))

# Stage 2 (optional until a model is trained)
CLS_MODEL = _path(env("CLS_MODEL", ""))  # e.g. models/scancar_cls_ncnn_model
CLS_IMGSZ = int(env("CLS_IMGSZ", "224"))
CLS_THRESHOLD = env("CLS_THRESHOLD", "")  # alert threshold; empty = threshold.json next to CLS_MODEL
# Which S3 folder a track lands in: scancar/ if P(scancar) >= this, else car/. Deliberately lower than
# the alert threshold: a false positive costs a glance, a false negative gets buried. No model = car/.
ROUTE_THRESHOLD = float(env("ROUTE_THRESHOLD", "0.3"))

# Discord alert when P(scancar) >= the alert threshold (only anonymized images are sent)
DISCORD_WEBHOOK_URL = env("DISCORD_WEBHOOK_URL", "")  # channel settings > Integrations > Webhooks
DISCORD_MIN_GAP = float(env("DISCORD_MIN_GAP", "60"))  # seconds; one pass can split into two tracks

# Empty-street frames (synth backgrounds, motion tuning)
EMPTY_EVERY = float(env("EMPTY_EVERY", "3600"))  # seconds between empty frames
EMPTY_QUIET = float(env("EMPTY_QUIET", "60"))  # no motion in the ROI for this long first

# Spool (SD card) + S3
ANNOTATE = env("ANNOTATE", "1") == "1"  # also upload annotated.jpg (frame + boxes + path) per track

SPOOL_DIR = _path(env("SPOOL_DIR", "~/scanwatch/spool"))
MIN_FREE_GB = float(env("MIN_FREE_GB", "2"))  # below this, drop the oldest spooled days
S3_ENDPOINT = env("S3_ENDPOINT", "")
S3_BUCKET = env("S3_BUCKET", "")
S3_REGION = env("S3_REGION", "us-east-1")
S3_ACCESS_KEY_ID = env("S3_ACCESS_KEY_ID", "")
S3_SECRET_ACCESS_KEY = env("S3_SECRET_ACCESS_KEY", "")
S3_PREFIX = env("S3_PREFIX", "").strip("/")
UPLOAD = env("UPLOAD", "1") == "1"  # 0 = only spool locally
