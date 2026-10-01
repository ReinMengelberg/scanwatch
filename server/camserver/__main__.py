"""python -m camserver  (from server/): camera + motion gate + detector + uploader + web."""
import threading
from http.server import ThreadingHTTPServer

from . import camera, uploader, web
from . import config as C
from .detect import Detector, load_classifier
from .motion import Motion
from .pipeline import Pipeline, run_live


def main():
    camera.load_state()
    w = int(C.VIEW_WIDTH)
    sw, sh = map(int, C.SIZE.split("x"))
    h = round(w * sh / sw / 2) * 2  # what ffmpeg's scale=W:-2 produces

    motion = Motion()
    pipe = Pipeline(Detector(), load_classifier(), w, h, framing=lambda: dict(camera.state))
    web.ctx.update(pipeline=pipe, motion=motion)

    threading.Thread(target=uploader.run, daemon=True).start()
    threading.Thread(target=camera.capture, args=(motion.update,), daemon=True).start()
    threading.Thread(target=run_live, args=(pipe, motion, camera), daemon=True).start()
    print(f"Serving on http://{C.BIND}:{C.PORT} (spool {C.SPOOL_DIR}, upload {'on' if C.UPLOAD else 'off'})", flush=True)
    ThreadingHTTPServer((C.BIND, C.PORT), web.Handler).serve_forever()


if __name__ == "__main__":
    main()
