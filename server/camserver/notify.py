"""Discord alerts via a channel webhook.

Test the webhook from the Pi:
  .venv/bin/python -m camserver.notify                 text only
  .venv/bin/python -m camserver.notify path/to/img.jpg with an image
"""
import json
import queue
import sys
import threading
import time

import requests

from . import config as C


class Discord:
    def __init__(self, url: str = C.DISCORD_WEBHOOK_URL):
        self.url = url
        self.q: queue.Queue = queue.Queue(maxsize=20)
        self.last_sent = -1e18
        self.status = {"sent": 0, "skipped": 0, "errors": 0, "last_error": ""}
        threading.Thread(target=self._run, daemon=True).start()

    def alert(self, track_id: str, t: float, score: float, images: dict[str, bytes]):
        """Queue an alert for a track starting at epoch time t; never blocks the pipeline. One scan car
        can split into two tracks, so alerts within DISCORD_MIN_GAP seconds of the previous one are dropped."""
        if t - self.last_sent < C.DISCORD_MIN_GAP:
            self.status["skipped"] += 1
            print(f"discord: skipped {track_id}, within {C.DISCORD_MIN_GAP:.0f}s of the previous alert", flush=True)
            return
        self.last_sent = t
        text = f"🚨 **Scan car** spotted at {time.strftime('%H:%M:%S', time.localtime(t))} (P = {score:.0%})"
        try:
            self.q.put_nowait((text, images, track_id))
        except queue.Full:
            self.status["skipped"] += 1

    def _run(self):
        while True:
            text, images, track_id = self.q.get()
            for attempt in range(5):
                try:
                    send(self.url, text, images)
                    self.status["sent"] += 1
                    print(f"discord: sent {track_id}", flush=True)
                    break
                except RateLimited as e:
                    time.sleep(e.retry_after)
                except Exception as e:  # network down: retry a few times, then give up on this one
                    self.status["errors"] += 1
                    self.status["last_error"] = f"{time.strftime('%H:%M:%S')} {type(e).__name__}: {e}"[:300]
                    time.sleep(10 * (attempt + 1))


class RateLimited(Exception):
    def __init__(self, retry_after: float):
        self.retry_after = retry_after


def send(url: str, text: str, images: dict[str, bytes] | None = None):
    images = images or {}
    names = list(images)
    payload = {"content": text, "allowed_mentions": {"parse": []}}
    if names:  # first image large in an embed, the rest as attachments below it
        payload["embeds"] = [{"image": {"url": f"attachment://{names[0]}"}, "color": 0xE5484D}]
    files = {f"files[{i}]": (n, images[n], "image/jpeg") for i, n in enumerate(names)}
    r = requests.post(url, data={"payload_json": json.dumps(payload)}, files=files or None, timeout=20)
    if r.status_code == 429:
        raise RateLimited(float(r.json().get("retry_after", 5)))
    r.raise_for_status()


def load():
    if not C.DISCORD_WEBHOOK_URL:
        print("no DISCORD_WEBHOOK_URL: no alerts", flush=True)
        return None
    return Discord()


if __name__ == "__main__":
    if not C.DISCORD_WEBHOOK_URL:
        sys.exit("set DISCORD_WEBHOOK_URL in server/.env first")
    imgs = {"test.jpg": open(sys.argv[1], "rb").read()} if len(sys.argv) > 1 else {}
    send(C.DISCORD_WEBHOOK_URL, "✅ scanwatch test message: the webhook works", imgs)
    print("sent")
