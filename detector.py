"""Always-on motion detector that sends a photo to Telegram when motion is seen.

Config (environment variables):
    TG_BOT_TOKEN   bot token from @BotFather (required)
    TG_CHAT_ID     chat to send to; if unset, run once and message the bot,
                   the script will print your chat id
    CAMERA_INDEX   camera number, default 0
    MIN_AREA       min changed area in pixels to count as motion, default 1500
    COOLDOWN       seconds between alerts, default 10
"""

import os
import sys
import threading
import time
from datetime import datetime

import cv2
import requests


def load_env(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")):
    """Load KEY=VALUE lines from .env next to this script into os.environ."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ.setdefault(key.strip(), value.strip())


load_env()

TOKEN = os.environ.get("TG_BOT_TOKEN", "")
CHAT_ID = os.environ.get("TG_CHAT_ID", "")
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
MIN_AREA = int(os.environ.get("MIN_AREA", "1500"))
COOLDOWN = float(os.environ.get("COOLDOWN", "30"))

API = "https://api.telegram.org/bot{}/".format(TOKEN)


def find_chat_id():
    """Print chat ids of people who messaged the bot recently."""
    r = requests.get(API + "getUpdates", timeout=10).json()
    chats = {}
    for u in r.get("result", []):
        msg = u.get("message") or u.get("channel_post") or {}
        chat = msg.get("chat")
        if chat:
            chats[chat["id"]] = chat.get("username") or chat.get("title") or ""
    if not chats:
        print("No messages found. Send any message to your bot, then run again.")
    for cid, name in chats.items():
        print("chat id: {}  ({})".format(cid, name))


def send_photo(frame, caption):
    ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        return
    # Retry so a short network/DNS dropout doesn't lose the photo
    for attempt in range(5):
        try:
            r = requests.post(
                API + "sendPhoto",
                data={"chat_id": CHAT_ID, "caption": caption},
                files={"photo": ("motion.jpg", jpg.tobytes(), "image/jpeg")},
                timeout=30,
            )
            if not r.ok:
                print("Telegram error:", r.text)
            return
        except requests.RequestException as e:
            print("Send failed (attempt {}/5): {}".format(attempt + 1, e))
            time.sleep(5 * (attempt + 1))


def open_camera():
    while True:
        cap = cv2.VideoCapture(CAMERA_INDEX)
        if cap.isOpened():
            # Let auto exposure settle so the first frames don't look like motion
            for _ in range(30):
                cap.read()
            return cap
        print("Camera not available, retrying in 5s...")
        time.sleep(5)


def prepare(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (21, 21), 0)


def run():
    cap = open_camera()
    background = None
    last_alert = 0.0
    print("Motion detector running. Ctrl+C to stop.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Lost camera, reconnecting...")
            cap.release()
            cap = open_camera()
            background = None
            continue

        gray = prepare(frame)
        if background is None:
            background = gray.astype("float")
            continue

        # Running average background adapts to slow light changes
        cv2.accumulateWeighted(gray, background, 0.05)
        delta = cv2.absdiff(gray, cv2.convertScaleAbs(background))
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        moving = [c for c in contours if cv2.contourArea(c) >= MIN_AREA]
        if moving and time.time() - last_alert >= COOLDOWN:
            last_alert = time.time()
            shot = frame.copy()
            for c in moving:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(shot, (x, y), (x + w, y + h), (0, 0, 255), 2)
            caption = "Motion detected {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            print(caption)
            # Send in background so detection keeps running during upload
            threading.Thread(target=send_photo, args=(shot, caption), daemon=True).start()

        time.sleep(0.05)


if __name__ == "__main__":
    if not TOKEN:
        sys.exit("Set TG_BOT_TOKEN environment variable.")
    if not CHAT_ID:
        find_chat_id()
        sys.exit("Set TG_CHAT_ID and run again.")
    try:
        run()
    except KeyboardInterrupt:
        print("Stopped.")
