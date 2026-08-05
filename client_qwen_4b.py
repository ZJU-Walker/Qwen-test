"""
Real-time camera client for the Qwen3-VL-4B task-planner server (qwen_server_4b.py).

Co-designed with the server for train/inference parity:
  - sends exactly NUM_FRAMES (10) frames per request, OLDEST-FIRST
  - samples them at STRIDE (10) camera frames apart, matching training
    (PrototypeRobotDataset: num_frames=10, stride=10 on 30 fps video)

A background thread streams the rolling 10-frame window to the H200 and updates
the latest prediction; the main thread captures the camera and draws the dashboard.
"""

import cv2
import requests
import time
import io
import threading
import numpy as np
from collections import deque

# --- Configuration ---
# NOTE: update the IP to your H200's address; port 8002 matches qwen_server_4b.py.
SERVER_URL = "http://10.79.12.149:8002/predict"
PROMPT = "which colored block did the human hand point to?"

CAMERA_INDEX = 8
CAMERA_FPS = 30
NUM_FRAMES = 10               # MUST equal the server's NUM_FRAMES
STRIDE = 10                   # training stride; at 30 fps -> frames 0.333s apart, ~3s window
INFERENCE_INTERVAL = 0.001    # ping as fast as possible
BUFFER_SIZE = NUM_FRAMES
RATE_WINDOW = 10

# --- Shared State ---
# Each buffer entry is a tuple: (raw_numpy_frame_for_gui, compressed_jpeg_bytes_for_network)
frame_buffer = deque(maxlen=BUFFER_SIZE)
latest_prediction = "Initializing connection to H200..."

# Lock protects both the buffer and the prediction text
state_lock = threading.Lock()


def build_dashboard(frames, prediction):
    """Creates a composite image with the newest frame large, and history small."""
    main_w, main_h = 640, 480
    history_count = BUFFER_SIZE - 1          # 9 smaller history frames
    hist_w = main_w // history_count         # 71
    hist_h = int(hist_w * (main_h / main_w)) # 53

    # Canvas = header + main image + history row + footer
    canvas = np.zeros((main_h + hist_h + 80, main_w, 3), dtype=np.uint8)

    if not frames:
        return canvas

    # 1. Top Header
    cv2.putText(canvas, "Qwen3-VL-4B Edge Inference Stream", (15, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # 2. Main Frame (the newest frame, index -1)
    main_img = cv2.resize(frames[-1], (main_w, main_h))
    canvas[40:40 + main_h, 0:main_w] = main_img

    # 3. History Grid (the older frames, oldest -> newest left to right)
    for i in range(min(history_count, len(frames) - 1)):
        h_img = cv2.resize(frames[i], (hist_w, hist_h))
        x_offset = i * hist_w
        y_offset = 40 + main_h
        canvas[y_offset:y_offset + hist_h, x_offset:x_offset + hist_w] = h_img
        # Label how many strided steps back this frame is
        cv2.putText(canvas, f"t-{history_count - i}", (x_offset + 5, y_offset + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    # 4. Bottom Footer (the Qwen output)
    cv2.putText(canvas, f"Output: {prediction}", (15, main_h + hist_h + 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)  # Cyan text

    return canvas


def inference_consumer():
    """Background thread: sends the 10-frame window, updates prediction, reports rate."""
    global latest_prediction
    inference_times = deque(maxlen=RATE_WINDOW)

    while len(frame_buffer) < BUFFER_SIZE:
        time.sleep(0.1)

    print(f"Network thread active: pinging {SERVER_URL} every {INFERENCE_INTERVAL}s")

    while True:
        start_time = time.time()

        # Grab the compressed JPEGs, oldest-first (deque iterates left=oldest -> right=newest)
        with state_lock:
            buffer_snapshot = [item[1] for item in frame_buffer]

        files_payload = []
        for i, img_bytes in enumerate(buffer_snapshot):
            files_payload.append(
                ('files', (f'frame_{i}.jpg', io.BytesIO(img_bytes), 'image/jpeg'))
            )

        try:
            response = requests.post(
                SERVER_URL,
                files=files_payload,
                data={"prompt": PROMPT},
                timeout=5.0,  # 4B is heavier than 3B; allow a little more headroom
            )
            if response.status_code == 200:
                with state_lock:
                    latest_prediction = response.json()['prediction']
                elapsed = time.time() - start_time
                inference_times.append(time.time())
                if len(inference_times) >= 2:
                    qwen_rate_hz = (len(inference_times) - 1) / (inference_times[-1] - inference_times[0])
                else:
                    qwen_rate_hz = 0.0
                print(
                    f"[{time.strftime('%X')}] Server: {latest_prediction} "
                    f"| latency={elapsed * 1000:.0f} ms | qwen_rate={qwen_rate_hz:.2f} Hz"
                )
            else:
                with state_lock:
                    latest_prediction = f"Server error {response.status_code}: {response.text[:80]}"
                print(f"Server returned {response.status_code}: {response.text[:200]}")

        except requests.exceptions.RequestException as e:
            with state_lock:
                latest_prediction = "Network Lag / Error connecting to H200"
            print(f"Network fault: {e}")

        elapsed = time.time() - start_time
        sleep_duration = max(0.0, INFERENCE_INTERVAL - elapsed)
        time.sleep(sleep_duration)


def main_gui_loop():
    """Main thread: captures video, manages stride, draws OpenCV dashboard."""
    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FPS, CAMERA_FPS)

    if not cap.isOpened():
        print(f"Camera fault (index {CAMERA_INDEX}). Exiting.")
        return

    frame_count = 0
    is_initialized = False

    print(f"GUI started: {CAMERA_FPS}Hz hardware, sending {NUM_FRAMES} frames @ stride {STRIDE}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                continue

            frame_count += 1

            # Compress for the network
            _, encoded_img = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            img_bytes = encoded_img.tobytes()

            with state_lock:
                # Rule 1: prime the buffer by duplicating the first frame
                # (mirrors training's clamping of negative indices to frame 0).
                if not is_initialized:
                    for _ in range(BUFFER_SIZE):
                        frame_buffer.append((frame, img_bytes))
                    is_initialized = True

                # Rule 2: only enqueue every STRIDE-th camera frame
                elif frame_count % STRIDE == 0:
                    frame_buffer.append((frame, img_bytes))

                # Snapshot for drawing
                current_frames = [item[0] for item in frame_buffer]
                current_pred = latest_prediction

            dashboard = build_dashboard(current_frames, current_pred)
            cv2.imshow("Qwen3-VL-4B Real-Time Robotics View", dashboard)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    net_thread = threading.Thread(target=inference_consumer, daemon=True)
    net_thread.start()
    main_gui_loop()
    print("\nStream closed safely.")
