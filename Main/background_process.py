import time
import cv2
import asyncio
import json
import websockets
from ultralytics import YOLO
from concurrent.futures import ThreadPoolExecutor

# Config
HOST = "0.0.0.0"
PORT = 8765
MODEL_PATH = "C:/Users/dayse/VSCODE files/Projects/Yolov11_PL/Models/best.pt"
#-------

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(0)
data_queue = asyncio.Queue()
connected_clients = set()
executor = ThreadPoolExecutor()

#Websocket broadcaster
async def handler(websocket):
        connected_clients.add(websocket)
        print("Client Connected")
        try:
            await websocket.wait_closed()
        finally:
            connected_clients.remove(websocket)
            print("Client Disconnected")

#Data broadcaster Co-routine
async def broadcaster():
    #Continuously send detections to connected WebSocket clients
    while True:
        frame_data = await data_queue.get()
        if frame_data is None:
            break

        if connected_clients:
            message = json.dumps(frame_data)
            await asyncio.gather(
                *[client.send(message) for client in connected_clients],
                return_exceptions=True
            )

        data_queue.task_done()

#YOLO detection loop
def detect_and_enqueue(loop):
    #Capture webcam frames, run YOLO, and enqueue structured results
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1

        results = model(frame, verbose=False)
        detections = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = results[0].names[cls]
            detections.append({
                "label": label,
                "confidence": conf,
                "bbox": [x1, y1, x2, y2]
            })

        frame_data = {
            "timestamp": time.time(),
            "frame_id": frame_count,
            "detections": detections
        }

        asyncio.run_coroutine_threadsafe(data_queue.put(frame_data), loop)

        annotated = results[0].plot()
        cv2.imshow("Live", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    asyncio.run_coroutine_threadsafe(data_queue.put(None), loop)
    cap.release()
    cv2.destroyAllWindows()

async def main():

    ws_server = await websockets.serve(handler, HOST, PORT)
    print(f"Websocket server running on ws://127.0.0.1:{PORT}")

    broadcaster_task = asyncio.create_task(broadcaster())

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(executor, detect_and_enqueue, loop)

    await broadcaster_task

    ws_server.close()
    await ws_server.wait_closed()


if __name__ == "__main__":
    asyncio.run(main())
