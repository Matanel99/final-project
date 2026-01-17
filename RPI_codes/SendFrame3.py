import os
import cv2
import httpx
import asyncio
import time

SERVER_ENDPOINT = os.environ.get("RPI_YOLO_ENDPOINT", "http://192.168.68.109:3000/analyzeImage")
SEND_IMG_SIZE = int(os.environ.get("RPI_SEND_IMG_SIZE", "480"))
JPEG_QUALITY  = int(os.environ.get("RPI_JPEG_QUALITY", "60"))

_http_client = httpx.AsyncClient(http2=False, timeout=10.0)

print(f"[SendFrame3] Using endpoint: {SERVER_ENDPOINT}")

def _downscale_bgr(img_bgr, out_size):
    if img_bgr is None:
        return None
    if out_size and out_size > 0 and (img_bgr.shape[1] != out_size or img_bgr.shape[0] != out_size):
        return cv2.resize(img_bgr, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return img_bgr

def _encode_jpeg(img_bgr, quality):
    ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None

async def handleRemoteDetection(frame_bgr, detectionQueue):
    try:
        small = _downscale_bgr(frame_bgr, SEND_IMG_SIZE)
        jpg_bytes = _encode_jpeg(small, JPEG_QUALITY)
        if jpg_bytes is None:
            await detectionQueue.put(None)
            return

        files = {"image": ("frame.jpg", jpg_bytes, "image/jpeg")}

        start = time.perf_counter()
        r = await _http_client.post(SERVER_ENDPOINT, files=files)
        e2e_ms = (time.perf_counter() - start) * 1000.0

        r.raise_for_status()
        data = r.json()

        if data and data.get("success"):
            dets = data.get("result", [])
            await detectionQueue.put(dets if dets else None)
        else:
            await detectionQueue.put(None)

    except httpx.HTTPError:
        await detectionQueue.put(None)
    except Exception:
        await detectionQueue.put(None)

if __name__ == "__main__":
    import numpy as np
    async def _test():
        q = asyncio.Queue()
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        await handleRemoteDetection(dummy, q)
        print("Result:", await q.get())
        await _http_client.aclose()
    asyncio.run(_test())
