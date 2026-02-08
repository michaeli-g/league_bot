import mmap
import numpy as np
import cv2
import time

def verify_shm():
    shm_name = "Global\\LoLBotFrame"
    shm_size = 1024 * 1024 * 64
    
    print(f"Connecting to shared memory: {shm_name}...")
    
    try:
        # Open existing shared memory created by C++
        # Note: In Windows, mmap.mmap needs a handle or just the name
        shm = mmap.mmap(-1, shm_size, tagname=shm_name, access=mmap.ACCESS_READ)
    except FileNotFoundError:
        print("Error: Shared memory not found. Make sure the C++ capturer is running!")
        return

    print("Connected! Reading frames...")

    while True:
        # Reset pointer
        shm.seek(0)
        
        # Read raw bytes (assuming 4K max, we can detect resolution in header if we add one)
        # For now, let's assume 1920x1080 for testing
        w, h = 1920, 1080
        frame_bytes = shm.read(w * h * 4)
        
        # Convert to numpy array
        frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 4))
        
        # Convert BGRA to BGR for display
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        
        # Resize for preview
        preview = cv2.resize(frame_bgr, (960, 540))
        
        cv2.imshow("Mothership View", preview)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    shm.close()

if __name__ == "__main__":
    verify_shm()
