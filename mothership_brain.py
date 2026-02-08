import mmap
import numpy as np
import cv2
import time
import ollama
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Optional
import json

@dataclass
class Entity:
    label: str
    confidence: float
    bbox: List[float]  # [x1, y1, x2, y2]
    
class MothershipBrain:
    def __init__(self, shm_name="Global\\LoLBotFrame", shm_size=1024*1024*64):
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.shm = None
        self.vision_model = None
        self.tactical_model = "llama3.2-vision" # Highly recommended for 5080
        
    def connect_to_eyes(self) -> bool:
        """Connects to the C++ Shared Memory bridge."""
        try:
            self.shm = mmap.mmap(-1, self.shm_size, tagname=self.shm_name, access=mmap.ACCESS_READ)
            print(f"Connected to RAM bridge: {self.shm_name}")
            return True
        except FileNotFoundError:
            print(f"Error: Shared memory '{self.shm_name}' not found. Run the C++ capturer first!")
            return False

    def load_models(self):
        """Loads both the fast vision (YOLO) and strategic (Ollama) models."""
        print("Loading YOLO vision engine...")
        self.vision_model = YOLO("yolov10n.pt")
        
        print(f"Checking local Ollama for '{self.tactical_model}'...")
        # We assume the user has Ollama running locally
        try:
            ollama.list() 
        except Exception as e:
            print(f"Warning: Local Ollama server not detected. Tactical reasoning will be disabled. {e}")

    def process_loop(self, w=1920, h=1080):
        """Main loop for vision processing and tactical decision making."""
        if not self.shm:
            return

        print("Mothership Brain active. Commencing full-stack AI processing...")
        last_tactical_update = 0
        
        while True:
            start_time = time.time()
            
            # 1. READ FRAME FROM RAM
            self.shm.seek(0)
            frame_bytes = self.shm.read(w * h * 4) 
            frame = np.frombuffer(frame_bytes, dtype=np.uint8).reshape((h, w, 4))
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            
            # 2. RUN FAST VISION (YOLO)
            results = self.vision_model(frame_bgr, verbose=False, device=0) 
            entities = []
            for result in results:
                for box in result.boxes:
                    entities.append(Entity(
                        label=self.vision_model.names[int(box.cls)],
                        confidence=float(box.conf),
                        bbox=box.xyxy[0].tolist()
                    ))
            
            # 3. TACTICAL REASONING (Ollama) - Run every 1 second (High-level strategy)
            if time.time() - last_tactical_update > 1.0:
                decision = self._ask_tactical_expert(frame_bgr, entities)
                print(f"\n[TACTICAL DECISION]: {decision}")
                last_tactical_update = time.time()

            # 4. REACTIVE LOGIC (Immediate output based on YOLO)
            # This would call the C++ Input DLL eventually
            self._reactive_execution(entities)

            latency = (time.time() - start_time) * 1000
            print(f"Vision Latency: {latency:.1f}ms | Entities: {len(entities)}", end='\r')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    def _ask_tactical_expert(self, frame_bgr, entities: List[Entity]) -> str:
        """Sends game state to local Ollama for strategic analysis."""
        # Convert entities to a readable summary
        entity_summary = [f"{e.label} at {e.bbox}" for e in entities if e.confidence > 0.5]
        
        prompt = f"""
        Game State: {json.dumps(entity_summary)}
        Goal: You are playing League of Legends. Analyze the view and entities.
        Task: Provide a ONE-SENTENCE tactical command (e.g., 'Retreat to tower', 'Engage Garen', 'Farm minions').
        """
        
        try:
            # We can also send the image if using a vision model!
            _, buffer = cv2.imencode('.jpg', cv2.resize(frame_bgr, (640, 360)))
            
            response = ollama.chat(
                model=self.tactical_model,
                messages=[{
                    'role': 'user',
                    'content': prompt,
                    'images': [buffer.tobytes()]
                }]
            )
            return response['message']['content']
        except Exception as e:
            return f"Ollama Error: {e}"

    def _reactive_execution(self, entities: List[Entity]):
        """Reactive logic based on immediate YOLO detections (Skill dodging, etc.)"""
        # Placeholder for calling the C++ DLL for Input
        pass

if __name__ == "__main__":
    brain = MothershipBrain()
    if brain.connect_to_eyes():
        brain.load_models()
        brain.process_loop(w=1920, h=1080)
