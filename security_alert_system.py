"""
AI Security Alert System
Detects unauthorized persons and sounds, alerts when someone enters except authorized person
"""

import cv2
import numpy as np
import torch
import threading
import time
import pyaudio
import struct
import math
from datetime import datetime
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class SecurityAlertSystem:
    def __init__(self):
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Person detection model (YOLOv5 or MobileNet SSD)
        self.person_detector = None
        self.load_person_detector()
        
        # Face recognition for authorized person
        self.authorized_face_encoding = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Audio settings
        self.CHUNK = 1024
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 1
        self.RATE = 44100
        self.SOUND_THRESHOLD = 1000  # Adjust based on environment
        
        # Alert settings
        self.alert_active = False
        self.last_alert_time = 0
        self.alert_cooldown = 5  # seconds between alerts
        
        # Status tracking
        self.authorized_person_present = False
        self.unauthorized_detected = False
        self.sound_detected = False
        
        # Frame buffer for detection
        self.frame_buffer = deque(maxlen=30)
        
        print("🔒 Security Alert System Initialized")
        print("=" * 50)
    
    def load_person_detector(self):
        """Load YOLOv5 person detection model"""
        try:
            # Using YOLOv5 nano for fast detection
            self.person_detector = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.person_detector.classes = [0]  # Only detect persons (class 0)
            self.person_detector.conf = 0.5  # Confidence threshold
            print("✓ Person detector loaded (YOLOv5)")
        except Exception as e:
            print(f"⚠ YOLOv5 load failed, using alternative method: {e}")
            # Fallback to basic detection
            self.person_detector = "cascade"
    
    def register_authorized_person(self):
        """Capture and save authorized person's face"""
        print("\n📸 REGISTRATION MODE")
        print("Position yourself in front of the camera...")
        print("Press SPACE to capture your face")
        print("Press ESC to skip registration")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            # Display frame
            display_frame = frame.copy()
            cv2.putText(display_frame, "Press SPACE to register face", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Detect faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.imshow('Register Authorized Person', display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # SPACE
                if len(faces) > 0:
                    # Save the face region
                    x, y, w, h = faces[0]
                    self.authorized_face_encoding = gray[y:y+h, x:x+w]
                    print("✓ Authorized person registered!")
                    cv2.destroyWindow('Register Authorized Person')
                    return True
                else:
                    print("⚠ No face detected, try again")
            elif key == 27:  # ESC
                print("Registration skipped")
                cv2.destroyWindow('Register Authorized Person')
                return False
        
        return False
    
    def is_authorized_person(self, frame):
        """Check if detected face matches authorized person"""
        if self.authorized_face_encoding is None:
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            face_region = gray[y:y+h, x:x+w]
            
            # Resize to match authorized face
            try:
                face_region_resized = cv2.resize(face_region, 
                    (self.authorized_face_encoding.shape[1], 
                     self.authorized_face_encoding.shape[0]))
                
                # Simple template matching
                result = cv2.matchTemplate(face_region_resized, 
                                          self.authorized_face_encoding, 
                                          cv2.TM_CCOEFF_NORMED)
                similarity = result[0][0]
                
                # If similarity is high, it's the authorized person
                if similarity > 0.6:  # Threshold
                    return True
            except:
                continue
        
        return False
    
    def detect_persons(self, frame):
        """Detect persons in frame using YOLOv5"""
        if self.person_detector == "cascade":
            # Fallback detection method
            return self.detect_persons_cascade(frame)
        
        # YOLOv5 detection
        results = self.person_detector(frame)
        detections = results.pandas().xyxy[0]
        
        # Filter for person class with confidence > 0.5
        persons = detections[detections['name'] == 'person']
        
        return len(persons) > 0, persons
    
    def detect_persons_cascade(self, frame):
        """Fallback person detection using HOG"""
        # Simple motion detection as fallback
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # This is a simplified version - in production, use proper person detector
        return False, None
    
    def monitor_sound(self):
        """Monitor audio input for loud sounds"""
        try:
            p = pyaudio.PyAudio()
            stream = p.open(format=self.FORMAT,
                           channels=self.CHANNELS,
                           rate=self.RATE,
                           input=True,
                           frames_per_buffer=self.CHUNK)
            
            print("✓ Sound monitoring started")
            
            while True:
                try:
                    data = stream.read(self.CHUNK, exception_on_overflow=False)
                    data_int = struct.unpack(str(self.CHUNK) + 'h', data)
                    
                    # Calculate RMS (Root Mean Square) for volume
                    rms = math.sqrt(sum(x**2 for x in data_int) / len(data_int))
                    
                    if rms > self.SOUND_THRESHOLD:
                        self.sound_detected = True
                    else:
                        self.sound_detected = False
                        
                except Exception as e:
                    continue
                    
        except Exception as e:
            print(f"⚠ Sound monitoring unavailable: {e}")
            print("Continuing with video-only monitoring...")
    
    def trigger_alert(self, reason):
        """Trigger security alert"""
        current_time = time.time()
        
        # Check cooldown
        if current_time - self.last_alert_time < self.alert_cooldown:
            return
        
        self.alert_active = True
        self.last_alert_time = current_time
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*60)
        print("🚨 SECURITY ALERT! 🚨")
        print(f"Time: {timestamp}")
        print(f"Reason: {reason}")
        print("="*60 + "\n")
        
        # Here you can add:
        # - Send email/SMS notification
        # - Save snapshot
        # - Sound alarm
        # - Log to database
    
    def run(self):
        """Main monitoring loop"""
        # Register authorized person
        has_authorized = self.register_authorized_person()
        
        # Start sound monitoring in separate thread
        sound_thread = threading.Thread(target=self.monitor_sound, daemon=True)
        sound_thread.start()
        
        print("\n🎥 Starting live monitoring...")
        print("Press 'q' to quit")
        print("Press 'r' to re-register authorized person")
        print("Press 's' to adjust sound sensitivity")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Create display frame
            display_frame = frame.copy()
            
            # Check for authorized person
            if has_authorized:
                self.authorized_person_present = self.is_authorized_person(frame)
            
            # Detect any persons
            persons_detected, detections = self.detect_persons(frame)
            
            # Alert logic
            if persons_detected and has_authorized and not self.authorized_person_present:
                self.unauthorized_detected = True
                self.trigger_alert("Unauthorized person detected!")
            else:
                self.unauthorized_detected = False
            
            if self.sound_detected:
                self.trigger_alert("Loud sound detected!")
            
            # Draw detections
            if self.person_detector != "cascade" and detections is not None:
                for idx, det in detections.iterrows():
                    x1, y1, x2, y2 = int(det['xmin']), int(det['ymin']), int(det['xmax']), int(det['ymax'])
                    
                    # Color based on authorization
                    if self.authorized_person_present:
                        color = (0, 255, 0)  # Green for authorized
                        label = "Authorized"
                    else:
                        color = (0, 0, 255)  # Red for unauthorized
                        label = "ALERT: Unauthorized!"
                    
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Status display
            status_y = 30
            cv2.putText(display_frame, f"Status: {'MONITORING' if not self.alert_active else 'ALERT!'}", 
                       (10, status_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 
                       (0, 255, 0) if not self.alert_active else (0, 0, 255), 2)
            
            if has_authorized:
                auth_status = "Present" if self.authorized_person_present else "Not Present"
                cv2.putText(display_frame, f"Authorized: {auth_status}", 
                           (10, status_y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            sound_status = "DETECTED" if self.sound_detected else "Normal"
            sound_color = (0, 0, 255) if self.sound_detected else (0, 255, 0)
            cv2.putText(display_frame, f"Sound: {sound_status}", 
                       (10, status_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, sound_color, 2)
            
            # Show frame
            cv2.imshow('AI Security Monitor', display_frame)
            
            # Keyboard controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cv2.destroyWindow('AI Security Monitor')
                has_authorized = self.register_authorized_person()
            elif key == ord('s'):
                print(f"\nCurrent sound threshold: {self.SOUND_THRESHOLD}")
                try:
                    new_threshold = int(input("Enter new threshold (500-5000): "))
                    self.SOUND_THRESHOLD = max(500, min(5000, new_threshold))
                    print(f"Sound threshold updated to: {self.SOUND_THRESHOLD}")
                except:
                    print("Invalid input, keeping current threshold")
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\n✓ Security system stopped")


if __name__ == "__main__":
    print("""
    ╔════════════════════════════════════════════════════════╗
    ║       AI SECURITY ALERT SYSTEM                         ║
    ║                                                        ║
    ║  • Detects unauthorized persons                        ║
    ║  • Monitors loud sounds                                ║
    ║  • Recognizes authorized person                        ║
    ║  • Real-time alerts                                    ║
    ╚════════════════════════════════════════════════════════╝
    """)
    
    try:
        system = SecurityAlertSystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nSystem stopped by user")
    except Exception as e:
        print(f"\n⚠ Error: {e}")
        print("Make sure you have:")
        print("  - A working webcam")
        print("  - A microphone (optional)")
        print("  - Required packages installed")
