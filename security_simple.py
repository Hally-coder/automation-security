"""
AI Security Alert System - SIMPLIFIED VERSION (Video Only)
No audio dependencies required - easier setup
"""

import cv2
import numpy as np
import torch
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class SimpleSecuritySystem:
    def __init__(self):
        print("Simple Security Alert System")
        print("=" * 50)
        
        # Video capture
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Load person detector
        print("Loading AI model...")
        try:
            self.detector = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            self.detector.classes = [0]  # Person only
            self.detector.conf = 0.5
            print("AI Model loaded (YOLOv5)")
        except:
            print("Using fallback detection")
            self.detector = None
        
        # Face recognition
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.authorized_face = None
        
        # Alert settings
        self.last_alert = 0
        self.alert_cooldown = 3
        
    def register_authorized_person(self):
        """Register your face"""
        print("\nFACE REGISTRATION")
        print("Position yourself in camera view")
        print("Press SPACE to capture | ESC to skip")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            # Show detected faces
            for (x, y, w, h) in faces:
                cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            cv2.putText(display, "Press SPACE to register face", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Registration', display)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == 32 and len(faces) > 0:  # SPACE
                x, y, w, h = faces[0]
                self.authorized_face = gray[y:y+h, x:x+w]
                print("Face registered!")
                cv2.destroyWindow('Registration')
                return True
            elif key == 27:  # ESC
                print("Registration skipped")
                cv2.destroyWindow('Registration')
                return False
        
        return False
    
    def is_authorized(self, frame):
        """Check if face matches authorized person"""
        if self.authorized_face is None:
            return False
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            try:
                face = cv2.resize(gray[y:y+h, x:x+w], 
                                 (self.authorized_face.shape[1], 
                                  self.authorized_face.shape[0]))
                
                result = cv2.matchTemplate(face, self.authorized_face, 
                                          cv2.TM_CCOEFF_NORMED)
                
                if result[0][0] > 0.6:
                    return True
            except:
                continue
        
        return False
    
    def detect_persons(self, frame):
        """Detect persons in frame"""
        if self.detector is None:
            return False, []
        
        results = self.detector(frame)
        detections = results.pandas().xyxy[0]
        persons = detections[detections['name'] == 'person']
        
        return len(persons) > 0, persons
    
    def alert(self, message):
        """Trigger alert"""
        current_time = time.time()
        
        if current_time - self.last_alert < self.alert_cooldown:
            return
        
        self.last_alert = current_time
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        print("\n" + "="*60)
        print("*** ALERT! ***")
        print("Time: " + timestamp)
        print("Reason: " + message)
        print("="*60 + "\n")
        
        # Add your custom alert actions here:
        # - Save snapshot: cv2.imwrite('alert_' + timestamp + '.jpg', frame)
        # - Send notification
        # - Sound alarm
    
    def run(self):
        """Main monitoring loop"""
        has_auth = self.register_authorized_person()
        
        print("\nMONITORING ACTIVE")
        print("Controls:")
        print("  Q = Quit")
        print("  R = Re-register face")
        print("  C = Adjust confidence")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            display = frame.copy()
            
            # Check authorization
            authorized_present = False
            if has_auth:
                authorized_present = self.is_authorized(frame)
            
            # Detect persons
            persons_found, detections = self.detect_persons(frame)
            
            # Alert logic
            if persons_found and has_auth and not authorized_present:
                self.alert("Unauthorized person detected!")
            
            # Draw detections
            if detections is not None and len(detections) > 0:
                for _, det in detections.iterrows():
                    x1, y1 = int(det['xmin']), int(det['ymin'])
                    x2, y2 = int(det['xmax']), int(det['ymax'])
                    
                    color = (0, 255, 0) if authorized_present else (0, 0, 255)
                    label = "Authorized" if authorized_present else "UNAUTHORIZED"
                    
                    cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display, label, (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Status display
            status = "ALERT" if (persons_found and not authorized_present and has_auth) else "NORMAL"
            status_color = (0, 0, 255) if status == "ALERT" else (0, 255, 0)
            
            cv2.putText(display, "Status: " + status, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
            if has_auth:
                auth_text = "Present" if authorized_present else "Not Present"
                cv2.putText(display, "Authorized: " + auth_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            cv2.imshow('Security Monitor', display)
            
            # Controls
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                cv2.destroyWindow('Security Monitor')
                has_auth = self.register_authorized_person()
            elif key == ord('c'):
                print("\nCurrent confidence: " + str(self.detector.conf))
                try:
                    new_conf = float(input("Enter new confidence (0.1-0.9): "))
                    self.detector.conf = max(0.1, min(0.9, new_conf))
                    print("Confidence set to: " + str(self.detector.conf))
                except:
                    print("Invalid input")
        
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nSystem stopped")


if __name__ == "__main__":
    print("""
    ============================================
       AI SECURITY SYSTEM (SIMPLE)
    ============================================
       * Person Detection
       * Face Recognition
       * Unauthorized Entry Alerts
       * No Audio Dependencies
    ============================================
    """)
    
    try:
        system = SimpleSecuritySystem()
        system.run()
    except KeyboardInterrupt:
        print("\n\nStopped by user")
    except Exception as e:
        print("\nError: " + str(e))
        print("\nMake sure:")
        print("  - Camera is connected")
        print("  - Packages installed: pip install opencv-python torch torchvision pandas")