# AI Security Alert System - Setup & Usage Guide

## 📋 Overview
This AI-powered security system monitors your camera and microphone to:
- Detect when anyone enters the camera view
- Recognize authorized persons (you)
- Alert when unauthorized persons are detected
- Monitor for loud sounds or unusual noises
- Provide real-time visual alerts

## 🔧 Installation

### Step 1: Install Python Dependencies

```bash
# Install main requirements
pip install opencv-python numpy torch torchvision pyaudio pandas

# Or use the requirements file
pip install -r requirements.txt
```

### Step 2: Platform-Specific Setup

#### **Windows:**
```bash
# PyAudio installation on Windows
pip install pipwin
pipwin install pyaudio

# Or download wheel from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio
```

#### **macOS:**
```bash
# Install PortAudio first
brew install portaudio

# Then install PyAudio
pip install pyaudio
```

#### **Linux (Ubuntu/Debian):**
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install python3-pyaudio portaudio19-dev

# Install Python packages
pip install pyaudio opencv-python
```

### Step 3: Test Your Setup

```bash
# Test camera
python -c "import cv2; print('Camera:', cv2.VideoCapture(0).isOpened())"

# Test microphone
python -c "import pyaudio; p = pyaudio.PyAudio(); print('Microphone:', p.get_default_input_device_info())"
```

## 🚀 Usage

### Basic Usage

```bash
python security_alert_system.py
```

### First-Time Setup

1. **Registration Phase:**
   - Position yourself in front of the camera
   - Press **SPACE** to register your face as authorized
   - Press **ESC** to skip registration (system will alert for all persons)

2. **Monitoring Phase:**
   - The system starts monitoring automatically
   - Green box = Authorized person detected
   - Red box = Unauthorized person detected (ALERT!)
   - Sound monitoring runs in background

### Keyboard Controls

| Key | Action |
|-----|--------|
| **q** | Quit the system |
| **r** | Re-register authorized person |
| **s** | Adjust sound sensitivity |

## ⚙️ Configuration

### Adjust Sound Sensitivity

During runtime, press **'s'** and enter a value:
- **500-1500**: Very sensitive (detects whispers)
- **1500-3000**: Normal (detects conversations)
- **3000-5000**: Less sensitive (only loud sounds)

### Modify Alert Settings

Edit in `security_alert_system.py`:

```python
# Line ~31: Alert cooldown (seconds between alerts)
self.alert_cooldown = 5  # Change to your preference

# Line ~26: Sound threshold
self.SOUND_THRESHOLD = 1000  # Adjust sensitivity

# Line ~19: Confidence threshold for person detection
self.person_detector.conf = 0.5  # 0.1 (sensitive) to 0.9 (strict)
```

## 📊 How It Works

### 1. Person Detection
- Uses **YOLOv5** (You Only Look Once) for real-time person detection
- Processes each frame to identify human figures
- Draws bounding boxes around detected persons

### 2. Face Recognition
- Uses **Haar Cascade Classifier** for face detection
- Compares detected faces with registered authorized person
- Template matching to verify identity

### 3. Sound Monitoring
- Captures audio in real-time using PyAudio
- Calculates RMS (Root Mean Square) of audio signal
- Triggers alert when volume exceeds threshold

### 4. Alert System
- Visual alerts on screen (red boxes, text warnings)
- Console alerts with timestamp
- Cooldown period to prevent alert spam

## 🎯 Use Cases

### Home Security
```python
# Run continuously
python security_alert_system.py
```

### Office Monitoring
```python
# Detect when someone enters your workspace
# Register yourself as authorized person
```

### Baby Monitor
```python
# Detect movement and sounds in nursery
# Set lower sound threshold for sensitivity
```

## 🔒 Privacy & Security

- **Local Processing:** All AI processing happens on your device
- **No Cloud Upload:** Video/audio never sent to external servers
- **No Recording:** Doesn't save video unless you modify the code
- **Authorized Person:** Only you can register as authorized

## 🛠️ Customization Examples

### Example 1: Save Alerts to File

Add to `trigger_alert()` method:

```python
def trigger_alert(self, reason):
    # ... existing code ...
    
    # Save to log file
    with open('security_log.txt', 'a') as f:
        f.write(f"{timestamp} - {reason}\n")
```

### Example 2: Save Snapshot on Alert

```python
def trigger_alert(self, reason):
    # ... existing code ...
    
    # Save current frame
    cv2.imwrite(f'alert_{timestamp.replace(":", "-")}.jpg', self.frame_buffer[-1])
```

### Example 3: Send Email Alert

```python
import smtplib
from email.mime.text import MIMEText

def trigger_alert(self, reason):
    # ... existing code ...
    
    # Send email
    msg = MIMEText(f"Security Alert: {reason}")
    msg['Subject'] = 'Security Alert!'
    msg['From'] = 'your-email@gmail.com'
    msg['To'] = 'recipient@gmail.com'
    
    with smtplib.SMTP('smtp.gmail.com', 587) as server:
        server.starttls()
        server.login('your-email@gmail.com', 'your-password')
        server.send_message(msg)
```

### Example 4: Multiple Authorized Persons

```python
def register_authorized_person(self):
    # Modify to store multiple faces
    self.authorized_faces = []  # List of authorized faces
    
    # Allow registering multiple times
    # Store each face in the list
```

## 🐛 Troubleshooting

### Camera Not Opening
```bash
# Check camera permissions
# Windows: Settings > Privacy > Camera
# macOS: System Preferences > Security & Privacy > Camera
# Linux: Check /dev/video0 permissions
```

### PyAudio Installation Fails
```bash
# Windows: Use pipwin or download wheel
# macOS: Install portaudio via brew
# Linux: Install portaudio19-dev
```

### YOLOv5 Download Slow
```bash
# It will auto-download on first run (~14MB)
# Be patient, it only happens once
# Or download manually from: https://github.com/ultralytics/yolov5
```

### High CPU Usage
```python
# Reduce frame processing in code:
self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Lower resolution
self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

# Or add frame skip:
if frame_count % 2 == 0:  # Process every 2nd frame
    # ... detection code ...
```

### False Alarms
```python
# Increase confidence threshold
self.person_detector.conf = 0.7  # Higher = stricter

# Adjust face matching threshold
if similarity > 0.7:  # Higher = stricter matching
```

## 📈 Performance Tips

1. **Use USB Camera:** Better quality than laptop webcam
2. **Good Lighting:** Improves person and face detection
3. **Stable Position:** Mount camera to reduce motion blur
4. **Background:** Plain backgrounds work better
5. **Distance:** Keep 3-6 feet from camera for best results

## 📝 System Requirements

- **Python:** 3.7 or higher
- **RAM:** Minimum 4GB (8GB recommended)
- **CPU:** Any modern processor (GPU optional)
- **Camera:** Any USB or built-in webcam
- **Microphone:** Optional but recommended
- **OS:** Windows 10/11, macOS 10.14+, Linux

## 🎓 Advanced Features (Optional)

### Add SMS Alerts (Twilio)
```bash
pip install twilio
```

### Add Database Logging (SQLite)
```python
import sqlite3
# Store alerts in database
```

### Add Web Dashboard (Flask)
```bash
pip install flask
# Create web interface to view alerts
```

### Add Motion Detection
```python
# Use frame differencing to detect movement
# Trigger alerts on significant motion
```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Ensure all dependencies are installed
3. Verify camera and microphone permissions
4. Test each component individually

## 🔄 Updates & Improvements

To enhance the system:
- Replace Haar Cascade with deep learning face recognition
- Add object detection (not just persons)
- Implement motion tracking
- Add night vision support (IR camera)
- Create mobile app for remote monitoring

## ⚖️ Legal Notice

**Important:** Check local laws before using this system:
- Audio recording may require consent
- Video surveillance has different rules by location
- Always respect privacy rights
- Use responsibly and ethically

---

**Happy Monitoring! 🔒**
