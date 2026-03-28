# Body Tracker

A real-time body pose tracking and motion analysis system using computer vision. Detects and tracks 33 full-body landmarks, calculates joint angles, and provides live analytics for fitness, sports, and rehabilitation applications.

## Architecture

![Architecture Diagram](https://mermaid.ink/svg/Zmxvd2NoYXJ0IFRECiBBWyBJbnB1dCBTb3VyY2VcbldlYmNhbSAvIFZpZGVvIEZpbGUgLyBSVFNQXSAtLT4gQltGcmFtZSBDYXB0dXJlXG5PcGVuQ1YgVmlkZW9DYXB0dXJlXQogQiAtLT4gQ1tJbWFnZSBQcmVwcm9jZXNzb3JcblJlc2l6ZSArIE5vcm1hbGl6ZV0KIEMgLS0-IERbUG9zZSBFc3RpbWF0b3Jcbk1lZGlhUGlwZSBCbGF6ZVBvc2VdCiBEIC0tPiBFe1Bvc2UgRGV0ZWN0ZWQ_fQogRSAtLSBObyAtLT4gRlsgTm8gUGVyc29uIEZvdW5kXQogRSAtLSBZZXMgLS0-IEdbMzMgTGFuZG1hcmsgUG9pbnRzXG5Xb3JsZCArIEltYWdlIENvb3Jkc10KIAogRyAtLT4gSFtKb2ludCBBbmdsZSBDYWxjdWxhdG9yXG5TaG91bGRlciAvIEVsYm93IC8gSGlwIC8gS25lZV0KIEcgLS0-IElbU2tlbGV0b24gUmVuZGVyZXJcbkRyYXcgQ29ubmVjdGlvbnMgKyBQb2ludHNdCiBHIC0tPiBKW01vdGlvbiBBbmFseXplclxuVmVsb2NpdHkgKyBBY2NlbGVyYXRpb25dCiAKIEggLS0-IEtbUmVwIENvdW50ZXJcbkFuZ2xlIFRocmVzaG9sZCBMb2dpY10KIEggLS0-IExbUG9zdHVyZSBWYWxpZGF0b3JcbkZvcm0gQ2hlY2tlcl0KIEogLS0-IE1bQWN0aXZpdHkgQ2xhc3NpZmllclxuU1ZNIC8gTUxQIE1vZGVsXQogCiBLIC0tPiBOWyBMaXZlIERhc2hib2FyZFxuT3BlbkNWIE92ZXJsYXkgLyBTdHJlYW1saXRdCiBMIC0tPiBOCiBNIC0tPiBOCiBJIC0tPiBOCiBOIC0tPiBPWyBTZXNzaW9uIExvZ2dlclxuSlNPTiAvIENTViBFeHBvcnRd)

## Features

- 33-point full-body landmark detection at 30+ FPS
- Joint angle calculation for all major joints
- Exercise rep counting with form validation
- Posture analysis and correction alerts
- Activity classification (squat, pushup, curl, etc.)
- Real-time overlay visualization
- Session recording and playback
- CSV/JSON data export for analysis
- RTSP stream support for IP cameras

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Pose Estimation | MediaPipe BlazePose |
| Computer Vision | OpenCV 4.x |
| Numerical Analysis | NumPy, SciPy |
| Classification | scikit-learn |
| Visualization | OpenCV GUI, Matplotlib |
| Data Export | Pandas, JSON |
| UI (optional) | Streamlit |

## How to Run

```bash
# 1. Clone and install
git clone https://github.com/jadfarhat-cell/body-tracker.git
cd body-tracker
pip install -r requirements.txt

# 2. Run with webcam (default)
python tracker.py

# 3. Run on a video file
python tracker.py --input workout.mp4 --output tracked.mp4

# 4. Run specific exercise counter
python tracker.py --exercise squat
python tracker.py --exercise pushup
python tracker.py --exercise curl

# 5. Run Streamlit analytics dashboard
streamlit run dashboard.py

# 6. Export session data
python tracker.py --export-csv session_data.csv
```

## Project Structure

```
body-tracker/
├── tracker.py # Main tracking script
├── dashboard.py # Streamlit dashboard
├── modules/
│ ├── pose_detector.py # MediaPipe wrapper
│ ├── angle_calculator.py # Joint angle math
│ ├── rep_counter.py # Exercise rep logic
│ ├── form_checker.py # Posture validation
│ ├── classifier.py # Activity recognition
│ └── renderer.py # Visualization
├── models/
│ └── activity_classifier.pkl
├── data/
│ └── sessions/ # Saved session data
├── requirements.txt
└── config.yaml
```