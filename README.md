# Sports2D Motion Analysis UI

A modern desktop application for biomechanics motion analysis built on top of [Sports2D](https://github.com/davidpagnon/Sports2D). Analyze human motion from video with real-time skeleton overlays, joint angle tracking, and interactive kinematics graphs.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyQt5](https://img.shields.io/badge/GUI-PyQt5-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## Features

- **One-Click Analysis** — Run Sports2D pose estimation directly from the UI (no terminal needed)
- **Skeleton Overlay** — Real-time skeleton visualization with joint connections
- **Click-to-Select** — Click any joint on the video to select it for analysis
- **4 Interactive Graphs** — Linear velocity (Vx, Vy, Vtotal), linear acceleration (Ax, Ay, Atotal), angular velocity, angular acceleration
- **Calibration Tool** — Draw a reference line on the video to convert from pixels to real-world units (meters)
- **Trajectory Visualization** — Absolute and relative (hip-centered) motion trails
- **Joint Angle Overlay** — Displays the angle value directly on the video
- **Dark Mode UI** — Modern Catppuccin-themed interface

## Installation

### Prerequisites
- Python 3.10 or higher

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/sports2d-ui.git
cd sports2d-ui

# 2. Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

## Usage

```bash
python sports2d_ui.py
```

### Quick Start

1. **Analyze a new video**: Click **⚡ Analyze Video** → browse for your video → set slowmo factor → click **Run Analysis**. The app runs Sports2D in the background and auto-loads results.

2. **Load existing analysis**: If you already ran Sports2D on a video (e.g. `slowmo.mp4` with a `slowmo_Sports2D/` folder), click **Load Existing** and select the video.

3. **Select a joint**: Click directly on any joint dot in the video. The side panel and all 4 graphs update instantly.

4. **Calibrate**: Click **📏 Calibrate** → draw a line across a known distance → enter its real-world length in meters. All units convert from px to m.

5. **Trajectories**: Use the **Show Trajectory** / **Show Relative Traj.** buttons to visualize motion paths.

## Project Structure

```
sports2d-ui/
├── sports2d_ui.py        # Main application (single file)
├── requirements.txt      # Python dependencies  
├── .gitignore
└── README.md
```

When you analyze a video (e.g. `video.mp4`), Sports2D creates:

```
video_Sports2D/
├── video_Sports2D_px_person00.trc    # Joint positions (pixels)
├── video_Sports2D_angles_person00.mot # Joint angles (degrees)
├── video_Sports2D_m_person00.trc     # Joint positions (meters)
└── ...
```

## Dependencies

| Package | Purpose |
|---------|---------|
| `PyQt5` | GUI framework |
| `pyqtgraph` | High-performance interactive graphs |
| `QtAwesome` | Font Awesome icons |
| `opencv-python` | Video playback and drawing |
| `scipy` | Savitzky-Golay smoothing filter |
| `pandas` / `numpy` | Data loading and computation |
| `sports2d` | Pose estimation engine |

## License

MIT License
