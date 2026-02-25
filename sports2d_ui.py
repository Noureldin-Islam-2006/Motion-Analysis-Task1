import sys
import os
import cv2
import subprocess
import pandas as pd
import numpy as np
import pyqtgraph as pg
from PyQt5 import QtWidgets, QtCore, QtGui
import qtawesome as qta
from scipy.signal import savgol_filter

# ─── Data Loaders ───────────────────────────────────────────────────────────────

class Sports2DLoader:
    @staticmethod
    def load_trc(filepath):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            header_vals = lines[2].split()
            data_rate = float(header_vals[0])
            num_frames = int(header_vals[2])
            marker_names_raw = lines[3].split('\t')
            marker_names = [n.strip() for n in marker_names_raw if n.strip() and n.strip() not in ['Frame#', 'Time']]
            data = pd.read_csv(filepath, sep='\t', skiprows=5, header=None)
            markers_data = {}
            for i, name in enumerate(marker_names):
                x_col = i * 3 + 2
                y_col = i * 3 + 3
                markers_data[name] = {
                    'x': data.iloc[:, x_col].values.astype(float),
                    'y': data.iloc[:, y_col].values.astype(float)
                }
            return {
                'frame_count': num_frames, 'data_rate': data_rate,
                'markers': markers_data, 'marker_list': marker_names,
                'time': data.iloc[:, 1].values.astype(float)
            }
        except Exception as e:
            print(f"Error loading TRC: {e}")
            return None

    @staticmethod
    def load_mot(filepath):
        try:
            with open(filepath, 'r') as f:
                lines = f.readlines()
            start_row = 0
            for i, line in enumerate(lines):
                if 'endheader' in line:
                    start_row = i + 1
                    break
            data = pd.read_csv(filepath, sep='\t', skiprows=start_row)
            headers = list(data.columns)
            angles_data = {}
            for header in headers[1:]:
                angles_data[header] = data[header].values.astype(float)
            return {
                'angles': angles_data, 'angle_list': headers[1:],
                'time': data[headers[0]].values.astype(float)
            }
        except Exception as e:
            print(f"Error loading MOT: {e}")
            return None

# ─── Marker → Angle mapping ────────────────────────────────────────────────────

MARKER_TO_ANGLE = {
    'RAnkle': 'right ankle', 'LAnkle': 'left ankle',
    'RKnee': 'right knee', 'LKnee': 'left knee',
    'RHip': 'right hip', 'LHip': 'left hip',
    'RShoulder': 'right shoulder', 'LShoulder': 'left shoulder',
    'RElbow': 'right elbow', 'LElbow': 'left elbow',
    'RWrist': 'right forearm', 'LWrist': 'left forearm',
    'RBigToe': 'right foot', 'RSmallToe': 'right foot', 'RHeel': 'right foot',
    'LBigToe': 'left foot', 'LSmallToe': 'left foot', 'LHeel': 'left foot',
    'Hip': 'pelvis', 'Neck': 'trunk', 'Head': 'head', 'Nose': 'head',
}

def smooth(arr, win=11, poly=3):
    if len(arr) > win:
        return savgol_filter(arr, win, poly)
    return arr

# ─── Sports2D Analysis Worker (runs in background thread) ──────────────────────

class AnalysisWorker(QtCore.QThread):
    """Runs sports2d CLI in a background thread and emits progress."""
    progress = QtCore.pyqtSignal(str)     # status text
    finished = QtCore.pyqtSignal(bool, str)  # success, message

    def __init__(self, video_path, slowmo_factor=1):
        super().__init__()
        self.video_path = video_path
        self.slowmo_factor = slowmo_factor

    def run(self):
        try:
            # Find sports2d executable
            scripts_dir = os.path.join(os.path.dirname(sys.executable), 'Scripts')
            sports2d_exe = os.path.join(scripts_dir, 'sports2d.exe')
            if not os.path.exists(sports2d_exe):
                # Fallback: try PATH
                sports2d_exe = 'sports2d'

            cmd = [
                sports2d_exe,
                '--video_input', self.video_path,
                '--slowmo_factor', str(self.slowmo_factor),
                '--fill_large_gaps_with', 'zeros',
                '--display_angle_values_on', 'none',
            ]

            self.progress.emit(f"Running: {' '.join(os.path.basename(c) for c in cmd[:3])}...")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=os.path.dirname(self.video_path),
                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            )

            # Stream output
            for line in process.stdout:
                line = line.strip()
                if line:
                    self.progress.emit(line)

            process.wait()
            if process.returncode == 0:
                self.finished.emit(True, "Analysis completed successfully!")
            else:
                self.finished.emit(False, f"sports2d exited with code {process.returncode}")
        except FileNotFoundError:
            self.finished.emit(False, "sports2d not found. Install with: pip install sports2d")
        except Exception as e:
            self.finished.emit(False, f"Error: {str(e)}")

# ─── Interactive video label ────────────────────────────────────────────────────

class VideoLabel(QtWidgets.QLabel):
    clicked = QtCore.pyqtSignal(QtCore.QPoint)
    calibration_done = QtCore.pyqtSignal(QtCore.QPoint, QtCore.QPoint)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.draw_mode = False
        self._cal_start = None
        self._cal_end = None
        self._drawing = False
        self.cal_text = ""  # Text to display on the line

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            if self.draw_mode:
                self._cal_start = event.pos()
                self._cal_end = event.pos()
                self._drawing = True
            else:
                self.clicked.emit(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drawing:
            self._cal_end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drawing:
            self._drawing = False
            self._cal_end = event.pos()
            self.update()
            if self._cal_start and self._cal_end:
                self.calibration_done.emit(self._cal_start, self._cal_end)
            self.draw_mode = False

    def paintEvent(self, event):
        super().paintEvent(event)
        if self._cal_start and self._cal_end:
            painter = QtGui.QPainter(self)
            pen = QtGui.QPen(QtGui.QColor('#F9E2AF'), 3, QtCore.Qt.DashLine)
            painter.setPen(pen)
            painter.drawLine(self._cal_start, self._cal_end)
            painter.setBrush(QtGui.QColor('#F9E2AF'))
            painter.drawEllipse(self._cal_start, 5, 5)
            painter.drawEllipse(self._cal_end, 5, 5)
            mid = (self._cal_start + self._cal_end) / 2
            painter.setPen(QtGui.QColor('#F9E2AF'))
            font = painter.font()
            font.setBold(True)
            font.setPointSize(11)
            painter.setFont(font)
            painter.drawText(int(mid.x()) + 10, int(mid.y()) - 10, self.cal_text)
            painter.end()

    def clear_cal_line(self):
        self._cal_start = None
        self._cal_end = None
        self.cal_text = ""
        self.update()


# ─── Analysis Settings Dialog ──────────────────────────────────────────────────

class AnalysisDialog(QtWidgets.QDialog):
    """Dialog to configure and start Sports2D analysis."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Run Sports2D Analysis")
        self.setMinimumWidth(450)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; font-size: 13px; }
            QLineEdit, QSpinBox, QDoubleSpinBox { 
                background-color: #313244; color: #cdd6f4; border: 1px solid #45475a; 
                border-radius: 6px; padding: 8px; font-size: 13px; 
            }
            QPushButton { 
                background-color: #4CAF50; color: white; border-radius: 6px; 
                padding: 10px 20px; font-weight: bold; font-size: 14px; border: none;
            }
            QPushButton:hover { background-color: #66BB6A; }
            QPushButton#cancelBtn { background-color: #45475a; }
            QPushButton#cancelBtn:hover { background-color: #585b70; }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QtWidgets.QLabel("⚡ Sports2D Analysis")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #A6E3A1;")
        layout.addWidget(title)

        desc = QtWidgets.QLabel("Select a video and configure analysis settings.\nThe analysis will run in the background.")
        desc.setStyleSheet("color: #a6adc8; font-size: 12px;")
        layout.addWidget(desc)

        # Video path
        file_row = QtWidgets.QHBoxLayout()
        self.video_edit = QtWidgets.QLineEdit()
        self.video_edit.setPlaceholderText("Select a video file...")
        self.video_edit.setReadOnly(True)
        browse_btn = QtWidgets.QPushButton("Browse")
        browse_btn.setStyleSheet("background-color: #313244; padding: 8px 16px;")
        browse_btn.clicked.connect(self._browse)
        file_row.addWidget(self.video_edit, 1)
        file_row.addWidget(browse_btn)
        layout.addLayout(file_row)

        # Slowmo factor
        slowmo_row = QtWidgets.QHBoxLayout()
        slowmo_row.addWidget(QtWidgets.QLabel("Slowmo Factor:"))
        self.slowmo_spin = QtWidgets.QSpinBox()
        self.slowmo_spin.setRange(1, 100)
        self.slowmo_spin.setValue(1)
        self.slowmo_spin.setToolTip("Set > 1 if video is recorded in slow motion")
        slowmo_row.addWidget(self.slowmo_spin)
        layout.addLayout(slowmo_row)

        # Buttons
        btn_row = QtWidgets.QHBoxLayout()
        cancel_btn = QtWidgets.QPushButton("Cancel")
        cancel_btn.setObjectName("cancelBtn")
        cancel_btn.clicked.connect(self.reject)
        self.run_btn = QtWidgets.QPushButton("▶  Run Analysis")
        self.run_btn.clicked.connect(self.accept)
        self.run_btn.setEnabled(False)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(self.run_btn)
        layout.addLayout(btn_row)

        self.video_path = None

    def _browse(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov *.mkv)")
        if path:
            self.video_path = path
            self.video_edit.setText(path)
            self.run_btn.setEnabled(True)


# ─── Progress Dialog ───────────────────────────────────────────────────────────

class ProgressDialog(QtWidgets.QDialog):
    """Shows real-time output from sports2d."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Sports2D Analysis Running...")
        self.setMinimumSize(600, 400)
        self.setStyleSheet("""
            QDialog { background-color: #1e1e2e; }
            QLabel { color: #cdd6f4; }
            QTextEdit { 
                background-color: #11111b; color: #A6E3A1; border: 1px solid #313244; 
                border-radius: 8px; font-family: 'Consolas', monospace; font-size: 12px; padding: 8px;
            }
            QPushButton { 
                background-color: #F38BA8; color: #1e1e2e; border-radius: 6px; 
                padding: 8px 20px; font-weight: bold; border: none; 
            }
        """)

        layout = QtWidgets.QVBoxLayout(self)
        self.status_lbl = QtWidgets.QLabel("⏳ Starting analysis...")
        self.status_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #F9E2AF;")
        layout.addWidget(self.status_lbl)

        self.log_text = QtWidgets.QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)

        self.cancel_btn = QtWidgets.QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        layout.addWidget(self.cancel_btn)

    def append_log(self, text):
        self.log_text.append(text)
        # Auto-scroll to bottom
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def set_finished(self, success, msg):
        if success:
            self.status_lbl.setText("✅ " + msg)
            self.status_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #A6E3A1;")
        else:
            self.status_lbl.setText("❌ " + msg)
            self.status_lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #F38BA8;")
        self.cancel_btn.setText("Close")


# ─── Main Application ──────────────────────────────────────────────────────────

class Sports2DApp(QtWidgets.QMainWindow):
    SKELETON_CONNECTIONS = [
        ('Nose', 'Neck'), ('Neck', 'Head'),
        ('Neck', 'RShoulder'), ('RShoulder', 'RElbow'), ('RElbow', 'RWrist'),
        ('Neck', 'LShoulder'), ('LShoulder', 'LElbow'), ('LElbow', 'LWrist'),
        ('Neck', 'Hip'),
        ('Hip', 'RHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
        ('RAnkle', 'RHeel'), ('RAnkle', 'RBigToe'), ('RAnkle', 'RSmallToe'),
        ('Hip', 'LHip'), ('LHip', 'LKnee'), ('LKnee', 'LAnkle'),
        ('LAnkle', 'LHeel'), ('LAnkle', 'LBigToe'), ('LAnkle', 'LSmallToe'),
    ]

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports2D Advanced Motion Analysis")
        self.resize(1440, 900)
        self.setWindowIcon(qta.icon('fa5s.running', color='#4CAF50'))

        self.trc_data = None
        self.mot_data = None
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30.0
        self.video_path = None
        self.cap = None
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._advance_frame)
        self.selected_joint = None

        # Calibration
        self.px_per_unit = None
        self.unit_name = "px"

        # Trajectory
        self.show_trajectory = False
        self.show_relative_trajectory = False

        # Analysis worker
        self._worker = None

        # Cached kinematics
        self._cache_time = None
        self._cache_vx = None
        self._cache_vy = None
        self._cache_vtotal = None
        self._cache_ax = None
        self._cache_ay = None
        self._cache_atotal = None
        self._cache_angle = None
        self._cache_ang_vel = None
        self._cache_ang_acc = None
        self._cache_angle_name = None

        self.scale_factor = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self.v_lines = []

        self._build_ui()
        self._apply_styles()

    # ── UI ──────────────────────────────────────────────────────────────────

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root = QtWidgets.QHBoxLayout(central)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(12)

        # LEFT: Video
        left = QtWidgets.QVBoxLayout()
        left.setSpacing(8)

        vid_frame = QtWidgets.QFrame()
        vid_frame.setObjectName("videoFrame")
        vid_lay = QtWidgets.QVBoxLayout(vid_frame)
        self.video_label = VideoLabel("Load a video to begin analysis")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setMinimumSize(750, 480)
        self.video_label.setMouseTracking(True)
        self.video_label.setStyleSheet("color: #6c7086; font-size: 15px; border: none;")
        self.video_label.clicked.connect(self._on_video_click)
        self.video_label.calibration_done.connect(self._on_calibration_line)
        vid_lay.addWidget(self.video_label)
        left.addWidget(vid_frame, 1)

        # Playback bar
        pb = QtWidgets.QFrame()
        pb.setObjectName("controlBar")
        pb_lay = QtWidgets.QHBoxLayout(pb)
        pb_lay.setContentsMargins(8, 4, 8, 4)
        self.play_btn = QtWidgets.QPushButton(qta.icon('fa5s.play', color='white'), "")
        self.play_btn.setFixedSize(40, 40)
        self.play_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.play_btn.clicked.connect(self._toggle_play)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setCursor(QtCore.Qt.PointingHandCursor)
        self.slider.sliderMoved.connect(self._seek)
        self.slider.sliderPressed.connect(lambda: self.timer.stop())
        self.time_lbl = QtWidgets.QLabel("0.00 / 0.00")
        self.time_lbl.setFixedWidth(110)
        self.time_lbl.setAlignment(QtCore.Qt.AlignCenter)
        pb_lay.addWidget(self.play_btn)
        pb_lay.addWidget(self.slider, 1)
        pb_lay.addWidget(self.time_lbl)
        left.addWidget(pb)

        # Actions bar
        act = QtWidgets.QHBoxLayout()

        analyze_btn = QtWidgets.QPushButton(qta.icon('fa5s.bolt', color='#1e1e2e'), "  Analyze Video")
        analyze_btn.setObjectName("analyzeBtn")
        analyze_btn.setToolTip("Run Sports2D analysis on a video")
        analyze_btn.clicked.connect(self._open_analysis_dialog)
        act.addWidget(analyze_btn)

        load_btn = QtWidgets.QPushButton(qta.icon('fa5s.folder-open', color='white'), "  Load Existing")
        load_btn.setObjectName("loadBtn")
        load_btn.setToolTip("Load a video that already has Sports2D analysis")
        load_btn.clicked.connect(self._load_video)
        act.addWidget(load_btn)

        self.cal_btn = QtWidgets.QPushButton(qta.icon('fa5s.ruler', color='#1e1e2e'), "  Calibrate")
        self.cal_btn.setObjectName("calBtn")
        self.cal_btn.setToolTip("Draw a line on the video and enter its real-world length")
        self.cal_btn.clicked.connect(self._start_calibration)
        act.addWidget(self.cal_btn)

        self.status_lbl = QtWidgets.QLabel("Ready")
        self.status_lbl.setStyleSheet("color: #585b70; font-size: 12px;")
        act.addWidget(self.status_lbl, 1)
        left.addLayout(act)

        root.addLayout(left, 7)

        # RIGHT: Side Panel
        right = QtWidgets.QVBoxLayout()
        right.setSpacing(10)

        self.sel_header = QtWidgets.QLabel("Selected: —")
        self.sel_header.setObjectName("selHeader")
        right.addWidget(self.sel_header)

        self.cal_info = QtWidgets.QLabel("Units: px (uncalibrated)")
        self.cal_info.setStyleSheet("color: #F9E2AF; font-size: 11px; border: none;")
        right.addWidget(self.cal_info)

        # Current Frame Data card
        data_card = QtWidgets.QFrame()
        data_card.setObjectName("dataCard")
        dc_lay = QtWidgets.QVBoxLayout(data_card)
        dc_lay.setSpacing(6)
        dc_title = QtWidgets.QLabel("Current Frame Data")
        dc_title.setStyleSheet("font-weight: bold; font-size: 14px; color: #cdd6f4; border: none;")
        dc_lay.addWidget(dc_title)

        self.stat_pos_label = QtWidgets.QLabel("Position (px):")
        self.stat_pos = self._make_stat_row(dc_lay, self.stat_pos_label)
        self.stat_vel_label = QtWidgets.QLabel("Linear Velocity (px/s):")
        self.stat_lin_vel = self._make_stat_row(dc_lay, self.stat_vel_label)
        self.stat_acc_label = QtWidgets.QLabel("Linear Acceleration (px/s²):")
        self.stat_lin_acc = self._make_stat_row(dc_lay, self.stat_acc_label)
        self.stat_angle_label = QtWidgets.QLabel("Joint Angle (deg):")
        self.stat_angle = self._make_stat_row(dc_lay, self.stat_angle_label)
        self.stat_angvel_label = QtWidgets.QLabel("Angular Velocity (deg/s):")
        self.stat_ang_vel = self._make_stat_row(dc_lay, self.stat_angvel_label)
        self.stat_angacc_label = QtWidgets.QLabel("Angular Accel (deg/s²):")
        self.stat_ang_acc = self._make_stat_row(dc_lay, self.stat_angacc_label)
        right.addWidget(data_card)

        # Trajectory buttons
        self.traj_btn = QtWidgets.QPushButton("Show Trajectory")
        self.traj_btn.setObjectName("trajBtn")
        self.traj_btn.clicked.connect(self._toggle_trajectory)
        right.addWidget(self.traj_btn)

        self.rel_traj_btn = QtWidgets.QPushButton("Show Relative Traj.")
        self.rel_traj_btn.setObjectName("relTrajBtn")
        self.rel_traj_btn.clicked.connect(self._toggle_relative_trajectory)
        right.addWidget(self.rel_traj_btn)

        # Graphs
        graphs_scroll = QtWidgets.QScrollArea()
        graphs_scroll.setWidgetResizable(True)
        graphs_scroll.setObjectName("graphScroll")
        graphs_inner = QtWidgets.QWidget()
        graphs_lay = QtWidgets.QVBoxLayout(graphs_inner)
        graphs_lay.setSpacing(6)

        self.graph_vel = pg.PlotWidget(title="Linear Velocity")
        self._style_graph(self.graph_vel, 'Speed', 'px/s')
        self.graph_vel.addLegend(offset=(60, 5))
        graphs_lay.addWidget(self.graph_vel)

        self.graph_acc = pg.PlotWidget(title="Linear Acceleration")
        self._style_graph(self.graph_acc, 'Accel', 'px/s²')
        self.graph_acc.addLegend(offset=(60, 5))
        graphs_lay.addWidget(self.graph_acc)

        self.graph_ang_vel = pg.PlotWidget(title="Angular Velocity (deg/s)")
        self._style_graph(self.graph_ang_vel, 'ω', 'deg/s')
        graphs_lay.addWidget(self.graph_ang_vel)

        self.graph_ang_acc = pg.PlotWidget(title="Angular Acceleration (deg/s²)")
        self._style_graph(self.graph_ang_acc, 'α', 'deg/s²')
        graphs_lay.addWidget(self.graph_ang_acc)

        for g in [self.graph_vel, self.graph_acc, self.graph_ang_vel, self.graph_ang_acc]:
            vl = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen('#F38BA8', width=2))
            g.addItem(vl)
            self.v_lines.append(vl)

        graphs_scroll.setWidget(graphs_inner)
        right.addWidget(graphs_scroll, 1)

        right_widget = QtWidgets.QWidget()
        right_widget.setFixedWidth(440)
        right_widget.setLayout(right)
        root.addWidget(right_widget, 3)

    def _make_stat_row(self, parent_layout, label_widget):
        row = QtWidgets.QHBoxLayout()
        label_widget.setStyleSheet("color: #a6adc8; font-size: 12px; border: none;")
        val = QtWidgets.QLabel("—")
        val.setStyleSheet("color: #cdd6f4; font-weight: bold; font-size: 13px; border: none;")
        val.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        row.addWidget(label_widget)
        row.addWidget(val)
        parent_layout.addLayout(row)
        return val

    def _style_graph(self, g, y_label, y_unit):
        g.setBackground('#181825')
        g.showGrid(x=True, y=True, alpha=0.15)
        g.setLabel('left', y_label, units=y_unit)
        g.setLabel('bottom', 'Time', units='s')
        g.setMinimumHeight(160)

    def _apply_styles(self):
        self.setStyleSheet("""
        QMainWindow { background-color: #11111b; }
        QWidget { color: #cdd6f4; font-family: 'Inter', 'Segoe UI', Arial; font-size: 13px; }
        #videoFrame { background-color: #181825; border-radius: 10px; border: 1px solid #313244; }
        #controlBar { background-color: #181825; border-radius: 8px; border: 1px solid #313244; }
        QPushButton { background-color: #313244; border-radius: 6px; padding: 8px 16px; color: white; border: none; }
        QPushButton:hover { background-color: #45475a; }
        QPushButton:pressed { background-color: #585b70; }
        #loadBtn { background-color: #45475a; padding: 10px 16px; }
        #loadBtn:hover { background-color: #585b70; }
        #analyzeBtn { background-color: #A6E3A1; color: #1e1e2e; font-weight: bold; padding: 10px 20px; }
        #analyzeBtn:hover { background-color: #94E298; }
        #calBtn { background-color: #F9E2AF; color: #1e1e2e; font-weight: bold; padding: 10px 16px; }
        #calBtn:hover { background-color: #FAD87D; }
        #selHeader { color: #89B4FA; font-size: 16px; font-weight: bold; border: none; }
        #dataCard { background-color: #181825; border-radius: 10px; border: 1px solid #313244; padding: 12px; }
        #trajBtn { background-color: #45475a; border-radius: 6px; padding: 10px; color: #cdd6f4; border: none; }
        #trajBtn:hover { background-color: #585b70; }
        #relTrajBtn { background-color: #45475a; border-radius: 6px; padding: 10px; color: #cdd6f4; border: none; }
        #relTrajBtn:hover { background-color: #585b70; }
        #graphScroll { border: none; background: transparent; }
        QScrollBar:vertical { background: #181825; width: 8px; }
        QScrollBar::handle:vertical { background: #45475a; border-radius: 4px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QSlider::groove:horizontal { border: none; height: 6px; background: #313244; border-radius: 3px; }
        QSlider::handle:horizontal { background: #4CAF50; border: none; width: 16px; height: 16px; margin: -5px 0; border-radius: 8px; }
        """)

    def _update_unit_labels(self):
        u = self.unit_name
        us = f"{u}/s"
        us2 = f"{u}/s²"
        self.stat_pos_label.setText(f"Position ({u}):")
        self.stat_vel_label.setText(f"Linear Velocity ({us}):")
        self.stat_acc_label.setText(f"Linear Acceleration ({us2}):")
        self.graph_vel.setTitle(f"Linear Velocity ({us})")
        self.graph_vel.setLabel('left', 'Speed', units=us)
        self.graph_acc.setTitle(f"Linear Acceleration ({us2})")
        self.graph_acc.setLabel('left', 'Accel', units=us2)
        if self.px_per_unit:
            self.cal_info.setText(f"✅ Calibrated: 1 {u} = {self.px_per_unit:.1f} px")
            self.cal_info.setStyleSheet("color: #A6E3A1; font-size: 11px; border: none;")
        else:
            self.cal_info.setText(f"Units: {u} (uncalibrated)")
            self.cal_info.setStyleSheet("color: #F9E2AF; font-size: 11px; border: none;")

    # ── Analysis Runner ─────────────────────────────────────────────────────

    def _open_analysis_dialog(self):
        dlg = AnalysisDialog(self)
        if dlg.exec_() == QtWidgets.QDialog.Accepted and dlg.video_path:
            self._run_analysis(dlg.video_path, dlg.slowmo_spin.value())

    def _run_analysis(self, video_path, slowmo_factor):
        # Open progress dialog
        self._progress_dlg = ProgressDialog(self)
        self._progress_dlg.show()

        # Start worker thread
        self._worker = AnalysisWorker(video_path, slowmo_factor)
        self._worker.progress.connect(self._progress_dlg.append_log)
        self._worker.finished.connect(self._on_analysis_finished)

        # Store video path for auto-loading
        self._analysis_video_path = video_path

        self._worker.start()

    def _on_analysis_finished(self, success, message):
        self._progress_dlg.set_finished(success, message)
        if success:
            # Auto-load the analyzed video
            self._load_video_from_path(self._analysis_video_path)
            self.status_lbl.setText("Analysis complete! Results loaded.")
            self.status_lbl.setStyleSheet("color: #A6E3A1; font-size: 12px; font-weight: bold;")

    # ── Calibration ─────────────────────────────────────────────────────────

    def _start_calibration(self):
        if not self.cap:
            QtWidgets.QMessageBox.warning(self, "No Video", "Load a video first.")
            return
        self.timer.stop()
        self.video_label.draw_mode = True
        self.video_label.clear_cal_line()
        self.status_lbl.setText("CALIBRATE: Draw a line on the video, then enter its real length.")
        self.status_lbl.setStyleSheet("color: #F9E2AF; font-size: 12px; font-weight: bold;")

    def _on_calibration_line(self, p1, p2):
        vx1 = (p1.x() - self.offset_x) / self.scale_factor
        vy1 = (p1.y() - self.offset_y) / self.scale_factor
        vx2 = (p2.x() - self.offset_x) / self.scale_factor
        vy2 = (p2.y() - self.offset_y) / self.scale_factor
        px_dist = np.sqrt((vx2 - vx1)**2 + (vy2 - vy1)**2)

        if px_dist < 5:
            self.status_lbl.setText("Line too short. Try again.")
            return

        val, ok = QtWidgets.QInputDialog.getDouble(
            self, "Calibration",
            f"Line is {px_dist:.1f} px.\nEnter the real-world length of this line (in meters):",
            value=1.0, min=0.001, decimals=4
        )
        if ok and val > 0:
            self.px_per_unit = px_dist / val
            self.unit_name = "m"
            # Update the calibration line text to show meters
            self.video_label.cal_text = f"{val:.4f} m"
            self.video_label.update()
            self._update_unit_labels()
            self.status_lbl.setText(f"Calibrated: {self.px_per_unit:.1f} px/m")
            self.status_lbl.setStyleSheet("color: #A6E3A1; font-size: 12px; font-weight: bold;")
            if self.selected_joint:
                self._compute_kinematics()
                self._update_all_graphs()
                self._set_frame(self.current_frame)
        else:
            self.status_lbl.setText("Calibration cancelled.")
            self.status_lbl.setStyleSheet("color: #585b70; font-size: 12px;")
            self.video_label.clear_cal_line()

    def _scale_val(self, px_val):
        if self.px_per_unit:
            return px_val / self.px_per_unit
        return px_val

    # ── Trajectory ──────────────────────────────────────────────────────────

    def _toggle_trajectory(self):
        self.show_trajectory = not self.show_trajectory
        if self.show_trajectory:
            self.traj_btn.setText("Hide Trajectory")
            self.traj_btn.setStyleSheet("background-color: #A6E3A1; color: #1e1e2e; font-weight: bold;")
        else:
            self.traj_btn.setText("Show Trajectory")
            self.traj_btn.setStyleSheet("")
        self._set_frame(self.current_frame)

    def _toggle_relative_trajectory(self):
        self.show_relative_trajectory = not self.show_relative_trajectory
        if self.show_relative_trajectory:
            self.rel_traj_btn.setText("Hide Relative Traj.")
            self.rel_traj_btn.setStyleSheet("background-color: #A6E3A1; color: #1e1e2e; font-weight: bold;")
        else:
            self.rel_traj_btn.setText("Show Relative Traj.")
            self.rel_traj_btn.setStyleSheet("")
        self._set_frame(self.current_frame)

    # ── Data Loading ────────────────────────────────────────────────────────

    def _load_video(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Open Video", "", "Videos (*.mp4 *.avi *.mov)")
        if path:
            self._load_video_from_path(path)

    def _load_video_from_path(self, path):
        self.video_path = path
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.slider.setMaximum(self.total_frames - 1)
        self.trc_data = None
        self.mot_data = None
        self.selected_joint = None
        self.video_label.clear_cal_line()

        base = os.path.splitext(os.path.basename(path))[0]
        d = os.path.dirname(path)
        analysis = os.path.join(d, f"{base}_Sports2D")
        if os.path.exists(analysis):
            trc = os.path.join(analysis, f"{base}_Sports2D_px_person00.trc")
            mot = os.path.join(analysis, f"{base}_Sports2D_angles_person00.mot")
            if os.path.exists(trc):
                self.trc_data = Sports2DLoader.load_trc(trc)
            if os.path.exists(mot):
                self.mot_data = Sports2DLoader.load_mot(mot)
            self.status_lbl.setText(f"Loaded: {os.path.basename(analysis)}")
            self.status_lbl.setStyleSheet("color: #A6E3A1; font-size: 12px;")
        else:
            self.status_lbl.setText("No analysis folder found")
            self.status_lbl.setStyleSheet("color: #585b70; font-size: 12px;")
        self._set_frame(0)

    # ── Frame Rendering ─────────────────────────────────────────────────────

    def _set_frame(self, idx):
        if not self.cap or idx < 0 or idx >= self.total_frames:
            return
        self.current_frame = idx
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = self._draw_overlays(frame, idx)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        lw, lh = self.video_label.width(), self.video_label.height()
        if lw <= 0 or lh <= 0:
            lw, lh = 750, 480
        ia = w / h
        la = lw / lh
        if ia > la:
            sw, sh = lw, int(lw / ia)
        else:
            sh, sw = lh, int(lh * ia)
        self.scale_factor = sw / w
        self.offset_x = (lw - sw) // 2
        self.offset_y = (lh - sh) // 2

        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg)
        self.video_label.setPixmap(pix.scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        ct = idx / self.fps
        tt = self.total_frames / self.fps
        self.time_lbl.setText(f"{ct:.2f} / {tt:.2f}")

        if self.trc_data and idx < len(self.trc_data['time']):
            tv = self.trc_data['time'][idx]
            for vl in self.v_lines:
                vl.setValue(tv)
        self._update_stats(idx)

    def _draw_overlays(self, frame, idx):
        if not self.trc_data:
            return frame
        markers = self.trc_data['markers']
        CLR_BONE = (180, 180, 180)
        CLR_DOT = (255, 255, 255)
        CLR_SEL = (80, 255, 80)

        # Trajectory
        if self.selected_joint and self.selected_joint in markers:
            c = markers[self.selected_joint]
            if self.show_trajectory:
                trail_len = min(60, idx)
                for i in range(max(0, idx - trail_len), idx):
                    if i < len(c['x']) and i + 1 < len(c['x']):
                        x1, y1 = c['x'][i], c['y'][i]
                        x2, y2 = c['x'][i+1], c['y'][i+1]
                        if x1 > 0 and x2 > 0 and not (np.isnan(x1) or np.isnan(x2)):
                            alpha = int(255 * (i - (idx - trail_len)) / max(trail_len, 1))
                            clr = (min(255, alpha), 200, 80)
                            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), clr, 2, cv2.LINE_AA)

            if self.show_relative_trajectory and 'Hip' in markers:
                if idx < len(markers['Hip']['x']):
                    hip_x, hip_y = markers['Hip']['x'][idx], markers['Hip']['y'][idx]
                    if hip_x > 0 and not np.isnan(hip_x):
                        trail_len = min(60, idx)
                        for i in range(max(0, idx - trail_len), idx):
                            if i >= len(c['x']) or i+1 >= len(c['x']) or i >= len(markers['Hip']['x']):
                                continue
                            hx_i, hy_i = markers['Hip']['x'][i], markers['Hip']['y'][i]
                            if hx_i <= 0 or np.isnan(hx_i):
                                continue
                            rx1 = c['x'][i] - hx_i + hip_x
                            ry1 = c['y'][i] - hy_i + hip_y
                            j = min(i+1, len(c['x'])-1)
                            hx_j = markers['Hip']['x'][min(j, len(markers['Hip']['x'])-1)]
                            hy_j = markers['Hip']['y'][min(j, len(markers['Hip']['y'])-1)]
                            rx2 = c['x'][j] - hx_j + hip_x
                            ry2 = c['y'][j] - hy_j + hip_y
                            cv2.line(frame, (int(rx1), int(ry1)), (int(rx2), int(ry2)),
                                     (80, 200, 255), 2, cv2.LINE_AA)

        # Skeleton
        for a, b in self.SKELETON_CONNECTIONS:
            if a in markers and b in markers:
                if idx < len(markers[a]['x']) and idx < len(markers[b]['x']):
                    x1, y1 = markers[a]['x'][idx], markers[a]['y'][idx]
                    x2, y2 = markers[b]['x'][idx], markers[b]['y'][idx]
                    if x1 > 0 and x2 > 0 and not (np.isnan(x1) or np.isnan(x2)):
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), CLR_BONE, 2, cv2.LINE_AA)

        # Dots & labels
        for name, c in markers.items():
            if idx < len(c['x']):
                x, y = c['x'][idx], c['y'][idx]
                if x <= 0 or np.isnan(x):
                    continue
                if name == self.selected_joint:
                    cv2.circle(frame, (int(x), int(y)), 9, CLR_SEL, -1, cv2.LINE_AA)
                    cv2.putText(frame, name, (int(x)+14, int(y)-14),
                                cv2.FONT_HERSHEY_DUPLEX, 0.65, CLR_SEL, 1, cv2.LINE_AA)
                    if self._cache_angle is not None and idx < len(self._cache_angle):
                        ang = self._cache_angle[idx]
                        if not np.isnan(ang):
                            cv2.putText(frame, f"{ang:.1f} deg", (int(x)+14, int(y)+18),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 200, 255), 1, cv2.LINE_AA)
                else:
                    cv2.circle(frame, (int(x), int(y)), 4, CLR_DOT, -1, cv2.LINE_AA)
        return frame

    # ── Click to select ─────────────────────────────────────────────────────

    def _on_video_click(self, pos):
        if not self.trc_data:
            return
        vx = (pos.x() - self.offset_x) / self.scale_factor
        vy = (pos.y() - self.offset_y) / self.scale_factor
        best, best_d = None, 50
        for name, c in self.trc_data['markers'].items():
            if self.current_frame < len(c['x']):
                mx, my = c['x'][self.current_frame], c['y'][self.current_frame]
                if mx <= 0 or np.isnan(mx):
                    continue
                d = np.hypot(mx - vx, my - vy)
                if d < best_d:
                    best_d = d
                    best = name
        if best:
            self.selected_joint = best
            self.sel_header.setText(f"Selected: {best}")
            self._compute_kinematics()
            self._update_all_graphs()
            self._set_frame(self.current_frame)

    # ── Kinematics ──────────────────────────────────────────────────────────

    def _compute_kinematics(self):
        if not self.selected_joint or not self.trc_data:
            return
        c = self.trc_data['markers'][self.selected_joint]
        dt = 1.0 / self.trc_data['data_rate']
        t = self.trc_data['time']
        self._cache_time = t

        s = 1.0 / self.px_per_unit if self.px_per_unit else 1.0

        vx = np.gradient(c['x'] * s, dt)
        vy = np.gradient(c['y'] * s, dt)
        self._cache_vx = smooth(vx)
        self._cache_vy = smooth(vy)
        self._cache_vtotal = smooth(np.sqrt(vx**2 + vy**2))

        ax = np.gradient(self._cache_vx, dt)
        ay = np.gradient(self._cache_vy, dt)
        self._cache_ax = smooth(ax)
        self._cache_ay = smooth(ay)
        self._cache_atotal = smooth(np.sqrt(ax**2 + ay**2))

        angle_col = MARKER_TO_ANGLE.get(self.selected_joint)
        if angle_col and self.mot_data and angle_col in self.mot_data['angles']:
            self._cache_angle_name = angle_col
            raw_angle = self.mot_data['angles'][angle_col]
            n = min(len(t), len(raw_angle))
            self._cache_angle = raw_angle[:n]
            ang_vel = np.gradient(self._cache_angle, dt)
            self._cache_ang_vel = smooth(ang_vel)
            ang_acc = np.gradient(self._cache_ang_vel, dt)
            self._cache_ang_acc = smooth(ang_acc)
        else:
            self._cache_angle_name = None
            self._cache_angle = None
            self._cache_ang_vel = None
            self._cache_ang_acc = None

    # ── Update graphs ───────────────────────────────────────────────────────

    def _update_all_graphs(self):
        if self._cache_time is None:
            return
        t = self._cache_time

        self.graph_vel.clear()
        self.graph_vel.addItem(self.v_lines[0])
        if self._cache_vx is not None:
            n = min(len(t), len(self._cache_vx))
            self.graph_vel.plot(t[:n], self._cache_vx[:n], pen=pg.mkPen('#F38BA8', width=1.5), name="Vx")
            self.graph_vel.plot(t[:n], self._cache_vy[:n], pen=pg.mkPen('#89B4FA', width=1.5), name="Vy")
            self.graph_vel.plot(t[:n], self._cache_vtotal[:n], pen=pg.mkPen('#A6E3A1', width=2), name="Vtotal")

        self.graph_acc.clear()
        self.graph_acc.addItem(self.v_lines[1])
        if self._cache_ax is not None:
            n = min(len(t), len(self._cache_ax))
            self.graph_acc.plot(t[:n], self._cache_ax[:n], pen=pg.mkPen('#F38BA8', width=1.5), name="Ax")
            self.graph_acc.plot(t[:n], self._cache_ay[:n], pen=pg.mkPen('#89B4FA', width=1.5), name="Ay")
            self.graph_acc.plot(t[:n], self._cache_atotal[:n], pen=pg.mkPen('#FAB387', width=2), name="Atotal")

        self.graph_ang_vel.clear()
        self.graph_ang_vel.addItem(self.v_lines[2])
        if self._cache_ang_vel is not None:
            label = self._cache_angle_name or "?"
            self.graph_ang_vel.setTitle(f"Angular Velocity: {label}")
            n = min(len(t), len(self._cache_ang_vel))
            self.graph_ang_vel.plot(t[:n], self._cache_ang_vel[:n], pen=pg.mkPen('#CBA6F7', width=2))
        else:
            self.graph_ang_vel.setTitle("Angular Velocity (no angle data)")

        self.graph_ang_acc.clear()
        self.graph_ang_acc.addItem(self.v_lines[3])
        if self._cache_ang_acc is not None:
            label = self._cache_angle_name or "?"
            self.graph_ang_acc.setTitle(f"Angular Acceleration: {label}")
            n = min(len(t), len(self._cache_ang_acc))
            self.graph_ang_acc.plot(t[:n], self._cache_ang_acc[:n], pen=pg.mkPen('#F9E2AF', width=2))
        else:
            self.graph_ang_acc.setTitle("Angular Acceleration (no angle data)")

    # ── Stats ───────────────────────────────────────────────────────────────

    def _update_stats(self, idx):
        if not self.selected_joint or not self.trc_data:
            return
        c = self.trc_data['markers'].get(self.selected_joint)
        if not c or idx >= len(c['x']):
            return
        x, y = c['x'][idx], c['y'][idx]
        if np.isnan(x):
            for w in [self.stat_pos, self.stat_lin_vel, self.stat_lin_acc,
                      self.stat_angle, self.stat_ang_vel, self.stat_ang_acc]:
                w.setText("N/A")
            return

        sx, sy = self._scale_val(x), self._scale_val(y)
        self.stat_pos.setText(f"({sx:.4f}, {sy:.4f})")

        if self._cache_vtotal is not None and idx < len(self._cache_vtotal):
            self.stat_lin_vel.setText(f"{self._cache_vtotal[idx]:.4f}")
        if self._cache_atotal is not None and idx < len(self._cache_atotal):
            self.stat_lin_acc.setText(f"{self._cache_atotal[idx]:.4f}")
        if self._cache_angle is not None and idx < len(self._cache_angle):
            self.stat_angle.setText(f"{self._cache_angle[idx]:.2f}")
        else:
            self.stat_angle.setText("—")
        if self._cache_ang_vel is not None and idx < len(self._cache_ang_vel):
            self.stat_ang_vel.setText(f"{self._cache_ang_vel[idx]:.2f}")
        else:
            self.stat_ang_vel.setText("—")
        if self._cache_ang_acc is not None and idx < len(self._cache_ang_acc):
            self.stat_ang_acc.setText(f"{self._cache_ang_acc[idx]:.2f}")
        else:
            self.stat_ang_acc.setText("—")

    # ── Playback ────────────────────────────────────────────────────────────

    def _toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_btn.setIcon(qta.icon('fa5s.play', color='white'))
        else:
            if not self.cap:
                return
            self.timer.start(int(1000 / self.fps))
            self.play_btn.setIcon(qta.icon('fa5s.pause', color='white'))

    def _advance_frame(self):
        if self.current_frame < self.total_frames - 1:
            self._set_frame(self.current_frame + 1)
        else:
            self.timer.stop()
            self.play_btn.setIcon(qta.icon('fa5s.play', color='white'))

    def _seek(self, val):
        self._set_frame(val)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    font = QtGui.QFont("Inter", 10)
    app.setFont(font)
    w = Sports2DApp()
    w.show()
    sys.exit(app.exec_())
