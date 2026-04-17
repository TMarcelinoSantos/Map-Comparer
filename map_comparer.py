import sys
import yaml
import numpy as np

from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget,
    QSplitter, QPushButton
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from scipy.spatial import cKDTree


# =========================
# ICP + ATE UTILITIES
# =========================
def best_fit_transform(A, B):
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    AA = A - centroid_A
    BB = B - centroid_B

    H = AA.T @ BB
    U, _, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T

    t = centroid_B - R @ centroid_A

    return R, t


def icp(A, B, max_iterations=20):
    A = A.copy()

    for _ in range(max_iterations):
        tree = cKDTree(B)
        _, indices = tree.query(A)

        matched_B = B[indices]

        R, t = best_fit_transform(A, matched_B)
        A = (R @ A.T).T + t

    return A


def symmetric_ate(gt_points, slam_points):
    if len(gt_points) == 0 or len(slam_points) == 0:
        return None

    tree_gt = cKDTree(gt_points)
    tree_slam = cKDTree(slam_points)

    d1, _ = tree_gt.query(slam_points)
    d2, _ = tree_slam.query(gt_points)

    return np.sqrt((np.mean(d1**2) + np.mean(d2**2)) / 2)


# =========================
# Matplotlib Canvas
# =========================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


# =========================
# Map Panel
# =========================
class MapPanel(QWidget):
    def __init__(self, title="Map"):
        super().__init__()

        self.title = title
        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def plot_track(self, file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)

        track = data["track"]
        left = track["left"]
        right = track["right"]

        left_x = [p["position"][0] for p in left]
        left_y = [p["position"][1] for p in left]

        right_x = [p["position"][0] for p in right]
        right_y = [p["position"][1] for p in right]

        # Store points
        self.left_points = np.column_stack((left_x, left_y))
        self.right_points = np.column_stack((right_x, right_y))

        # Plot
        self.canvas.ax.clear()
        self.canvas.ax.scatter(left_x, left_y, c="blue", s=10, label="Blue cones")
        self.canvas.ax.scatter(right_x, right_y, c="yellow", s=10, label="Yellow cones")

        self.canvas.ax.set_title(self.title)
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.grid()
        self.canvas.ax.legend()

        self.canvas.draw()

    def clear(self):
        self.canvas.ax.clear()
        self.canvas.draw()


# =========================
# Drop Panel
# =========================
class DropMapPanel(QWidget):
    def __init__(self, title, hint_text, color):
        super().__init__()

        self.map = MapPanel(title)

        self.label = QLabel(hint_text)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet(f"""
            QLabel {{
                border: 3px dashed {color};
                font-size: 18px;
                padding: 30px;
            }}
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.map)

        self.setLayout(layout)
        self.setAcceptDrops(True)

        self.map.hide()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()

        try:
            self.map.plot_track(file_path)
            self.label.hide()
            self.map.show()

            if hasattr(self, "controls"):
                self.controls.compute_ate()

        except Exception as e:
            self.label.setText(f"Error:\n{str(e)}")

    def reset(self):
        self.map.clear()
        self.map.hide()
        self.label.show()


# =========================
# Control Panel
# =========================
class ControlPanel(QWidget):
    def __init__(self, left, right):
        super().__init__()

        self.left = left
        self.right = right

        layout = QVBoxLayout()

        self.ate_label = QLabel("ATE: -")

        btn_left = QPushButton("Reset GT")
        btn_right = QPushButton("Reset SLAM")
        btn_all = QPushButton("Reset BOTH")

        btn_left.clicked.connect(self.left.reset)
        btn_right.clicked.connect(self.right.reset)
        btn_all.clicked.connect(self.reset_all)

        layout.addWidget(self.ate_label)
        layout.addWidget(btn_left)
        layout.addWidget(btn_right)
        layout.addWidget(btn_all)
        layout.addStretch()

        self.setLayout(layout)

    def compute_ate(self):
        if not hasattr(self.left.map, "left_points") or not hasattr(self.right.map, "left_points"):
            return

        gt_left = self.left.map.left_points
        gt_right = self.left.map.right_points

        slam_left = self.right.map.left_points
        slam_right = self.right.map.right_points

        # Align
        slam_left = icp(slam_left, gt_left)
        slam_right = icp(slam_right, gt_right)

        blue_ate = symmetric_ate(gt_left, slam_left)
        yellow_ate = symmetric_ate(gt_right, slam_right)

        self.ate_label.setText(
            f"Blue ATE: {blue_ate:.3f} m\n"
            f"Yellow ATE: {yellow_ate:.3f} m"
        )

    def reset_all(self):
        self.left.reset()
        self.right.reset()
        self.ate_label.setText("ATE: -")


# =========================
# Main Window
# =========================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Track Viewer")
        self.resize(1300, 700)

        self.left_panel = DropMapPanel(
            "GROUND TRUTH",
            "Drop Ground Truth map",
            "#2b6cb0"
        )

        self.right_panel = DropMapPanel(
            "SLAM MAP",
            "Drop SLAM map",
            "#c53030"
        )

        self.controls = ControlPanel(self.left_panel, self.right_panel)

        # Link controls
        self.left_panel.controls = self.controls
        self.right_panel.controls = self.controls

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.addWidget(self.controls)

        splitter.setSizes([500, 500, 200])

        layout = QVBoxLayout()
        layout.addWidget(splitter)
        self.setLayout(layout)


# =========================
# Run
# =========================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())