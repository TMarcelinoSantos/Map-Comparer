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


def icp_full(A, B, max_iterations=20):
    A = A.copy()

    R_total = np.eye(2)
    t_total = np.zeros(2)

    for _ in range(max_iterations):
        tree = cKDTree(B)
        _, indices = tree.query(A)

        matched_B = B[indices]

        R, t = best_fit_transform(A, matched_B)

        A = (R @ A.T).T + t

        R_total = R @ R_total
        t_total = R @ t_total + t

    return R_total, t_total


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

        # -------- Robust loading --------
        left = track.get("left", [])
        right = track.get("right", [])
        unknown = track.get("unknown", [])

        # Filter unknown
        unknown_blue = [c for c in unknown if c.get("class") == "blue"]
        unknown_yellow = [c for c in unknown if c.get("class") == "yellow"]

        # Merge both sources
        blue = left + unknown_blue
        yellow = right + unknown_yellow

        if (len(left) > 0 or len(right) > 0) and len(unknown) > 0:
            print("INFO: Merging 'left/right' with filtered 'unknown' cones")

        # Extract positions
        left_x = [p["position"][0] for p in blue]
        left_y = [p["position"][1] for p in blue]

        right_x = [p["position"][0] for p in yellow]
        right_y = [p["position"][1] for p in yellow]

        # Safe storage
        self.left_points = np.column_stack((left_x, left_y)) if left_x else np.empty((0, 2))
        self.right_points = np.column_stack((right_x, right_y)) if right_x else np.empty((0, 2))

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
# Overlay Panel
# =========================
class OverlayPanel(QWidget):
    def __init__(self):
        super().__init__()

        self.canvas = MplCanvas()
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)

    def plot_overlay(self, gt_left, gt_right, slam_left, slam_right):
        ax = self.canvas.ax
        ax.clear()

        ax.scatter(gt_left[:, 0], gt_left[:, 1], c="blue", s=20, label="GT Blue")
        ax.scatter(gt_right[:, 0], gt_right[:, 1], c="gold", s=20, label="GT Yellow")

        ax.scatter(slam_left[:, 0], slam_left[:, 1], c="cyan", s=10, label="SLAM Blue")
        ax.scatter(slam_right[:, 0], slam_right[:, 1], c="orange", s=10, label="SLAM Yellow")

        # Connections
        tree_blue = cKDTree(gt_left)
        _, idx_blue = tree_blue.query(slam_left)

        for i, j in enumerate(idx_blue):
            ax.plot(
                [slam_left[i][0], gt_left[j][0]],
                [slam_left[i][1], gt_left[j][1]],
                'gray', linewidth=0.5
            )

        tree_yellow = cKDTree(gt_right)
        _, idx_yellow = tree_yellow.query(slam_right)

        for i, j in enumerate(idx_yellow):
            ax.plot(
                [slam_right[i][0], gt_right[j][0]],
                [slam_right[i][1], gt_right[j][1]],
                'gray', linewidth=0.5
            )

        ax.set_title("Overlay + Correspondences")
        ax.set_aspect("equal")
        ax.grid()
        ax.legend()

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
# Metric Box
# =========================
class MetricBox(QLabel):
    def __init__(self, title, color):
        super().__init__()

        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"""
            QLabel {{
                border: 2px solid {color};
                border-radius: 10px;
                padding: 15px;
                font-size: 16px;
                background-color: #f9f9f9;
            }}
        """)

        self.setText(f"{title}\n---")

    def set_value(self, title, value):
        self.setText(f"{title}\n{value:.3f} m")


# =========================
# Control Panel
# =========================
class ControlPanel(QWidget):
    def __init__(self, left, right, overlay):
        super().__init__()

        self.left = left
        self.right = right
        self.overlay = overlay

        layout = QVBoxLayout()

        self.blue_box = MetricBox("Blue ATE", "#2b6cb0")
        self.yellow_box = MetricBox("Yellow ATE", "#d69e2e")
        self.total_box = MetricBox("Total ATE", "#444")

        btn_left = QPushButton("Reset GT")
        btn_right = QPushButton("Reset SLAM")
        btn_all = QPushButton("Reset BOTH")

        btn_left.clicked.connect(self.left.reset)
        btn_right.clicked.connect(self.right.reset)
        btn_all.clicked.connect(self.reset_all)

        layout.addWidget(self.blue_box)
        layout.addWidget(self.yellow_box)
        layout.addWidget(self.total_box)

        layout.addSpacing(20)

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

        gt_all = np.vstack((gt_left, gt_right))
        slam_all = np.vstack((slam_left, slam_right))

        R, t = icp_full(slam_all, gt_all)

        slam_left_aligned = (R @ slam_left.T).T + t
        slam_right_aligned = (R @ slam_right.T).T + t

        blue_ate = symmetric_ate(gt_left, slam_left_aligned)
        yellow_ate = symmetric_ate(gt_right, slam_right_aligned)
        total_ate = (blue_ate + yellow_ate) / 2

        self.blue_box.set_value("Blue ATE", blue_ate)
        self.yellow_box.set_value("Yellow ATE", yellow_ate)
        self.total_box.set_value("Total ATE", total_ate)

        self.overlay.plot_overlay(gt_left, gt_right, slam_left_aligned, slam_right_aligned)

    def reset_all(self):
        self.left.reset()
        self.right.reset()
        self.overlay.clear()

        self.blue_box.setText("Blue ATE\n---")
        self.yellow_box.setText("Yellow ATE\n---")
        self.total_box.setText("Total ATE\n---")


# =========================
# Main Window
# =========================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Track Viewer")
        self.resize(1500, 700)

        self.left_panel = DropMapPanel("GROUND TRUTH", "Drop Ground Truth map", "#2b6cb0")
        self.right_panel = DropMapPanel("SLAM MAP", "Drop SLAM map", "#c53030")
        self.overlay_panel = OverlayPanel()

        self.controls = ControlPanel(self.left_panel, self.right_panel, self.overlay_panel)

        self.left_panel.controls = self.controls
        self.right_panel.controls = self.controls

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.addWidget(self.overlay_panel)
        splitter.addWidget(self.controls)

        splitter.setSizes([400, 400, 500, 200])

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