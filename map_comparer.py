import sys
import yaml

from PyQt5.QtWidgets import (
    QApplication, QLabel, QVBoxLayout, QWidget, QSplitter
)
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


# =========================
# Matplotlib Canvas
# =========================
class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8, 6), dpi=100)
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

        self.canvas.ax.clear()

        self.canvas.ax.scatter(left_x, left_y, c="blue", marker="o", label="Left")
        self.canvas.ax.scatter(right_x, right_y, c="yellow", marker="o", label="Right")

        self.canvas.ax.set_title(self.title)
        self.canvas.ax.set_aspect("equal")
        self.canvas.ax.grid()
        self.canvas.ax.legend()

        self.canvas.draw()


# =========================
# Drop + Map container
# =========================
class DropMapPanel(QWidget):
    def __init__(self, title, hint_text, accent_color):
        super().__init__()

        self.map = MapPanel(title)

        self.drop_label = QLabel(hint_text)
        self.drop_label.setAlignment(Qt.AlignCenter)

        self.drop_label.setStyleSheet(f"""
            QLabel {{
                border: 3px dashed {accent_color};
                font-size: 18px;
                font-weight: bold;
                padding: 30px;
                color: #444;
                background-color: #fafafa;
            }}
        """)

        layout = QVBoxLayout()
        layout.addWidget(self.drop_label)
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
            self.drop_label.hide()
            self.map.show()

        except Exception as e:
            self.drop_label.setText(f"Error:\n{str(e)}")


# =========================
# Main Window
# =========================
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Track Viewer")
        self.resize(1200, 700)

        # LEFT = Ground Truth
        self.left_panel = DropMapPanel(
            title="GROUND TRUTH",
            hint_text="Drop GROUND TRUTH map here",
            accent_color="#2b6cb0"   # blue
        )

        # RIGHT = SLAM
        self.right_panel = DropMapPanel(
            title="SLAM MAP",
            hint_text="Drop SLAM map here",
            accent_color="#c53030"   # red
        )

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.left_panel)
        splitter.addWidget(self.right_panel)
        splitter.setSizes([600, 600])

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