import sys
import yaml

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class MplCanvas(FigureCanvas):
    def __init__(self):
        self.fig = Figure(figsize=(8,6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)


class DropWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.label = QLabel("\n\n Drag & Drop your track YAML file here \n\n")
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                font-size: 16px;
                padding: 20px;
            }
        """)

        # ✅ create canvas first
        self.canvas = MplCanvas()

        # ✅ THEN create toolbar (THIS WAS MISSING)
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        self.toolbar = NavigationToolbar(self.canvas, self)

        # layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)   # now it exists → no crash

        self.setLayout(layout)
        self.setAcceptDrops(True)
    
        self.canvas.hide()
        self.toolbar.hide()

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.label.setText(f"Loaded:\n{file_path}")

        try:
            self.plot_track(file_path)
            # Hide the label after successful plotting
            self.label.setVisible(False)
            self.canvas.show()
            self.toolbar.show()
        except Exception as e:
            self.label.setText(f"Error:\n{str(e)}")

    def plot_track(self, file_path):
        with open(file_path, 'r') as f:
            data = yaml.safe_load(f)

        track = data['track']
        left = track['left']
        right = track['right']

        left_x = [p['position'][0] for p in left]
        left_y = [p['position'][1] for p in left]

        right_x = [p['position'][0] for p in right]
        right_y = [p['position'][1] for p in right]

        # Clear previous plot
        self.canvas.ax.clear()

        # Plot boundaries
        self.canvas.ax.scatter(left_x, left_y, c='blue', marker='o', label='Left')
        self.canvas.ax.scatter(right_x, right_y, c='yellow', marker='o', label='Right')


        self.canvas.ax.set_title("Track Layout")
        self.canvas.ax.set_aspect('equal')
        self.canvas.ax.grid()
        self.canvas.ax.legend()

        self.canvas.draw()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Track Viewer")
        self.resize(800, 600)

        layout = QVBoxLayout()
        layout.addWidget(DropWidget())

        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())