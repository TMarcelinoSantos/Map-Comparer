import sys
import yaml
import matplotlib.pyplot as plt

from PyQt5.QtWidgets import QApplication, QLabel, QVBoxLayout, QWidget
from PyQt5.QtCore import Qt


class DropWidget(QLabel):
    def __init__(self):
        super().__init__()

        self.setText("\n\n Drag & Drop your track YAML file here \n\n")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            QLabel {
                border: 3px dashed #aaa;
                font-size: 18px;
                padding: 40px;
            }
        """)

        self.setAcceptDrops(True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        file_path = event.mimeData().urls()[0].toLocalFile()
        self.setText(f"Loaded:\n{file_path}")

        try:
            self.plot_track(file_path)
        except Exception as e:
            self.setText(f"Error:\n{str(e)}")

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

        plt.figure(figsize=(8, 8))
        plt.scatter(left_x, left_y, c='blue', s=10)
        plt.scatter(right_x, right_y, c='yellow', s=10)

        # Optional: centerline
        center_x = [(lx + rx)/2 for lx, rx in zip(left_x, right_x)]
        center_y = [(ly + ry)/2 for ly, ry in zip(left_y, right_y)]
        plt.plot(center_x, center_y, 'k--', label='Centerline')

        plt.axis('equal')
        plt.legend()
        plt.title("Track Layout")
        plt.grid()

        plt.show()


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Track Viewer")
        self.resize(400, 300)

        layout = QVBoxLayout()
        self.drop_widget = DropWidget()

        layout.addWidget(self.drop_widget)
        self.setLayout(layout)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())