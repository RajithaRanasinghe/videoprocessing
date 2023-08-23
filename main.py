import sys
import cv2
import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QAction, QStatusBar, QFileDialog, QHBoxLayout, QSlider, QDesktopWidget, QSplitter, QInputDialog, QMessageBox, QComboBox, QPushButton, QLineEdit, QGridLayout
from PySide2.QtGui import QImage, QPixmap, QColor
from PySide2.QtCore import QTimer, Qt

from skimage.filters import threshold_yen, threshold_triangle, threshold_otsu, threshold_minimum, threshold_mean, threshold_isodata


class CustomFilterDialog(QDialog):

    def __init__(self):
        super().__init__()

        self.mainLayout = QVBoxLayout()
        self.setLayout(self.mainLayout)
        self.kernel = None  # To store the numpy array

        # Create a dropdown (QComboBox) to select filter size
        self.comboBox = QComboBox(self)
        self.comboBox.addItems(['3x3', '4x4', '5x5'])
        self.mainLayout.addWidget(self.comboBox)

        # Create a button to set the grid for inputting filter values
        self.setButton = QPushButton('Set Filter Size', self)
        self.setButton.clicked.connect(self.createFilterGrid)
        self.mainLayout.addWidget(self.setButton)

    def createFilterGrid(self):
        size = int(self.comboBox.currentText().split('x')[0])

        # Clear existing grid if there is one
        for i in reversed(range(self.mainLayout.count())):
            widget = self.mainLayout.itemAt(i).widget()
            if isinstance(widget, QLineEdit) or isinstance(widget, QLabel) or isinstance(widget, QPushButton):
                widget.deleteLater()

        self.gridLayout = QGridLayout()
        self.entries = [[QLineEdit(self) for _ in range(size)] for _ in range(size)]

        for i in range(size):
            for j in range(size):
                self.gridLayout.addWidget(QLabel(f"R{i+1}C{j+1}:"), i, 2*j) 
                self.gridLayout.addWidget(self.entries[i][j], i, 2*j + 1)

        self.mainLayout.addLayout(self.gridLayout)

        # Add a submit button
        self.submitButton = QPushButton('Submit', self)
        self.submitButton.clicked.connect(self.getFilterValues)
        self.mainLayout.addWidget(self.submitButton)

    def getFilterValues(self):
        size = int(self.comboBox.currentText().split('x')[0])

        # Check if any of the entries are empty
        for i in range(size):
            for j in range(size):
                if self.entries[i][j].text().strip() == '':
                    # If an entry is empty, display an error message and return
                    self.errorLabel = QLabel("<font color='red'>All fields must be filled!</font>", self)
                    self.mainLayout.addWidget(self.errorLabel)
                    return

        filter_values = [[float(self.entries[i][j].text()) for j in range(size)] for i in range(size)]
        self.kernel = np.array(filter_values)  # Convert to numpy array
        self.accept()

    def get_kernel(self):
        return self.kernel




class FrameProcessor:
    """Class for frame-by-frame video processing."""

    def __init__(self):
        self.filter_type = "original"
        self.custom_kernel = None

    def set_filter(self, filter_type, threshold_method="binary"):
        self.filter_type = filter_type
        self.threshold_method = threshold_method

    def set_custom_kernel(self, kernel: np.ndarray):
        self.custom_kernel = kernel

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.filter_type == "original":
            return frame
        elif self.filter_type == "sobel":
            return self.sobel_filter(frame)
        elif self.filter_type == "gaussian":
            return cv2.GaussianBlur(frame, (15, 15), 0)
        elif self.filter_type == "canny":
            return self.canny_edge_detection(frame)
        elif self.filter_type == "sepia":
            return self.sepia_filter(frame)
        elif self.filter_type == "negative":
            return self.negative_filter(frame)
        elif self.filter_type == "threshold":
            return self.threshold_filter(frame, method=self.threshold_method)
        elif self.filter_type == "color_swap":
            return self.color_swap(frame)
        elif self.filter_type == "custom":
            return self.custom_filter(frame)
        else:
           print('Wrong Filter type: returning original')
           return frame
         
    def custom_filter(self, frame: np.ndarray) -> np.ndarray:
        if self.custom_kernel is None:
            print("No custom kernel provided. Returning original frame.")
            return frame
        return cv2.filter2D(frame, -1, self.custom_kernel)
          
    def canny_edge_detection(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)

    def sobel_filter(self, frame: np.ndarray) -> np.ndarray:
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Calculate the x and y gradients using the Sobel operator
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        grad_y = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        
        # Convert gradients back to 8 bit unsigned int
        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)
        
        # Combine the two gradients
        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        
        # Convert single channel grayscale image to three channel RGB
        sobel_colored = cv2.cvtColor(sobel, cv2.COLOR_GRAY2RGB)
        
        return sobel_colored

    def sepia_filter(self, frame: np.ndarray) -> np.ndarray:
        sepia_filter = np.array([[0.272, 0.534, 0.131],
                                 [0.349, 0.686, 0.168],
                                 [0.393, 0.769, 0.189]])
        sepia_frame = frame @ sepia_filter.T
        np.clip(sepia_frame, 0, 255, out=sepia_frame)
        sepia_frame = sepia_frame.astype(np.uint8)
        return sepia_frame

    def negative_filter(self, frame: np.ndarray) -> np.ndarray:
        return 255 - frame

    def threshold_filter(self, frame: np.ndarray, method="binary") -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        if method == "binary":
            _, thresholded = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        elif method == "yen":
            threshold_value = threshold_yen(gray)
            _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "triangle":
            threshold_value = threshold_triangle(gray)
            _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "otsu":
            _, thresholded = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == "minimum":
            threshold_value = threshold_minimum(gray)
            _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "mean":
            threshold_value = np.mean(gray)
            _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        elif method == "isodata":
            threshold_value = threshold_isodata(gray)
            _, thresholded = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)
        else:
            print(f"Warning: Unknown threshold method '{method}' provided!")
            return frame
        
        return cv2.cvtColor(thresholded, cv2.COLOR_GRAY2RGB)

    def color_swap(self, frame: np.ndarray) -> np.ndarray:
        b, g, r = cv2.split(frame)
        return cv2.merge([r, g, b])


class App(QMainWindow):
    def __init__(self):
        super().__init__()

        # Webcam and video file variables
        self.cap = None
        self.timer = None
        self.video_source = 'webcam'  # default to webcam
        self.processor = FrameProcessor()
        self.slider_being_dragged = False

        # Fetch screen resolution
        screen_geometry = QDesktopWidget().availableGeometry()
        self.screen_width, self.screen_height = screen_geometry.width(), screen_geometry.height()

        # GUI setup
        self.setup_ui()

    def setup_ui(self):
        # GUI elements setup
        self.input_label = QLabel(self)
        self.input_label.setScaledContents(True)
        self.processed_label = QLabel(self)
        self.processed_label.setScaledContents(True)

        # Set QMainWindow size
        self.setMinimumSize(self.screen_width, self.screen_height)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self.input_label)
        splitter.addWidget(self.processed_label)

        # Add video progress bar
        self.video_progress = QSlider(self)
        self.video_progress.setOrientation(Qt.Horizontal)
        self.video_progress.sliderPressed.connect(self.slider_pressed)
        self.video_progress.sliderReleased.connect(self.slider_released)
        self.video_progress.setVisible(False)  # Initially hidden

        main_layout = QVBoxLayout()
        main_layout.addWidget(splitter)
        main_layout.addWidget(self.video_progress)
        
    
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Menu bar setup
        self.setup_menu_bar()

        # Status bar setup with indicator
        self.setup_status_bar()

        # Window setup
        self.setWindowTitle("Webcam with PySide2 - Split View")
        self.resize(1600, 600)

    def setup_menu_bar(self):
        menu_bar = QMenuBar(self)

        # File Menu
        file_menu = QMenu("File", self)
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # Video Source submenu
        source_menu = QMenu("Video Source", self)
        webcam_action = QAction("Webcam", self)
        webcam_action.triggered.connect(self.set_webcam_source)
        file_action = QAction("FromFile", self)
        file_action.triggered.connect(self.set_file_source)
        source_menu.addActions([webcam_action, file_action])
        file_menu.addMenu(source_menu)

        menu_bar.addMenu(file_menu)

        # Run and Stop actions
        run_action = QAction("Run", self)
        run_action.triggered.connect(self.start_webcam)
        stop_action = QAction("Stop", self)
        stop_action.triggered.connect(self.stop_webcam)

        # Filters menu
        self.setup_filters_menu(menu_bar)

        menu_bar.addActions([run_action, stop_action])
        self.setMenuBar(menu_bar)

    def setup_filters_menu(self, menu_bar):
        filters_menu = QMenu("Filters", self)

        # Adding filter actions
        original_action = QAction("Original", self)
        original_action.triggered.connect(lambda: self.set_filter("original"))

        sobel_action = QAction("Sobel Filter", self)
        sobel_action.triggered.connect(lambda: self.set_filter("sobel"))

        gaussian_action = QAction("Gaussian Blur", self)
        gaussian_action.triggered.connect(lambda: self.set_filter("gaussian"))

        canny_action = QAction("Canny Edge Detection", self)
        canny_action.triggered.connect(lambda: self.set_filter("canny"))

        sepia_action = QAction("Sepia", self)
        sepia_action.triggered.connect(lambda: self.set_filter("sepia"))
        
        negative_action = QAction("Negative", self)
        negative_action.triggered.connect(lambda: self.set_filter("negative"))
           
        color_swap_action = QAction("Color Swap", self)
        color_swap_action.triggered.connect(lambda: self.set_filter("color_swap"))
        
        custom_action = QAction("Custom Kernel", self)
        custom_action.triggered.connect(self.set_custom_filter)

        filters_menu.addActions([original_action, sobel_action, gaussian_action, canny_action, sepia_action, negative_action, color_swap_action, custom_action])

        threshold_submenu = QMenu("Threshold", self)

        binary_action = QAction("Binary", self)
        binary_action.triggered.connect(lambda: self.set_filter("binary"))

        yen_action = QAction("Yen's Method", self)
        yen_action.triggered.connect(lambda: self.set_filter("yen"))

        triangle_action = QAction("Triangle Algorithm", self)
        triangle_action.triggered.connect(lambda: self.set_filter("triangle"))

        otsu_action = QAction("Otsu's Method", self)
        otsu_action.triggered.connect(lambda: self.set_filter("otsu"))

        minimum_action = QAction("Minimum Method", self)
        minimum_action.triggered.connect(lambda: self.set_filter("minimum"))

        mean_action = QAction("Mean of Grayscale", self)
        mean_action.triggered.connect(lambda: self.set_filter("mean"))

        isodata_action = QAction("ISODATA Method", self)
        isodata_action.triggered.connect(lambda: self.set_filter("isodata"))

        threshold_submenu.addActions([binary_action, yen_action, triangle_action, otsu_action, minimum_action, mean_action, isodata_action])
        filters_menu.addMenu(threshold_submenu)
        menu_bar.addMenu(filters_menu)


    def set_custom_filter(self):
        dialog = CustomFilterDialog()
        dialog.exec_()
        self.processor.set_custom_kernel(dialog.get_kernel())
        self.processor.set_filter('custom')

    def slider_pressed(self):
        self.slider_being_dragged = True

    def slider_released(self):
        if self.cap and self.video_source != "webcam":
            desired_frame = self.video_progress.value()
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, desired_frame)
            self.slider_being_dragged = False

    def set_filter(self, filter_type):
        if filter_type in ["binary", "yen", "triangle", "otsu", "minimum", "mean", "isodata"]:
            self.processor.set_filter("threshold", threshold_method=filter_type)
        else:
            self.processor.set_filter(filter_type)


    def setup_status_bar(self):
        self.status_indicator = QLabel(self)
        self.status_indicator.setFixedSize(16, 16)
        self.status_bar = QStatusBar(self)
        self.status_bar.addPermanentWidget(self.status_indicator)
        self.setStatusBar(self.status_bar)
        self.update_status(False)

    def set_webcam_source(self):
        self.video_source = 'webcam'

    def set_file_source(self):
        filepath, _ = QFileDialog.getOpenFileName(self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mkv *.mov);;All Files (*)")
        if filepath:
            self.video_source = filepath
            # If video file is selected, make the progress bar visible
            self.video_progress.setVisible(True)


    def start_webcam(self):
        if not self.cap:
            if self.video_source == 'webcam':
                self.cap = cv2.VideoCapture(0)

                # Timer for updating video feed
                interval = 30 #FPS
            else:
                self.cap = cv2.VideoCapture(self.video_source)
                
                # Adjust the range of the slider to match the video's duration in milliseconds
                total_duration = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.cap.get(cv2.CAP_PROP_FPS) * 1000)
                self.video_progress.setRange(0, total_duration)
                self.video_progress.setValue(0)
                frame_rate = int(self.cap.get(cv2.CAP_PROP_FPS))
                interval = int(1000 / frame_rate)  # Convert frame rate to interval in milliseconds

            if not self.cap.isOpened():
                print("Could not open video source!")
                self.cap = None
                return
            
            
            # Timer for updating video feed
            self.timer = QTimer(self)
            self.timer.timeout.connect(self.update_frame)
            self.timer.start(interval)  # Use the adjusted interval here
            self.update_status(True)

    def stop_webcam(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            self.cap = None
            if self.timer:
                self.timer.stop()
                self.timer = None
            self.input_label.clear()
            self.processed_label.clear()
            self.update_status(False)

    def update_frame(self):
        if self.cap:
            ret, frame = self.cap.read()
        
            # Update video progress slider based on video time
            if self.video_source != "webcam":
                elapsed_time = int(self.cap.get(cv2.CAP_PROP_POS_MSEC))
                # Update the slider only if it's not being dragged by the user
                if not self.slider_being_dragged:
                    self.video_progress.setValue(elapsed_time)

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Show the unprocessed frame
                self.display_frame(frame, self.input_label)

                # Process and show the processed frame
                processed_frame = self.processor.process(frame)
                self.display_frame(processed_frame, self.processed_label)
            else:
                # End of video file
                self.stop_webcam()

    def display_frame(self, frame, label):
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        label.setPixmap(pixmap)

    def update_status(self, camera_on):
        if camera_on:
            self.status_indicator.setStyleSheet("background-color: green; border-radius: 8px;")
            self.status_bar.showMessage("Video is ON")
        else:
            self.status_indicator.setStyleSheet("background-color: red; border-radius: 8px;")
            self.status_bar.showMessage("Video is OFF")

    def closeEvent(self, event):
        self.stop_webcam()
        super().closeEvent(event)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = App()
    window.showMaximized()
    sys.exit(app.exec_())
