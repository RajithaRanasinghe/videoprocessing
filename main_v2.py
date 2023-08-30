import sys
import cv2
import numpy as np
from PySide2.QtWidgets import QApplication, QMainWindow, QLabel, QDialog, QVBoxLayout, QWidget, QMenuBar, QMenu, QAction, QStatusBar, QFileDialog, QHBoxLayout, QSlider, QDesktopWidget, QSplitter, QInputDialog, QMessageBox, QComboBox, QPushButton, QLineEdit, QGridLayout
from PySide2.QtGui import QImage, QPixmap, QColor
from PySide2.QtCore import QTimer, Qt
import os

from skimage.filters import threshold_yen, threshold_triangle, threshold_otsu, threshold_minimum, threshold_mean, threshold_isodata

import torch.nn as nn
import torch.optim as optim
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models
import torch.nn.functional as F
from pytorch_msssim import ssim

class SimpleANN(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(SimpleANN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

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



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        self.enc_conv1 = DoubleConv(in_channels, 64)
        self.enc_conv2 = DoubleConv(64, 128)
        self.enc_conv3 = DoubleConv(128, 256)
        self.enc_conv4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        self.up_trans1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec_conv1 = DoubleConv(512, 256)
        self.up_trans2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec_conv2 = DoubleConv(256, 128)
        self.up_trans3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec_conv3 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.pool(x1)
        x3 = self.enc_conv2(x2)
        x4 = self.pool(x3)
        x5 = self.enc_conv3(x4)
        encoded = self.pool(x5)
        x = self.enc_conv4(encoded)

        # Decoder with skip connections
        x = self.up_trans1(x)
        x = torch.cat([x, x5], dim=1)
        x = self.dec_conv1(x)

        x = self.up_trans2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.dec_conv2(x)

        x = self.up_trans3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.dec_conv3(x)

        return self.out_conv(x)

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=32):
        super(VAE, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim)  # for input 256 x 256
        self.fc_var = nn.Linear(256 * 32 * 32, latent_dim)


        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 256 * 4 * 4)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
        )


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.fc_mu(encoded)
        log_var = self.fc_var(encoded)
        z = self.reparameterize(mu, log_var)
        decoded = self.decoder(self.decoder_input(z).view(-1, 256, 4, 4))
        return decoded, mu, log_var
    
    def decode(self, z):
        # Decoding the latent variable z to get the reconstructed data
        decoded = self.decoder(self.decoder_input(z).view(-1, 256, 4, 4))
        return decoded


class FrameProcessor:
    """Class for frame-by-frame video processing."""

    def __init__(self):
        self.filter_type = "original"
        self.custom_kernel = None
        self.latent_variable = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model = SimpleANN(input_size=224*224*3, hidden_sizes=[500, 300, 100], output_size=224*224*3).to(self.device)
        self.input_size = 1024 * 768 * 3  # replace W, H, and C with the actual values.
        


        # Define preprocessing pipeline for the frame before feeding it to the model
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256,256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def vae_loss(self, recon_x, x, mu, log_var):
        BCE = F.mse_loss(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
           
    def denormalize(self, tensor):
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1).to(self.device)
        return torch.clamp(tensor * std + mean, 0, 1)  # Clamp values between 0 and 1


    def set_cnn_model(self):
        # Initialize the UNet model
        #self.model = UNet(in_channels=3, out_channels=3).to(self.device)
        self.model = VAE(image_channels=3, latent_dim=32).to(self.device)

        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def cnn_filter(self, frame: np.ndarray) -> np.ndarray:

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = self.transform(frame)
        input_rgb = frame_tensor.unsqueeze(0).float().to(self.device)

        # Pass the tensor through the VAE model
        recon_frame, mu, log_var = self.model(input_rgb)

        # Denormalize the tensor
        denormalized_output = self.denormalize(recon_frame)

        # Make sure the tensor shape is [H, W, C] for cv2
        output_rgb = denormalized_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # Convert the output to uint8 format as cv2 expects this
        output_rgb = (output_rgb * 255).astype(np.uint8)

        processed_frame = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)
        
        return processed_frame

    
    def train_on_feedback(self, frame, feedback):
        if feedback > 0:
            strength = feedback
            for strength in range(strength):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_tensor = self.transform(frame)
                input_rgb = frame_tensor.unsqueeze(0).float().to(self.device)
                
                self.optimizer.zero_grad()

                # Pass the tensor through the VAE
                recon_frame, mu, log_var = self.model(input_rgb)

                # Calculate the VAE loss
                loss = self.vae_loss(recon_frame, input_rgb, mu, log_var)

                loss.backward()
                self.optimizer.step()

                # Save the latent variable
                self.latent_variable = mu.detach()


    def generate_from_latent(self, model, latent_variable):
        # Ensure the model is set to evaluation mode
        model.eval()

        # Generate a frame from the given latent variable
        with torch.no_grad():
            generated_frame_tensor = model.decode(latent_variable).squeeze(0)

        # Denormalize the generated tensor
        denormalized_output = self.denormalize(generated_frame_tensor)


        # Convert the tensor format to [H, W, C] for cv2 compatibility
        output_rgb = denormalized_output.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()

        # Convert the tensor values to uint8 format as expected by cv2
        output_rgb = (output_rgb * 255).astype(np.uint8)

        # Convert from RGB to BGR for cv2 compatibility
        processed_frame = cv2.cvtColor(output_rgb, cv2.COLOR_RGB2BGR)

        return processed_frame

    def save_model(self, model_path="saved_model.pth", latent_path="saved_latent.pth"):
        """
        Save the model and its latent variable to the specified paths.

        :param model_path: Path to save the model.
        :param latent_path: Path to save the latent variable.
        """
        torch.save(self.model.state_dict(), model_path)
        torch.save(self.latent_variable, latent_path)
        print(f"Model saved to {model_path} and Latent variable saved to {latent_path}")

    def load_model(self, model_path="saved_model.pth", latent_path="saved_latent.pth"):
        """
        Load the saved model and its latent variable from the specified paths.

        :param model_path: Path to load the model from.
        :param latent_path: Path to load the latent variable from.
        :return: Processed frame after generating from loaded latent variable.
        """
        if not (os.path.exists(model_path) and os.path.exists(latent_path)):
            print("Model or latent variable file does not exist. Please check the paths provided.")
            return None

        self.model.load_state_dict(torch.load(model_path))
        self.model.to(self.device)
        self.model.eval()  # Set model to evaluation mode

        self.latent_variable = torch.load(latent_path).to(self.device)


    def set_filter(self, filter_type, threshold_method="binary"):
        self.filter_type = filter_type
        self.threshold_method = threshold_method

    def set_custom_kernel(self, kernel: np.ndarray):
        self.custom_kernel = kernel

    def process(self, frame: np.ndarray) -> np.ndarray:
        if self.filter_type == "original":
            return frame
        elif self.filter_type == "cnn":
            return self.cnn_filter(frame)
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
        self.current_frame = None
        self.recall_flag = False

        # Fetch screen resolution
        screen_geometry = QDesktopWidget().availableGeometry()
        self.screen_width, self.screen_height = screen_geometry.width(), screen_geometry.height()

        self.feedback_btn = QPushButton("Provide Positive Feedback", self)
        self.feedback_btn.clicked.connect(self.on_feedback)
        
        self.remember_btn = QPushButton("Remember", self)
        self.remember_btn.clicked.connect(self.on_remember)

        self.recall_btn = QPushButton("Recall", self)
        self.recall_btn.clicked.connect(self.on_recall)

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
        #main_layout.addWidget(self.feedback_btn)
        main_layout.addWidget(self.remember_btn)
        main_layout.addWidget(self.recall_btn)
        
    
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Menu bar setup
        self.setup_menu_bar()

        # Status bar setup with indicator
        self.setup_status_bar()

        # Window setup
        self.setWindowTitle("HMI v0.1")
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

        cnn_action = QAction("cnn", self)
        cnn_action.triggered.connect(lambda: self.set_filter("cnn"))

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

        filters_menu.addActions([original_action, cnn_action, sobel_action, gaussian_action, canny_action, sepia_action, negative_action, color_swap_action, custom_action])

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

    def on_remember(self):
        self.processor.train_on_feedback(self.current_frame, 30) # 30 iterations
        self.processor.save_model()

    def on_recall(self):
        self.processor.load_model()
        if self.recall_flag:
            self.recall_flag = False
            self.recall_indicator.setStyleSheet("background-color: red; border-radius: 8px;")
            self.status_bar.showMessage("Recall is OFF")
        else:
            self.recall_flag = True
            self.recall_indicator.setStyleSheet("background-color: grenn; border-radius: 8px;")
            self.status_bar.showMessage("Recall is ON")

        self.recall_indicator.setText('Recalling {}'.format(self.recall_flag))

    def on_feedback(self):
        # We assume 1 represents positive feedback and -1 represents negative feedback
        feedback = 1
        self.processor.train_on_feedback(self.current_frame, feedback)


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
        # Status Indicator
        self.status_indicator = QLabel(self)
        self.status_indicator.setMinimumWidth(100)
        
        
        # Recall Indicator
        self.recall_indicator = QLabel(self)
        self.recall_indicator.setText('Recalling {}'.format(self.recall_flag))
        self.recall_indicator.setStyleSheet("background-color: red; border-radius: 8px;")
        self.recall_indicator.setMinimumWidth(100)

        # Create a horizontal layout
        layout = QHBoxLayout()
        layout.addWidget(self.recall_indicator)
        layout.addWidget(self.status_indicator)
        
        layout.setSpacing(10)  # Add space between widgets
        layout.setContentsMargins(5, 5, 5, 5)  # Add margins to the layout

        # Create a container widget for the layout
        container_widget = QWidget(self)
        container_widget.setLayout(layout)
        #container_widget.setFixedSize(100, 25)  # Adjust the size as needed

        # Create and set up the status bar
        self.status_bar = QStatusBar(self)
        self.status_bar.addPermanentWidget(container_widget)
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
                interval = 5 #FPS
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

            W = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Width of the frames
            H = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) # Height of the frames
            C = 3  # Number of channels (3 for color frames, 1 for grayscale)

            input_size = W * H * C
            self.processor.input_size = input_size
            #self.processor.set_ann_model()
            self.processor.set_cnn_model()       
            
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
            self.current_frame = frame
        
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

                #self.processor.train_on_feedback(self.current_frame, 1)

                if self.recall_flag:
                    # Process and show the processed frame
                    processed_frame = self.processor.generate_from_latent(self.processor.model, self.processor.latent_variable)
                    self.display_frame(processed_frame, self.processed_label)
                else:
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
    app.setStyle('Fusion')

    # Load the stylesheet
    STYLE_FILE_NAME = "style_dark.qss"
    FOLDER_NAME = "/app_data"
    file_path = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)) +
        FOLDER_NAME,
        STYLE_FILE_NAME)
    with open(file_path, "r", encoding="utf-8") as f:
        app.setStyleSheet(f.read())

    window = App()
    window.showMaximized()
    sys.exit(app.exec_())
