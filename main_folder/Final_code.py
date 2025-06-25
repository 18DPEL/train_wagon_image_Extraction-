import sys
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from PyQt6.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, 
                            QWidget, QPushButton, QFileDialog, QHBoxLayout, 
                            QMessageBox, QProgressBar)
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread

class DetectionThread(QThread):
    update_signal = pyqtSignal(object, object)
    finished_signal = pyqtSignal()
    
    def __init__(self, model, cap, class_list, area, output_dir):
        super().__init__()
        self.model = model
        self.cap = cap
        self.class_list = class_list
        self.area = area
        self.output_dir = output_dir
        self.cropped_wagons = set()
        self.paused = False
        self.running = True
        
    def run(self):
        count = 0
        while self.running:
            if self.paused:
                continue
                
            ret, frame = self.cap.read()
            if not ret:
                break
                
            count += 1
            if count % 2 != 0:
                continue
                
            frame = cv2.resize(frame, (1020, 500))
            results = self.model.predict(frame)
            
            try:
                boxes_data = results[0].boxes.data.cpu().numpy()
                boxes_df = pd.DataFrame(boxes_data).astype("float")
                
                for index, row in boxes_df.iterrows():
                    x1, y1, x2, y2, _, d = map(int, row)
                    class_name = self.class_list[d]
                    if 'wagon' in class_name:
                        result = cv2.pointPolygonTest(np.array(self.area, np.int32), ((x2, y2)), False)
                        if result >= 0:
                            unique_id = f'{x1}-{y1}-{x2}-{y2}'
                            if unique_id not in self.cropped_wagons:
                                self.cropped_wagons.add(unique_id)
                                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.circle(frame, (x2, y2), 5, (255, 0, 255), -1)
                                cv2.putText(frame, str(class_name), (x1, y1), 
                                           cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)
                                
                                cropped_image = frame[y1:y2, x1:x2]
                                crop_filename = os.path.join(self.output_dir, f'cropped_wagon_{count}_{index}.jpg')
                                cv2.imwrite(crop_filename, cropped_image)
                
                cv2.polylines(frame, [np.array(self.area, np.int32)], True, (255, 0, 0), 2)
                cv2.putText(frame, str('Detection Area'), (882, 407), 
                          cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 2)
                
                self.update_signal.emit(frame, count)
                
            except Exception as e:
                print(f"Error processing frame: {e}")
                
        self.finished_signal.emit()
        
    def stop(self):
        self.running = False

class WagonDetectionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Train Wagon Detection and Cropping")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize variables
        self.model = None
        self.cap = None
        self.class_list = []
        self.area = [(945, 23), (958, 23), (958, 390), (945, 390)]
        self.output_dir = ""
        self.detection_thread = None
        
        # UI Setup
        self.init_ui()
        self.load_default_model()
        
    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, 4)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 2px solid grey;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #05B8CC;
                width: 10px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Control buttons
        control_layout = QHBoxLayout()
        
        self.load_video_btn = QPushButton("Load Video")
        self.load_video_btn.clicked.connect(self.load_video)
        self.load_video_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        
        self.start_btn = QPushButton("Start Detection")
        self.start_btn.clicked.connect(self.start_detection)
        self.start_btn.setEnabled(False)
        self.start_btn.setStyleSheet("""
            QPushButton {
                background-color: #008CBA;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #007B9E;
            }
        """)
        
        self.pause_btn = QPushButton("Pause")
        self.pause_btn.clicked.connect(self.toggle_pause)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setStyleSheet("""
            QPushButton {
                background-color: #f44336;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #d32f2f;
            }
        """)
        
        self.select_output_btn = QPushButton("Select Output Folder")
        self.select_output_btn.clicked.connect(self.select_output_folder)
        self.select_output_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                border: none;
                padding: 8px 16px;
                font-size: 14px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
        """)
        
        control_layout.addWidget(self.load_video_btn)
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.pause_btn)
        control_layout.addWidget(self.select_output_btn)
        main_layout.addLayout(control_layout)
        
        # Status label
        self.status_label = QLabel("Status: Ready")
        self.status_label.setStyleSheet("""
            QLabel {
                font-weight: bold;
                color: #333;
                font-size: 14px;
            }
        """)
        main_layout.addWidget(self.status_label)
        
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)
        
    def load_default_model(self):
        try:
            model_path = r'D:\train_wagon_image_Extraction--main\train_wagon_image_Extraction--main\main_folder\best (5).pt'
            self.model = YOLO(model_path)
            
            class_file = r"D:\train_wagon_image_Extraction--main\train_wagon_image_Extraction--main\main_folder\coco.txt"
            with open(class_file, "r") as my_file:
                self.class_list = my_file.read().split("\n")
            
            self.status_label.setText("Status: Default model loaded successfully")
        except Exception as e:
            self.status_label.setText(f"Status: Error loading default model - {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to load default model:\n{str(e)}")
    
    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Open Video File", "", 
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_name:
            self.cap = cv2.VideoCapture(file_name)
            if not self.cap.isOpened():
                QMessageBox.critical(self, "Error", "Could not open video file")
                return
            
            self.video_file = file_name
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.progress_bar.setMaximum(self.total_frames)
            self.progress_bar.setValue(0)
            
            self.start_btn.setEnabled(True)
            self.status_label.setText(f"Status: Video loaded - {os.path.basename(file_name)}")
            
            ret, frame = self.cap.read()
            if ret:
                self.display_frame(frame)
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            os.makedirs(self.output_dir, exist_ok=True)
            self.status_label.setText(f"Status: Output folder set to {folder}")
    
    def start_detection(self):
        if not self.cap:
            QMessageBox.warning(self, "Warning", "Please load a video first")
            return
        
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output folder first")
            return
        
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.progress_bar.setValue(0)
        
        self.detection_thread = DetectionThread(
            self.model, self.cap, self.class_list, self.area, self.output_dir
        )
        self.detection_thread.update_signal.connect(self.update_display)
        self.detection_thread.finished_signal.connect(self.detection_finished)
        
        self.start_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.load_video_btn.setEnabled(False)
        self.status_label.setText("Status: Detection running...")
        
        self.detection_thread.start()
    
    def toggle_pause(self):
        if self.detection_thread:
            self.detection_thread.paused = not self.detection_thread.paused
            if self.detection_thread.paused:
                self.pause_btn.setText("Resume")
                self.status_label.setText("Status: Paused")
            else:
                self.pause_btn.setText("Pause")
                self.status_label.setText("Status: Detection running...")
    
    def update_display(self, frame, frame_count):
        self.progress_bar.setValue(frame_count)
        self.display_frame(frame)
    
    def detection_finished(self):
        self.start_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.load_video_btn.setEnabled(True)
        self.status_label.setText("Status: Detection completed")
        
        if self.detection_thread:
            self.detection_thread.stop()
            self.detection_thread.wait()
            self.detection_thread = None
    
    def display_frame(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        
        q_img = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        self.video_label.setPixmap(pixmap.scaled(
            self.video_label.width(), self.video_label.height(),
            Qt.AspectRatioMode.KeepAspectRatio
        ))
    
    def closeEvent(self, event):
        if self.detection_thread and self.detection_thread.isRunning():
            self.detection_thread.stop()
            self.detection_thread.wait()
        
        if self.cap:
            self.cap.release()
        
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = WagonDetectionApp()
    window.show()
    sys.exit(app.exec())