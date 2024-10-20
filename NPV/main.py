import os
import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO  # Import YOLOv8
import datetime
import pytz  # Import pytz for timezone handling

class NepalPlateVisionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Nepal Plate Vision")
        self.master.geometry("600x400")  # Width x Height
        self.master.minsize(600, 400)  # Minimum size
        self.master.configure(bg='lightblue')  # Background color

        self.label = tk.Label(master, text="Nepal Plate Vision", font=("Helvetica", 16), bg='lightblue')
        self.label.pack(pady=20)

        self.time_label = tk.Label(master, font=("Helvetica", 12), bg='lightblue')
        self.time_label.pack(side=tk.TOP, anchor=tk.NE, padx=10, pady=10)  # Align to the top right corner

        self.button_frame = tk.Frame(master, bg='lightblue')
        self.button_frame.pack(pady=10, fill=tk.BOTH, expand=True)

        self.btn_real_time = tk.Button(self.button_frame, text="Real-Time Video Capture", command=self.open_real_time_window,
                                        font=("Helvetica", 12), bg='blue', fg='gold', activebackground='darkblue', activeforeground='gold', height=2)
        self.btn_real_time.pack(fill=tk.X, padx=10, pady=5)

        self.btn_upload = tk.Button(self.button_frame, text="Upload Images/Videos", command=self.upload_file,
                                     font=("Helvetica", 12), bg='blue', fg='gold', activebackground='darkblue', activeforeground='gold', height=2)
        self.btn_upload.pack(fill=tk.X, padx=10, pady=5)

        self.cap = None  # Placeholder for video capture
        self.video_running = False  # Flag to check if video is running
        self.model_plate = YOLO('/Users/dikshantthapa/Desktop/NepalPlateVision/NPV 1.0/YOLO/best_plate.pt')  # Load YOLOv8 model
        
        # Create a directory for detected plates with today's date
        today = datetime.datetime.now().strftime("%Y%m%d")
        self.save_dir = f'/Users/dikshantthapa/Desktop/NepalPlateVision/NPV/detected_plates_{today}'
        os.makedirs(self.save_dir, exist_ok=True)

        self.update_time()  # Start the time update

    def update_time(self):
        # Get the current time in Nepal
        nepal_tz = pytz.timezone('Asia/Kathmandu')
        current_time = datetime.datetime.now(nepal_tz).strftime('%Y-%m-%d %H:%M:%S')
        self.time_label.config(text=current_time)
        self.master.after(1000, self.update_time)  # Update every second

    def open_real_time_window(self):
        if self.video_running:
            messagebox.showinfo("Info", "Real-Time Video Capture is already running.")
            return

        self.real_time_window = tk.Toplevel(self.master)
        self.real_time_window.title("Real-Time Video Capture")
        self.real_time_window.geometry("800x600")
        self.real_time_window.protocol("WM_DELETE_WINDOW", self.stop_video)  # Stop video on close

        self.cap = cv2.VideoCapture(0)
        self.video_running = True
        self.update_frame()

    def update_frame(self):
        if self.video_running and self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                # YOLOv8 model inference
                results = self.model_plate(frame)
                frame = self.plot_boxes(results, frame)  # Plot boxes on the frame

                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(img)

                # Clear previous label (if any) and display the new frame
                for widget in self.real_time_window.pack_slaves():
                    if isinstance(widget, tk.Label):
                        widget.destroy()

                label = tk.Label(self.real_time_window, image=img)
                label.image = img  # Keep a reference
                label.pack()

                # Call this method again after 10 milliseconds
                self.real_time_window.after(10, self.update_frame)
            else:
                self.stop_video()

    def plot_boxes(self, results, frame):
        # Extract bounding boxes and labels from results
        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Get bounding boxes
            confidences = result.boxes.conf.numpy()  # Get confidences
            classes = result.boxes.cls.numpy()  # Get class indices

            for box, conf, cls in zip(boxes, confidences, classes):
                x1, y1, x2, y2 = box
                # Draw bounding boxes in red color
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)  # Red box
                # Display confidence percentage inside the box
                cv2.putText(frame, f'{conf * 100:.1f}%', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

                # Save cropped image if confidence is above a threshold (e.g., 50%)
                if conf > 0.5:
                    self.save_cropped_image(frame, int(x1), int(y1), int(x2), int(y2))

        return frame

    def save_cropped_image(self, frame, x1, y1, x2, y2):
        # Crop the detected license plate from the frame
        cropped_plate = frame[y1:y2, x1:x2]

        # Create a unique filename using timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.save_dir, f'plate_{timestamp}.png')

        # Save the cropped image
        cv2.imwrite(filename, cropped_plate)
        print(f'Saved cropped plate image: {filename}')

    def stop_video(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_running = False
        if hasattr(self, 'real_time_window'):
            self.real_time_window.destroy()  # Close the real-time window

        # Flush the contents of the detected plates folder
        for filename in os.listdir(self.save_dir):
            file_path = os.path.join(self.save_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                print(f'Deleted image: {file_path}')

    def upload_file(self):
        file_path = filedialog.askopenfilename(
            title="Select Image or Video",
            filetypes=[
                ("Image Files", "*.jpg"),
                ("Image Files", "*.jpeg"),
                ("Image Files", "*.png"),
                ("Video Files", "*.mp4"),
                ("All Files", "*.*")  # Allow all file types for testing
            ]
        )
        if file_path:
            print(f"File uploaded: {file_path}")
            # Implement your YOLO model for uploaded files if needed
        else:
            print("No file selected.")

if __name__ == "__main__":
    root = tk.Tk()
    app = NepalPlateVisionApp(root)
    root.mainloop()
