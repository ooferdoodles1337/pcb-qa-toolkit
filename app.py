import os
import time
from datetime import datetime
import queue
import threading
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from PIL import Image, ImageTk
from skimage.exposure import match_histograms
from skimage.metrics import structural_similarity

from changechip import pipeline
from widgets import PanZoomCanvas


class PCBQualityAssuranceApp:
    def __init__(self, root, camera_id, camera_frame_width, camera_frame_height):
        print("Starting App")
        self.root = root
        self.root.title("PCB Quality Assurance Toolkit")
        self.window_width = 1600
        self.window_height = 900
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.minsize(1400, 800)
        self.root.config()

        self.font = "Segoe UI"
        self.fontsize = 12

        self.reference_image = None
        self.current_frame = None
        self.processed_frame = None
        self.frame_queue = queue.Queue(maxsize=1)  # Queue to hold frames for processing
        self.flicker_state = True

        self.cap = cv2.VideoCapture(
            camera_id, cv2.CAP_DSHOW
        )  # Changed to use default camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_frame_height)
        print("Initialized Webcam")

        # Set up the GUI
        self.setup_gui()
        print("GUI Setup Complete")

        # Start video capture
        self.capture_thread = threading.Thread(target=self.capture_webcam)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Start the image processing thread
        self.processing_thread = threading.Thread(target=self.process_output)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        # Release the video capture when the app closes
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=3)

        # Create left and right frames
        self.left_frame = tk.Frame(self.root, bg="azure2")
        self.left_frame.grid(row=0, column=0, sticky="nswe")
        self.left_frame.pack_propagate(False)

        self.right_frame = tk.Frame(self.root, bg="azure2")
        self.right_frame.grid(row=0, column=1, sticky="nswe")
        self.right_frame.pack_propagate(False)

        # Left Frame Widgets
        self.create_left_frame_widgets()

        # Right Frame Widgets
        self.create_right_frame_widgets()

        # Bind resize event and schedule display update
        self.root.bind("<Configure>", self.resize_all_canvases)
        self.resize_all_canvases(None)
        self.root.after(10, self.update_display)

    def create_left_frame_widgets(self):
        # Cameras Section
        self.input_label = tk.Label(
            self.left_frame,
            text="Input Image",
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        self.input_label.pack(fill="x", padx=5, pady=(5, 0))
        self.input_canvas = tk.Canvas(self.left_frame)
        self.input_canvas.pack(fill="x", padx=5, pady=(5, 0))

        self.reference_label = tk.Label(
            self.left_frame,
            text="Reference Image",
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        self.reference_label.pack(fill="x", padx=5, pady=(5, 0))
        self.reference_canvas = tk.Canvas(self.left_frame)
        self.reference_canvas.pack(fill="x", padx=5, pady=(5, 0))

        # Buttons Section
        self.button_frame = tk.Frame(self.left_frame)
        self.button_frame.pack(fill="x", padx=5, pady=(5, 0))

        self.setup_button(self.button_frame, "Capture", self.capture_reference)
        self.setup_button(self.button_frame, "Clear", self.clear_reference)
        self.setup_button(self.button_frame, "Upload", self.upload_reference)

        # Mode Radio Buttons
        self.mode_label = tk.Label(
            self.left_frame,
            text="Output Mode",
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        self.mode_label.pack(fill="x", padx=5, pady=(5, 0))
        self.setup_radio_buttons()

        # Preprocessing Checkboxes
        self.preprocess_label = tk.Label(
            self.left_frame,
            text="Preprocessing Options",
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        self.preprocess_label.pack(fill="x", padx=5, pady=(5, 0))
        self.setup_checkboxes()

    def setup_button(self, parent, text, command):
        button = tk.Button(
            parent,
            text=text,
            command=command,
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        button.pack(side=tk.LEFT, expand=True, fill="both")

    def setup_radio_buttons(self):
        self.mode = tk.StringVar(value="none")
        modes = (
            ("None", "none"),
            ("Overlay", "overlay"),
            ("Difference", "difference"),
            ("SSIM", "ssim"),
            ("Flicker", "flicker"),
            ("ChangeChip", "changechip"),
        )

        for mode_text, mode_value in modes:
            r = tk.Radiobutton(
                self.left_frame,
                text=mode_text,
                value=mode_value,
                variable=self.mode,
                bg="azure1",
                font=(self.font, self.fontsize),
            )
            r.pack(fill="x", padx=5)

    def setup_checkboxes(self):
        self.homography_var = tk.IntVar()
        self.histogram_var = tk.IntVar()

        self.setup_checkbox("Align Images", self.homography_var)
        self.setup_checkbox("Match Colors", self.histogram_var)

    def setup_checkbox(self, text, variable):
        checkbox = tk.Checkbutton(
            self.left_frame,
            text=text,
            variable=variable,
            onvalue=1,
            offvalue=0,
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        checkbox.pack(fill="x", padx=5, pady=(0, 0))

    def create_right_frame_widgets(self):
        # Output Image Section
        self.output_label = tk.Label(
            self.right_frame,
            text="Output Image",
            bg="azure1",
            font=(self.font, self.fontsize),
        )
        self.output_label.pack(fill="x", padx=5, pady=(5, 0))
        self.output_canvas = PanZoomCanvas(master=self.right_frame)
        self.defect_capture_button = tk.Button(
            self.right_frame,
            text="Capture Defect",
            bg="azure1",
            command=self.capture_defect,
            font=(self.font, self.fontsize),
        )
        self.defect_capture_button.pack(fill="x", padx=5, pady=(5, 5))

    def capture_webcam(self):
        """
        Continuously captures frames from the webcam. The latest frame is always stored in `self.current_frame`.
        Frames are also placed in a queue for further processing, ensuring that only the most recent frame is kept
        if the queue becomes full.
        """
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                # Only put the latest frame into the queue, discard the old one if the queue is full
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)
                else:
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    self.frame_queue.put(frame)

    def resize_all_canvases(self, event):
        """
        Resizes all canvases in the GUI to maintain a 16:9 aspect ratio whenever the window is resized.

        This function calculates the new dimensions for each canvas based on the width of the frame it is
        contained in, adjusted for a padding of 10 pixels, and then updates the canvas size accordingly.

        Args:
            event (tk.Event): The event object containing information about the resize event.
        """

        def resize_canvas(canvas, frame):
            # Calculate canvas dimensions with 16:9 aspect ratio
            frame_width = frame.winfo_width() - 10  # Adjust for padding
            canvas_width = frame_width
            canvas_height = int(canvas_width * 9 / 16)
            canvas.config(width=canvas_width, height=canvas_height)

        # Resize each canvas
        resize_canvas(self.input_canvas, self.left_frame)
        resize_canvas(self.reference_canvas, self.left_frame)

    # ------------------------- Display Update Functions ------------------------- #

    def update_display(self):
        """
        Continuously updates the display of the input, reference, and output canvases. This function schedules
        the next update after a short delay to ensure a smooth and responsive interface.

        If the current frame is not None, it attempts to update the input, reference, and output displays.
        In case of any exceptions during the update process, the error is printed to the console.

        This function is designed to be called repeatedly using `root.after`, creating a loop that updates
        the display every 10 milliseconds.
        """
        if self.current_frame is not None:
            try:
                self.update_input_display()
                self.update_reference_display()
                self.update_output_display()
            except Exception as e:
                print(f"Error updating display: {e}")

        # Schedule the next display update
        self.root.after(10, self.update_display)

    def update_canvas_display(self, canvas, frame):
        """
        Updates a given canvas with the provided frame. The frame is converted to the appropriate format and
        resized to fit the canvas dimensions before being displayed.

        Args:
            canvas (tk.Canvas): The canvas widget to update.
            frame (np.array): The frame to display on the canvas, expected in numpy array format.

        This function resizes the frame to match the canvas size and converts it to a format suitable for
        display in the Tkinter canvas. It then creates an image on the canvas at the top-left corner (0, 0).
        The converted frame is stored in the canvas to prevent it from being garbage collected.
        """
        canvas_size = (canvas.winfo_width(), canvas.winfo_height())
        converted_frame = self.convert_frame_format(frame, canvas_size)
        canvas.create_image(0, 0, anchor=tk.NW, image=converted_frame)
        canvas.image = converted_frame

    def update_input_display(self):
        self.update_canvas_display(self.input_canvas, self.current_frame)

    def update_reference_display(self):
        image_source = (
            self.current_frame if self.reference_image is None else self.reference_image
        )
        self.update_canvas_display(self.reference_canvas, image_source)

    def update_output_display(self):
        if self.reference_image is not None:
            self.output_canvas.set_image(
                self.convert_frame_format(self.processed_frame, convert_to_tk=False)
            )
        else:
            self.output_canvas.remove_image()

    # ------------------------- Image Processing Functions ------------------------- #

    def process_output(self):
        """
        Continuously processes frames from the frame queue. If no reference image is available,
        it skips processing and waits briefly to avoid high CPU usage.
        """
        while True:
            frame = self.frame_queue.get()  # Wait for a frame to be available
            if frame is not None and self.reference_image is not None:
                try:
                    self.processed_frame = self.process_current_frame(frame)
                except Exception as e:
                    print(f"Error processing output frame: {e}")

    def process_current_frame(self, frame):
        """
        Processes the current frame based on selected options and mode.

        This function applies color histogram matching and homography transformation
        if they are active, then processes the frame according to the selected mode.

        Args:
            frame (np.array): The current frame to be processed.

        Returns:
            np.array: The processed frame based on the selected mode.
        """
        histogram_active = self.histogram_var.get() == 1
        homography_active = self.homography_var.get() == 1
        mode = self.mode.get()

        if histogram_active:
            frame = self.match_colors(self.reference_image, frame)

        if homography_active:
            frame = self.apply_homography(self.reference_image, frame)

        mode_functions = {
            "overlay": self.process_overlay,
            "difference": self.process_difference,
            "ssim": self.process_ssim,
            "flicker": self.process_flicker,
            "changechip": self.process_changechip,
        }

        output = mode_functions.get(mode, lambda ref, frm: frm)(
            self.reference_image, frame
        )
        return output

    def process_overlay(self, reference_image, current_frame, alpha=0.5):
        return cv2.addWeighted(reference_image, alpha, current_frame, 1 - alpha, 0)

    def process_difference(
        self,
        reference_image,
        current_frame,
        min_contour_area=300,
        alpha=0.5,
        min_ratio=0.005,
        max_ratio=0.5,
    ):
        diff = cv2.absdiff(reference_image, current_frame)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

        # Threshold the grayscale difference image to create a binary mask
        _, binary_mask = cv2.threshold(gray_diff, 40, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Create a copy of the current frame to draw the contours
        red_diff = current_frame.copy()

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_contour_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue  # Avoid division by zero

            ratio = area / (perimeter**2)
            if min_ratio <= ratio <= max_ratio:
                cv2.drawContours(
                    red_diff, [contour], -1, (0, 0, 255), thickness=cv2.FILLED
                )

        output_frame = cv2.addWeighted(current_frame, 1 - alpha, red_diff, alpha, 0)
        return output_frame

    def process_ssim(self, reference_image, current_frame):
        gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        _, diff = structural_similarity(gray_reference, gray_frame, full=True)
        diff = (diff * 255).astype("uint8")
        diff_color = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
        return diff_color

    def process_flicker(self, reference_image, frame, delay=0.2):
        time.sleep(delay)
        self.flicker_state = not self.flicker_state
        return reference_image if self.flicker_state else frame

    def process_changechip(self, reference_image, frame):
        output = pipeline((frame, reference_image), resize_factor=0.5)
        output = cv2.resize(output, None, fx=2, fy=2)
        return output

    # ------------------------- Feature-Based Homography ------------------------- #

    def apply_homography(self, reference_image, current_frame):
        # Convert images to grayscale
        gray_reference = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

        # Detect ORB keypoints and descriptors
        orb = cv2.ORB_create()
        keypoints1, descriptors1 = orb.detectAndCompute(gray_reference, None)
        keypoints2, descriptors2 = orb.detectAndCompute(gray_frame, None)

        # Match descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(descriptors1, descriptors2)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)

        for i, match in enumerate(matches):
            points1[i, :] = keypoints1[match.queryIdx].pt
            points2[i, :] = keypoints2[match.trainIdx].pt

        # Find homography
        h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

        # Use homography to warp current frame
        height, width, channels = reference_image.shape
        aligned_frame = cv2.warpPerspective(current_frame, h, (width, height))

        return aligned_frame

    def match_colors(self, reference_image, current_frame):
        return match_histograms(current_frame, reference_image, channel_axis=-1)

    # ----------------------------- Button Functions ----------------------------- #

    def capture_reference(self):
        try:
            # Ensure the references directory exists
            dir = os.path.join("images", "reference_images")
            os.makedirs(dir, exist_ok=True)

            # Copy the current frame to use as the reference image
            self.reference_image = self.current_frame.copy()

            # Generate a timestamped filename for the reference image
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            reference_image_filename = f"reference_image_{current_time_str}.png"
            reference_image_path = os.path.join(dir, reference_image_filename)

            # Save the reference image to the specified path
            cv2.imwrite(reference_image_path, self.reference_image)
            print(f"Reference image saved at {reference_image_path}")

        except Exception as e:
            print(f"An error occurred while capturing the reference image: {e}")

    def clear_reference(self):
        self.reference_image = None
        print("Reference image cleared")

    def upload_reference(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.reference_image = cv2.imread(file_path)

    def capture_defect(self):
        try:
            # Ensure the defects directory exists
            dir = os.path.join("images", "defect_images")
            os.makedirs(dir, exist_ok=True)

            # Copy the current frame to use as the defect image
            defect_image = self.current_frame.copy()

            # Generate a timestamped filename for the defect image
            current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            defect_image_filename = f"defect_image_{current_time_str}.png"
            defect_image_path = os.path.join(dir, defect_image_filename)

            # Save the defect image to the specified path
            cv2.imwrite(defect_image_path, defect_image)
            print(f"Defect image saved at {defect_image_path}")

        except Exception as e:
            print(f"An error occurred while capturing the defect image: {e}")

    # ------------------------------ Other Functions ----------------------------- #

    def convert_frame_format(
        self, frame, target_size=None, convert_to_tk=True, rotation_angle=180
    ):
        """
        Converts a frame from OpenCV format to PIL format and optionally to a Tkinter-compatible format,
        with an optional rotation applied.

        Args:
            frame (np.array): The input frame in OpenCV format (BGR color space).
            target_size (tuple, optional): The target size (width, height) for resizing the frame. Defaults to None.
            convert_to_tk (bool, optional): Whether to convert the PIL image to a Tkinter PhotoImage. Defaults to True.
            rotation_angle (int, optional): The angle by which to rotate the PIL image. Defaults to 180 degrees.

        Returns:
            ImageTk.PhotoImage or PIL.Image: If convert_to_tk is True, returns a Tkinter PhotoImage object.
                                            Otherwise, returns a PIL Image object.

        Notes:
            - If target_size is specified, the frame is resized to the target dimensions using OpenCV's resize function.
            - Converts the frame from BGR to RGB color space using OpenCV and then creates a PIL Image from the array.
            - Rotates the PIL image by the specified rotation_angle before returning to adjust for image orientation.
        """
        # Resize the frame if target_size is specified
        resized_frame = cv2.resize(frame, target_size) if target_size else frame

        # Convert BGR to RGB and create a PIL Image
        pil_image = Image.fromarray(
            cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        ).rotate(rotation_angle)

        # Convert PIL Image to Tkinter PhotoImage if requested
        return ImageTk.PhotoImage(pil_image) if convert_to_tk else pil_image

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = PCBQualityAssuranceApp(
        root, camera_id=1, camera_frame_width=2560, camera_frame_height=1440
    )
    root.mainloop()
