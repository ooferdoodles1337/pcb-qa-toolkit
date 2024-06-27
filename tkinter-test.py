import cv2
from PIL import Image as PILImage, ImageTk
from tkinter import *
from tkinter import filedialog
import threading


class PCBQualityAssuranceApp:
    def __init__(self, root):
        print("Starting App")
        self.root = root
        self.root.title("PCB Quality Assurance Toolkit")
        self.window_width = 1600
        self.window_height = 900
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.config(bg="skyblue")
        root.minsize(800, 600)

        self.reference_image = None  # Variable to store the reference image
        self.current_frame = None  # Variable to store the current frame

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Changed to use default camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print("Initialized Webcam")

        # Set up the GUI
        self.setup_gui()
        print("GUI Setup Complete")

        # Start video capture
        self.capture_thread = threading.Thread(target=self.capture_webcam)
        self.capture_thread.daemon = True
        self.capture_thread.start()

        # Release the video capture when the app closes
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def setup_gui(self):
        # Configure grid layout
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Create frames
        self.left_frame = Frame(
            self.root,
            bg="grey",
            width=int(self.window_width * 0.35),
            height=self.window_height,
        )
        self.left_frame.grid(row=0, column=0, padx=5, pady=10, sticky="nswe")
        self.left_frame.grid_propagate(False)

        self.right_frame = Frame(
            self.root,
            bg="grey",
            width=int(self.window_width * 0.65),
            height=self.window_height,
        )
        self.right_frame.grid(row=0, column=1, padx=5, pady=10, sticky="nswe")
        self.right_frame.grid_propagate(False)

        # Left Frame
        self.input_display = Label(self.left_frame)
        self.input_display.grid(row=0, column=0, sticky="nswe")

        self.reference_display = Label(self.left_frame)
        self.reference_display.grid(row=1, column=0, sticky="nswe")

        self.button_frame = Frame(self.left_frame)
        self.button_frame.grid(row=2, column=0, sticky="nswe")

        self.capture_reference_button = Button(
            self.button_frame, text="Capture Reference", command=self.capture_reference
        )
        self.capture_reference_button.grid(row=0, column=0, sticky="ew")

        self.clear_reference_button = Button(
            self.button_frame,
            text="Clear Reference",
            command=self.clear_reference,
        )
        self.clear_reference_button.grid(row=0, column=1, sticky="ew")

        self.upload_file_button = Button(
            self.button_frame,
            text="Upload Reference",
            command=self.upload_reference,
        )
        self.upload_file_button.grid(row=0, column=2, sticky="ew")

        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)
        self.button_frame.grid_columnconfigure(2, weight=1)

        self.mode_frame = Frame(self.left_frame)
        self.mode_frame.grid(row=3, column=0, sticky="nswe")

        self.mode = StringVar(value="none")

        self.none_radio = Radiobutton(
            self.mode_frame, text="None", variable=self.mode, value="none"
        )
        self.none_radio.grid(row=0, column=0, padx=5, sticky="w")
        self.overlay_radio = Radiobutton(
            self.mode_frame, text="Overlay", variable=self.mode, value="overlay"
        )
        self.overlay_radio.grid(row=1, column=0, padx=5, sticky="w")

        self.difference_radio = Radiobutton(
            self.mode_frame, text="Difference", variable=self.mode, value="difference"
        )
        self.difference_radio.grid(row=2, column=0, padx=5, sticky="w")

        # Right Frame
        self.output_display = Label(
            self.right_frame,
        )
        self.output_display.grid(row=0, column=0, sticky="nswe")

        # Schedule the initial display update
        self.root.after(10, self.update_display)

    def capture_webcam(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame

    def update_display(self):
        if self.current_frame is not None:
            try:
                self.update_input_display()
                self.update_reference_display()
                self.update_output_display()
            except Exception as e:
                print(f"Error updating display: {e}")

        # Schedule the next display update
        self.root.after(10, self.update_display)

    def update_input_display(self):
        target_size = (
            self.left_frame.winfo_width() - 5,
            int((self.left_frame.winfo_width() - 5) * 9 / 16),
        )
        converted_frame = self.convert_frame_to_photoimage(
            self.current_frame, target_size
        )
        self.input_display.photo_image = converted_frame
        self.input_display.configure(image=converted_frame)

    def update_reference_display(self):
        target_size = (
            self.left_frame.winfo_width() - 5,
            int((self.left_frame.winfo_width() - 5) * 9 / 16),
        )
        if self.reference_image is None:
            converted_frame = self.convert_frame_to_photoimage(
                self.current_frame, target_size
            )
            self.reference_display.photo_image = converted_frame
            self.reference_display.configure(image=converted_frame)
            return

        converted_frame = self.convert_frame_to_photoimage(
            self.reference_image, target_size
        )
        self.reference_display.photo_image = converted_frame
        self.reference_display.configure(image=converted_frame)

    def update_output_display(self):
        target_size = (
            self.right_frame.winfo_width() - 5,
            int((self.right_frame.winfo_width() - 5) * 9 / 16),
        )
        if self.reference_image is None:
            self.output_display.configure(image=None)
            self.output_display.photo_image = None
            return

        frame = self.current_frame
        try:
            if self.mode.get() == "overlay":
                output = self.process_overlay(frame, self.reference_image)
            elif self.mode.get() == "difference":
                output = self.process_difference(frame, self.reference_image)
            else:
                output = frame
        except Exception as e:
            print(f"Error processing image: {e}")

        converted_frame = self.convert_frame_to_photoimage(output, target_size)
        self.output_display.photo_image = converted_frame
        self.output_display.configure(image=converted_frame)

    def process_overlay(self, input, reference, transparency=0.5):
        return cv2.addWeighted(input, transparency, reference, transparency, 0)

    def process_difference(self, input, reference):
        return 255 - cv2.absdiff(input, reference)

    def capture_reference(self):
        if self.current_frame is not None:
            self.reference_image = self.current_frame
            print("Reference image captured")

    def clear_reference(self):
        self.reference_image = None

    def upload_reference(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp;*.tiff")]
        )
        if file_path:
            self.reference_image = cv2.imread(file_path)
            print("Reference image uploaded")
            self.update_reference_display()

    def convert_frame_to_photoimage(self, frame, target_size):
        resized_frame = cv2.resize(frame, target_size)
        converted_frame = ImageTk.PhotoImage(
            image=PILImage.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
        )
        return converted_frame

    def on_closing(self):
        self.cap.release()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = PCBQualityAssuranceApp(root)
    root.mainloop()
