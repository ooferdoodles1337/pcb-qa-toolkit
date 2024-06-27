import cv2
from PIL import Image as PILImage, ImageTk
from tkinter import *


class PCBQualityAssuranceApp:
    def __init__(self, root):
        print("Starting App")
        self.root = root
        self.root.title("PCB Quality Assurance Toolkit")
        self.window_width = 1200
        self.window_height = 900
        self.root.geometry(f"{self.window_width}x{self.window_height}")
        self.root.config(bg="skyblue")
        root.minsize(800, 600)
        root.maxsize(1920, 1080)

        self.reference_image = None  # Variable to store the reference image

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        print("Initialized Webcam")

        # Set up the GUI
        self.setup_gui()
        print("GUI Setup Complete")

        # Start video capture
        self.root.after(200, self.capture_webcam_stream)

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
        self.left_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nswe")
        self.left_frame.grid_propagate(False)

        self.right_frame = Frame(
            self.root,
            bg="grey",
            width=int(self.window_width * 0.65),
            height=self.window_height,
        )
        self.right_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nswe")
        self.right_frame.grid_propagate(False)

        # Left Frame
        self.input_display = Label(self.left_frame)
        self.input_display.grid(row=0, column=0, sticky="nswe")

        self.reference_display = Label(self.left_frame)
        self.reference_display.grid(row=1, column=0, sticky="nswe")

        self.button_frame = Frame(self.left_frame)
        self.button_frame.grid(row=2, column=0, sticky="nswe")

        self.save_reference_button = Button(
            self.button_frame, text="Take Reference", command=self.take_reference
        )
        self.save_reference_button.grid(row=0, column=0, sticky="ew")

        self.clear_reference_button = Button(
            self.button_frame,
            text="Clear Reference",
            command=self.clear_reference,
        )
        self.clear_reference_button.grid(row=0, column=1, sticky="ew")

        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        self.mode_frame = Frame(self.left_frame)
        self.mode_frame.grid(row=3, column=0, sticky="nswe")

        self.mode = StringVar(value="overlay")

        self.overlay_radio = Radiobutton(
            self.mode_frame, text="Overlay", variable=self.mode, value="overlay"
        )
        self.overlay_radio.grid(row=0, column=0, padx=5, sticky="w")

        self.difference_radio = Radiobutton(
            self.mode_frame, text="Difference", variable=self.mode, value="difference"
        )
        self.difference_radio.grid(row=1, column=0, padx=5, sticky="w")

        # Right Frame
        self.output = Label(self.right_frame, width=self.left_frame.winfo_width())
        self.output.grid(row=0, column=0, sticky="nswe")

    def capture_webcam_stream(self):
        ret, frame = self.cap.read()
        if ret:
            # Convert the captured frame to RGBA
            opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            captured_image = PILImage.fromarray(opencv_image)

            # Get the dimensions of the left frame
            left_frame_width = self.left_frame.winfo_width() - 5

            # Calculate the height maintaining the 16:9 aspect ratio
            left_frame_height = int(left_frame_width * 9 / 16)

            # Resize the captured image to fit within the left frame dimensions
            resized_image = captured_image.resize((left_frame_width, left_frame_height))
            photo_image = ImageTk.PhotoImage(image=resized_image)

            # Update the input display with the resized image
            self.input_display.photo_image = photo_image
            self.input_display.configure(image=photo_image)

            # Schedule the next frame capture
            self.input_display.after(10, self.capture_webcam_stream)

            if self.reference_image is None:
                self.reference_display.photo_image = photo_image
                self.reference_display.configure(image=photo_image)

    def take_reference(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.reference_image = frame

    def clear_reference(self):
        self.reference_image = None


if __name__ == "__main__":
    root = Tk()
    app = PCBQualityAssuranceApp(root)
    root.mainloop()
