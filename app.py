import os
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
        self.root.minsize(1000, 600)
        # root.resizable(False, False)
        self.root.config()

        self.reference_image = None
        self.current_frame = None

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
        self.root.grid_columnconfigure(1, weight=2)

        # Create frames
        self.left_frame = Frame(self.root, bg="azure2")
        self.left_frame.grid(row=0, column=0, sticky="nswe")
        self.left_frame.pack_propagate(False)

        self.right_frame = Frame(self.root, bg="azure3")
        self.right_frame.grid(row=0, column=1, sticky="nswe")
        self.right_frame.pack_propagate(False)

        # Left Frame

        ## Input Frame
        self.input_label = Label(self.left_frame, text="Input Image", bg="azure1")
        self.input_label.pack(fill="x", padx=5, pady=(5, 0))
        self.input_canvas = Canvas(self.left_frame)
        self.input_canvas.pack(fill="x", padx=5, pady=(5, 0))

        self.reference_label = Label(
            self.left_frame, text="Reference Image", bg="azure1"
        )
        self.reference_label.pack(fill="x", padx=5, pady=(5, 0))
        self.reference_canvas = Canvas(self.left_frame)
        self.reference_canvas.pack(fill="x", padx=5, pady=(5, 0))

        ## Button Frame
        self.button_frame = Frame(self.left_frame)
        self.button_frame.pack(fill="x", padx=5, pady=(5, 0))

        self.capture_reference_button = Button(
            self.button_frame,
            text="Capture Reference",
            command=self.capture_reference,
            bg="azure1",
        )
        self.capture_reference_button.pack(side=LEFT, expand=True, fill="both")

        self.clear_reference_button = Button(
            self.button_frame,
            text="Clear Reference",
            command=self.clear_reference,
            bg="azure1",
        )
        self.clear_reference_button.pack(side=LEFT, expand=True, fill="both")

        self.upload_file_button = Button(
            self.button_frame,
            text="Upload Reference",
            command=self.upload_reference,
            bg="azure1",
        )
        self.upload_file_button.pack(side=LEFT, expand=True, fill="both")

        ## Mode Frame
        self.mode_label = Label(self.left_frame, text="Output Mode", bg="azure1")
        self.mode_label.pack(fill="x", padx=5, pady=(5, 0))
        self.mode = StringVar(value="none")
        modes = (("None", "none"), ("Overlay", "overlay"), ("Difference", "difference"))

        for mode_text, mode_value in modes:
            r = Radiobutton(
                self.left_frame, text=mode_text, value=mode_value, variable=self.mode, bg="azure1"
            )
            r.pack(fill="x", padx=5)

        # Right Frame
        self.output_label = Label(self.right_frame, text="Output Image", bg="azure2").pack(
            fill="x", padx=5, pady=(5, 0)
        )
        self.output_canvas = Canvas(self.right_frame, bg="azure2")
        self.output_canvas.pack(fill="both", padx=5, pady=5)

        self.root.bind("<Configure>", self.resize_all_canvases)
        self.resize_all_canvases(None)

        # Schedule the initial display update
        self.root.after(10, self.update_display)

    def capture_webcam(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame

    def resize_all_canvases(self, event):
        # Calculate canvas dimensions with 16:9 aspect ratio
        def resize_canvas(canvas, frame):
            frame_width = frame.winfo_width() - 10  # Adjust for padding
            canvas_width = frame_width
            canvas_height = int(canvas_width * 9 / 16)
            canvas.config(width=canvas_width, height=canvas_height)

        # Resize each canvas
        resize_canvas(self.output_canvas, self.right_frame)
        resize_canvas(self.input_canvas, self.left_frame)
        resize_canvas(self.reference_canvas, self.left_frame)

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
        canvas_width = self.input_canvas.winfo_width()
        canvas_height = self.input_canvas.winfo_height()
        target_size = (canvas_width, canvas_height)
        converted_frame = self.convert_frame_to_photoimage(
            self.current_frame, target_size
        )
        self.input_canvas.create_image(0, 0, anchor=NW, image=converted_frame)
        self.input_canvas.image = converted_frame

    def update_reference_display(self):
        canvas_width = self.reference_canvas.winfo_width()
        canvas_height = self.reference_canvas.winfo_height()
        target_size = (canvas_width, canvas_height)
        if self.reference_image is None:
            converted_frame = self.convert_frame_to_photoimage(
                self.current_frame, target_size
            )
            self.reference_canvas.create_image(0, 0, anchor=NW, image=converted_frame)
            self.reference_canvas.image = converted_frame
            return

        converted_frame = self.convert_frame_to_photoimage(
            self.reference_image, target_size
        )
        self.reference_canvas.create_image(0, 0, anchor=NW, image=converted_frame)
        self.reference_canvas.image = converted_frame

    def update_output_display(self):
        canvas_width = self.output_canvas.winfo_width()
        canvas_height = self.output_canvas.winfo_height()
        target_size = (canvas_width, canvas_height)
        if self.reference_image is None:
            self.output_canvas.create_image(0, 0, anchor=NW, image=None)
            self.output_canvas.image = None
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
        self.output_canvas.create_image(0, 0, anchor=NW, image=converted_frame)
        self.output_canvas.image = converted_frame

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
