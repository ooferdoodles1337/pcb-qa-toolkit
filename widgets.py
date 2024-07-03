import tkinter as tk
import numpy as np
from PIL import Image, ImageTk


class PanZoomCanvas(tk.Frame):
    def __init__(self, master):
        super().__init__(master)
        self.pil_image = None  # Image data to be displayed
        self.zoom_cycle = 0
        self.create_widget()  # Create canvas

        # Initial affine transformation matrix
        self.reset_transform()

    def create_widget(self):
        # Canvas
        self.canvas = tk.Canvas(self.master, background="black")
        self.canvas.pack(
            fill=tk.BOTH, expand=True, padx=5, pady=(5,0)
        )  # Fill the window and expand with it

        # Controls
        self.canvas.bind("<Button-1>", self.mouse_down_left)  # MouseDown
        self.canvas.bind("<B1-Motion>", self.mouse_move_left)  # MouseDrag
        self.canvas.bind(
            "<Double-Button-1>", self.mouse_double_click_left
        )  # MouseDoubleClick
        self.canvas.bind("<MouseWheel>", self.mouse_wheel)  # MouseWheel

    def set_image(self, image):
        # PIL.Image
        self.pil_image = image
        # Preserve current zoom level
        current_scale = self.scale_factor
        current_offset_x = self.offset_x
        current_offset_y = self.offset_y

        # Set the affine transformation matrix to display the entire image.
        self.zoom_fit(self.pil_image.width, self.pil_image.height)

        # Restore previous zoom level
        self.scale_factor = current_scale
        self.offset_x = current_offset_x
        self.offset_y = current_offset_y

        # Redraw the image on the canvas
        self.redraw_image()

    def remove_image(self):
        self.pil_image = None
        self.canvas.delete("all")

    # -------------------------------------------------------------------------------
    # Mouse events
    # -------------------------------------------------------------------------------
    def mouse_down_left(self, event):
        self.__old_event = event

    def mouse_move_left(self, event):
        if self.pil_image is None:
            return

        self.translate(event.x - self.__old_event.x, event.y - self.__old_event.y)
        self.redraw_image()
        self.__old_event = event

    def mouse_double_click_left(self, event):
        if self.pil_image is None:
            return
        self.zoom_fit(self.pil_image.width, self.pil_image.height)
        self.redraw_image()

    def mouse_wheel(self, event):
        if self.pil_image is None:
            return

        if event.delta < 0:
            if self.zoom_cycle <= -5:
                return
            self.scale_at(0.8, event.x, event.y)
            self.zoom_cycle -= 1

        else:
            if self.zoom_cycle >= 9:
                return
            self.scale_at(1.25, event.x, event.y)
            self.zoom_cycle += 1

        self.redraw_image()  # Refresh

    # -------------------------------------------------------------------------------
    # Affine Transformation for Image Display
    # -------------------------------------------------------------------------------

    def reset_transform(self):
        self.mat_affine = np.eye(3)  # 3x3 identity matrix
        self.scale_factor = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def translate(self, offset_x, offset_y, zoom=False):
        mat = np.eye(3)  # 3x3 identity matrix
        mat[0, 2] = float(offset_x)
        mat[1, 2] = float(offset_y)
        self.offset_x += offset_x
        self.offset_y += offset_y
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        scale = self.scale_factor
        max_y = scale * 3072
        max_x = scale * 4096
        self.mat_affine = np.dot(mat, self.mat_affine)

        if not zoom:
            if abs(self.offset_x) > abs(max_x - canvas_width):
                self.offset_x = -(max_x - canvas_width)
            if abs(self.offset_y) > abs(max_y - canvas_height):
                self.offset_y = -(max_y - canvas_height)

        if self.offset_x > 0.0:
            self.offset_x = 0.0
        if self.offset_y > 0.0:
            self.offset_y = 0.0

    def scale(self, scale_factor):
        mat = np.eye(3)  # 3x3 identity matrix

        mat[0, 0] = scale_factor
        mat[1, 1] = scale_factor
        self.scale_factor *= scale_factor
        self.mat_affine = np.dot(mat, self.mat_affine)

    def scale_at(self, scale_factor, cx, cy):
        self.translate(-cx, -cy, True)
        self.scale(scale_factor)
        self.translate(cx, cy)

    def zoom_fit(self, image_width, image_height):
        self.master.update()
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        if image_width * image_height <= 0 or canvas_width * canvas_height <= 0:
            return

        # self.reset_transform()

        scale = 1.0
        offsetx = 0.0
        offsety = 0.0

        # if (canvas_width * image_height) > (image_width * canvas_height):
        #     scale = canvas_height / image_height
        #     offsetx = (canvas_width - image_width * scale) / 2
        # else:s
        #     scale = canvas_width / image_width
        #     offsety = (canvas_height - image_height * scale) / 2

        self.scale_factor = scale
        self.offset_x = offsetx
        self.offset_y = offsety
        self.scale(scale)
        self.translate(offsetx, offsety)

    def to_image_point(self, x, y):
        if self.pil_image is None:
            return []
        mat_inv = np.linalg.inv(self.mat_affine)
        image_point = np.dot(mat_inv, (x, y, 1.0))
        if (
            image_point[0] < 0
            or image_point[1] < 0
            or image_point[0] > self.pil_image.width
            or image_point[1] > self.pil_image.height
        ):
            return []

        return image_point

    # -------------------------------------------------------------------------------
    # Drawing
    # -------------------------------------------------------------------------------

    def draw_image(self):
        if self.pil_image is None:
            return

        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()

        mat_inv = np.linalg.inv(self.mat_affine)
        affine_inv = (
            mat_inv[0, 0],
            mat_inv[0, 1],
            mat_inv[0, 2],
            mat_inv[1, 0],
            mat_inv[1, 1],
            mat_inv[1, 2],
        )

        dst = self.pil_image.transform(
            (canvas_width, canvas_height), Image.AFFINE, affine_inv, Image.NEAREST
        )

        im = ImageTk.PhotoImage(image=dst)

        self.canvas.create_image(0, 0, anchor="nw", image=im)
        self.image = im

    def redraw_image(self):
        if self.pil_image is None:
            return
        self.draw_image()
