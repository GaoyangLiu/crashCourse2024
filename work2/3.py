import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
import numpy as np

class CrackRecognitionGUI:
    def __init__(self, master):
        self.master = master
        master.title("Crack Recognition")

        self.left_frame = tk.Frame(master)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.right_frame = tk.Frame(master)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.label_left = tk.Label(self.left_frame)
        self.label_left.pack()

        self.label_right = tk.Label(self.right_frame)
        self.label_right.pack()

        self.load_button = tk.Button(master, text="Load Image", command=self.load_image)
        self.load_button.pack(pady=10)

        self.run_button = tk.Button(master, text="Run", command=self.detect_cracks)
        self.run_button.pack(pady=10)

        self.image_path = None
        self.image = None

    def load_image(self):
        self.image_path = filedialog.askopenfilename()
        if self.image_path:
            self.image = cv2.imread(self.image_path)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
            self.image_tk = self._cv_image_to_tk(self.image)
            self.label_left.config(image=self.image_tk)
            self.label_left.image = self.image_tk

    def detect_cracks(self):
        if self.image is not None:
            # Simulate crack detection by finding edges
            edges = cv2.Canny(self.image, 100, 200)  # Using Canny Edge Detection
            color_edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            edges_tk = self._cv_image_to_tk(color_edges)
            self.label_right.config(image=edges_tk)
            self.label_right.image = edges_tk

    @staticmethod
    def _cv_image_to_tk(cv_image):
        pil_image = Image.fromarray(cv_image)
        tk_image = ImageTk.PhotoImage(image=pil_image)
        return tk_image

if __name__ == "__main__":
    root = tk.Tk()
    app = CrackRecognitionGUI(root)
    root.mainloop()