import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk

class ImageProcess:
    def __init__(self, root):
        self.root = root
        self.root.title("图像显示及处理")
        self.root.geometry("900x600")

        self.left_image_label = Label(root, bg="gray")
        self.left_image_label.place(x=100, y=100, width=200, height=200)

        self.right_image_label = Label(root, bg="gray")
        self.right_image_label.place(x=550, y=100, width=200, height=200)

        self.select_button = Button(root, text="选择图片", command=self.load_image)
        self.select_button.place(x=150, y=450, width=100, height=40)

        self.run_button = Button(root, text="运行", command=self.process_image)
        self.run_button.place(x=600, y=450, width=100, height=40)

        self.image = None
        self.processed_image = None

        #self.root.grid_rowconfigure(0, weight=1)
        #self.root.grid_columnconfigure(0, weight=1)
        #self.root.grid_columnconfigure(1, weight=1)

    def display_image(self, img, label):
        img.thumbnail((200, 200))
        img_tk = ImageTk.PhotoImage(img)
        label.config(image=img_tk)
        label.image = img_tk


    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("图片文件", "*.jpg;*.png;*.jpeg;*.bmp"), ("所有文件", "*.*")])
        if file_path:
            self.image = Image.open(file_path)
            self.display_image(self.image, self.left_image_label)

    def process_image(self):
        if self.image:
            self.processed_image = self.image.transpose(Image.FLIP_LEFT_RIGHT)
            self.display_image(self.processed_image, self.right_image_label)

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcess(root)
    root.mainloop()
