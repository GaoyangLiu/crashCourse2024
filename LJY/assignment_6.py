import tkinter as tk
from tkinter import filedialog, messagebox

class TextEditor:

    def __init__(self, root):
        self.root = root
        self.root.title("文本编辑器")
        self.file_path = None

        self.text_area = tk.Text(root, wrap = 'word')
        self.text_area.pack(expand = 1, fill = 'both')

        self.status_bar = tk.Label(root, text = "行数：1", anchor = 'w')
        self.status_bar.pack(side = tk.BOTTOM, fill = tk.X)

        self.menu_bar = tk.Menu(root)
        self.root.config(menu = self.menu_bar)

        file_menu = tk.Menu(self.menu_bar, tearoff = 0)
        file_menu.add_command(label="新建", command=self.new_file)
        file_menu.add_command(label="打开", command=self.open_file)
        file_menu.add_command(label="保存", command=self.save_file)
        file_menu.add_command(label="另存为...", command=self.save_as_file)
        file_menu.add_separator()
        file_menu.add_command(label="退出", command=root.quit)
        self.menu_bar.add_cascade(label="文件", menu=file_menu)

        self.text_area.bind('<KeyRelease>', self.update_line_count)

    def new_file(self):
        self.text_area.delete(1.0, tk.END)
        self.root.title("新文件 - 文本编辑器")

    def open_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
        if file_path:
            with open(file_path,'r') as file:
                content = file.read()
            self.text_area.delete(1.0, tk.END)
            self.text_area.insert(tk.INSERT, content)
            self.root.title(f"{file_path} - 文本编辑器")

    def save_file(self):
        if self.file_path:
            try:
                with open(self.file_path, 'w') as file:
                    content = self.text_area.get(1.0, tk.END)
                    file.write(content)
                self.root.title(f"{self.file_path} - 文件编辑器")
            except Exception as e:
                messagebox.showerror("错误", f"无法保存文件：{e}")
        else:
            self.save_as_file()

    def save_as_file(self):
        try:
            file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("文本文件", "*.txt"), ("所有文件", "*.*")])
            if file_path:
                with open(file_path, 'w') as file:
                    content = self.text_area.get(1.0, tk.END)
                    file.write(content)
                self.root.title(f"{file_path} - 文本编辑器")
        except Exception as e:
            messagebox.showerror("错误", f"无法保存文件：{e}")



    def update_line_count(self, event = None):
        line_count = int(self.text_area.index('end-1c').split('.')[0])
        self.status_bar.config(text=f"行数: {line_count}")

if __name__ == "__main__":
    root = tk.Tk()
    editor = TextEditor(root)
    root.mainloop()