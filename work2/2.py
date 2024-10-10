import tkinter as tk
from tkinter import filedialog, Text, END

class TextEditor:
    def __init__(self, master):
        self.master = master
        master.title("Simple Text Editor")

        self.text = Text(master)
        self.text.pack(expand=True, fill='both')

        self.status_bar = tk.Label(master, text="Total Lines: 0", bd=1, relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.menu = tk.Menu(master)
        master.config(menu=self.menu)

        file_menu = tk.Menu(self.menu, tearoff=0)
        self.menu.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save As...", command=self.save_as)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=master.quit)

        self.text.bind('<<Modified>>', self.update_status)

    def save_as(self):
        file_path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")])
        if file_path:
            with open(file_path, 'w') as file:
                file.write(self.text.get(1.0, END))
                self.text.edit_modified(False)  # Clear the modified flag

    def update_status(self, event=None):
        total_lines = len(self.text.get(1.0, END).splitlines())
        self.status_bar.config(text=f"Total Lines: {total_lines}")

if __name__ == "__main__":
    root = tk.Tk()
    app = TextEditor(root)
    root.mainloop()