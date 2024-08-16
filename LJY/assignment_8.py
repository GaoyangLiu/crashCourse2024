import tkinter as tk
import random

class SquishGame:
    def __init__(self, root):
        self.root = root
        self.root.title("Squish Game")
        self.root.geometry("800x600")

        self.target_button = tk.Button(root, text="Squish Me!", command=self.squish_target)
        self.target_button.place(x=100, y=100, width=100, height=50)

        self.score = 0
        self.score_label = tk.Label(root, text=f"得分: {self.score}")
        self.score_label.pack()

        self.move_target()

    def move_target(self):
        new_x = random.randint(0, 700)
        new_y = random.randint(0, 550)
        self.target_button.place(x=new_x, y=new_y)
        self.root.after(1000, self.move_target)

    def squish_target(self):
        self.score += 1
        self.score_label.config(text=f"得分: {self.score}")

if __name__ == "__main__":
    root = tk.Tk()
    game = SquishGame(root)
    root.mainloop()
