import os
import tkinter as tk
from PIL import Image, EpsImagePlugin

from PyTorchModel import Classifier


can_size = [400, 400]


class Paint:
    def __init__(self, master):
        self.master = master
        self.controls_frame = tk.Frame(self.master, padx=5, pady=5)
        self.output_frame = tk.Frame(self.master, padx=5, pady=5)
        self.c = tk.Canvas(self.master, heigh=can_size[0], width=can_size[1], bg="black")
        self.clear_b = tk.Button(
            self.controls_frame,
            text="Clear",
            font=("Verdana", 14, "bold"),
            command=self.clear,
        )
        self.ready_b = tk.Button(
            self.controls_frame,
            text="Ready",
            font=("Verdana", 14, "bold"),
            command=self.ready,
        )

        self.old_x = None
        self.old_y = None
        self.penwidth = 30
        self.pred_number = tk.StringVar()
        self.pred_number.set("---")
        self.draw_widgets()
        self.c.bind('<B1-Motion>', self.paint)
        self.c.bind('<ButtonRelease-1>', self.reset)
        self.Classifier = Classifier()

    def paint(self, e):
        if self.old_x and self.old_y:
            self.c.create_line(
                self.old_x,
                self.old_y,
                e.x,
                e.y,
                width=self.penwidth,
                fill="white",
                capstyle=tk.ROUND,
                smooth=True
            )

        self.old_x = e.x
        self.old_y = e.y

    def reset(self):
        self.old_x = None
        self.old_y = None      
           
    def clear(self):
        self.c.delete(tk.ALL)
        self.c.create_rectangle(0, 0, can_size[0] + 100, can_size[1] + 100, fill="black")

    def ready(self):
        tmp_canvas = "tmp_canvas.eps"
        self.c.postscript(file=tmp_canvas)
        EpsImagePlugin.gs_windows_binary = r"C:\Program Files\gs\gs9.55.0\bin\gswin64c"
        im = Image.open(tmp_canvas)
        val, _ = self.Classifier.predict(im)
        self.pred_number.set(str(val))
        del im
        os.remove(tmp_canvas)

    def draw_widgets(self):
        self.clear_b.pack(side=tk.LEFT)
        self.ready_b.pack(side=tk.RIGHT)
        self.controls_frame.pack(side=tk.TOP)

        self.c.pack(fill=tk.BOTH, expand=True)
        self.c.create_rectangle(0, 0, can_size[0] + 100, can_size[1] + 100, fill="black")

        self.output_frame.grid_rowconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(0, weight=1)
        self.output_frame.grid_columnconfigure(1, weight=1)
        tk.Label(
            self.output_frame, text="Result: ", font=("Verdana", 12)).grid(
            row=0, column=0, sticky="nse")
        tk.Label(
            self.output_frame, 
            textvariable=self.pred_number, 
            font=("Verdana", 14, "bold")
        ).grid(row=0, column=1, sticky="nsw")
        self.output_frame.pack(side=tk.BOTTOM)
        

if __name__ == '__main__':
    root = tk.Tk()
    root.geometry("+300+300")
    Paint(root)
    root.title('Paint')
    root.resizable(False, False)
    root.mainloop()
