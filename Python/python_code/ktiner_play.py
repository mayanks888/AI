# import tkinter as tk
# r = tk.Tk()
# r.title('Counting Seconds')
# button = tk.Button(r, text='Stop', width=25, command=r.destroy)
# button.pack()
# r.mainloop()

# from tkinter import *
#
# root = Tk()
# frame = Frame(root)
# frame.pack()
# bottomframe = Frame(root)
# bottomframe.pack(side=BOTTOM)
# redbutton = Button(frame, text='Red', fg='red')
# redbutton.pack(side=LEFT)
# greenbutton = Button(frame, text='Brown', fg='brown')
# greenbutton.pack(side=LEFT)
# bluebutton = Button(frame, text='Blue', fg='blue')
# bluebutton.pack(side=LEFT)
# blackbutton = Button(bottomframe, text='Black', fg='black')
# blackbutton.pack(side=BOTTOM)
# print(blackbutton)
# root.mainloop()

import tkinter as tk


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        frame = tk.Frame(self, bg="green",
                         height=100, width=100)
        frame.bind("<Button-1>", self.print_event)
        frame.bind("<Double-Button-1>", self.print_event)
        frame.bind("<ButtonRelease-1>", self.print_event)
        frame.bind("<B1-Motion>", self.print_event)
        frame.bind("<Enter>", self.print_event)
        frame.bind("<Leave>", self.print_event)
        frame.pack(padx=50, pady=50)

    def print_event(self, event):
        position = "(x={}, y={})".format(event.x, event.y)
        print(event.type, "event", position)


if __name__ == "__main__":
    app = App()
    app.mainloop()
