"""
Identity card demo
Author: liuyuan
2018.2
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import process


def on_select():
    file_path = filedialog.askopenfilename()
    image_path.set(file_path)
    process.process(file_path)


if __name__ == '__main__':
    main_frame = tk.Tk()
    main_frame.title('Identity card demo')

    ttk.Label(main_frame, text='Select identity card image file:').grid(column=0, row=0, sticky='W')
    image_path = tk.StringVar()
    image_path_edit = ttk.Entry(main_frame, width=60, textvariable=image_path)
    image_path_edit.grid(column=0, row=1, sticky='W')
    select_button = ttk.Button(main_frame, width=10, text='...', command=on_select)
    select_button.grid(column=1, row=1, sticky='W')

    main_frame.resizable(False, False)
    main_frame.mainloop()
