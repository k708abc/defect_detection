import tkinter.ttk as ttk
import tkinter as tk
import cv2
import numpy as np
import os
import glob
from Modules.maxima_detection_functions import (
    get_image_values,
    get_datatypes,
)
from Modules.fft_formation import fft_process
import math


def initial_setting_dfft(self):
    self.image_open = True
    self.dfft_FFT = True
    self.prev_mag = 1
    self.cb_choise_dfft.set(self.choise.get())
    set_imtype(self)
    self.cb_imtype_dfft.set(self.imtype_choise.get())
    cv2.namedWindow("Arrow")
    cv2.setMouseCallback("Arrow", mouse_event_arrow1, self)
    show_image_dfft(self)


def prepare_widgets(self):
    create_frame_ref_choise_dfft(self)
    create_frame_fft_dfft(self)
    create_frame_magnification_dfft(self)
    create_frame_size_dfft(self)
    create_frame_set_vector(self)
    create_frame_params_dfft(self)
    create_frame_buttons_dfft(self)


def create_frame_ref_choise_dfft(self):
    self.frame_choise_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_choise_dfft(self)
    create_layout_choise_dfft(self)
    self.frame_choise_dfft.pack()


def create_widgets_choise_dfft(self):
    self.image_list = glob.glob(self.dir_name + "*")
    self.image_list = [os.path.basename(pathname) for pathname in self.image_list]
    # reference
    self.var_cb_choise_dfft = tk.StringVar()
    self.cb_choise_dfft = ttk.Combobox(
        self.frame_choise_dfft,
        textvariable=self.var_cb_choise_dfft,
        values=self.image_list,
        width=40,
    )
    self.cb_choise_dfft.bind(
        "<<ComboboxSelected>>", lambda event, arg=self: ref_selected_dfft(event, arg)
    )
    #
    self.cblabel_ref_dfft = tk.Label(self.frame_choise_dfft, text="Reference")
    #
    self.imtype_list_dfft = []
    self.var_imtypes_dfft = tk.StringVar()
    self.cb_imtype_dfft = ttk.Combobox(
        self.frame_choise_dfft,
        textvariable=self.var_imtypes_dfft,
        values=self.imtype_list_dfft,
        width=40,
    )
    self.imtype_dfft_text = ttk.Label(self.frame_choise_dfft, text="Image type")
    #
    self.button_dfft_open = tk.Button(
        self.frame_choise_dfft,
        text="Open",
        command=lambda: ref_open_dfft(self),
        width=10,
    )
    self.button_dfft_open["state"] = tk.NORMAL


def create_layout_choise_dfft(self):
    self.cb_choise_dfft.grid(row=0, column=1, **self.padWE)
    self.cblabel_ref_dfft.grid(row=0, column=0, **self.padWE)
    self.cb_imtype_dfft.grid(row=1, column=1, **self.padWE)
    self.imtype_dfft_text.grid(row=1, column=0, **self.padWE)
    self.button_dfft_open.grid(
        rowspan=2, row=0, column=2, sticky=tk.N + tk.S, padx=15, pady=2
    )


def create_frame_fft_dfft(self):
    self.frame_fft_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_fft_dfft(self)
    create_layout_fft_dfft(self)
    self.frame_fft_dfft.pack()


def create_widgets_fft_dfft(self):
    self.fft_button_dfft = tk.Button(
        self.frame_fft_dfft,
        text="FFT→\rReal",
        command=lambda: fft_function_dfft(self),
        width=10,
    )
    #
    self.var_method_fft_dfft = tk.StringVar()
    self.method_fft_table = ["Linear", "Sqrt", "Log"]
    self.method_fft_cb_dfft = ttk.Combobox(
        self.frame_fft_dfft,
        textvariable=self.var_method_fft_dfft,
        values=self.method_fft_table,
    )
    self.method_fft_cb_dfft.bind(
        "<<ComboboxSelected>>",
        lambda event, arg=self: cb_method_selected_dfft(event, arg),
    )
    self.method_fft_cb_dfft.current(2)
    self.method_text_dfft = ttk.Label(self.frame_fft_dfft, text="Method")
    #
    self.var_window_dfft = tk.StringVar()
    self.window_table = ["Rect", "Hann", "Hamming", "Blackman"]
    self.window_cb_dfft = ttk.Combobox(
        self.frame_fft_dfft, textvariable=self.var_window_dfft, values=self.window_table
    )
    self.window_cb_dfft.bind(
        "<<ComboboxSelected>>",
        lambda event, arg=self: cb_window_selected_dfft(event, arg),
    )
    self.window_cb_dfft.current(1)
    self.window_text_dfft = ttk.Label(self.frame_fft_dfft, text="Window")


def create_layout_fft_dfft(self):
    self.fft_button_dfft.grid(rowspan=2, row=0, column=2, sticky=tk.N + tk.S)
    self.method_text_dfft.grid(row=0, column=0, **self.padWE)
    self.method_fft_cb_dfft.grid(row=0, column=1, **self.padWE)
    self.window_text_dfft.grid(row=1, column=0, **self.padWE)
    self.window_cb_dfft.grid(row=1, column=1, **self.padWE)


def create_frame_magnification_dfft(self):
    self.frame_mag_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_mag_dfft(self)
    create_layout_mag_dfft(self)
    self.frame_mag_dfft.pack()


def create_widgets_mag_dfft(self):
    self.label_dfft_mag = tk.Label(self.frame_mag_dfft, text="Magnification")
    self.dfft_mag_entry = ttk.Entry(self.frame_mag_dfft, width=7)
    self.dfft_mag_entry.bind("<Return>", lambda event, arg=self: mag_bind(event, arg))
    self.dfft_mag_entry.insert(tk.END, "1")


def create_layout_mag_dfft(self):
    self.label_dfft_mag.grid(row=0, column=0, **self.padWE)
    self.dfft_mag_entry.grid(row=0, column=1, **self.padWE)


def create_frame_size_dfft(self):
    self.frame_size_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_size_dfft(self)
    create_layout_size_dfft(self)
    self.frame_size_dfft.pack()


def create_widgets_size_dfft(self):
    self.label_dfft_size_x = tk.Label(self.frame_size_dfft, text="Size x (nm)")
    self.dfft_size_x_entry = ttk.Entry(self.frame_size_dfft, width=7)
    self.dfft_size_x_entry.insert(tk.END, "30")
    #
    self.label_dfft_size_y = tk.Label(self.frame_size_dfft, text="Size y (nm)")
    self.dfft_size_y_entry = ttk.Entry(self.frame_size_dfft, width=7)
    self.dfft_size_y_entry.insert(tk.END, "30")
    #
    self.label_dfft_size_x_FFT = tk.Label(self.frame_size_dfft, text="Size kx (nm-1)")
    self.dfft_size_x_FFT = tk.Label(self.frame_size_dfft, text="0")
    #
    self.label_dfft_size_y_FFT = tk.Label(self.frame_size_dfft, text="Size ky (nm-1)")
    self.dfft_size_y_FFT = tk.Label(self.frame_size_dfft, text="0")


def create_layout_size_dfft(self):
    self.label_dfft_size_x.grid(row=0, column=0, **self.padWE)
    self.dfft_size_x_entry.grid(row=0, column=1, **self.padWE)
    self.label_dfft_size_y.grid(row=0, column=2, **self.padWE)
    self.dfft_size_y_entry.grid(row=0, column=3, **self.padWE)
    self.label_dfft_size_x_FFT.grid(row=1, column=0, **self.padWE)
    self.dfft_size_x_FFT.grid(row=1, column=1, **self.padWE)
    self.label_dfft_size_y_FFT.grid(row=1, column=2, **self.padWE)
    self.dfft_size_y_FFT.grid(row=1, column=3, **self.padWE)


def create_frame_set_vector(self):
    self.frame_set_vector = ttk.Frame(self.fft_fix_window)
    create_widgets_set_vector(self)
    create_layout_set_vector(self)
    self.frame_set_vector.pack()


def create_widgets_set_vector(self):
    self.button_dfft_vec1 = tk.Button(
        self.frame_set_vector,
        text="Set k vector 1",
        command=lambda: set_vector_1(self),
        width=10,
    )
    self.button_dfft_vec1["state"] = tk.NORMAL
    #
    self.dfft_k1_label = ttk.Label(self.frame_set_vector, text="0 nm-1")
    self.dfft_k1_angle_label = ttk.Label(self.frame_set_vector, text="0 °")
    self.dfft_r1_label = ttk.Label(self.frame_set_vector, text="0 nm")
    self.dfft_r1_angle_label = ttk.Label(self.frame_set_vector, text="0 °")
    self.FFT_label_k1 = ttk.Label(self.frame_set_vector, text="FFT")
    self.FFT_label_r1 = ttk.Label(self.frame_set_vector, text="Real")
    #
    self.button_dfft_vec2 = tk.Button(
        self.frame_set_vector,
        text="Set k vector 2",
        command=lambda: set_vector_2(self),
        width=10,
    )
    self.button_dfft_vec2["state"] = tk.NORMAL
    #
    self.dfft_k2_label = ttk.Label(self.frame_set_vector, text="0 nm-1")
    self.dfft_k2_angle_label = ttk.Label(self.frame_set_vector, text="0 °")
    self.dfft_r2_label = ttk.Label(self.frame_set_vector, text="0 nm")
    self.dfft_r2_angle_label = ttk.Label(self.frame_set_vector, text="0 °")
    self.FFT_label_k2 = ttk.Label(self.frame_set_vector, text="FFT")
    self.FFT_label_r2 = ttk.Label(self.frame_set_vector, text="Real")


def create_layout_set_vector(self):
    self.button_dfft_vec1.grid(rowspan=2, row=0, column=0, sticky=tk.N + tk.S)
    self.dfft_k1_label.grid(row=0, column=2, **self.padWE)
    self.dfft_k1_angle_label.grid(row=0, column=3, **self.padWE)
    self.dfft_r1_label.grid(row=1, column=2, **self.padWE)
    self.dfft_r1_angle_label.grid(row=1, column=3, **self.padWE)
    self.FFT_label_k1.grid(row=0, column=1, **self.padWE)
    self.FFT_label_r1.grid(row=1, column=1, **self.padWE)
    #
    self.button_dfft_vec2.grid(rowspan=2, row=2, column=0, sticky=tk.N + tk.S)
    self.dfft_k2_label.grid(row=2, column=2, **self.padWE)
    self.dfft_k2_angle_label.grid(row=2, column=3, **self.padWE)
    self.dfft_r2_label.grid(row=3, column=2, **self.padWE)
    self.dfft_r2_angle_label.grid(row=3, column=3, **self.padWE)
    self.FFT_label_k2.grid(row=2, column=1, **self.padWE)
    self.FFT_label_r2.grid(row=3, column=1, **self.padWE)


def create_frame_params_dfft(self):
    self.frame_params_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_params_dfft(self)
    create_layout_params_dfft(self)
    self.frame_params_dfft.pack()


def create_widgets_params_dfft(self):
    self.dfft_ratio_label = ttk.Label(self.frame_params_dfft, text="Ratio (v2/v1)")
    self.dfft_dif_angle_label = ttk.Label(self.frame_params_dfft, text="Diff. angle")
    self.dfft_ratio_val = ttk.Label(self.frame_params_dfft, text="0")
    self.dfft_dif_angle_val = ttk.Label(self.frame_params_dfft, text="0 °")
    #
    self.dfft_set_ratio_label = ttk.Label(self.frame_params_dfft, text="Set ratio")
    self.dfft_set_dif_angle_label = ttk.Label(self.frame_params_dfft, text="Set angle")
    self.dfft_set_ratio_entry = ttk.Entry(self.frame_params_dfft, width=7)
    self.dfft_set_ratio_entry.insert(tk.END, "1")
    self.dfft_set_angle_entry = ttk.Entry(self.frame_params_dfft, width=7)
    self.dfft_set_angle_entry.insert(tk.END, "60")


def create_layout_params_dfft(self):
    self.dfft_ratio_label.grid(row=0, column=2, **self.padWE)
    self.dfft_dif_angle_label.grid(row=1, column=2, **self.padWE)
    self.dfft_ratio_val.grid(row=0, column=3, **self.padWE)
    self.dfft_dif_angle_val.grid(row=1, column=3, **self.padWE)
    #
    self.dfft_set_ratio_label.grid(row=0, column=0, **self.padWE)
    self.dfft_set_ratio_entry.grid(row=0, column=1, **self.padWE)
    self.dfft_set_dif_angle_label.grid(row=1, column=0, **self.padWE)
    self.dfft_set_angle_entry.grid(row=1, column=1, **self.padWE)


def create_frame_buttons_dfft(self):
    self.frame_buttons_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_buttons_dfft(self)
    create_layout_buttons_dfft(self)
    self.frame_buttons_dfft.pack()


def create_widgets_buttons_dfft(self):
    self.frame_buttons_dfft = ttk.Frame(self.fft_fix_window)
    create_widgets_buttons_dfft(self)
    create_layout_buttons_dfft(self)
    self.frame_buttons_dfft.pack()


def create_widgets_buttons_dfft(self):
    self.button_dfft_calculate = tk.Button(
        self.frame_buttons_dfft,
        text="Calculate",
        command=lambda: calculate_dfft(self),
        width=10,
    )
    self.button_dfft_calculate["state"] = tk.NORMAL
    #
    self.button_dfft_reset = tk.Button(
        self.frame_buttons_dfft,
        text="Reset",
        command=lambda: reset_dfft(self),
        width=10,
    )
    self.button_dfft_reset["state"] = tk.NORMAL
    #
    self.button_dfft_close = tk.Button(
        self.frame_buttons_dfft,
        text="Close",
        command=lambda: close_dfft(self),
        width=10,
    )
    self.button_dfft_close["state"] = tk.NORMAL


def create_layout_buttons_dfft(self):
    self.button_dfft_calculate.grid(row=0, column=0, **self.padWE)
    self.button_dfft_reset.grid(row=0, column=1, **self.padWE)
    self.button_dfft_close.grid(row=0, column=2, **self.padWE)


def set_imtype(self):
    self.data_path = self.dir_name + self.cb_choise_dfft.get()
    self.imtype_list_dfft = get_datatypes(self.data_path)
    self.cb_imtype_dfft["values"] = self.imtype_list_dfft


def ref_selected_dfft(event, self):
    set_imtype(self)
    self.cb_imtype_dfft.current(0)
    self.fft_fix_window.update()


def ref_open_dfft(self):
    show_image_dfft(self)


def mouse_event_arrow1(event, x, y, flags, self):
    if event == cv2.EVENT_LBUTTONUP and self.dfft_FFT:
        self.vec1_x = x
        self.vec1_y = y
        show_dfft(self)
        put_values_dfft(self)


def mouse_event_arrow2(event, x, y, flags, self):
    if event == cv2.EVENT_LBUTTONUP and self.dfft_FFT:
        self.vec2_x = x
        self.vec2_y = y
        show_dfft(self)
        put_values_dfft(self)


def fft_function_dfft(self):
    if self.dfft_FFT:
        self.dfft_FFT = False
        self.fft_button_dfft["text"] = "Real\r→FFT"

    else:
        self.dfft_FFT = True
        self.fft_button_dfft["text"] = "FFT→\rReal"
    show_image_dfft(self)


def cb_method_selected_dfft(event, self):
    if self.dfft_FFT:
        show_image_dfft(self)


def cb_window_selected_dfft(event, self):
    if self.dfft_FFT:
        show_image_dfft(self)


def using_vec1(self):
    self.button_dfft_vec1["state"] = tk.DISABLED
    self.button_dfft_vec1["text"] = "Selected"
    self.button_dfft_vec2["state"] = tk.NORMAL
    self.button_dfft_vec2["text"] = "set k vector 2"


def using_vec2(self):
    self.button_dfft_vec1["state"] = tk.NORMAL
    self.button_dfft_vec1["text"] = "set k vector 1"
    self.button_dfft_vec2["state"] = tk.DISABLED
    self.button_dfft_vec2["text"] = "Selected"


def set_vector_1(self):
    cv2.setMouseCallback("Arrow", mouse_event_arrow1, self)
    using_vec1(self)


def set_vector_2(self):
    cv2.setMouseCallback("Arrow", mouse_event_arrow2, self)
    using_vec2(self)


def reset_dfft(self):
    reset_arrow_dfft(self)
    show_image_dfft()


def reset_arrow_dfft(self):
    default_x = self.image_target.shape[1]
    default_y = self.image_target.shape[0]
    self.vec1_x = int(default_x / 4 * 3)
    self.vec1_y = int(default_y / 2)
    self.vec2_x = int(default_x / 2)
    self.vec2_y = int(default_y / 4)


def close_dfft(self):
    try:
        cv2.destroyWindow("Arrow")
    except:
        pass
    self.fft_fix_window.destroy()


def magnification(image, magnitude):
    height, width = image.shape[0], image.shape[1]
    width_x = int(width * magnitude)
    height_y = int(height * magnitude)
    image_re = cv2.resize(image, (width_x, height_y))
    return image_re


def mag_bind(event, self):
    show_image_dfft(self)


def show_image_dfft(self):
    read_reference_dfft(self)
    convert_FFT(self)
    if self.dfft_FFT:
        self.image_target = np.copy(self.image_dfft_FFT)
    else:
        self.image_target = np.copy(self.image_dfft_or)
    # magnification
    self.image_target = magnification(
        self.image_target, float(self.dfft_mag_entry.get())
    )

    if self.image_open:
        reset_arrow_dfft(self)
        set_vector_1(self)
        self.image_open = False
    self.vec1_x = int(self.vec1_x / self.prev_mag * float(self.dfft_mag_entry.get()))
    self.vec1_y = int(self.vec1_y / self.prev_mag * float(self.dfft_mag_entry.get()))
    self.vec2_x = int(self.vec2_x / self.prev_mag * float(self.dfft_mag_entry.get()))
    self.vec2_y = int(self.vec2_y / self.prev_mag * float(self.dfft_mag_entry.get()))

    self.prev_mag = float(self.dfft_mag_entry.get())
    colorize(self)
    show_dfft(self)
    put_values_dfft(self)


def colorize(self):
    self.image_target = cv2.cvtColor(
        self.image_target.astype(np.float32), cv2.COLOR_GRAY2BGR
    )


def read_reference_dfft(self):
    self.data_path_dfft = self.dir_name + self.cb_choise_dfft.get()
    self.channel_type_dfft = self.cb_imtype_dfft.current()
    #
    self.image_dfft_or, self.image_dfft_params, _ = get_image_values(
        self.data_path_dfft, self.channel_type_dfft
    )

    init_sizes_calculation(self)

    self.image_dfft_or = (self.image_dfft_or - np.min(self.image_dfft_or)) / (
        np.max(self.image_dfft_or) - np.min(self.image_dfft_or)
    )


def init_sizes_calculation(self):
    size_x_real = self.image_dfft_params[0]
    size_y_real = self.image_dfft_params[1]
    #
    self.fft_size_x_dfft = round(
        2 * math.pi / size_x_real * self.image_dfft_or.shape[1], 2
    )
    self.fft_size_y_dfft = round(
        2 * math.pi / size_y_real * self.image_dfft_or.shape[0], 2
    )
    #
    self.dfft_size_x_entry.delete(0, tk.END)
    self.dfft_size_x_entry.insert(tk.END, size_x_real)
    #
    self.dfft_size_y_entry.delete(0, tk.END)
    self.dfft_size_y_entry.insert(tk.END, size_y_real)
    #
    self.dfft_size_x_FFT["text"] = str(self.fft_size_x_dfft)
    self.dfft_size_y_FFT["text"] = str(self.fft_size_y_dfft)


def convert_FFT(self):
    self.image_dfft_FFT = fft_process(
        self.image_dfft_or, self.method_fft_cb_dfft.get(), self.window_cb_dfft.get()
    )
    self.image_dfft_FFT = (self.image_dfft_FFT - np.min(self.image_dfft_FFT)) / (
        np.max(self.image_dfft_FFT) - np.min(self.image_dfft_FFT)
    )


def show_dfft(self):
    self.image_dfft = np.copy(self.image_target)
    draw_arrow(self)
    cv2.imshow("Arrow", self.image_dfft)


def draw_arrow(self):
    center = (int(self.image_dfft.shape[1] / 2), int(self.image_dfft.shape[0] / 2))
    if self.dfft_FFT:
        cv2.arrowedLine(
            self.image_dfft,
            center,
            (self.vec1_x, self.vec1_y),
            (255, 0, 0),
            thickness=1,
            shift=0,
            tipLength=0.1,
        )
        cv2.arrowedLine(
            self.image_dfft,
            center,
            (self.vec2_x, self.vec2_y),
            (0, 255, 0),
            thickness=1,
            shift=0,
            tipLength=0.1,
        )
    else:
        cv2.arrowedLine(
            self.image_dfft,
            center,
            (
                center[0]
                + int(
                    self.x1_real
                    / float(self.dfft_size_x_entry.get())
                    * self.image_dfft.shape[1]
                ),
                center[1]
                + int(
                    self.y1_real
                    / float(self.dfft_size_y_entry.get())
                    * self.image_dfft.shape[0]
                ),
            ),
            (255, 0, 0),
            thickness=1,
            shift=0,
            tipLength=0.1,
        )
        cv2.arrowedLine(
            self.image_dfft,
            center,
            (
                center[0]
                + int(
                    self.x2_real
                    / float(self.dfft_size_x_entry.get())
                    * self.image_dfft.shape[1]
                ),
                center[1]
                + int(
                    self.y2_real
                    / float(self.dfft_size_y_entry.get())
                    * self.image_dfft.shape[0]
                ),
            ),
            (0, 255, 0),
            thickness=1,
            shift=0,
            tipLength=0.1,
        )


def convert_FFT_real(FFT_x1, FFT_y1, FFT_x2, FFT_y2):
    # unit vector in real space
    xr1 = 2 * math.pi / (FFT_x1 * FFT_y2 - FFT_y1 * FFT_x2) * FFT_y2
    yr1 = -2 * math.pi / (FFT_x1 * FFT_y2 - FFT_y1 * FFT_x2) * FFT_x2
    xr2 = 2 * math.pi / (FFT_x2 * FFT_y1 - FFT_y2 * FFT_x1) * FFT_y1
    yr2 = -2 * math.pi / (FFT_x2 * FFT_y1 - FFT_y2 * FFT_x1) * FFT_x1
    return xr1, yr1, xr2, yr2


def get_polar(x, y):
    r = math.sqrt(x ** 2 + y ** 2)
    theta = math.atan2(y, x)
    return r, theta


def put_values_dfft(self):
    center = (self.image_dfft.shape[1] / 2, self.image_dfft.shape[0] / 2)
    x_diff1 = (
        (self.vec1_x - center[0]) / self.image_dfft.shape[1] * self.fft_size_x_dfft
    )
    y_diff1 = (
        (self.vec1_y - center[1]) / self.image_dfft.shape[0] * self.fft_size_y_dfft
    )
    x_diff2 = (
        (self.vec2_x - center[0]) / self.image_dfft.shape[1] * self.fft_size_x_dfft
    )
    y_diff2 = (
        (self.vec2_y - center[1]) / self.image_dfft.shape[0] * self.fft_size_y_dfft
    )
    #
    r1, theta1 = get_polar(x_diff1, y_diff1)
    r2, theta2 = get_polar(x_diff1, y_diff2)

    #
    self.x1_real, self.y1_real, self.x2_real, self.y2_real = convert_FFT_real(
        x_diff1, y_diff1, x_diff2, y_diff2
    )
    #
    r1_real, theta1_real = get_polar(self.x1_real, self.y1_real)
    r2_real, theta2_real = get_polar(self.x2_real, self.y2_real)
    #
    self.dfft_k1_label["text"] = str(round(r1, 2)) + " nm-1"
    self.dfft_k1_angle_label["text"] = str(round(-theta1 / math.pi * 180, 2)) + " °"
    self.dfft_k2_label["text"] = str(round(r2, 2)) + " nm-1"
    self.dfft_k2_angle_label["text"] = str(round(-theta2 / math.pi * 180, 2)) + " °"
    #
    self.dfft_r1_label["text"] = str(round(r1_real, 2)) + " nm"
    self.dfft_r1_angle_label["text"] = (
        str(round(-theta1_real / math.pi * 180, 2)) + " °"
    )
    self.dfft_r2_label["text"] = str(round(r2_real, 2)) + " nm"
    self.dfft_r2_angle_label["text"] = (
        str(round(-theta2_real / math.pi * 180, 2)) + " °"
    )
    #
    self.dfft_ratio_val["text"] = str(round(r2_real / r1_real, 2))
    self.dfft_dif_angle_val["text"] = str(
        round((theta1_real - theta2_real) / math.pi * 180, 2)
    )


def drift_fft(self):
    self.fft_fix_window = tk.Toplevel()
    self.fft_fix_window.title("Drift fix (FFT)")
    prepare_widgets(self)
    initial_setting_dfft(self)


def get_theta(v, w, self):
    L = float(self.dfft_size_x_entry.get())
    a = self.x1_real
    b = self.y1_real
    c = self.x2_real
    d = self.y2_real

    av1x = a + b / L * v
    av1y = b + b / L * w
    av2x = c + d / L * v
    av2y = d + d / L * w

    atheta1 = math.atan2(av1y, av1x)
    atheta2 = math.atan2(av2y, av2x)

    dtheta = abs(-atheta1 + atheta2)
    otheta = float(self.dfft_set_angle_entry.get())

    theta_diff = abs(dtheta * 180 / math.pi - otheta)

    return theta_diff


def calculate_drift_dfft(self):
    L = float(self.dfft_size_y_entry.get())
    a = self.x1_real
    b = self.y1_real
    c = self.x2_real
    d = self.y2_real
    r = float(self.dfft_set_ratio_entry.get())
    k = math.cos(math.radians(float(self.dfft_set_angle_entry.get())))

    v = (k * r * (b * c + a * d) - r ** 2 * a * b - c * d) / (
        r ** 2 * b ** 2 - 2 * k * r * b * d + d ** 2
    )
    sqrt = (
        -(v ** 2)
        + 2 * (c * d - r ** 2 * a * b) * v / (r ** 2 * b ** 2 - d ** 2)
        + (c ** 2 - r ** 2 * a ** 2) / (r ** 2 * b ** 2 - d ** 2)
    )
    w1 = -math.sqrt(sqrt) - 1
    w2 = math.sqrt(sqrt) - 1

    v = L * v
    w1 = L * w1
    w2 = L * w2

    the1 = get_theta(v, w1, self)
    the2 = get_theta(v, w2, self)

    if the1 < the2:
        w = w1
    else:
        w = w2
    # fix_check(v, w, self)
    return v, w


def fix_check(v, w, self):
    v11 = 1
    v12 = v / float(self.dfft_size_y_entry.get())
    v21 = 0
    v22 = 1 + w / float(self.dfft_size_y_entry.get())
    n_vec1 = [
        self.x1_real * v11 + self.y1_real * v12,
        self.x1_real * v21 + self.y1_real * v22,
    ]
    n_vec2 = [
        self.x2_real * v11 + self.y2_real * v12,
        self.x2_real * v21 + self.y2_real * v22,
    ]
    abs_v1 = math.sqrt(n_vec1[0] ** 2 + n_vec1[1] ** 2)
    abs_v2 = math.sqrt(n_vec2[0] ** 2 + n_vec2[1] ** 2)

    ratio = abs_v2 / abs_v1
    angle = (
        math.acos((n_vec1[0] * n_vec2[0] + n_vec1[1] * n_vec2[1]) / abs_v1 / abs_v2)
        / math.pi
        * 180
    )


def calculate_dfft(self):
    self.v, self.w = calculate_drift_dfft(self)
    self.drift_dx.delete(0, tk.END)
    self.drift_dx.insert(tk.END, round(self.v, 2))
    self.drift_dy.delete(0, tk.END)
    self.drift_dy.insert(tk.END, round(self.w, 2))
    if self.drift_bool.get() is True:
        self.modify_image()
