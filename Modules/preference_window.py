#!python3.11

import tkinter as tk
import tkinter.ttk as ttk


class Window(ttk.Frame):
    def __init__(self, master) -> None:
        super().__init__(master, padding=2)
        self.init_setting()
        self.create_frame_image_choise()
        self.create_frame_contrast()
        self.create_frame_cb()
        self.create_frame_process()
        self.create_frame_checks()
        self.create_frame_size()
        self.create_frame_auto()
        self.create_frame_record()
        self.create_frame_buttons()
        self.create_frame_notes()
        self.create_frame_notes2()
        self.master = master
        master.title("Maxima finder")

    def create_frame_image_choise(self):
        self.frame_image_choice = ttk.Frame()
        self.create_widgets_choice()
        self.create_layout_choice()
        self.frame_image_choice.pack()

    def create_widgets_choice(self):
        self.fol_choice_text = ttk.Label(self.frame_image_choice, text="Folder")
        self.fol_name = ttk.Entry(self.frame_image_choice, width=20)
        self.button_folchoice = tk.Button(
            self.frame_image_choice,
            text="Folder choice",
            command=self.fol_choice_clicked,
            width=10,
        )
        #
        self.var_images = tk.StringVar()
        self.choice = ttk.Combobox(
            self.frame_image_choice,
            textvariable=self.var_images,
            values=[],
            width=57,
        )
        self.choice.bind("<<ComboboxSelected>>", self.choice_selected)
        self.choice_text = ttk.Label(self.frame_image_choice, text="Image")
        #
        self.var_imtypes = tk.StringVar()
        self.imtype_choice = ttk.Combobox(
            self.frame_image_choice,
            textvariable=self.var_imtypes,
            values=[],
            width=57,
        )
        self.imtype_choice.bind("<<ComboboxSelected>>", self.type_choice_selected)
        self.imtype_text = ttk.Label(self.frame_image_choice, text="Image type")
        #
        self.button_imopen = tk.Button(
            self.frame_image_choice,
            text="Open",
            command=self.image_open_clicked,
            width=10,
        )
        self.images_update(self.dir_name)

    def create_frame_contrast(self):
        self.frame_contrast = ttk.Frame()
        self.create_widgets_contrast()
        self.create_layout_contrast()
        self.frame_contrast.pack()

    def create_widgets_contrast(self):
        self.upper_val = tk.DoubleVar()
        self.def_max = 255
        self.upper_val.set(self.def_max)
        self.upper_val.trace("w", self.upper_value_change)
        self.scale_upper = ttk.Scale(
            self.frame_contrast,
            variable=self.upper_val,
            orient=tk.HORIZONTAL,
            length=300,
            from_=-50,
            to=300,
        )
        #
        self.lower_val = tk.DoubleVar()
        self.def_min = 0
        self.lower_val.set(self.def_min)
        self.lower_val.trace("w", self.lower_value_change)
        self.scale_lower = ttk.Scale(
            self.frame_contrast,
            variable=self.lower_val,
            orient=tk.HORIZONTAL,
            length=300,
            from_=-50,
            to=300,
        )
        #
        self.upper_text = ttk.Label(self.frame_contrast, text="Upper")
        self.lower_text = ttk.Label(self.frame_contrast, text="lower")
        self.default_button = tk.Button(
            self.frame_contrast,
            text="Back to \rDefault",
            command=self.default_function,
            width=10,
        )
        #
        self.set_as_default_button = tk.Button(
            self.frame_contrast,
            text="Set as \rDefault",
            command=self.set_default_function,
            width=10,
        )
        #

    def create_frame_cb(self):
        self.frame_cb = ttk.Frame()
        self.create_widgets_cb()
        self.create_layout_cb()
        self.frame_cb.pack()

    def create_widgets_cb(self):
        self.var_cb_color = tk.StringVar()
        self.colormap_table = [
            "gray",
            "AUTUMN",
            "BONE ",
            "JET",
            "WINTER",
            "RAINBOW",
            "OCEAN",
            "SUMMER",
            "SPRING",
            "COOL",
            "HSV",
            "PINK",
            "HOT",
            "PALULA",
            "MAGNA",
            "INFERNO",
            "PLASMA",
            "VIRDIS",
            "CIVIDIS",
            "TWILIGHT",
            "TWILIGHT_SHIFTED",
        ]
        self.cb_color = ttk.Combobox(
            self.frame_cb,
            textvariable=self.var_cb_color,
            values=self.colormap_table,
        )

        self.cb_color.bind("<<ComboboxSelected>>", self.cb_color_selected)
        self.cb_color_text = ttk.Label(self.frame_cb, text="Color")
        self.cb_color.current(2)
        #

    def create_frame_process(self):
        self.frame_process = ttk.Frame()
        self.create_widgets_process()
        self.create_layout_process()
        self.frame_process.pack()

    def create_widgets_process(self):
        self.range_text = ttk.Label(self.frame_process, text="Range of image (max: --)")
        self.upper_set_text = ttk.Label(self.frame_process, text="Upper")
        self.lower_set_text = ttk.Label(self.frame_process, text="lower")
        #
        self.upper_set_entry = ttk.Entry(self.frame_process, width=7)
        self.upper_set_entry.bind("<Return>", self.range_change)
        self.upper_set_entry.insert(tk.END, "100")
        self.upper_set_entry.bind("<Up>", self.upper_up)
        self.upper_set_entry.bind("<Down>", self.upper_down)

        self.lower_set_entry = ttk.Entry(self.frame_process, width=7)
        self.lower_set_entry.bind("<Return>", self.range_change)
        self.lower_set_entry.insert(tk.END, "0")
        self.lower_set_entry.bind("<Up>", self.lower_up)
        self.lower_set_entry.bind("<Down>", self.lower_down)
        #
        self.smooth_entry = ttk.Entry(self.frame_process, width=4)
        self.smooth_entry.insert(tk.END, "0")
        self.smooth_entry.bind("<Return>", self.smooth_change)
        self.cb_smooth_text = ttk.Label(self.frame_process, text="Smoothing (pix)")
        #
        self.median_entry = ttk.Entry(self.frame_process, width=4)
        self.median_entry.insert(tk.END, "1")
        self.median_entry.bind("<Return>", self.median_change)
        self.cb_median_text = ttk.Label(self.frame_process, text="Median (pix)")
        #
        self.analysis_range_text = ttk.Label(
            self.frame_process, text="Analysis radius (nm)"
        )
        self.analysis_range = ttk.Entry(self.frame_process, width=4)
        self.analysis_range.bind("<Return>", self.analysis_range_change)
        self.analysis_range.insert(tk.END, "1")
        #
        self.rescale_text = ttk.Label(self.frame_process, text="Rescale")
        self.rescale_all = ttk.Entry(self.frame_process, width=7)
        self.rescale_all.insert(tk.END, "1")
        self.rescale_all.bind("<Return>", self.rescale)

    def create_frame_checks(self):
        self.frame_check = ttk.Frame()
        self.create_widgets_checks()
        self.create_layout_checks()
        self.frame_check.pack()

    def create_widgets_checks(self):
        self.plane_bool = tk.BooleanVar()
        self.plane_bool.set(True)
        self.plane_check = tk.Checkbutton(
            self.frame_check,
            variable=self.plane_bool,
            text="Plane subtraction",
            command=self.plane_image,
        )

        self.ave_bool = tk.BooleanVar()
        self.ave_bool.set(True)
        self.ave_check = tk.Checkbutton(
            self.frame_check,
            variable=self.ave_bool,
            text="Average subtraction",
            command=self.ave_image,
        )

    def create_frame_auto(self):
        self.frame_auto = ttk.Frame()
        self.create_widgets_auto()
        self.create_layout_auto()
        self.frame_auto.pack()

    def create_widgets_auto(self):
        self.auto_button = tk.Button(
            self.frame_auto,
            text="Auto detection",
            command=self.auto_detection,
            width=18,
            height=3,
        )
        self.auto_range_text = ttk.Label(self.frame_auto, text="Detection Range (nm)")
        self.auto_range = ttk.Entry(self.frame_auto, width=7)
        self.auto_range.insert(tk.END, "3")
        #
        self.auto_thresh_text = ttk.Label(self.frame_auto, text="Threshhold (× std)")
        self.auto_thresh = ttk.Entry(self.frame_auto, width=7)
        self.auto_thresh.insert(tk.END, "3")
        #

        self.auto_bool = tk.BooleanVar()
        self.auto_bool.set(False)
        self.auto_check = tk.Checkbutton(
            self.frame_auto,
            variable=self.auto_bool,
            text="Run when open",
        )

    def create_frame_size(self):
        self.frame_size = ttk.Frame()
        self.create_widgets_size()
        self.create_layout_size()
        self.frame_size.pack()

    def create_widgets_size(self):
        self.original_size_text = ttk.Label(self.frame_size, text="Original size (nm)")
        self.current_size_text = ttk.Label(self.frame_size, text="Current size (nm)")
        self.original_x_text = ttk.Label(self.frame_size, text="x")
        self.current_x_text = ttk.Label(self.frame_size, text="x")
        self.original_y_text = ttk.Label(self.frame_size, text="y")
        self.current_y_text = ttk.Label(self.frame_size, text="y")
        #
        self.original_x = ttk.Entry(self.frame_size, width=7)
        self.original_x.insert(tk.END, "30")
        self.original_x.bind("<Return>", self.original_size_changed)
        #
        self.orpix_x = ttk.Label(self.frame_size, text="(- px)")
        #
        self.original_y = ttk.Entry(self.frame_size, width=7)
        self.original_y.insert(tk.END, "30")
        self.original_y.bind("<Return>", self.original_size_changed)
        #
        self.orpix_y = ttk.Label(self.frame_size, text="(- px)")
        #
        self.current_x = ttk.Label(self.frame_size, text="30")
        self.current_y = ttk.Label(self.frame_size, text="30")
        self.current_pxx = ttk.Label(self.frame_size, text="(- px)")
        self.current_pxy = ttk.Label(self.frame_size, text="(- px)")

    def create_frame_record(self):
        self.frame_record = ttk.Frame()
        self.create_widgets_record()
        self.create_layout_record()
        self.frame_record.pack()

    def create_widgets_record(self):
        self.rec_fol_name = ttk.Entry(self.frame_record, width=20)
        self.button_recfolchoice = tk.Button(
            self.frame_record,
            text="Rec. folder choice",
            command=self.rec_fol_choice_clicked,
            width=10,
        )

        self.fol_choice_text_record = ttk.Label(self.frame_record, text="Record folder")
        self.record_text = ttk.Label(self.frame_record, text="Record")
        self.record = ttk.Entry(self.frame_record, width=40)
        self.record.insert(tk.END, "---")
        self.record.bind("<Return>", self.record_name_changed)
        self.record_plus = ttk.Entry(self.frame_record, width=20)
        self.record_plus.insert(tk.END, "---")
        self.record_plus.bind("<Return>", self.record_name_changed)
        #
        self.dirdiv_bool = tk.BooleanVar()
        self.dirdiv_bool.set(True)
        self.dirdiv_check = tk.Checkbutton(
            self.frame_record,
            variable=self.dirdiv_bool,
            text="Divide folder",
        )

    def create_frame_buttons(self):
        self.frame_buttons = ttk.Frame()
        self.create_widgets_buttons()
        self.create_layout_buttons()
        self.frame_buttons.pack()

    def create_widgets_buttons(self):
        self.record_button = tk.Button(
            self.frame_buttons,
            text="Record",
            command=self.record_function,
            width=18,
            height=2,
        )
        self.record_next_button = tk.Button(
            self.frame_buttons,
            text="Rec. and next",
            command=self.record_next_function,
            width=18,
            height=2,
        )

        self.run_all_button = tk.Button(
            self.frame_buttons,
            text="Run all",
            command=self.run_all_function,
            width=18,
            height=2,
        )

        self.delete_all_button = tk.Button(
            self.frame_buttons,
            text="Delete all",
            command=self.delete_function,
            width=18,
            height=2,
        )

    def create_frame_notes(self):
        self.frame_notes = ttk.Frame()
        self.create_widgets_notes()
        self.create_layout_notes()
        self.frame_notes.pack()

    def create_widgets_notes(self):
        self.left_text = ttk.Label(self.frame_notes, text="Left click: Adding a point.")
        self.right_text = ttk.Label(
            self.frame_notes, text="Right click: Removing a point."
        )

    def create_frame_notes2(self):
        self.frame_notes2 = ttk.Frame()
        self.create_widgets_notes2()
        self.create_layout_notes2()
        self.frame_notes2.pack()

    def create_widgets_notes2(self):
        self.status_text = ttk.Label(self.frame_notes2, text="Status: None")
