#!python3.9

import tkinter as tk
import os
from tkinter import filedialog
import pathlib


class Events:
    def images_update(self, dir_name):
        self.image_list.dir_name = dir_name
        self.image_list.formlist()
        self.choice["values"] = self.image_list.images
        if len(self.image_list.images) > 0:
            self.choice.current(0)
            self.button_imopen["state"] = tk.NORMAL
            self.imtype_choice["values"] = self.image_list.types[0]
            if self.image_list.data_type[self.choice.current()] == "SM4":
                self.imtype_choice.current(self.image_list.default)
            else:
                self.imtype_choice.current(0)
        else:
            self.button_imopen["state"] = tk.DISABLED

    def record_fol_function(self):
        if self.rec_fol:
            self.dir_name_rec = self.image_list.dir_name
            self.rec_fol_name.delete(0, tk.END)
            self.rec_fol_name.insert(tk.END, self.dir_name_rec)

    def fol_choice_clicked(self):
        abs_pass = pathlib.Path(filedialog.askdirectory(initialdir=self.dir_name))
        if abs_pass == pathlib.Path("."):
            return
        fol_dir = os.path.relpath(abs_pass, self.init_dir)
        self.images_update(fol_dir)
        self.fol_name.delete(0, tk.END)
        self.fol_name.insert(tk.END, fol_dir)
        self.record_fol_function()
        self.master.update()

    def choice_selected(self, event):
        self.button_imopen["state"] = tk.NORMAL
        self.imtype_choice["values"] = self.image_list.types[self.choice.current()]
        if self.image_list.data_type[self.choice.current()] == "SM4":
            self.imtype_choice.current(self.image_list.default)
        else:
            self.imtype_choice.current(0)
        self.master.update()

    def type_choice_selected(self, event):
        if self.image_list.data_type[self.choice.current()] == "SM4":
            self.image_list.default = self.imtype_choice.current()

    def image_open_clicked(self):
        self.image_open()

    def image_open(self):
        self.update_all_params()
        self.myimage.data_path = self.image_list.dir_name + "\\" + self.choice.get()
        self.myimage.channel_val = self.imtype_choice.current()
        self.myimage.channel_name = self.imtype_choice.get()
        self.myimage.points = []
        self.myimage.read_image()
        self.update_max()
        self.status_text["text"] = "Status: Image opened"
        self.myimage.show_image()
        self.update_after_show()
        self.update_size()
        if self.auto_bool.get():
            self.myimage.show_image()
            self.auto_detection()
            self.status_text["text"] = "Status: Image opened with auto detection."

    def cb_color_selected(self, event):
        self.myimage.color_num = self.colormap_table.index(self.cb_color.get())
        self.myimage.show_image()

    def upper_value_change(self, *args):
        self.myimage.upper = int(self.upper_val.get())
        self.myimage.show_image()

    def lower_value_change(self, *args):
        self.myimage.lower = int(self.lower_val.get())
        self.myimage.show_image()

    def default_function(self):
        self.upper_val.set(255)
        self.lower_val.set(0)
        self.myimage.back_default()
        self.myimage.show_image()

    def set_default_function(self):
        self.myimage.set_default()
        self.upper_val.set(255)
        self.lower_val.set(0)

    def smooth_change(self, event):
        if self.smooth_entry.get() == "":
            self.smooth_entry.insert(tk.END, "0")
        self.myimage.smooth_val = float(self.smooth_entry.get())
        self.myimage.show_image()

    def median_change(self, event):
        if self.median_entry.get() == "":
            self.median_entry.insert(tk.END, "0")
        self.myimage.median_val = float(self.median_entry.get())
        self.myimage.show_image()

    def analysis_range_change(self, event):
        if self.analysis_range.get() == "":
            self.analysis_range.insert(tk.END, "1")
        self.myimage.analysis_range = float(self.analysis_range.get())
        self.myimage.show_image()

    def range_change(self, evemt):
        self.myimage.range_u = int(self.upper_set_entry.get())
        self.myimage.range_l = int(self.lower_set_entry.get())
        if self.myimage.open_bool:
            self.myimage.show_image()
            self.update_size()

    def update_all_params(self):
        self.myimage.smooth_val = float(self.smooth_entry.get())
        self.myimage.median_val = float(self.median_entry.get())
        self.myimage.analysis_range = float(self.analysis_range.get())
        self.myimage.analysis_ex = float(self.analysis_ex.get())
        self.myimage.plane_bool = self.plane_bool.get()
        self.myimage.ave_bool = self.ave_bool.get()

    def update_max(self):
        self.range_text["text"] = (
            "Range of image (max: " + str(self.myimage.range_max) + ")"
        )
        self.master.update()

    def upper_up(self, evemt):
        val = int(self.upper_set_entry.get()) + 1
        if val > self.myimage.range_max:
            val = self.myimage.range_max
        self.upper_set_entry.delete(0, tk.END)
        self.upper_set_entry.insert(tk.END, val)
        self.range_change(0)

    def upper_down(self, evemt):
        val = int(self.upper_set_entry.get()) - 1
        if val < int(self.lower_set_entry.get()):
            val = int(self.lower_set_entry.get())
        self.upper_set_entry.delete(0, tk.END)
        self.upper_set_entry.insert(tk.END, val)
        self.range_change(0)

    def lower_up(self, evemt):
        val = int(self.lower_set_entry.get()) + 1
        if val > int(self.upper_set_entry.get()):
            val = int(self.upper_set_entry.get())
        self.lower_set_entry.delete(0, tk.END)
        self.lower_set_entry.insert(tk.END, val)
        self.range_change(0)

    def lower_down(self, evemt):
        val = int(self.lower_set_entry.get()) - 1
        if val < 0:
            val = 0
        self.lower_set_entry.delete(0, tk.END)
        self.lower_set_entry.insert(tk.END, val)
        self.range_change(0)

    def rescale(self, event):
        prev = self.myimage.mag
        self.myimage.mag = float(self.rescale_all.get())
        up = int(self.upper_set_entry.get()) / prev * float(self.rescale_all.get())
        low = int(self.lower_set_entry.get()) / prev * float(self.rescale_all.get())
        self.myimage.range_max = int(
            self.myimage.range_max / prev * float(self.rescale_all.get())
        )
        points = []
        for point in self.myimage.points:
            points.append(
                (
                    int(point[0] / prev * float(self.rescale_all.get())),
                    int(point[1] / prev * float(self.rescale_all.get())),
                )
            )
        self.myimage.points = points
        self.upper_set_entry.delete(0, tk.END)
        self.upper_set_entry.insert(tk.END, int(up))
        self.lower_set_entry.delete(0, tk.END)
        self.lower_set_entry.insert(tk.END, int(low))
        self.range_change(0)
        self.update_max()

    def plane_image(self):
        self.myimage.plane_bool = self.plane_bool.get()
        self.myimage.show_image()

    def ave_image(self):
        self.myimage.ave_bool = self.ave_bool.get()
        self.myimage.show_image()

    def auto_detection(self):
        if self.myimage.open_bool:
            self.myimage.auto_range = float(self.auto_range.get())
            self.myimage.auto_thresh = float(self.auto_thresh.get())
            self.myimage.auto_dup = float(self.auto_dup.get())
            self.update_all_params()
            self.myimage.auto_detection()
            self.status_text["text"] = "Status: Auto detection done"

    def original_size_changed(self, event):
        self.myimage.x_size_or = float(self.original_x.get())
        self.myimage.y_size_or = float(self.original_y.get())
        self.update_size()
        self.myimage.show_image()

    def rec_fol_choice_clicked(self):
        self.rec_fol = False
        abs_pass = pathlib.Path(filedialog.askdirectory(initialdir=self.dir_name_rec))
        if abs_pass == pathlib.Path("."):
            return
        self.dir_name_rec = os.path.relpath(abs_pass, self.init_dir)
        self.rec_fol_name.delete(0, tk.END)
        self.rec_fol_name.insert(tk.END, self.dir_name_rec)

    def record_name_changed(self, event):
        self.name_change = True

    def record_function(self):
        if self.myimage.open_bool:
            self.update_all_params()
            self.status_text["text"] = "Status: Recording..."
            self.myimage.prepare_cut_image()
            self.myimage.defect_analysis()
            self.rec_text()
            self.rec_image()
            self.status_text["text"] = "Status: Recorded."

    def record_next_function(self):
        if self.myimage.open_bool:
            self.myimage.prepare_cut_image()
            self.myimage.defect_analysis()
            self.rec_text()
            self.rec_image()
            num = self.image_list.images.index(self.choice.get()) + 1
            if num >= len(self.image_list.images):
                num = 0
            self.choice.current(num)
            self.image_open()

    def run_all_function(self):
        self.auto_bool.set(True)
        self.status_text["text"] = "Status: Run all runnning..."
        for i in range(len(self.image_list.images)):
            self.choice.current(i)
            self.choice_selected(0)
            self.image_open()
            self.record_function()
        self.status_text["text"] = "Status: Run all finished"

    def delete_function(self):
        self.myimage.points = []
        self.myimage.show_image()

    def rec_text(self):
        self.recording_text()
        self.recording_density()
