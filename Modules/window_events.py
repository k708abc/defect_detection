#!python3.11

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
        self.imtype_choice.current(0)
        self.master.update()

    def image_open_clicked(self):
        self.image_open()

    def image_open(self):
        self.myimage.data_path = self.image_list.dir_name + "\\" + self.choice.get()
        self.myimage.channel_val = self.imtype_choice.current()
        self.myimage.channel_name = self.imtype_choice.get()
        self.myimage.points = []
        self.myimage.read_image()
        if self.auto_bool.get() is False:
            self.myimage.show_image()
        self.update_after_show()
        if self.auto_bool.get():
            self.auto_detection()

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

    def auto_set_function(self):
        pass

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

    def range_change(self, evemt):
        self.myimage.range_u = int(self.upper_set_entry.get())
        self.myimage.range_l = int(self.lower_set_entry.get())
        self.myimage.show_image()
        self.update_size()

    def upper_up(self, evemt):
        val = int(self.upper_set_entry.get()) + 1
        self.upper_set_entry.delete(0, tk.END)
        self.upper_set_entry.insert(tk.END, val)
        self.range_change(0)

    def upper_down(self, evemt):
        val = int(self.upper_set_entry.get()) - 1
        self.upper_set_entry.delete(0, tk.END)
        self.upper_set_entry.insert(tk.END, val)
        self.range_change(0)

    def lower_up(self, evemt):
        val = int(self.lower_set_entry.get()) + 1
        self.lower_set_entry.delete(0, tk.END)
        self.lower_set_entry.insert(tk.END, val)
        self.range_change(0)

    def lower_down(self, evemt):
        val = int(self.lower_set_entry.get()) - 1
        self.lower_set_entry.delete(0, tk.END)
        self.lower_set_entry.insert(tk.END, val)
        self.range_change(0)

    def rescale(self, event):
        prev = self.myimage.mag
        self.myimage.mag = float(self.rescale_all.get())
        up = int(self.upper_set_entry.get()) / prev * float(self.rescale_all.get())
        low = int(self.lower_set_entry.get()) / prev * float(self.rescale_all.get())
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

    def plane_image(self):
        self.myimage.plane = self.plane_bool.get()
        self.myimage.show_image()

    def auto_detection(self):
        self.myimage.auto_range = float(self.auto_range.get())
        self.myimage.auto_thresh = float(self.auto_thresh.get())
        self.myimage.auto_detection()

    def original_size_changed(self, event):
        self.myimage.x_size_or = float(self.original_x.get())
        self.myimage.y_size_or = float(self.original_y.get())

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
        self.myimage.prepare_cut_image()
        self.rec_text()
        self.rec_image()

    def record_next_function(self):
        self.myimage.prepare_cut_image()
        self.rec_text()
        self.rec_image()
        num = self.image_list.images.index(self.choice.get()) + 1
        if num >= len(self.image_list.images):
            num = 0
        self.choice.current(num)
        self.image_open()

    def run_all_function(self):
        self.auto_bool.set(True)
        for i in range(len(self.image_list.images)):
            self.choice.current(i)
            self.image_open()
            self.record_function()

    def delete_function(self):
        self.myimage.points = []
        self.myimage.show_image()

    def rec_text(self):
        self.recording_text()
