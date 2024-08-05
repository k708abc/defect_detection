#!python3.12

import os
import tkinter as tk
from Modules.image_class import MyImage, ImageList


class Functions:
    def run(self) -> None:
        self.mainloop()

    def init_setting(self):
        self.myimage = MyImage()
        self.image_list = ImageList()
        self.init_dir = os.getcwd()
        self.dir_name = os.getcwd()
        self.dir_name_rec = os.getcwd()
        self.rec_fol = True
        self.name_change = False

    def record_name_base(self):
        if self.name_change is False:
            self.image_name = self.choice.get()
            self.channel_type = self.imtype_choice.current()
            self.rec_name = (
                self.image_name.replace(os.path.splitext(self.myimage.data_path)[1], "")
                + "_"
                + str(self.channel_type)
            )
            self.record.delete(0, tk.END)
            self.record.insert(tk.END, self.rec_name)

    def check_name(self):
        if self.record_plus.get() not in ("_processed", "_FFT", "---"):
            self.name_change = True

    def record_name_real(self):
        self.check_name()
        if self.name_change is False:
            self.record_plus.delete(0, tk.END)
            self.record_plus.insert(tk.END, "_processed")

    def update_after_show(self):
        self.record_name_base()
        self.record_name_real()
        self.original_x.delete(0, tk.END)
        self.original_x.insert(tk.END, self.myimage.x_size_or)
        self.original_y.delete(0, tk.END)
        self.original_y.insert(tk.END, self.myimage.y_size_or)
        self.orpix_x["text"] = "(" + str(self.myimage.x_pix_or) + " px)"
        self.orpix_y["text"] = "(" + str(self.myimage.y_pix_or) + " px)"
        self.upper_set_entry.delete(0, tk.END)
        self.upper_set_entry.insert(tk.END, self.myimage.y_current_pix)
        self.lower_set_entry.delete(0, tk.END)
        self.lower_set_entry.insert(tk.END, "0")
        self.current_x["text"] = self.myimage.x_size_or
        self.current_y["text"] = self.myimage.y_size_or
        self.current_pxx["text"] = "(" + str(self.myimage.x_pix_or) + " px)"
        self.current_pxy["text"] = "(" + str(self.myimage.y_pix_or) + " px)"

    def update_size(self):
        self.current_x["text"] = str(round(self.myimage.x_current, 2))
        self.current_y["text"] = str(round(self.myimage.y_current, 2))
        self.current_pxx["text"] = "(" + str(self.myimage.x_current_pix) + " px)"
        self.current_pxy["text"] = "(" + str(self.myimage.y_current_pix) + " px)"
