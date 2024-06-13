import cv2
import os
import matplotlib.pyplot as plt


class Recording:
    def folder_check(self, folder):
        if os.path.isdir(folder):
            pass
        else:
            os.makedirs(folder)

    def recording_text(self):
        if self.dirdiv_bool.get():
            fol_add = "textIm_" + self.record_plus.get() + "\\"
            self.folder_check(self.dir_name_rec + "\\" + fol_add)
            txt_name = (
                self.dir_name_rec
                + "\\"
                + fol_add
                + self.record.get()
                + self.record_plus.get()
                + ".txt"
            )
        else:
            txt_name = (
                self.dir_name_rec
                + "\\"
                + self.record.get()
                + self.record_plus.get()
                + ".txt"
            )
        with open(txt_name, mode="w") as f:
            f.write(self.record.get() + self.record_plus.get() + "\n")
            f.write("Image_name:" + "\t" + self.myimage.data_path + "\n")
            f.write("Channel:" + "\t" + self.myimage.channel_name + "\n\n")
            f.write("Original_size_X:" + "\t" + str(self.myimage.x_size_or) + "\n")
            f.write("Original_size_Y:" + "\t" + str(self.myimage.y_size_or) + "\n")
            f.write("Current_size_X:" + "\t" + str(self.myimage.x_current) + "\n")
            f.write("Current_size_Y:" + "\t" + str(self.myimage.x_current) + "\n")
            f.write("Pixcel_X" + "\t" + str(self.myimage.x_current_pix) + "\n")
            f.write("Pixcel_Y" + "\t" + str(self.myimage.y_current_pix) + "\n\n")
            f.write("Smoothing" + "\t" + str(self.myimage.smooth) + "\n")
            f.write("Median" + "\t" + str(self.myimage.median) + "\n")
            f.write("Rescale" + "\t" + str(self.myimage.mag) + "\n")
            f.write("Subtraction" + "\t" + str(self.myimage.plane) + "\n")
            f.write("Range_upper" + "\t" + str(self.myimage.range_u) + "\n")
            f.write("Range_lower" + "\t" + str(self.myimage.range_l) + "\n")
            f.write("Auto_range" + "\t" + str(self.myimage.auto_range) + "\n\n")
            f.write(
                "Total_area"
                + "\t"
                + str(round(self.myimage.total_area, 2))
                + "\t"
                + "nm2"
                + "\n"
            )
            f.write("Defect_number" + "\t" + str(self.myimage.defect_number) + "\n")
            f.write(
                "Density"
                + "\t"
                + str(round(self.myimage.density, 3))
                + "\t"
                + "/nm2"
                + "\n\n"
            )
            f.write("Defect_position:" + "\n")
            for point in self.myimage.cut_points:
                f.write(str(point[0]) + "\t" + str(point[1] + "\n"))
            f.write("\n")
            #
            f.write("Data:" + "\n")
            if self.real_shown:
                for row in self.real_image.image_mod:
                    for values in row:
                        f.write(str(values) + "\t")
                    f.write("\n")
            else:
                for row in self.fft_image.image_mod:
                    for values in row:
                        f.write(str(values) + "\t")
                    f.write("\n")

        if self.dirdiv_bool.get():
            fol_add = "Process_" + self.record_plus.get() + "\\"
            self.folder_check(self.dir_name_rec + "\\" + fol_add)
            txt_name = (
                self.dir_name_rec
                + "\\"
                + fol_add
                + self.record.get()
                + self.record_plus.get()
                + "_process.txt"
            )
        else:
            txt_name = (
                self.dir_name_rec
                + "\\"
                + self.record.get()
                + self.record_plus.get()
                + "_process.txt"
            )
        with open(txt_name, mode="w") as f:
            f.write("Process_record" + "\n")
            f.write(self.record.get() + self.record_plus.get() + "\n")
            f.write("Image_name:" + "\t" + self.image_name + "\n")
            f.write("Name_tag:" + "\t" + self.record_plus.get() + "\n\n")
            f.write("Real_params:" + "\n")
            for pro in self.processes:
                f.write(pro.rec())
            f.write("\n")

            if self.fft_func.image is not None:
                f.write(self.fft_func.rec(self.real_shown))
                for pro in self.processes_FFT:
                    f.write(pro.rec())

    def rec_image(self):
        if self.dirdiv_bool.get():
            fol_add = "BMP_" + self.record_plus.get() + "\\"
            self.folder_check(self.dir_name_rec + "\\" + fol_add)
            img_name = (
                self.dir_name_rec
                + "\\"
                + fol_add
                + self.record.get()
                + self.record_plus.get()
                + ".bmp"
            )
        else:
            img_name = (
                self.dir_name_rec
                + "\\"
                + self.record.get()
                + self.record_plus.get()
                + ".bmp"
            )
        if self.real_shown:
            cv2.imwrite(img_name, self.real_image.image_show)
        else:
            cv2.imwrite(img_name, self.fft_image.image_show)
