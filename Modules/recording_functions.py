import cv2
import os
import csv


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
            f.write(
                "Current_size_Y:" + "\t" + str(round(self.myimage.y_current, 4)) + "\n"
            )
            f.write("Pixcel_X" + "\t" + str(self.myimage.x_current_pix) + "\n")
            f.write("Pixcel_Y" + "\t" + str(self.myimage.y_current_pix) + "\n\n")
            f.write("Smoothing" + "\t" + str(self.myimage.smooth_val) + "\n")
            f.write("Median" + "\t" + str(self.myimage.median_val) + "\n")
            f.write("Rescale" + "\t" + str(self.myimage.mag) + "\n")
            f.write("Plane_Subtraction" + "\t" + str(self.myimage.plane_bool) + "\n")
            f.write("Average_Subtraction" + "\t" + str(self.myimage.ave_bool) + "\n")
            f.write("Range_upper" + "\t" + str(self.myimage.range_u) + "\n")
            f.write("Range_lower" + "\t" + str(self.myimage.range_l) + "\n")
            f.write("Auto_range" + "\t" + str(self.myimage.auto_range) + "\n")
            f.write("Auto_thresh" + "\t" + str(self.myimage.auto_thresh) + "\n")
            f.write("Auto_duplicate" + "\t" + str(self.myimage.auto_dup) + "\n")
            f.write("Analysis_range" + "\t" + str(self.myimage.analysis_range) + "\n")
            f.write(
                "Analysis_exclusive_range"
                + "\t"
                + str(self.myimage.analysis_ex)
                + "\n\n"
            )
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
                + str(round(self.myimage.density, 5))
                + "\t"
                + "/nm2"
                + "\n\n"
            )
            f.write("Height_average" + "\t" + str(self.myimage.height_ave) + "\n")
            f.write("Height_std" + "\t" + str(self.myimage.height_std) + "\n")
            f.write("Normarized_std" + "\t" + str(self.myimage.norm_std) + "\n\n")
            f.write(
                "Defect_position:"
                + "\t"
                + "X"
                + "\t"
                + "Y"
                + "\t"
                + "Absolute_height"
                + "\t"
                + "Relative_height"
                + "\t"
                + "Analyzed_or_not"
                + "\n"
            )
            for point, abs_h, rel_h, ana_bool in zip(
                self.myimage.cut_points,
                self.myimage.abs_height,
                self.myimage.rel_height,
                self.myimage.analyzed,
            ):
                f.write(
                    str(point[0])
                    + "\t"
                    + str(point[1])
                    + "\t"
                    + str(round(abs_h, 5))
                    + "\t"
                    + str(round(rel_h, 5))
                    + "\t"
                    + str(ana_bool)
                    + "\n"
                )
            f.write("\n")
            #
            f.write("Data:" + "\n")
            for row in self.myimage.cut_image_gray:
                for values in row:
                    f.write(str(values) + "\t")
                f.write("\n")

    def recording_density(self):
        if self.dirdiv_bool.get():
            fol_add = "defect_density" + "\\"
            self.folder_check(self.dir_name_rec + "\\" + fol_add)
            txt_name = self.dir_name_rec + "\\" + fol_add + "\\density_summary" + ".csv"
        else:
            txt_name = self.dir_name_rec + "\\density_summary" + ".csv"
        if os.path.isfile(txt_name):
            pass
        else:
            with open(txt_name, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "Data name",
                        "Total area (nm2)",
                        "Defect number",
                        "Density (number/nm2)",
                        "Height average",
                        "Height std",
                        "Normarized std",
                    ]
                )

        with open(txt_name, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    self.myimage.data_path,
                    self.myimage.total_area,
                    self.myimage.defect_number,
                    self.myimage.density,
                    self.myimage.height_ave,
                    self.myimage.height_std,
                    self.myimage.norm_std,
                ]
            )

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
                + "_full.bmp"
            )
            img_name_full = (
                self.dir_name_rec
                + "\\"
                + fol_add
                + self.record.get()
                + self.record_plus.get()
                + "_cut.bmp"
            )
        else:
            img_name = (
                self.dir_name_rec
                + "\\"
                + self.record.get()
                + self.record_plus.get()
                + "_full.bmp"
            )
            img_name_full = (
                self.dir_name_rec
                + "\\"
                + self.record.get()
                + self.record_plus.get()
                + "_cut.bmp"
            )
        cv2.imwrite(img_name, self.myimage.image_show)
        cv2.imwrite(img_name_full, self.myimage.cut_image_p)
