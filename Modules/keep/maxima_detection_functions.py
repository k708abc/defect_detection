import numpy as np
import cv2
import os
from Modules.read_sm4 import datatypes, sm4_getdata
from Modules.read_images import bmp_getdata, txt_getdata
import matplotlib.pyplot as plt
import math


def get_datatypes(data_path):
    data_type = os.path.splitext(data_path)
    if data_type[1] == ".bmp":
        return ["bmp"]
    elif data_type[1] == ".txt":
        return ["txt"]
    elif data_type[1] == ".SM4":
        return datatypes(data_path)
    elif os.path.isdir(data_type[0]):
        return "folder"


def get_image_values(data_path, channel_type):
    data_type = os.path.splitext(data_path)
    read_type = 0

    if data_type[1] == ".SM4":
        data, scan_params = sm4_getdata(data_path, channel_type)
        read_type = 1
    elif data_type[1] == ".bmp":
        data, scan_params = bmp_getdata(data_path)
        read_type = 2

    elif data_type[1] == ".txt":
        data, scan_params = txt_getdata(data_path)
        read_type = 3
    return data, scan_params, read_type


def fix_drift(image, v, w, size_x, size_y):
    ps_y = image.shape[0]
    ps_x = image.shape[1]
    v11 = 1 + v / 2 / size_x / ps_y
    v12 = v / size_y
    v21 = w / 2 / ps_y / size_x
    v22 = 1 + w / size_y
    #
    if v12 < 0:
        x_shift = -v12 * ps_y
    else:
        x_shift = 0
    # set matrix for fix thermal drift
    mat = np.array([[v11, v12, x_shift], [v21, v22, 0]], dtype=np.float32)

    # corrct the thermal drift of target image
    h, w = image.shape
    affine_img = cv2.warpAffine(image, mat, (2 * w, 2 * h))

    # calculate the edge of the image
    ax = v11 * ps_x
    ay = v21 * ps_x
    bx = v12 * ps_y
    by = v22 * ps_y
    #
    x0 = int(min(bx, 0) + x_shift)
    y0 = int(min(ay, 0))
    xmax = int(max(ax, bx, ax + bx) + x_shift)
    ymax = int(max(by, by + ay, ay))

    # crop the image
    im_crop = affine_img[y0:ymax, x0:xmax]
    x_size = 1 / ps_x * (xmax - x0)
    y_size = 1 / ps_y * (ymax - y0)
    return im_crop, x_size, y_size


def rescale(image, x_mag, y_mag, all_mag, or_x, or_y):
    width_x = int(or_y * x_mag * all_mag)
    height_y = int(or_x * y_mag * all_mag)

    modified_image = cv2.resize(image, (width_x, height_y))
    return modified_image


def average_subtraction(image):
    for rows in range(len(image)):
        average = np.average(image[rows])
        image[rows] = image[rows] - average

    return image


def get_LUT(maximum, minimum):
    LUT = LUT = np.zeros((256, 1), dtype="uint8")
    if maximum == minimum:
        for i in range(-100, 356):
            if i < maximum:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 0
            elif i == maximum:
                if i >= 0 and i <= 255:
                    LUT[i][0] = maximum
            else:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 255
    elif maximum > minimum:
        diff = 255 / (maximum - minimum)
        k = 0
        for i in range(-100, 356):
            if i < minimum:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 0
            elif i <= maximum:
                if i >= 0 and i <= 255:
                    LUT[i][0] = int(diff * k)
                k = k + 1
            else:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 255
    else:
        diff = 255 / (maximum - minimum)
        k = 0
        for i in range(-100, 356):
            if i < maximum:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 255
            elif i <= minimum:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 255 + int(diff * k)
                k = k + 1
            else:
                if i >= 0 and i <= 255:
                    LUT[i][0] = 0
    return LUT


def contrast_change(image, upper, lower):
    LUT = get_LUT(upper, lower)
    modified_image = cv2.LUT(image, LUT)
    return modified_image


def color_change(image, color_num):
    if color_num == 0:
        modified_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        modified_image = cv2.applyColorMap(image, color_num - 1)
    return modified_image


def image_form_rec(self):
    image_rec = contrast_change(
        self.image_mod,
        int(self.upper_val.get()),
        int(self.lower_val.get()),
    )
    image_rec = color_change(image_rec, self.colormap_table.index(self.cb_color.get()))
    for point in self.selected_list:
        if point != None:
            cv2.circle(image_rec, (point[0], point[1]), 2, point[2], -1)
    return image_rec


def record_point_sets(txt_name, self):
    with open(txt_name, mode="w") as f:
        f.write("Lobe_positions" + "\n")
        f.write(self.record.get() + "\n")
        for set in self.selected_points:
            f.write(set["type"] + "\n")
            f.write(
                "pixels:"
                + "\t"
                + str(set["pixels_x"])
                + "\t"
                + str(set["pixels_y"])
                + "\n"
            )
            f.write("size:" + "\t" + str(set["x"]) + "\t" + str(set["y"]) + "\n")
            f.write("Fe:" + "\t" + str(set["Fe"][0]) + "\t" + str(set["Fe"][1]) + "\n")
            f.write(
                "lobe_A:"
                + "\t"
                + str(set["lobe A"][0])
                + "\t"
                + str(set["lobe A"][1])
                + "\n"
            )
            f.write(
                "lobe_B:"
                + "\t"
                + str(set["lobe B"][0])
                + "\t"
                + str(set["lobe B"][1])
                + "\n"
            )
            f.write(
                "lobe_C:"
                + "\t"
                + str(set["lobe C"][0])
                + "\t"
                + str(set["lobe C"][1])
                + "\n"
            )
            f.write(
                "lobe_D:"
                + "\t"
                + str(set["lobe D"][0])
                + "\t"
                + str(set["lobe D"][1])
                + "\n"
            )


def record_sets(
    Fe_x,
    Fe_y,
    B_x,
    B_y,
    C_x,
    C_y,
    D_x,
    D_y,
    alpha,
    beta,
    theta,
    phi,
    t,
    h,
    error,
    txt_name,
):
    with open(txt_name, mode="w") as f:
        for i in range(len(Fe_x)):
            f.write("Molecule_" + str(i) + "\n")
            f.write("Experimental:" + "\n")
            f.write("\t" + "Fe:" + "\t" + str(Fe_x[i]) + ", " + str(Fe_y[i]) + "\n")
            f.write("\t" + "B :" + "\t" + str(B_x[i]) + ", " + str(B_y[i]) + "\n")
            f.write("\t" + "C :" + "\t" + str(C_x[i]) + ", " + str(C_y[i]) + "\n")
            f.write("\t" + "D :" + "\t" + str(D_x[i]) + ", " + str(D_y[i]) + "\n\n")
            f.write("Simulated:" + "\n")
            f.write("\t" + "Alpha" + "\t" + str(alpha[i]) + "\n")
            f.write("\t" + "Beta" + "\t" + str(beta[i]) + "\n")
            f.write("\t" + "t" + "\t" + str(t[i]) + "\n")
            f.write("\t" + "h" + "\t" + str(h[i]) + "\n")
            f.write("\t" + "error" + "\t" + str(error[i]) + "\n")
            f.write("\t" + "theta" + "\t" + str(theta[i]) + "\n")
            f.write("\t" + "phi" + "\t" + str(phi[i]) + "\n")


def record_point_set2(self):
    if self.terrace_selected:
        txt_name = "Records/" + self.record.get() + "_set_terrace.txt"
        record_sets(
            self.terrace_Fe_x,
            self.terrace_Fe_y,
            self.terrace_lobeB_x,
            self.terrace_lobeB_y,
            self.terrace_lobeC_x,
            self.terrace_lobeC_y,
            self.terrace_lobeD_x,
            self.terrace_lobeD_y,
            self.terrace_alpha_list,
            self.terrace_beta_list,
            self.terrace_theta_list,
            self.terrace_phi_list,
            self.terrace_t_list,
            self.terrace_h_list,
            self.terrace_error_list,
            txt_name,
        )
    #
    if self.edge1_selected:
        txt_name = "Records/" + self.record.get() + "_set_edge1.txt"
        record_sets(
            self.edge1_Fe_x,
            self.edge1_Fe_y,
            self.edge1_lobeB_x,
            self.edge1_lobeB_y,
            self.edge1_lobeC_x,
            self.edge1_lobeC_y,
            self.edge1_lobeD_x,
            self.edge1_lobeD_y,
            self.edge1_alpha_list,
            self.edge1_beta_list,
            self.edge1_theta_list,
            self.edge1_phi_list,
            self.edge1_t_list,
            self.edge1_h_list,
            self.edge1_error_list,
            txt_name,
        )
    #
    if self.edge2_selected:
        txt_name = "Records/" + self.record.get() + "_set_edge2.txt"
        record_sets(
            self.edge2_Fe_x,
            self.edge2_Fe_y,
            self.edge2_lobeB_x,
            self.edge2_lobeB_y,
            self.edge2_lobeC_x,
            self.edge2_lobeC_y,
            self.edge2_lobeD_x,
            self.edge2_lobeD_y,
            self.edge2_alpha_list,
            self.edge2_beta_list,
            self.edge2_theta_list,
            self.edge2_phi_list,
            self.edge2_t_list,
            self.edge2_h_list,
            self.edge2_error_list,
            txt_name,
        )


def record_images_scatter(image_name, self):
    image = image_form_rec(self)
    image_name = "Records/" + self.record.get() + ".bmp"
    cv2.imwrite(image_name, image)


def record_average_positions(txt_name, self):
    with open(txt_name, mode="w") as f:
        f.write("Average (experimrntal)" + self.record.get() + "\n\n")
        f.write("Terrace:" + "\n")
        f.write(
            "\t"
            + "A:"
            + "\t"
            + str(self.ave_terrace_A[0])
            + "\t"
            + str(self.ave_terrace_A[1])
            + "\n"
        )
        f.write(
            "\t"
            + "B:"
            + "\t"
            + str(self.ave_terrace_B[0])
            + "\t"
            + str(self.ave_terrace_B[1])
            + "\n"
        )
        f.write(
            "\t"
            + "C:"
            + "\t"
            + str(self.ave_terrace_C[0])
            + "\t"
            + str(self.ave_terrace_C[1])
            + "\n"
        )
        f.write(
            "\t"
            + "D:"
            + "\t"
            + str(self.ave_terrace_D[0])
            + "\t"
            + str(self.ave_terrace_D[1])
            + "\n"
        )
        f.write(
            "\t"
            + "Fe:"
            + "\t"
            + str(self.ave_terrace_Fe[0])
            + "\t"
            + str(self.ave_terrace_Fe[1])
            + "\n"
        )

        f.write("Edge 1:" + "\n")
        f.write(
            "\t"
            + "A:"
            + "\t"
            + str(self.ave_edge1_A[0])
            + "\t"
            + str(self.ave_edge1_A[1])
            + "\n"
        )
        f.write(
            "\t"
            + "B:"
            + "\t"
            + str(self.ave_edge1_B[0])
            + "\t"
            + str(self.ave_edge1_B[1])
            + "\n"
        )
        f.write(
            "\t"
            + "C:"
            + "\t"
            + str(self.ave_edge1_C[0])
            + "\t"
            + str(self.ave_edge1_C[1])
            + "\n"
        )
        f.write(
            "\t"
            + "D:"
            + "\t"
            + str(self.ave_edge1_D[0])
            + "\t"
            + str(self.ave_edge1_D[1])
            + "\n"
        )
        f.write(
            "\t"
            + "Fe:"
            + "\t"
            + str(self.ave_edge1_Fe[0])
            + "\t"
            + str(self.ave_edge1_Fe[1])
            + "\n"
        )
        f.write("Edge 2:" + "\n")
        f.write(
            "\t"
            + "A:"
            + "\t"
            + str(self.ave_edge2_A[0])
            + "\t"
            + str(self.ave_edge2_A[1])
            + "\n"
        )
        f.write(
            "\t"
            + "B:"
            + "\t"
            + str(self.ave_edge2_B[0])
            + "\t"
            + str(self.ave_edge2_B[1])
            + "\n"
        )
        f.write(
            "\t"
            + "C:"
            + "\t"
            + str(self.ave_edge2_C[0])
            + "\t"
            + str(self.ave_edge2_C[1])
            + "\n"
        )
        f.write(
            "\t"
            + "D:"
            + "\t"
            + str(self.ave_edge2_D[0])
            + "\t"
            + str(self.ave_edge2_D[1])
            + "\n"
        )
        f.write(
            "\t"
            + "Fe:"
            + "\t"
            + str(self.ave_edge2_Fe[0])
            + "\t"
            + str(self.ave_edge2_Fe[1])
            + "\n"
        )


def record_all_positions(self):
    if self.terrace_selected:
        txt_name = "Records/" + self.record.get() + "_lists_terrace.txt"
        with open(txt_name, mode="w") as f:
            f.write("Experimental position of lobes" + "\n")
            f.write("x" + "\t" + "y" + "\n")
            f.write("lobe A:" + "\n")
            for pos_x, pos_y in zip(self.terrace_lobeA_x, self.terrace_lobeA_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe B:" + "\n")
            for pos_x, pos_y in zip(self.terrace_lobeB_x, self.terrace_lobeB_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe C:" + "\n")
            for pos_x, pos_y in zip(self.terrace_lobeC_x, self.terrace_lobeC_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe D:" + "\n")
            for pos_x, pos_y in zip(self.terrace_lobeD_x, self.terrace_lobeD_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe Fe:" + "\n")
            for pos_x, pos_y in zip(self.terrace_Fe_x, self.terrace_Fe_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
    #
    if self.edge1_selected:
        txt_name = "Records/" + self.record.get() + "_lists_edge1.txt"
        with open(txt_name, mode="w") as f:
            f.write("Experimental position of lobes" + "\n")
            f.write("x" + "\t" + "y" + "\n")
            f.write("lobe A:" + "\n")
            for pos_x, pos_y in zip(self.edge1_lobeA_x, self.edge1_lobeA_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe B:" + "\n")
            for pos_x, pos_y in zip(self.edge1_lobeB_x, self.edge1_lobeB_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe C:" + "\n")
            for pos_x, pos_y in zip(self.edge1_lobeC_x, self.edge1_lobeC_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe D:" + "\n")
            for pos_x, pos_y in zip(self.edge1_lobeD_x, self.edge1_lobeD_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("Fe:" + "\n")
            for pos_x, pos_y in zip(self.edge1_Fe_x, self.edge1_Fe_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
    #
    if self.edge2_selected:
        txt_name = "Records/" + self.record.get() + "_lists_edge2.txt"
        with open(txt_name, mode="w") as f:
            f.write("Experimental position of lobes" + "\n")
            f.write("x" + "\t" + "y" + "\n")
            f.write("lobe A:" + "\n")
            for pos_x, pos_y in zip(self.edge2_lobeA_x, self.edge2_lobeA_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe B:" + "\n")
            for pos_x, pos_y in zip(self.edge2_lobeB_x, self.edge2_lobeB_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe C:" + "\n")
            for pos_x, pos_y in zip(self.edge2_lobeC_x, self.edge2_lobeC_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("lobe D:" + "\n")
            for pos_x, pos_y in zip(self.edge2_lobeD_x, self.edge2_lobeD_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")
            f.write("Fe:" + "\n")
            for pos_x, pos_y in zip(self.edge2_Fe_x, self.edge2_Fe_y):
                f.write(str(pos_x) + "\t" + str(pos_y) + "\n")


def recording_scatter(self):
    if os.path.exists("Records/"):
        pass
    else:
        os.mkdir("Records/")
    #
    txt_name = "Records/" + self.record.get() + ".txt"
    record_point_sets(txt_name, self)
    image_name = "Records/" + self.record.get() + ".bmp"
    record_images_scatter(image_name, self)
    txt_name = "Records/" + self.record.get() + "_ave.txt"
    record_average_positions(txt_name, self)
    record_all_positions(self)


def angular_scatter_img(x_list, y_list, ave_x, ave_y, x_label, y_label, fig_name):
    fig = plt.figure()
    plt.scatter(x_list, y_list)
    plt.scatter([ave_x], [ave_y], marker="x")
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.axis("square")
    plt.savefig(fig_name)


def angular_scatter_txt(
    x_list, y_list, ave_x, ave_y, x_label, y_label, txt_name, label
):
    with open(txt_name, mode="w") as f:
        f.write("Label" + "\t" + x_label + "\t" + y_label + "\n")
        i = 1
        for x, y in zip(x_list, y_list):
            f.write(label + "_" + str(i) + "\t" + str(x) + "\t" + str(y) + "\n")
            i += 1


def angular_scatter(self):
    if self.terrace_selected:
        fig_name = "Records/" + self.record.get() + "_alpha_beta_teracce.png"
        angular_scatter_img(
            self.terrace_alpha_list,
            self.terrace_beta_list,
            self.terrace_alpha_ave,
            self.terrace_beta_ave,
            "Alpha",
            "Beta",
            fig_name,
        )
        txt_name = "Records/" + self.record.get() + "_alpha_beta_teracce.txt"
        angular_scatter_txt(
            self.terrace_alpha_list,
            self.terrace_beta_list,
            self.terrace_alpha_ave,
            self.terrace_beta_ave,
            "Alpha",
            "Beta",
            txt_name,
            "terrace",
        )
        #
        fig_name = "Records/" + self.record.get() + "_theta_phi_teracce.png"
        #
        angular_scatter_img(
            self.terrace_theta_list,
            self.terrace_phi_list,
            self.terrace_theta_ave,
            self.terrace_phi_ave,
            "Theta",
            "Phi",
            fig_name,
        )
        txt_name = "Records/" + self.record.get() + "_theta_phi_teracce.txt"
        angular_scatter_txt(
            self.terrace_theta_list,
            self.terrace_phi_list,
            self.terrace_theta_ave,
            self.terrace_phi_ave,
            "Theta",
            "Phi",
            txt_name,
            "terrace",
        )
    #
    if self.edge1_selected:
        fig_name = "Records/" + self.record.get() + "_alpha_beta_edge1.png"
        angular_scatter_img(
            self.edge1_alpha_list,
            self.edge1_beta_list,
            self.edge1_alpha_ave,
            self.edge1_beta_ave,
            "Alpha",
            "Beta",
            fig_name,
        )
        txt_name = "Records/" + self.record.get() + "_alpha_beta_edge1.txt"
        angular_scatter_txt(
            self.edge1_alpha_list,
            self.edge1_beta_list,
            self.edge1_alpha_ave,
            self.edge1_beta_ave,
            "Alpha",
            "Beta",
            txt_name,
            "edge1",
        )
        #
        fig_name = "Records/" + self.record.get() + "_theta_phi_edge1.png"
        #
        angular_scatter_img(
            self.edge1_theta_list,
            self.edge1_phi_list,
            self.edge1_theta_ave,
            self.edge1_phi_ave,
            "Theta",
            "Phi",
            fig_name,
        )
        txt_name = "Records/" + self.record.get() + "_theta_phi_edge1.txt"
        angular_scatter_txt(
            self.edge1_theta_list,
            self.edge1_phi_list,
            self.edge1_theta_ave,
            self.edge1_phi_ave,
            "Theta",
            "Phi",
            txt_name,
            "edge1",
        )
    #
    #
    if self.edge2_selected:
        fig_name = "Records/" + self.record.get() + "_alpha_beta_edge2.png"
        angular_scatter_img(
            self.edge2_alpha_list,
            self.edge2_beta_list,
            self.edge2_alpha_ave,
            self.edge2_beta_ave,
            "Alpha",
            "Beta",
            fig_name,
        )
        txt_name = "Records/" + self.record.get() + "_alpha_beta_edge2.txt"
        angular_scatter_txt(
            self.edge2_alpha_list,
            self.edge2_beta_list,
            self.edge2_alpha_ave,
            self.edge2_beta_ave,
            "Alpha",
            "Beta",
            txt_name,
            "edge2",
        )
        #
        fig_name = "Records/" + self.record.get() + "_theta_phi_edge2.png"
        #
        angular_scatter_img(
            self.edge2_theta_list,
            self.edge2_phi_list,
            self.edge2_theta_ave,
            self.edge2_phi_ave,
            "Theta",
            "Phi",
            fig_name,
        )
        txt_name = "Records/" + self.record.get() + "_theta_phi_edge2.txt"
        angular_scatter_txt(
            self.edge2_theta_list,
            self.edge2_phi_list,
            self.edge2_theta_ave,
            self.edge2_phi_ave,
            "Theta",
            "Phi",
            txt_name,
            "edge2",
        )
    #


def results_summary(self):
    txt_name = "Records/" + self.record.get() + "_summary.txt"
    with open(txt_name, mode="w") as f:
        f.write("Image name:" + "\t" + self.choise.get() + "\n")
        f.write("Channel:" + "\t" + self.imtype_choise.get() + "\n")
        f.write("color:" + "\t" + self.cb_color.get() + "\n")
        f.write("Smoothing:" + "\t" + self.smooth_entry.get() + "\n")
        f.write("Median:" + "\t" + self.cb_median.get() + "\n")
        f.write("Fix drift:" + "\t" + str(self.drift_bool.get()) + "\n")
        f.write("\t" + "v:" + "\t" + self.drift_dx.get() + "\n")
        f.write("\t" + "w:" + "\t" + self.drift_dy.get() + "\n")
        f.write("Rescale All:" + "\t" + self.rescale_all.get() + "\n")
        f.write("Rescale x:" + "\t" + self.rescale_x.get() + "\n")
        f.write("Rescale y:" + "\t" + self.rescale_y.get() + "\n")
        f.write("Average subtravt:" + "\t" + str(self.average_bool.get()) + "\n")
        f.write("Original x:" + "\t" + self.original_x.get() + "\n")
        f.write("Original y:" + "\t" + self.original_y.get() + "\n")
        if self.old_method_bool.get():
            f.write("Method:" + "\t" + "From three lobes" + "\n")
        else:
            f.write("Method:" + "\t" + "From all points" + "\n")
        if self.from_fe_bool.get():
            f.write("Graph style:" + "\t" + "From Fe")
        else:
            f.write("Graph style:" + "\t" + "From center of CD")
        if self.rotate_CD_bool.get():
            f.write(", " + "set CD along y")
        f.write("\n\n")

        if self.terrace_selected:
            f.write("Terrace" + "\n")
            f.write("Number of molecules: " + "\t" + str(len(self.terrace_Fe_x)) + "\n")
            f.write("Average position (experiment)" + "\n")
            f.write(
                "B:"
                + "\t"
                + "("
                + str(self.ave_terrace_B[0])
                + ", "
                + str(self.ave_terrace_B[1])
                + ")"
                + "\n"
            )
            f.write(
                "C:"
                + "\t"
                + "("
                + str(self.ave_terrace_C[0])
                + ", "
                + str(self.ave_terrace_C[1])
                + ")"
                + "\n"
            )
            f.write(
                "D:"
                + "\t"
                + "("
                + str(self.ave_terrace_D[0])
                + ", "
                + str(self.ave_terrace_D[1])
                + ")"
                + "\n"
            )
            f.write(
                "Fe:"
                + "\t"
                + "("
                + str(self.ave_terrace_Fe[0])
                + ", "
                + str(self.ave_terrace_Fe[1])
                + ")"
                + "\n"
            )
            f.write("Averaged simulated values:" + "\n")
            f.write(
                "B:"
                + "\t"
                + str(self.terrace_ave_pos_x[0])
                + ", "
                + str(self.terrace_ave_pos_y[0])
                + "\n"
            )
            f.write(
                "C:"
                + "\t"
                + str(self.terrace_ave_pos_x[1])
                + ", "
                + str(self.terrace_ave_pos_y[1])
                + "\n"
            )
            f.write(
                "D:"
                + "\t"
                + str(self.terrace_ave_pos_x[2])
                + ", "
                + str(self.terrace_ave_pos_y[2])
                + "\n"
            )
            f.write(
                "T:"
                + "\t"
                + str(self.terrace_ave_pos_x[3])
                + ", "
                + str(self.terrace_ave_pos_y[3])
                + "\n"
            )
            f.write(
                "Alpha:"
                + "\t"
                + str(self.terrace_alpha_ave)
                + "\t"
                + "("
                + str(self.terrace_alpha_std)
                + ")"
                + "\n"
            )
            f.write(
                "Beta:"
                + "\t"
                + str(self.terrace_beta_ave)
                + "\t"
                + "("
                + str(self.terrace_beta_std)
                + ")"
                + "\n"
            )
            f.write(
                "t:"
                + "\t"
                + str(self.terrace_t_ave)
                + "\t"
                + "("
                + str(self.terrace_t_std)
                + ")"
                + "\n"
            )
            f.write(
                "h:"
                + "\t"
                + str(self.terrace_h_ave)
                + "\t"
                + "("
                + str(self.terrace_h_std)
                + ")"
                + "\n"
            )
            f.write(
                "error:"
                + "\t"
                + str(self.terrace_error_ave)
                + "\t"
                + "("
                + str(self.terrace_error_std)
                + ")"
                + "\n"
            )
            f.write(
                "theta:"
                + "\t"
                + str(self.terrace_theta_ave)
                + "\t"
                + "("
                + str(self.terrace_theta_std)
                + ")"
                + "\n"
            )
            f.write(
                "phi:"
                + "\t"
                + str(self.terrace_phi_ave)
                + "\t"
                + "("
                + str(self.terrace_phi_std)
                + ")"
                + "\n\n"
            )
        if self.edge1_selected:
            f.write("Edge1" + "\n")
            f.write("Number of molecules: " + "\t" + str(len(self.edge1_Fe_x)) + "\n")
            f.write("Average position (experiment)" + "\n")
            f.write(
                "B:"
                + "\t"
                + "("
                + str(self.ave_edge1_B[0])
                + ", "
                + str(self.ave_edge1_B[1])
                + ")"
                + "\n"
            )
            f.write(
                "C:"
                + "\t"
                + "("
                + str(self.ave_edge1_C[0])
                + ", "
                + str(self.ave_edge1_C[1])
                + ")"
                + "\n"
            )
            f.write(
                "D:"
                + "\t"
                + "("
                + str(self.ave_edge1_D[0])
                + ", "
                + str(self.ave_edge1_D[1])
                + ")"
                + "\n"
            )
            f.write(
                "Fe:"
                + "\t"
                + "("
                + str(self.ave_edge1_Fe[0])
                + ", "
                + str(self.ave_edge1_Fe[1])
                + ")"
                + "\n"
            )
            f.write("Averaged simulated values:" + "\n")
            f.write(
                "B:"
                + "\t"
                + str(self.edge1_ave_pos_x[0])
                + ", "
                + str(self.edge1_ave_pos_y[0])
                + "\n"
            )
            f.write(
                "C:"
                + "\t"
                + str(self.edge1_ave_pos_x[1])
                + ", "
                + str(self.edge1_ave_pos_y[1])
                + "\n"
            )
            f.write(
                "D:"
                + "\t"
                + str(self.edge1_ave_pos_x[2])
                + ", "
                + str(self.edge1_ave_pos_y[2])
                + "\n"
            )
            f.write(
                "T:"
                + "\t"
                + str(self.edge1_ave_pos_x[3])
                + ", "
                + str(self.edge1_ave_pos_y[3])
                + "\n"
            )
            f.write(
                "Alpha:"
                + "\t"
                + str(self.edge1_alpha_ave)
                + "\t"
                + "("
                + str(self.edge1_alpha_std)
                + ")"
                + "\n"
            )
            f.write(
                "Beta:"
                + "\t"
                + str(self.edge1_beta_ave)
                + "\t"
                + "("
                + str(self.edge1_beta_std)
                + ")"
                + "\n"
            )
            f.write(
                "t:"
                + "\t"
                + str(self.edge1_t_ave)
                + "\t"
                + "("
                + str(self.edge1_t_std)
                + ")"
                + "\n"
            )
            f.write(
                "h:"
                + "\t"
                + str(self.edge1_h_ave)
                + "\t"
                + "("
                + str(self.edge1_h_std)
                + ")"
                + "\n"
            )
            f.write(
                "error:"
                + "\t"
                + str(self.edge1_error_ave)
                + "\t"
                + "("
                + str(self.edge1_error_std)
                + ")"
                + "\n"
            )
            f.write(
                "theta:"
                + "\t"
                + str(self.edge1_theta_ave)
                + "\t"
                + "("
                + str(self.edge1_theta_std)
                + ")"
                + "\n"
            )
            f.write(
                "phi:"
                + "\t"
                + str(self.edge1_phi_ave)
                + "\t"
                + "("
                + str(self.edge1_phi_std)
                + ")"
                + "\n\n"
            )
        if self.edge2_selected:
            f.write("Edge2" + "\n")
            f.write("Number of molecules: " + "\t" + str(len(self.edge2_Fe_x)) + "\n")
            f.write("Average position (experiment)" + "\n")
            f.write(
                "B:"
                + "\t"
                + "("
                + str(self.ave_edge2_B[0])
                + ", "
                + str(self.ave_edge2_B[1])
                + ")"
                + "\n"
            )
            f.write(
                "C:"
                + "\t"
                + "("
                + str(self.ave_edge2_C[0])
                + ", "
                + str(self.ave_edge2_C[1])
                + ")"
                + "\n"
            )
            f.write(
                "D:"
                + "\t"
                + "("
                + str(self.ave_edge2_D[0])
                + ", "
                + str(self.ave_edge2_D[1])
                + ")"
                + "\n"
            )
            f.write(
                "Fe:"
                + "\t"
                + "("
                + str(self.ave_edge2_Fe[0])
                + ", "
                + str(self.ave_edge2_Fe[1])
                + ")"
                + "\n"
            )
            f.write("Averaged simulated values:" + "\n")
            f.write(
                "B:"
                + "\t"
                + str(self.edge2_ave_pos_x[0])
                + ", "
                + str(self.edge2_ave_pos_y[0])
                + "\n"
            )
            f.write(
                "C:"
                + "\t"
                + str(self.edge2_ave_pos_x[1])
                + ", "
                + str(self.edge2_ave_pos_y[1])
                + "\n"
            )
            f.write(
                "D:"
                + "\t"
                + str(self.edge2_ave_pos_x[2])
                + ", "
                + str(self.edge2_ave_pos_y[2])
                + "\n"
            )
            f.write(
                "T:"
                + "\t"
                + str(self.edge2_ave_pos_x[3])
                + ", "
                + str(self.edge2_ave_pos_y[3])
                + "\n"
            )
            f.write(
                "Alpha:"
                + "\t"
                + str(self.edge2_alpha_ave)
                + "\t"
                + "("
                + str(self.edge2_alpha_std)
                + ")"
                + "\n"
            )
            f.write(
                "Beta:"
                + "\t"
                + str(self.edge2_beta_ave)
                + "\t"
                + "("
                + str(self.edge2_beta_std)
                + ")"
                + "\n"
            )
            f.write(
                "t:"
                + "\t"
                + str(self.edge2_t_ave)
                + "\t"
                + "("
                + str(self.edge2_t_std)
                + ")"
                + "\n"
            )
            f.write(
                "h:"
                + "\t"
                + str(self.edge2_h_ave)
                + "\t"
                + "("
                + str(self.edge2_h_std)
                + ")"
                + "\n"
            )
            f.write(
                "error:"
                + "\t"
                + str(self.edge2_error_ave)
                + "\t"
                + "("
                + str(self.edge2_error_std)
                + ")"
                + "\n"
            )
            f.write(
                "theta:"
                + "\t"
                + str(self.edge2_theta_ave)
                + "\t"
                + "("
                + str(self.edge2_theta_std)
                + ")"
                + "\n"
            )
            f.write(
                "phi:"
                + "\t"
                + str(self.edge2_phi_ave)
                + "\t"
                + "("
                + str(self.edge2_phi_std)
                + ")"
                + "\n"
            )


def cal_Fe_pos(alpha_list, beta_list, h_list):
    x_list = []
    y_list = []
    for alpha, beta, h in zip(alpha_list, beta_list, h_list):
        x_list.append(-h * math.sin(beta))
        y_list.append(-h * math.sin(alpha) * math.cos(beta))
    return x_list, y_list


def Fe_scatter_fig(alpha_list, beta_list, h_list, fig_name):
    fig = plt.figure()
    x_list, y_list = cal_Fe_pos(alpha_list, beta_list, h_list)
    plt.scatter(x_list, y_list)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("square")
    plt.savefig(fig_name)


def Fe_scatter(self):
    if self.terrace_selected:
        fig_name = "Records/" + self.record.get() + "_Fe_terrace.png"
        Fe_scatter_fig(
            self.terrace_alpha_list,
            self.terrace_beta_list,
            self.terrace_h_list,
            fig_name,
        )
    if self.edge1_selected:
        fig_name = "Records/" + self.record.get() + "_Fe_edge1.png"
        Fe_scatter_fig(
            self.edge1_alpha_list, self.edge1_beta_list, self.edge1_h_list, fig_name
        )
    if self.edge2_selected:
        fig_name = "Records/" + self.record.get() + "_Fe_edge2.png"
        Fe_scatter_fig(
            self.edge2_alpha_list, self.edge2_beta_list, self.edge2_h_list, fig_name
        )


def record_estimates(self):
    Fe_scatter(self)
    angular_scatter(self)
    record_point_set2(self)
    results_summary(self)
