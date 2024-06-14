#!python3.11
import numpy as np
import os
from spym.io import rhksm4
from PIL import Image
import cv2
import glob
from scipy import ndimage


class ImageList:
    dir_name = None
    images = []
    types = []

    def formlist(self):
        image_list = [
            os.path.basename(pathname) for pathname in glob.glob(self.dir_name + "\*")
        ]
        self.images = []
        self.types = []
        for file in image_list:
            data_path = self.dir_name + "\\" + file
            data_type = os.path.splitext(data_path)
            if data_type[1] == ".bmp":
                self.images.append(file)
                self.types.append(["bmp"])
            elif data_type[1] == ".txt":
                check = self.check_text(data_path)
                if check:
                    self.images.append(file)
                    self.types.append(["txt"])
            elif data_type[1] == ".SM4":
                self.images.append(file)
                self.types.append(self.datatypes(data_path))
            else:
                pass

    def check_text(self, data_path):
        with open(data_path) as f:
            lines = f.readlines()
            for line in lines:
                values = line.split()
                if "Data:" in values:
                    return True
        return False

    def datatypes(self, data_path):
        sm4data = rhksm4.load(data_path)
        data_type_list = []
        data_num = 0
        while True:
            try:
                data = sm4data[data_num]
                data_type_list.append(
                    data.attrs["RHK_Label"] + "(" + data.attrs["RHK_ScanTypeName"] + ")"
                )
                data_num += 1
            except:
                break
        return data_type_list


class MyImage:
    image_or = None
    image_show = None
    x_pix_or = None
    y_pix_or = None
    x_size_or = None
    y_size_or = None
    upper = 255
    lower = 0
    data_path = None
    channel_name = None
    channel_val = None
    open_bool = False
    max_contrast = 255
    min_contrast = 0
    color_num = 2
    image_name = "Target"
    mouse_x = 0
    mouse_y = 0
    #
    smooth_val = 1
    median_val = 1
    plane_bool = False
    mag = 1
    range_u = 100
    range_l = 0
    points = []
    cut_points = []
    rel_height = []
    c_size = 5
    auto_range = 2
    auto_thresh = 2
    #
    rel_height = []
    height_ave = None
    height_std = None
    norm_std = None

    @property
    def x_current(self):
        return self.x_size_or

    @property
    def y_current(self):
        return self.y_size_or * (self.y_current_pix) / (self.y_pix_or * self.mag)

    @property
    def x_current_pix(self):
        return int(self.x_pix_or * self.mag)

    @property
    def y_current_pix(self):
        return abs(self.range_u - self.range_l)

    @property
    def total_area(self):
        return self.x_current * self.y_current

    @property
    def defect_number(self):
        return len(self.points)

    @property
    def density(self):
        return self.defect_number / self.total_area

    @property
    def pix_range(self):
        return int(self.auto_range / self.x_current * self.x_current_pix / 2)

    def initialize(self):
        self.upper = 255
        self.lower = 0
        self.x_mag = 1
        self.y_mag = 1

    def read_image(self):
        self.image_or, self.params = self.get_image_values()
        self.x_size_or = self.params[0]
        self.y_size_or = self.params[1]
        self.y_pix_or, self.x_pix_or = self.image_or.shape[:2]
        self.range_u = self.y_pix_or
        self.range_l = 0
        self.open_bool = True
        self.image_pl = np.copy(self.image_or)
        self.image_mod = np.copy(self.image_or)
        self.default_contrast()
        cv2.namedWindow(self.image_name)
        cv2.setMouseCallback(self.image_name, self.mouse_event_point)

    def default_contrast(self):
        self.max_contrast = np.max(self.image_mod)
        self.min_contrast = np.min(self.image_mod)

    def get_image_values(self):
        data_type = os.path.splitext(self.data_path)
        if data_type[1] == ".SM4":
            data, scan_params = self.sm4_getdata()
        elif data_type[1] == ".bmp":
            data, scan_params = self.bmp_getdata()
        elif data_type[1] == ".txt":
            data, scan_params = self.txt_getdata()
        return data, scan_params

    def sm4_getdata(self):
        sm4data_set = rhksm4.load(self.data_path)
        sm4data_im = sm4data_set[self.channel_val]
        image_data = sm4data_im.data
        image_data = image_data.astype(np.float32)
        scan_params = [
            -round(
                sm4data_im.attrs["RHK_Xscale"]
                * sm4data_im.attrs["RHK_Xsize"]
                * 1000000000,
                1,
            ),
            -round(
                sm4data_im.attrs["RHK_Yscale"]
                * sm4data_im.attrs["RHK_Ysize"]
                * 1000000000,
                1,
            ),
        ]
        return image_data, scan_params

    def bmp_getdata(self):
        bmpdata_im = np.array(Image.open(self.data_path).convert("L"), dtype=np.float32)
        scan_params = [
            30,
            round(30 / bmpdata_im.shape[1] * bmpdata_im.shape[0], 2),
        ]
        return bmpdata_im, scan_params

    def txt_getdata(self):
        with open(self.data_path) as f:
            lines = f.readlines()
            read_check = False
            point_check = False
            text_data = []
            self.points = []
            for line in lines:
                values = line.split()
                if "Peaks:" in values:
                    read_check = False
                if read_check is True:
                    text_data.append([])
                    for val in values:
                        if val == "\n":
                            pass
                        else:
                            text_data[-1].append(float(val))

                if "Data:" in values:
                    read_check = True
                    point_check = False
                if point_check is True:
                    if len(values) >= 2:
                        self.points.append((int(values[0]), int(values[1])))
                if "Defect_position:" in values:
                    point_check = True
                if "Current_size_X:" in values:
                    xsize = float(values[1])
                if "Current_size_Y:" in values:
                    ysize = float(values[1])

        data_np = np.array(text_data, dtype=np.float32)
        if read_check:
            return data_np, [xsize, ysize]
        else:
            return False, False

    def show_image(
        self,
    ):
        if self.open_bool:
            self.plane()
            self.smoothing()
            self.median()
            self.rescale()
            self.contrast_adjust()
            self.contrast_change()
            self.color_change()
            self.draw_point()
            cv2.imshow(self.image_name, self.image_show)

    def plane(self):
        if self.plane_bool is True:
            self.image_pl = np.copy(self.image_or)
        else:
            self.image_pl = np.copy(self.image_or)

    def smoothing(self):
        self.image_sm = ndimage.gaussian_filter(self.image_pl, float(self.smooth_val))

    def median(self):
        self.image_med = ndimage.gaussian_filter(self.image_sm, float(self.median_val))

    def rescale(self):
        width_x = int(self.x_pix_or * float(self.mag))
        height_y = int(self.y_pix_or * float(self.mag))
        self.image_mod = cv2.resize(self.image_med, (width_x, height_y))

    def draw_point(self):
        self.image_show = np.copy(self.image_cl)
        for point in self.points:
            if point[1] <= self.low_pix and point[1] >= self.up_pix:
                self.image_show = cv2.circle(
                    self.image_show, point, self.c_size, (0, 0, 255), 1
                )

    def add_point(self):
        val = (self.mouse_x, self.mouse_y)
        if val not in self.points:
            self.points.append(val)
        self.draw_point()

    def remove_point(self):
        min_dis = 10000000
        cand_num = None
        for i in range(len(self.points)):
            dis = (self.mouse_x - self.points[i][0]) ** 2 + (
                self.mouse_y - self.points[i][1]
            ) ** 2
            if dis < min_dis:
                min_dis = dis
                cand_num = i
        if min_dis <= self.c_size**2:
            self.points.pop(cand_num)
        self.draw_point()

    def mouse_event_point(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONUP:
            self.mouse_x = x
            self.mouse_y = y
            self.add_point()
        if event == cv2.EVENT_RBUTTONUP:
            self.mouse_x = x
            self.mouse_y = y
            self.remove_point()

        cv2.imshow(self.image_name, self.image_show)

    def contrast_adjust(self):
        image = (
            (self.image_mod - self.min_contrast)
            / (self.max_contrast - self.min_contrast)
            * 255
        )
        image[image > 255] = 255
        image[image < 0] = 0
        self.image_cad = image.astype(np.uint8)

    def contrast_change(self):
        LUT = self.get_LUT()
        self.image_cad = cv2.LUT(self.image_cad, LUT)

    def get_LUT(self):
        LUT = np.zeros((256, 1), dtype="uint8")
        maximum = int(self.upper)
        minimum = int(self.lower)
        if maximum == minimum:
            for i in range(-50, 301):
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
            for i in range(-50, 301):
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
            for i in range(-50, 301):
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

    def color_change(self):
        height = len(self.image_cad)
        width = len(self.image_cad[1])
        if self.color_num == 0:
            self.image_cl = cv2.cvtColor(self.image_cad, cv2.COLOR_GRAY2BGR)
        else:
            self.image_cl = cv2.applyColorMap(self.image_cad, self.color_num - 1)
        self.up_pix = height - self.range_u
        self.low_pix = height - self.range_l
        if self.up_pix >= 1:
            u_image = self.image_cad[
                0 : self.up_pix,
                0:width,
            ]
            u_image_show = cv2.cvtColor(u_image, cv2.COLOR_GRAY2BGR)
            self.image_cl[0 : self.up_pix, 0:width] = u_image_show
        if self.low_pix <= height - 1:
            l_image = self.image_cad[self.low_pix : height, 0:width]
            l_image_show = cv2.cvtColor(l_image, cv2.COLOR_GRAY2BGR)
            self.image_cl[self.low_pix : height, 0:width] = l_image_show

    def prepare_cut_image(self):
        width = len(self.image_cad[1])
        image_cl_gray = cv2.cvtColor(self.image_cad, cv2.COLOR_GRAY2BGR)
        self.cut_image_gray = image_cl_gray[self.up_pix : self.low_pix - 1, 0:width]
        self.cut_image = self.image_cl[self.up_pix : self.low_pix - 1, 0:width]
        self.cut_points = []
        for point in self.points:
            x, y = point
            y = y - self.up_pix
            self.cut_points.append((x, y))
        self.cut_image_p = np.copy(self.cut_image)
        for point in self.cut_points:
            if point[1] <= self.low_pix - self.up_pix and point[1] >= 0:
                self.cut_image_p = cv2.circle(
                    self.cut_image_p, point, self.c_size, (0, 0, 255), 1
                )

    def auto_detection(self):
        height = len(self.image_mod)
        width = len(self.image_mod[1])
        x_list = np.arange(0, width, int(self.pix_range / 2))
        y_list = np.arange(0, height, int(self.pix_range / 2))
        points = []

        for w in x_list:
            for h in y_list:
                x0 = max(w - self.pix_range, 0)
                x1 = min(w + self.pix_range, width - 1)
                y0 = max(h - self.pix_range, 0)
                y1 = min(h + self.pix_range, height - 1)
                vals = self.image_mod[y0:y1, x0:x1]
                vw = len(vals[1])
                vh = len(vals)
                min_val = np.min(vals)
                cand = np.where(vals == min_val)
                for i in range(len(cand) - 1):
                    valy = cand[0][i]
                    valx = cand[1][i]
                    if (valx in (0, vw - 1)) or (valy in (0, vh - 1)):
                        pass
                    elif (valx + x0, valy + y0) not in points:
                        points.append((valx + x0, valy + y0))
        angles = np.linspace(0, 2 * np.pi, 30)
        circle_points = [
            (int(self.pix_range * np.sin(t)), int(self.pix_range * np.cos(t)))
            for t in angles
        ]
        # self.points = points
        self.points = []
        for point in points:
            x, y = point
            point_val = self.image_mod[y][x]
            c_vals = []
            for c in circle_points:
                cx = c[1] + x
                cy = c[0] + y
                if (cx < 0) or (cx > width - 1) or (cy < 0) or (cy > height - 1):
                    pass
                else:
                    c_vals.append(self.image_mod[cy][cx])
            c_ave = np.average(c_vals)
            c_std = np.std(c_vals)
            if point_val < c_ave - c_std * self.auto_thresh:
                self.points.append((x, y))

        self.show_image()

    def defect_analysis(self):
        height = len(self.image_mod)
        width = len(self.image_mod[1])
        angles = np.linspace(0, 2 * np.pi, 30)
        circle_points = [
            (int(self.pix_range * np.sin(t)), int(self.pix_range * np.cos(t)))
            for t in angles
        ]
        self.rel_height = []
        for point in self.points:
            x, y = point
            point_val = self.image_mod[y][x]
            c_vals = []
            for c in circle_points:
                cx = c[1] + x
                cy = c[0] + y
                if (cx < 0) or (cx > width - 1) or (cy < 0) or (cy > height - 1):
                    pass
                else:
                    c_vals.append(self.image_mod[cy][cx])
            c_ave = np.average(c_vals)
            self.rel_height.append(c_ave - point_val)
        self.height_ave = np.average(self.rel_height)
        self.height_std = np.std(self.rel_height)
        self.norm_std = self.height_std / self.height_ave

    def set_default(self):
        if self.open_bool:
            self.max_contrast, self.min_contrast = (
                self.max_contrast - self.min_contrast
            ) / 255 * self.upper + self.min_contrast, (
                self.max_contrast - self.min_contrast
            ) / 255 * self.lower + self.min_contrast

    def back_default(self):
        if self.open_bool:
            self.max_contrast = np.max(self.image_mod)
            self.min_contrast = np.min(self.image_mod)

    def des_image(self):
        cv2.destroyWindow(self.image_name)
        self.open_bool = False
