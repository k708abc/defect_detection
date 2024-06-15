#!python3.11
from scipy import ndimage
import numpy as np
import cv2


class ImOpen:
    name = "Im_open"
    image = None
    switch = True
    prev_mag_x = 1
    prev_mag_y = 1
    mag_rate_x = 1
    mag_rate_y = 1

    def mag_update(self):
        return [self.prev_mag_x * self.mag_rate_x, self.prev_mag_y * self.mag_rate_y]

    def run(self):
        return self.image

    def rewrite(self, params):
        pass

    def rec(self):
        return ""

    def read(self, vals):
        pass


class Smoothing:
    name = "Smoothing"
    range = None
    params = ["range"]
    params_type = ["entry"]
    image = None
    switch = True
    prev_mag_x = 1
    prev_mag_y = 1
    mag_rate_x = 1
    mag_rate_y = 1

    def mag_update(self):
        return self.prev_mag_x * self.mag_rate_x, self.prev_mag_y * self.mag_rate_y

    def rewrite(self, params):
        self.range = float(params[0])

    def getval(self, p_name):
        if p_name == "range":
            return self.range

    def run(self):
        image_mod = ndimage.gaussian_filter(self.image, float(self.range))
        return image_mod

    def rec(self):
        txt = self.name + "\t"
        for param in self.params:
            txt += param + "\t" + str(self.getval(param)) + "\t"
        txt += str(self.switch)
        return txt

    def read(self, vals):
        for i in range(len(vals)):
            if vals[i] == self.params[0]:
                self.range = float(vals[i + 1])
        if vals[-1] == "True":
            self.switch = True
        elif vals[-1] == "False":
            self.switch = False


class Median:
    name = "Median"
    range = None
    params = ["range"]
    params_type = ["entry"]
    image = None
    switch = True
    prev_mag_x = 1
    prev_mag_y = 1
    mag_rate_x = 1
    mag_rate_y = 1

    def mag_update(self):
        return [self.prev_mag_x * self.mag_rate_x, self.prev_mag_y * self.mag_rate_y]

    def rewrite(self, params):
        self.range = int(float(params[0]))

    def getval(self, p_name):
        if p_name == "range":
            return self.range

    def run(self):
        if int(float(self.range)) <= 0:
            image_mod = self.image
        else:
            image_mod = ndimage.median_filter(self.image, int(float(self.range)))
        return image_mod

    def rec(self):
        txt = self.name + "\t"
        for param in self.params:
            txt += param + "\t" + str(self.getval(param)) + "\t"
        txt += str(self.switch)
        return txt

    def read(self, vals):
        for i in range(len(vals)):
            if vals[i] == self.params[0]:
                self.range = int(float(vals[i + 1]))
        if vals[-1] == "True":
            self.switch = True
        elif vals[-1] == "False":
            self.switch = False


class Rescale:
    name = "Rescale"
    image = None
    all = 1
    x = 1
    y = 1
    params = ["All", "X", "Y"]
    params_type = ["entry", "entry", "entry"]
    cal = None
    switch = True
    prev_mag_x = 1
    prev_mag_y = 1
    mag_rate_x = 1
    mag_rate_y = 1

    def mag_update(self):
        return [self.prev_mag_x * self.mag_rate_x, self.prev_mag_y * self.mag_rate_y]

    def rewrite(self, params):
        self.all = float(params[0])
        self.x = float(params[1])
        self.y = float(params[2])

    def getval(self, p_name):
        if p_name == "All":
            return self.all
        if p_name == "X":
            return self.x
        if p_name == "Y":
            return self.y

    def run(self):
        or_y, or_x = self.image.shape[:2]
        width_x = int(or_x * float(self.x) * float(self.all))
        height_y = int(or_y * float(self.y) * float(self.all))
        modified_image = cv2.resize(self.image, (width_x, height_y))
        self.mag_rate_x = self.x
        self.mag_rate_y = self.y
        return modified_image

    def rec(self):
        txt = self.name + "\t"
        for param in self.params:
            txt += param + "\t" + str(self.getval(param)) + "\t"
        txt += str(self.switch)
        return txt

    def read(self, vals):
        for i in range(len(vals)):
            if vals[i] == self.params[0]:
                self.all = float(vals[i + 1])
            if vals[i] == self.params[1]:
                self.x = float(vals[i + 1])
            if vals[i] == self.params[2]:
                self.y = float(vals[i + 1])
        if vals[-1] == "True":
            self.switch = True
        elif vals[-1] == "False":
            self.switch = False


class Cut:
    name = "Cut"
    image = None
    ratio = None
    params = ["ratio"]
    params_type = ["entry"]
    cal = None
    switch = True
    prev_mag_x = 1
    prev_mag_y = 1
    mag_rate_x = 1
    mag_rate_y = 1

    def mag_update(self):
        return [self.prev_mag_x * self.mag_rate_x, self.prev_mag_y * self.mag_rate_y]

    def rewrite(self, params):
        self.ratio = float(params[0])

    def getval(self, p_name):
        if p_name == "ratio":
            return self.ratio

    def run(self):
        if self.ratio == 100:
            self.mag_rate_x = 1
            self.mag_rate_y = 1
            return self.image
        else:
            half_ratio = self.ratio / 200
            height, width = self.image.shape[:2]
            diff_h = int(height * half_ratio)
            diff_w = int(width * half_ratio)
            image_cropped = self.image[
                diff_h : height - diff_h, diff_w : width - diff_w
            ]
            self.mag_rate_x = 2 * diff_w / width
            self.mag_rate_y = 2 * diff_h / height
            return image_cropped

    def rec(self):
        txt = self.name + "\t"
        for param in self.params:
            txt += param + "\t" + str(self.getval(param)) + "\t"
        txt += str(self.switch)
        return txt

    def read(self, vals):
        for i in range(len(vals)):
            if vals[i] == self.params[0]:
                self.ratio = float(vals[i + 1])
        if vals[-1] == "True":
            self.switch = True
        elif vals[-1] == "False":
            self.switch = False


class Average:
    name = "Ave. sub."
    image = None
    params = []
    params_type = []
    cal = None
    switch = True
    prev_mag_x = 1
    prev_mag_y = 1
    mag_rate_x = 1
    mag_rate_y = 1

    def mag_update(self):
        return [self.prev_mag_x * self.mag_rate_x, self.prev_mag_y * self.mag_rate_y]

    def rewrite(self, params):
        pass

    def getval(self, p_name):
        return

    def run(self):
        for rows in range(len(self.image)):
            average = np.average(self.image[rows])
            self.image[rows] = self.image[rows] - average
        image_sub = self.image + np.min(self.image)
        return image_sub

    def rec(self):
        txt = self.name + "\t"
        for param in self.params:
            txt += param + "\t" + str(self.getval(param)) + "\t"
        txt += str(self.switch)
        return txt

    def read(self, vals):
        if vals[-1] == "True":
            self.switch = True
        elif vals[-1] == "False":
            self.switch = False
