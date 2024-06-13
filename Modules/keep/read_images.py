import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def bmp_getdata(data_path):
    bmpdata_im = np.array(Image.open(data_path).convert("L"), dtype=np.float32)
    scan_params = [
        30,
        round(30 / bmpdata_im.shape[1] * bmpdata_im.shape[0], 2),
        0,
        np.max(bmpdata_im),
        np.min(bmpdata_im),
    ]
    return bmpdata_im, scan_params


def txt_getdata(data_path):
    with open(data_path) as f:
        lines = f.readlines()
        read_check = False
        text_data = []
        for line in lines:
            values = line.split()
            if read_check is True:
                text_data.append([])
                for val in values:
                    if val == "\n":
                        pass
                    else:
                        text_data[-1].append(float(val))
            if "Data:" in values:
                read_check = True
            if "Current_size_X:" in values:
                xsize = float(values[1])
            if "Current_size_Y:" in values:
                ysize = float(values[1])
            if "STM_bias:" in values:
                bias = float(values[1]) / 1000
    data_np = np.array(text_data, dtype=np.float32)
    return data_np, [xsize, ysize, bias, np.max(data_np), np.min(data_np)]
