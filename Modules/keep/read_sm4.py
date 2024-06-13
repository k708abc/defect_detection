from spym.io import rhksm4
import cv2
import numpy as np
import matplotlib.pyplot as plt


def datatypes(data_path):
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


def sm4_getdata(data_path, data_type):
    sm4data_set = rhksm4.load(data_path)
    sm4data_im = sm4data_set[data_type]
    image_data = sm4data_im.data
    image_data = image_data.astype(np.uint16)
    scan_params = [
        -round(
            sm4data_im.attrs["RHK_Xscale"] * sm4data_im.attrs["RHK_Xsize"] * 1000000000,
            1,
        ),
        -round(
            sm4data_im.attrs["RHK_Yscale"] * sm4data_im.attrs["RHK_Ysize"] * 1000000000,
            1,
        ),
        sm4data_im.attrs["RHK_Bias"],
        np.max(image_data),
        np.min(image_data),
    ]
    return image_data, scan_params
