import numpy as np
from numpy import fft
from scipy import signal


def apply_window(target, window):
    target_copy = np.copy(target)
    if window == "Hann":
        wfunc = signal.hann(target.shape[0])
        wfunc2 = signal.hann(target.shape[1])
    elif window == "Hamming":
        wfunc = signal.hamming(target.shape[0])
        wfunc2 = signal.hamming(target.shape[1])
    elif window == "Blackman":
        wfunc = signal.blackman(target.shape[0])
        wfunc2 = signal.blackman(target.shape[1])
    else:
        wfunc = signal.boxcar(target.shape[0])
        wfunc2 = signal.boxcar(target.shape[1])
    for i in range(target.shape[0]):
        for k in range(target.shape[1]):
            target_copy[i][k] = target[i][k] * wfunc[i] * wfunc2[k]
    return target_copy


def fft_processing(w_image):
    fimage_or = np.fft.fft2(w_image)
    fimage_or = np.fft.fftshift(fimage_or)
    fimage_or = np.abs(fimage_or)
    return fimage_or


def fft_scaling(image, method):
    if method == "Linear":
        fimage = image
    elif method == "Log":
        fimage = np.log(image, out=np.zeros_like(image), where=(image != 0))
    elif method == "Sqrt":
        fimage = np.sqrt(image)
    return fimage


def fft_process(target, method, window):
    w_image = apply_window(target, window)
    fft_image = fft_processing(w_image)
    fft_image = fft_scaling(fft_image, method)
    fft_image = fft_image.astype(np.float32)
    return fft_image
