import numpy as np
from scipy import signal

def horizontal_median_filter(A, N):
    return signal.medfilt(A, (1, N))

def horizontal_average_filter(A, N):
    return signal.convolve2d(A, np.ones((1, N)) / N, mode='same')