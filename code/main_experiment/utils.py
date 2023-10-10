import numpy as np
import matplotlib.pyplot as plt
import os
import nibabel as nib
import torch
import torchvision.transforms as transforms
import cv2
from PIL import Image
from sklearn.linear_model import LinearRegression
from scipy.ndimage import gaussian_filter
from scipy.signal import convolve
import scipy.signal as signal
import random
from sklearn.metrics import r2_score
import seaborn as sns
import pickle


def save_object(obj, filename):
    # Overwrites any existing file.
    with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "src", filename), 'wb') as outp:
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)


def load_object(filename):
    with open(os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), "src", filename), 'rb') as inp:
        obj = pickle.load(inp)
        return obj


def preprocess(v):
    return (v - v.min()) / (v.max() - v.min())


def MSE(A):
    m, n = A.shape
    return (np.linalg.norm(A, "fro") ** 2) / (m * n)
