import os
import sys
import re
import json
import glob
import colorsys
import rasterio
import cv2
import random
import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from matplotlib import pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 150)
pd.set_option('display.float_format', '{:.4f}'.format)

pj = os.path.join
ls = glob.glob

DATA_SET_URL = "https://iastate.app.box.com/index.php?folder_id=271502960797&q%5Bshared_item%5D%5Bshared_name%5D=p8nj1ukvwx3yo7off8y8yspdruc0mjna&rm=box_v2_zip_shared_folder"

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = pj(PROJECT_ROOT, "data")
MODELS_DIR = pj(PROJECT_ROOT, "models")
RESULTS_DIR = pj(PROJECT_ROOT, "results")
CACHE_DIR = pj(PROJECT_ROOT, "cache")

DATA_SET_ZIP = pj(DATA_ROOT, "2023.zip")

for dir in [MODELS_DIR, RESULTS_DIR, CACHE_DIR]:
    os.makedirs(dir, exist_ok=1)

os.makedirs(pj(DATA_ROOT, "processed"), exist_ok=1)

METADATA_CSV = pj(DATA_ROOT, "processed", "metadata.csv")
TRAINING_DATA_CSV = pj(DATA_ROOT, "processed", "training_data.csv")
TEST_DATA_CSV = pj(DATA_ROOT, "processed", "test_data.csv")
CANONICAL_DATA_CSV = pj(DATA_ROOT, "processed", "canonical.csv")
SPECTRAL_DATA_CSV = pj(DATA_ROOT, "processed", "spectral_data.csv")
SPECTRAL_STATS_JSON = pj(PROJECT_ROOT, "spectral_stats.json")

SUBMISSION_CSV = pj(DATA_ROOT, "processed", "test_HIPS_HYBRIDS_2023_V2.3.csv")

MAX_TP_DIFF = 10
MAX_GROWTH_CYCLE_DAYS = 150

# Model Parameters

RNET_IMAGE_CHANNELS = 9
RNET_DEFAULT_WEIGHTS = "ResNet18_Weights.IMAGENET1K_V1"
RNET_TRAIN_EPOCHS = 500
RNET_BATCH_SIZE = 300

AUG_FLIP = 0.1
AUG_SHIFT = 0.1
AUG_INTENS = 0.1

SPECTRAL_STATS = None
if os.path.isfile(SPECTRAL_STATS_JSON):
    with open(SPECTRAL_STATS_JSON) as f:
        SPECTRAL_STATS = json.load(f) 

BAND_INDEX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "nir": 3,
    "re":4,
    "db":5
}

SI_INDEX = {
    "h":0,
    "s":1,
    "l":2,
    "ndvi":3,
    "npci":4,
    "psri":5,
    "gli":6,
    "w":7,
}

BAND_QL = 0.0001
BAND_QH = 0.999

NUM_BANDS = 6

EPSILON = 1e-6

def show_image(frame : np.array, cmap='gray'):
    if len(frame.shape) > 2:
        rgb = frame[:, :, ::-1]
        plt.imshow(rgb)
    else:
        plt.imshow(frame, cmap=cmap)
    plt.axis('off')
    current_fig = plt.gcf()
    current_fig.set_size_inches(16, 9)
    plt.show()

def get_corresponding_uav_file(sat_image_file):
    uav_file = sat_image_file.replace("Satellite", "UAV").replace(".TIF", ".png")
    if os.path.isfile(uav_file):
        return uav_file
    else:
        return None

def scalar_to_rgb(scalar):
    rgb = np.zeros((scalar.shape[0], 3))
    
    # Interpolate between red and yellow for values in [0, 0.5]
    mask = scalar <= 0.5
    rgb[mask, 0] = 1
    rgb[mask, 1] = scalar[mask] * 2
    
    # Interpolate between yellow and green for values in [0.5, 1]
    mask = scalar > 0.5
    rgb[mask, 0] = 2 * (1 - scalar[mask])
    rgb[mask, 1] = 1
    
    return (rgb * 255).astype(np.uint8)

class SatImage:
    def __init__(self, image_path) -> None:
        self.filename = os.path.basename(image_path)
        with rasterio.open(image_path, 'r') as tif:
            self.data = tif.read()
            self.bbox = tif.bounds
        
        if SPECTRAL_STATS is not None:
            self.calulate_spectral_indices()

    def __getitem__(self, idx):
        if idx in BAND_INDEX.keys():
            return self.data[BAND_INDEX[idx]]
        elif type(idx) == int and idx > 0 and idx < self.data.shape[2]:
            return self.data[idx]
        else:
            print("[WARN] Invalid satellite image index.")
            return None
    
    def calulate_spectral_indices(self):

        r = self.get_channel_norm("red")
        g = self.get_channel_norm("green")
        b = self.get_channel_norm("blue")
        n = self.get_channel_norm("nir")
        rr = self.get_channel_norm("re")
        
        self.spectral_indices = np.zeros(shape=(len(SI_INDEX.keys()), r.shape[0], r.shape[1]))

        for i in range(0, r.shape[0]):
            for j in range(0, r.shape[1]):
                _h,_l,_s = colorsys.rgb_to_hls(r[i,j], g[i,j], b[i,j])
                self.spectral_indices[SI_INDEX["h"],i,j] = _h
                self.spectral_indices[SI_INDEX["s"],i,j] = _s
                self.spectral_indices[SI_INDEX["l"],i,j] = _l

        self.spectral_indices[SI_INDEX["ndvi"]] = (n-r)/(n+r+EPSILON)
        self.spectral_indices[SI_INDEX["npci"]] = (r-b)/(r+b+EPSILON)
        self.spectral_indices[SI_INDEX["psri"]] = (r-b)/(n+EPSILON)
        self.spectral_indices[SI_INDEX["gli"]] = (2*g-r-b)/(2*g+r+b+EPSILON)
        self.spectral_indices[SI_INDEX["w"]] = n/(rr+EPSILON)

    def get_rgb(self):
        rgb_data = np.dstack((self.data[0], self.data[1], self.data[2])).astype(np.float32)
        for band in ["red", "green", "blue"]:
            i = BAND_INDEX[band]
            rgb_data[:,:,i] = self.get_channel_norm(band) * 255.0
        rgb_data = rgb_data.astype(np.uint8)
        return rgb_data

    def get_channel_norm(self, band):
        stats = SPECTRAL_STATS[band]
        band_data = self.data[BAND_INDEX[band]].astype(np.float32)
        band_data = np.clip(band_data, stats["ql"], stats["qh"])
        band_data = (band_data - stats["ql"]) / (stats["qh"]-stats["ql"])
        return band_data