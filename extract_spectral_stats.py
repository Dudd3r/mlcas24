from common import *
from scipy.stats import skewnorm

image_files = pd.read_csv(TRAINING_DATA_CSV)["image_path"]

def extract_spectral_data_long_format():
    # Load images
    print("[INFO] Loading images.")
    satellite_images = []

    n_image_files = len(image_files)
    n_5perc = n_image_files // 20
    n = 0

    for i in range(n_image_files):
        if i > 0 and i % n_5perc == 0:
            print("[INFO] Progress: {}%, reading images: {}/{}".format(i//n_5perc*5, i, n_image_files))
        s = SatImage(image_files[i])
        satellite_images.append(s)
        n+=(s.data.shape[1] * s.data.shape[2])

    print("[INFO] Total number of datapoints: {}".format(n))

    # init the global array
    spectral_data = np.zeros(shape=(n, NUM_BANDS), dtype=np.int32)
    row_offset = 0

    # Unwrap into one giant 2D array
    print("[INFO] Reshaping data.")
    for i in range(n_image_files):
        original_shape = satellite_images[i].data.shape
        reshaped = satellite_images[i].data.reshape(
            (NUM_BANDS, original_shape[1] * original_shape[2])).T
        reshaped = reshaped[~np.all(reshaped == 0, axis=1)]
        spectral_data[row_offset: row_offset+reshaped.shape[0]] = reshaped
        row_offset += reshaped.shape[0]

    # Cut the extra elements
    spectral_data = spectral_data[:row_offset]

    print("[INFO] Saving data.")
    pd.DataFrame(spectral_data, columns=list(BAND_INDEX)).to_csv(SPECTRAL_DATA_CSV, index=False)

def extract_band_statistics():
    print("[INFO] Extracting spectral stats.")
    spectral_statistics = {}

    sd = pd.read_csv(SPECTRAL_DATA_CSV)
    for band in BAND_INDEX.keys():
        
        print("[INFO] From spectral band: {}".format(band))

        bd = sd[band]
        
        ql = bd.quantile(BAND_QL)
        qh = bd.quantile(BAND_QH)
        clipped_data = bd.loc[(bd>ql)&(bd<qh)]
        
        spectral_statistics[band] = {
            "ql":ql, "qh":qh,
            "mean": bd.mean(),
            "median": bd.median(),
            "pdf_parameters": skewnorm.fit(clipped_data)
        }

    with open(SPECTRAL_STATS_JSON, 'w') as f:
        json.dump(spectral_statistics, f, indent=4)

if __name__ == "__main__":
    if not os.path.isfile(SPECTRAL_DATA_CSV):
        extract_spectral_data_long_format()
    if not os.path.isfile(SPECTRAL_STATS_JSON):
        extract_band_statistics()
    extract_band_statistics()