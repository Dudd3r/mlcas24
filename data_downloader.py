from common import *

import shutil
import requests
import zipfile
from tqdm import tqdm

def download_data(download_url, out_path):

    print("[INFO] Downloading file: {}".format(out_path))

    response = requests.get(download_url, stream=True)
    total_size = int(response.headers.get('content-length', 0))

    with open(out_path, 'wb') as file, tqdm(
            desc='Downloading',
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
        for data in response.iter_content(chunk_size=1024):
            file.write(data)
            bar.update(len(data))

def unzip(zip_path, out_path):
    if not os.path.isdir(out_path):
        os.makedirs(out_path)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(path=out_path)

def extract_data():
    print("[INFO] Extracting files.")
    
    unzip(pj(DATA_ROOT, "2023.zip"), pj(DATA_ROOT, "2023_tmp"))
    
    # Train data
    unzip(pj(DATA_ROOT, "2023_tmp", "MLCAS24_Competition_data", "train", "2022.zip"),
          pj(DATA_ROOT, "train"))
    unzip(pj(DATA_ROOT, "2023_tmp", "MLCAS24_Competition_data", "train", "2023.zip"),
          pj(DATA_ROOT, "train"))
    
    # Validation
    unzip(pj(DATA_ROOT, "2023_tmp", "MLCAS24_Competition_data", "validation", "2023.zip"),
    pj(DATA_ROOT, "validation"))

    # Test
    unzip(pj(DATA_ROOT, "2023_tmp", "MLCAS24_Competition_data", "test", "Test.zip"),
    pj(DATA_ROOT, "test", "2023"))
    if not os.path.isdir(pj(DATA_ROOT, "test", "2023", "DataPublication_final")):
        os.rename(pj(DATA_ROOT, "test", "2023", "Test"),
                pj(DATA_ROOT, "test", "2023", "DataPublication_final"))
    
    shutil.copy(pj(DATA_ROOT, "2023_tmp", "MLCAS24_Competition_data", "validation", "DateofCollection.xlsx"), 
                pj(DATA_ROOT, "validation", "2023", "GroundTruth", "DateofCollection.xlsx"))
    
    shutil.copy(pj(DATA_ROOT, "2023_tmp", "MLCAS24_Competition_data", "validation", "DateofCollection.xlsx"), 
                pj(DATA_ROOT, "test", "2023", "DataPublication_final", "GroundTruth", "DateofCollection.xlsx"))
    
    #shutil.rmtree(pj(DATA_ROOT, "2023_tmp"))

def run():
    print("[INFO] Downloading dataset.")
    resp = requests.get(DATA_SET_URL)
    resp_json = json.loads(resp.text)
    url = resp_json["download_url"]
    print("[INFO] URL: {}".format(url))

    if os.path.isfile(DATA_SET_ZIP):
        print("[WARN] {} already exists".format(DATA_SET_ZIP))
        overwrite = None
        while overwrite not in ["y", "n"]:
            overwrite = input("[INPUT] Overwrite (y/n)?")
        if overwrite == "n": 
            return

    download_data(url, DATA_SET_ZIP)
    extract_data()

if __name__ == "__main__":
    run()
    