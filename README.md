# MLCAS24 - Maize yield prediction using satellite images

The repo contains multiple small tools that help build a maize yield prediction pipeline,
based on satellite images.

Tested on:
- Windows 11, python 3.10.11 CUDA11.8, 3.12.5 CPU
- Ubuntu 22.04, python 3.10.12

Recommended environment:
- Windows11, python 3.10.11, CUDA11.8, pandas==2.0.3, torch==2.1.0, torchvision==0.16.0 

## Approach

To be written. (Technical documentation of the pipeline).

In a nutshell: 
- A custom, shallow ResNet that combines the raw image features 
with some predictors that is used to forecast on the plot level.

Predictors:
- Normalized genotype ranking
- Seasonal bias
- Normalized time interval between planting date and image collection date
- Nitrogen amount

Spectral indices used:
- NVDI
- GLI
- Hue

No external datasets were used.

## How to reproduce

### Install

- Python install (>3.9) is required.
- If you have CUDA and cuDNN installed then the model will utilize the GPU otherwise it runs on the CPU.

```shell
python -m pip install -r requirements.txt
```

### Autorun

The simplest way to run the pipeline is executing the autorun script. It will internally
run the data downloader, data formatter and will start the model in inference mode.

```shell
python autorun.py
```

The autorun pipeline will also load the test CSV and perform the predictions on it. The results are stored under the "results" directory.

### Components

#### Data downloader

The data downloader, apart from acquiring the data, extracts the files and reformats the structure so that it is compatible
with the pipeline.

```shell
python data_downloader.py
```

#### Data formatting & Project Setup

The data formatter will create the neccessary project structure and will create
a number of data views stored as CSVs. The canonical data format uses the satellite image file names as its primary key.

```shell
python data_formatter.py
```

#### (Optional) Spectral Statistics Extraction

The following scrip is used to extract some statistics about the satellite spectra for normalization. The values are stored in the JSON file: spectral_stats.json

The repo already contains the JSON because the extraction takes a while.

```shell
python extract_spectral_stats.py
```

#### (Optional) Full Image Reconstruction

A small script exists that creates full UAV and Satellite images from the 
segmented pieces with yield overlay for visualization and ideation.
Not needed to reproduce the results in any way.
```shell
python construct_full_images.py
```

#### Training & Inference

When run for the first time the script will build cache files of the 
datasets so retraining will require less time.

The script expects a "model_id" as a command line parameter.

If a model already exists with the model_id provided then the script will start in inference mode.

If a new model_id is provided then the script starts training a new model. Once the training terminates the weights will be saved in the models folder.

Resuming the training is not supported at the moment.

The repo contains the weights for the model used in the competition:
"resnet5_200_fin"

Inference:
```shell
python resnet_mono.py resnet5_200_fin
```
Training:
```shell
python resnet_mono.py my_new_model
```

In inference mode the script will generate predictions on the test data
and save the resulting csv in the results folder.

### Notes

This repo contains the bare minimum to reproduce the results submitted to the competition. The feature engineering, feature selection, model selection and EDA is scattered accross notebooks and small scripts which I am not comfortable sharing at this moment. 

If you have any issues running the pipeline or find any bugs don't hesitate to contanct me.

Cheers,
Andras

---

**Author:** Andras Toth<br>
**E-mail:** andras.toth.hun@gmail.com<br>
**LinkedIn:** https://www.linkedin.com/in/andr%C3%A1s-t%C3%B3th-54b6151aa/<br>
**Company:** Phenospex B.V.<br>
**Year:** 2024
