# MLCAS24 - Maize yield prediction using satellite images

The repo contains multiple small tools that help build a yield prediction pipeline.

## Data

The data is expected to be located in the repo folder inside the "data" directory in the following structure:

data/\<dataset>\/\<year>\/

where dataset may be train, validation or test. Under the "year" directory the data is expected to follow the
MLCAS data standard. (e.g. 
- data/train/2023/DataPublication_final/GroundTruth/\*.csv
- data/train/2023/DataPublication_final/Satellite/\<location\>/\<timepoint\>/\*.tif)

## Approach

To be written. (Technical documentation of the pipeline).

In a nutshell: 
- A custom, shallow ResNet that combines the raw image features 
with some predictors, is used to forecast on the plot level.

Predictors:
- Normalized genotype ranking (based on the traning data)
- Small environment bias based on the yearly yield differences (based on the training data)
- Normalized time interval between planting date and image collection date
- Nitrogen amount

Spectral indices used:
- NVDI
- GLI
- Hue

## How to reproduce

### Install

- Some modernish version (3.9<) of python is required.
- If you have CUDA and cuDNN installed then the model will utilize the GPU otherwise it runs on the CPU 

```shell
python -m pip install -r requirements.txt
```

Copy the data into the repo directory as explained above.

### Run data formatting & project setup

The data formatter will create the neccessary project structure and will create
a number of data views stored as CSVs.

```shell
python data_formatter.py
```

### (Optional) Run spectral statistics extraction

The repo already contains the JSON because the extraction takes a while.
If you want to verify:
```shell
python extract_spectral_stats.py
```

### (Optional) Reconstruct full images

A small script exists that creates full UAV and Satellite images from the 
segmented pieces with yield overlay for visualization and ideation.
Not needed to reproduce the results in any way.
```shell
python construct_full_images.py
```

### Training & Inference

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

P.S.: The pipeline could use a good refactoring. :>. I will take some time to clean things up a bit.

Cheers,
Andras

---

**Author:** Andras Toth<br>
**LinkedIn:** https://www.linkedin.com/in/andr%C3%A1s-t%C3%B3th-54b6151aa/<br>
**Company:** Phenospex B.V.<br>
**Year:** 2024
