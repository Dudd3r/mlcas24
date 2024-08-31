from common import *

class MLCASDataFormatter:
    def __init__(self, data_root, data_split="train") -> None:
        self.data_root = data_root

        if data_split not in ["train", "validation", "test"]:
            print("Unknown data split: {}".format(data_split))
            quit(1)
        
        self.data_split = data_split

        # from MLCAS data folder structure
        mlcas_sat_image_pattern\
        = [self.data_root, data_split,"**/*.TIF"]
        
        metadata_csv_pattern\
        = [self.data_root, data_split,"**/GroundTruth/*HIPS*.csv"]

        collection_date_xlsx_pattern\
        = [self.data_root, data_split,"**/GroundTruth/DateofCollection.xlsx"]

        self.satellite_image_files = ls(pj(*mlcas_sat_image_pattern), recursive=True)
        print(f"[INFO] Total satellite images found: {len(self.satellite_image_files)}")
        
        self.metadata_files = ls(pj(*metadata_csv_pattern), recursive=True)
        self.collection_date_files = ls(pj(*collection_date_xlsx_pattern), recursive=True)

        self.collection_date_available = len(self.collection_date_files) > 0

        self.training_data = pd.DataFrame()

    def write_data_to_csv(self, filename):
        self.data.to_csv(filename, index=False)

    def build(self):
        # Build a central metadata CSV
        metadata = None
        
        # Read all the metadata CSVs
        for file in self.metadata_files:
            df = pd.read_csv(file)
            if 'index' in df.columns:
                df = df.drop('index', axis=1)
            if metadata is None:
                metadata = df
            else:
                metadata = pd.concat((metadata, df))

        # Only want rows where we have a yield record for training
        if self.data_split == "train":
            metadata = metadata.loc[~metadata['yieldPerAcre'].isna()]
        
        # Fix typo in MOValley experiment
        typo_idx = metadata.loc[(metadata["location"] == "MOValley")&(metadata["experiment"] == "Hyrbrids")].index
        metadata.loc[typo_idx, "experiment"] = "Hybrids"

        # Create a merge id based on location, experiment, col, row
        metadata["merge_id"] = metadata["location"] + "_" + metadata["experiment"].astype(str)\
        + "_" + metadata["range"].astype(str) + "_" + metadata["row"].astype(str)

        self.metadata = metadata

        # Do the same with the collection date files
        if self.collection_date_available:

            collection_dates = None

            for file in self.collection_date_files:
                df = pd.read_excel(file)
                # Fix link to MOValley data
                df.loc[df["Location"] == "Missouri Valley", "Location"] = "MOValley"
                collection_dates = df if collection_dates is None else pd.concat((collection_dates, df))
            
            collection_dates = collection_dates.loc[collection_dates["Image"] == "Satellite"]
            collection_dates['year'] = collection_dates["Date"].astype(str).apply(lambda x: x.split("-")[0])
            collection_dates = collection_dates.rename({"time":"timepoint", 
                                                        "Location":"location",
                                                        "Date": "image_collection_date"}, axis=1)
            collection_dates.loc[collection_dates["location"] == "Missouri Valley", "location"] = "MOValley"
            collection_dates = collection_dates.drop("Image", axis=1)

        # Create a frame from the satellite image filenames
        # filename pattern: location-timepoint-experiment_id-range-row.tif
        df = pd.DataFrame({"image_path": self.satellite_image_files})
        df["filename"] = df["image_path"].apply(os.path.basename)
        df["year"] = df["image_path"].apply(lambda x: re.search(r'\\(\d{4})\\', x).group(1))
        df["location"] = df["filename"].apply(lambda x: x.split("-")[0])
        df["timepoint"] = df["filename"].apply(lambda x: x.split("-")[1])
        df["experiment"] = df["filename"].apply(lambda x: x.split("-")[2].split("_")[0])
        df["range"] = df["filename"].apply(lambda x: x.split("-")[2].split("_")[1])
        df["row"] = df["filename"].apply(lambda x: x.split("-")[2].split("_")[2].split(".")[0])
        df["merge_id"] = df["location"] + "_" + df["experiment"] + "_" +df["range"] + "_" + df["row"]
        df = df.drop(["filename", "location", "experiment", "row", "range"], axis=1)
        
        # Link images to the metadata
        df = pd.merge(df, metadata, on="merge_id")

        # Link collection date
        if self.collection_date_available:
            df = pd.merge(df, collection_dates, on=["location", "year", "timepoint"])

        # Correct stand count for ScottsBluff
        if "totalStandCount" in df.columns:
            df.loc[df["plotLength"] != 17.5, "totalStandCount"] *= (17.5/22.5)
        
        # Normalize nitrogen amount
        df["poundsOfNitrogenPerAcre"] = (df["poundsOfNitrogenPerAcre"] - df["poundsOfNitrogenPerAcre"].min())\
        /(df["poundsOfNitrogenPerAcre"].max() - df["poundsOfNitrogenPerAcre"].min())

        # Drop irrelevant columns
        # anthesis data not available in Ames
        # irrigation only in ScottsBluff
        junk_cols = ["qrCode", "merge_id", "plotNumber", "daysToAnthesis", 
                      "GDDToAnthesis", "irrigationProvided", "plotLength", "nitrogenTreatment"]
        for jc in junk_cols:
            if jc in df.columns:
                df = df.drop(jc, axis=1)
        
        # Rename stuff for clarity
        rename_mapper = {"poundsOfNitrogenPerAcre": "nitrogen_amount",
                         "range": "column",
                         "plantingDate": "planting_date",
                         "totalStandCount": "stand_count",
                         "yieldPerAcre": "yield"}
        
        for k,v in rename_mapper.items():
            if k in df.columns:
                df = df.rename({k:v}, axis=1)

        df["planting_date"] = pd.to_datetime(df["planting_date"], format='mixed')
        
        if self.collection_date_available:
            df["image_collection_date"] = pd.to_datetime(df["image_collection_date"], format='mixed')
            df["p2i_days"] = (df["image_collection_date"] - df["planting_date"]).dt.days

        # Create a simple sample - and experiment id for convenience
        df["experiment_id"] =   df["year"].astype(str) + "_" +\
                                df["location"].astype(str) + "_"  +\
                                df["experiment"].astype(str)


        df["sample_id"] =   df["location"].astype(str) + "_" +\
                            df["experiment"].astype(str) + "_" +\
                            df["column"].astype(str) + "_" +\
                            df["row"].astype(str)

        self.data = df
        print("[INFO] Done.")

def extract_image_parameters(image_path):
    sat_image = SatImage(image_path)
    image_parameters = {}
    
    for spectral_index, i in SI_INDEX.items():
        image_data = sat_image.spectral_indices[i]

        flat_data = image_data.flatten()
        flat_data = flat_data[flat_data!=0]
        image_parameters["{}_avg".format(spectral_index)] = flat_data.mean()
        image_parameters["{}_med".format(spectral_index)] = np.median(flat_data)
        image_parameters["{}_min".format(spectral_index)] = flat_data.min()
        image_parameters["{}_max".format(spectral_index)] = flat_data.max()
        image_parameters["{}_std".format(spectral_index)] = np.std(flat_data)

    return image_parameters

def transform_target_experiment():
    print("[INFO] Transforming.")
    train_data = pd.read_csv(TRAINING_DATA_CSV)
    train_data["set"] = "train"
    train_data = train_data.loc[(train_data["location"] == "Ames") & 
                   (train_data["year"] == 2022) & 
                   ((train_data["timepoint"] == "TP1")|(train_data["timepoint"] == "TP2"))].copy()

    print("[INFO] Target samples: {}".format(train_data.shape[0]))

    tp1_idx = train_data.loc[train_data["timepoint"] == "TP1"].index
    tp2_idx = train_data.loc[train_data["timepoint"] == "TP2"].index

    val_n = 25

    train_data.loc[tp1_idx[:val_n], "set"] = "val"
    train_data.loc[tp2_idx[:val_n], "set"] = "val"

    test_data = pd.read_csv(TEST_DATA_CSV)
    test_data["set"] = "test"
    canon_data = pd.concat((train_data, test_data))

    # Add norm genotype ranking
    cf = canon_data.sort_values(by="yield", ascending=False)
    cf['rank'] = cf.groupby(['year', 'location'])['yield'].rank(method='average', ascending=False)
    genotype_rank = cf.groupby('genotype')['rank'].sum().rank(method='min')
    canon_data['rank'] = canon_data['genotype'].map(genotype_rank) / len(cf["genotype"].unique())

    canon_data.to_csv(CANONICAL_DATA_CSV, index=False)

def transform_non_tp():
    print("[INFO] Transforming.")
    train_data = pd.read_csv(TRAINING_DATA_CSV)
    train_data["set"] = "train"

    tp1_idx = train_data.loc[(train_data["location"] == "Ames") & 
                   (train_data["year"] == 2022) & 
                   (train_data["timepoint"] == "TP1")].index
    tp2_idx = train_data.loc[(train_data["location"] == "Ames") & 
                   (train_data["year"] == 2022) & 
                   (train_data["timepoint"] == "TP2")].index

    val_n = 50
    train_data.loc[tp1_idx[:val_n], "set"] = "val"
    train_data.loc[tp2_idx[:val_n], "set"] = "val"

    print("[INFO] Validation samples: {}".format(train_data.loc[train_data["set"] == "val"].shape[0]))

    test_data = pd.read_csv(TEST_DATA_CSV)
    test_data["set"] = "test"
    canon_data = pd.concat((train_data, test_data))

    # Add norm genotype ranking
    cf = canon_data.sort_values(by="yield", ascending=False)
    cf['rank'] = cf.groupby(['year', 'location'])['yield'].rank(method='average', ascending=False)
    genotype_rank = cf.groupby('genotype')['rank'].sum().rank(method='min')
    canon_data['rank'] = canon_data['genotype'].map(genotype_rank) / len(cf["genotype"].unique())

    canon_data.to_csv(CANONICAL_DATA_CSV, index=False)

def transform_discrete_tp():
    print("[INFO] Transforming.")

    train_data = pd.read_csv(TRAINING_DATA_CSV)
    test_data = pd.read_csv(TEST_DATA_CSV)

    metadata_cols = ["year", "location", "experiment", "block", "row", "column", "genotype"]
    statdata_cols = ["nitrogen_amount", "stand_count", "yield"]

    # Find out for which days we have sat data in the test set.
    sat_collection_days = test_data["p2i_days"].unique()

    # Create a small lut for the time series
    tp_lut = {"TP{}".format(i+1) : sat_collection_days[i] for i in range(0, len(sat_collection_days))}
    # Keep track of the day differences in collection
    tp_diff_cols = ["TP{}_diff".format(i+1) for i in range(0, len(sat_collection_days))]

    # Set the columns for the new frame
    all_cols = metadata_cols + list(tp_lut.keys()) + tp_diff_cols + statdata_cols
    canon_data = pd.DataFrame(columns=all_cols)

    # Add the test data
    test_samples = test_data["sample_id"].unique()
    for test_sample in test_samples:
        sample_rows = test_data.loc[test_data["sample_id"] == test_sample]
        if sample_rows.shape[0] != 3:
            print("[WARN] Inconsistent collection dates in test data.")
            continue
        # copy over the metadata
        canon_entry = pd.DataFrame(index=range(0, 1), columns=all_cols)
        canon_entry.iloc[0][metadata_cols] = sample_rows.iloc[0][metadata_cols]
        canon_entry.iloc[0][statdata_cols] = sample_rows.iloc[0][statdata_cols]
        canon_entry.iloc[0][tp_diff_cols] = 0

        # set the image paths
        for tp_col, tp_val in tp_lut.items():
            corresp_row = sample_rows.loc[sample_rows["p2i_days"] == tp_val]
            if len(corresp_row) == 0:
                print("[WARN] Inconsistent collection dates in test data.")
                continue
            canon_entry.iloc[0][tp_col] = corresp_row["image_path"].values[0]
        canon_data = pd.concat((canon_data, canon_entry))
    
    canon_data["set"] = "test"

    # Add the training data
    train_samples = train_data["sample_id"].unique()
    for train_sample in train_samples:
        sample_rows = train_data.loc[train_data["sample_id"] == train_sample]

        # check if the yield data is consistent
        if not len(sample_rows["yield"].unique()):
            print("[WARN] Inconsisten yield sample in training data.")
            continue

        # copy over the metadata
        canon_entry = pd.DataFrame(index=range(0, 1), columns=all_cols + ["set"])
        canon_entry.iloc[0][metadata_cols] = sample_rows.iloc[0][metadata_cols]
        canon_entry.iloc[0][statdata_cols] = sample_rows.iloc[0][statdata_cols]

        # strategy 1: Pick the closest timepoint to the test timepoints
        # IDEA: do image blending by bilinear interpolation

        for tp_col, tp_val in tp_lut.items():
            closest_row = sample_rows.loc[(tp_val-sample_rows["p2i_days"]).abs().idxmin()]
            diff = (tp_val - closest_row["p2i_days"])

            if diff >= MAX_TP_DIFF:
                canon_entry.iloc[0][tp_col] = ""
            else:
                canon_entry.iloc[0][tp_col] = closest_row["image_path"]
            canon_entry.iloc[0][tp_col + "_diff"] = diff

        canon_entry["set"] = "train"
        canon_data = pd.concat((canon_data, canon_entry))

    canon_data.to_csv(CANONICAL_DATA_CSV, index=False)

if __name__ == "__main__":

    # some data wrangling    
    for data_split, filename in {"train": TRAINING_DATA_CSV,
                                 "test": TEST_DATA_CSV}.items():
        print("[INFO] Formatting {} data.".format(data_split))
        df = MLCASDataFormatter(DATA_ROOT, data_split=data_split)
        df.build()
        df.write_data_to_csv(filename)
    
    # build the canonical data representation
    transform_non_tp()