import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from torch.utils.data import Dataset, DataLoader

from common import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class ResNet(nn.Module):
    def __init__(self, num_channels=9):
        super(ResNet, self).__init__()
        
        self.resnet = models.resnet18(weights="ResNet18_Weights.IMAGENET1K_V1")
        self.resnet.conv1 = nn.Conv2d(num_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet.fc = nn.Identity()  # Removing the final fully connected layer
        
        self.float_fc = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU()
        )

        self.fc_combined = nn.Sequential(
            nn.Linear(512 + 32, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def forward(self, image, float_value):
        image_features = self.resnet(image)
        float_features = self.float_fc(float_value)
        combined_features = torch.cat((image_features, float_features), dim=1)
        output = self.fc_combined(combined_features)
        return output
    
class MLDataset(Dataset):
    def __init__(self, train_data, mode="train"):
        
        print("[INFO] Loading dataset.")
        self.mode = mode

        if os.path.isfile(pj(CACHE_DIR, f"{mode}_image_data.npy")):
            self.image_data = np.load(pj(CACHE_DIR, f"{mode}_image_data.npy"))
            self.image_sizes = np.load(pj(CACHE_DIR, f"{mode}_image_sizes.npy"))
            self.metadata = np.load(pj(CACHE_DIR, f"{mode}_metadata.npy"))
            self.targets = np.load(pj(CACHE_DIR, f"{mode}_targets.npy"))
        else:
            n = train_data.shape[0]
            channels = 9

            self.image_data = np.zeros(shape=(n, channels, 30, 30), dtype=np.float32)
            self.image_sizes = np.zeros(shape=(n, 2), dtype=np.int32)
            self.metadata = np.zeros(shape=(n, 3), dtype=np.float32)
            self.targets = np.zeros(shape=(n, 1), dtype=np.float32)

            for i in range(n):
                sample = train_data.iloc[i]
                image_path = str(sample["image_path"])

                if os.path.isfile(image_path):
                    image = SatImage(image_path)

                    h, w = image["red"].shape

                    # Add the bands
                    self.image_data[i, 0, 0:h, 0:w] = image.get_channel_norm("red")
                    self.image_data[i, 1, 0:h, 0:w] = image.get_channel_norm("green")
                    self.image_data[i, 2, 0:h, 0:w] = image.get_channel_norm("blue")
                    self.image_data[i, 3, 0:h, 0:w] = image.get_channel_norm("nir")
                    self.image_data[i, 4, 0:h, 0:w] = image.get_channel_norm("re")
                    self.image_data[i, 5, 0:h, 0:w] = image.get_channel_norm("db")

                    # Add the indices
                    self.image_data[i, 6, 0:h, 0:w] = image.spectral_indices[SI_INDEX["ndvi"]]
                    self.image_data[i, 7, 0:h, 0:w] = image.spectral_indices[SI_INDEX["gli"]]
                    self.image_data[i, 8, 0:h, 0:w] = image.spectral_indices[SI_INDEX["h"]]

                    if w > h:
                        self.image_data[i] = np.rot90(self.image_data[i], k=-1, axes=(1, 2))
                        tmp_w = w
                        w = h
                        h = tmp_w

                    self.image_sizes[i, 0] = h
                    self.image_sizes[i, 1] = w

                self.metadata[i, 0] = sample["p2i_days"]
                self.metadata[i, 1] = sample["nitrogen_amount"]
                self.metadata[i, 2] = sample["rank"]

                # Add the target
                self.targets[i] = sample["yield"]

            np.save(pj(CACHE_DIR, f"{mode}_image_data.npy"), self.image_data)
            np.save(pj(CACHE_DIR, f"{mode}_image_sizes.npy"), self.image_sizes)
            np.save(pj(CACHE_DIR, f"{mode}_metadata.npy"), self.metadata)
            np.save(pj(CACHE_DIR, f"{mode}_targets.npy"), self.targets)

        print("[INFO] Dataset ready.")

    def __len__(self):
        return len(self.image_data)

    def __getitem__(self, idx):
        image = self.image_data[idx]
        image_size = self.image_sizes[idx]
        metadata = self.metadata[idx]
        target = self.targets[idx]

        # Apply augmentation
        if self.mode == "train":
            image = self.augment_image(image, image_size)

        # Convert numpy arrays to torch tensors
        image = torch.tensor(image, dtype=torch.float32)
        metadata = torch.tensor(metadata, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)

        return image, metadata, target

    def shift_image(self, image, image_size):
        shift_options = ["tl", "tr", "bl", "br"]
        option = random.choice(shift_options)

        if option == "tl":
            return image
        
        else:
            real_image = image[:, :image_size[0]+1, :image_size[1]+1]
            shifted_image = np.zeros_like(image)

            if option == "tr":
                shifted_image[:, image.shape[1]-image_size[0]-1:, :image_size[1]+1] = real_image
            elif option == "bl":
                shifted_image[:, :image_size[0]+1, image.shape[2]-image_size[1]-1:] = real_image
            elif option == "br":
                shifted_image[:, image.shape[1]-image_size[0]-1:, image.shape[2]-image_size[1]-1:] = real_image

            return shifted_image

    def shift_intensity(self, image):
        # Preserve the ratio by multiplying with a random factor
        factor = np.random.uniform(0.8, 1.2)
        image[:6, :, :] = image[:6, :, :] * factor
        image[:6, :, :] = np.clip(image[:6, :, :], 0, 1)
        return image

    def augment_image(self, image, image_size):
        # Flip horizontally and/or vertically
        if random.random() < AUG_FLIP:
            image = np.flip(image, axis=2).copy()
        if random.random() < AUG_FLIP:
            image = np.flip(image, axis=1).copy()
        if random.random() < AUG_SHIFT:
            image = self.shift_image(image, image_size)
        if random.random() < AUG_INTENS:
            image = self.shift_intensity(image)
        return image

def save_model(model, filename):
    torch.save(model.state_dict(), filename)
    print("Saving model {}.".format(filename))

def load_model(model_class, filename):
    model = model_class()
    model.load_state_dict(torch.load(filename))
    model.eval()  # Set the model to evaluation mode
    print("[INFO] Loading weights: {}".format(filename))
    return model

def train_model(model, dataloaders, criterion, optimizer, device, epochs):
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        for (images, float_values, targets) in dataloaders["train"]:
            images = images.to(device)
            float_values = float_values.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            
            output = model(images, float_values)
            loss = criterion(output, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * targets.size(0)
        
        # Calculate metric for the train set
        epoch_loss = running_loss / len(dataloaders["train"].dataset)
        rmse = torch.sqrt(torch.tensor(epoch_loss))
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        output_sum = 0.0

        with torch.no_grad():
            for images, float_values, targets in dataloaders["val"]:
                images = images.to(device)
                float_values = float_values.to(device)
                targets = targets.to(device)

                output = model(images, float_values)
                output_sum += output
                loss = criterion(output, targets)

                val_loss += loss.item() * targets.size(0)

        val_loss /= len(dataloaders["val"].dataset)
        val_rmse = torch.sqrt(torch.tensor(val_loss))
        val_mean = torch.mean(output)

        print(f"[{device}] Epoch {epoch+1}/{epochs}, RMSE (train): {rmse:.4f}, RMSE (val): {val_rmse:.4f}, mean: {val_mean:.2f}")

def train(filename, epochs, batch_size):
    print(filename)
    if os.path.isfile(filename):
        print("[WARN] Model already exists with the same name: {}".format(filename))
        return

    canon_data = pd.read_csv(CANONICAL_DATA_CSV)
    train_data = canon_data.loc[canon_data["set"] == "train"]
    val_data = canon_data.loc[canon_data["set"] == "val"]
    test_data = canon_data.loc[canon_data["set"] == "test"]
    
    train_dataset = MLDataset(pd.concat((train_data, val_data)), mode="train")
    val_dataset = MLDataset(val_data, mode="val")
    test_dataset = MLDataset(test_data, mode="test")

    print("[INFO] Training on device: {}".format(device))
    cnn = ResNet().to(device)

    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.001)

    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
        "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=True),
        "test": DataLoader(test_dataset, batch_size=test_data.shape[0], shuffle=False)
    }

    # start training
    train_model(cnn, dataloaders, criterion, optimizer, device, epochs)
    save_model(cnn, filename)

def predict(model, data_loader):
    model.to(device)
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, float_values, _ in data_loader:
            images = images.to(device)
            float_values = float_values.to(device)

            # Forward pass
            output = model(images, float_values)
            predictions.extend(output.cpu().numpy())
    
    return np.array(predictions).flatten()

def infer(model_file):
    model = load_model(ResNet, model_file)
    canon_data = pd.read_csv(CANONICAL_DATA_CSV)
    test_data = canon_data.loc[canon_data["set"] == "test"].copy()
    dataset = MLDataset(test_data, mode="test")
    dataloader = DataLoader(dataset, batch_size=test_data.shape[0], shuffle=False)
    prediction = predict(model, dataloader)
    
    test_data["prediction"] = prediction
    return test_data

def generate_results(prediction, model_id):
    test_file = pj(DATA_ROOT, "test", "2023", "DataPublication_final", "GroundTruth", "test_HIPS_HYBRIDS_2023_V2.3.csv")
    tf = pd.read_csv(test_file)
    tf["range"] = tf["range"].astype(int)
    tf["row"] = tf["row"].astype(int)

    for i in range(0, tf.shape[0]):
        tr = tf.iloc[i]
        pr = prediction.loc[(prediction["column"] == tr["range"]) & (prediction["row"] == tr["row"]) & 
               (prediction["block"] == tr["block"])]
        for j in range(0, pr.shape[0]):
            tf.loc[i, "prediction_{}".format(j+1)] = pr.iloc[j]["prediction"]

    train_data = pd.read_csv(TRAINING_DATA_CSV)
    df = train_data[["year", "location", "genotype", "yield"]]
    df = df.loc[(df["location"] == "MOValley")]
    year_average = df[["year", "yield"]].groupby(by="year").median()
    year_bias = year_average["yield"].max() - year_average["yield"].min()

    tf["yieldPerAcre"] = tf[["prediction_1", "prediction_2", "prediction_3"]].mean(axis=1) + year_bias
    
    tf = tf.drop(["prediction_{}".format(i) for i in range(1, 4)], axis=1)

    result_file = pj(RESULTS_DIR, "{}.csv".format(model_id))
    print("[INFO] Saving results to: {}".format(result_file))
    tf.to_csv(result_file, index=False)

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Please provide a model_id.")
        quit()

    model_id = str(sys.argv[1]).replace(" ", "")
    epochs = 500
    batch_size = 300

    train(pj(MODELS_DIR,"{}.pth".format(model_id)), epochs, batch_size)
    prediction = infer(pj(MODELS_DIR,"{}.pth".format(model_id)))
    generate_results(prediction, model_id)