import pandas as pd

from src.Utils import *
import matplotlib.pyplot as plt
from src.config import *


print(f"--- Using device: {device} ---")

# Specify params for training
train_params = {"batch_size": 16, "input_length": 512, "lr": 0.00002, "shuffle": True, "epochs": 5, "warmup": 50}

# ---- Loading the Dataset ----

# specify the train/val split for this dataset
splits = {'train': 'cosqa-train.json', 'validation': 'cosqa-dev.json'}

# get train and val CoSQA datasets
print("Loading CoSQA dataset...")
train_raw = pd.read_json("hf://datasets/gonglinyuan/CoSQA/" + splits["train"])
val_raw = pd.read_json("hf://datasets/gonglinyuan/CoSQA/" + splits["validation"])

train_set = format_train_dataset(val_raw, display_data=DEBUGGING)
val_set = format_test_dataset(val_raw, display_data=DEBUGGING)

# ---- Examining the dataset ----

code_lengths = [len(x) for x in train_raw["code"].values]

print(train_raw["code"][0])
print("Code example:")

print(train_raw["doc"][0])
print("Code example:")


print("MIN LENGTH   |   MAX LENGTH: ", min(code_lengths), max(code_lengths))

# Get the number of bins we should group the counts in. Bins are in intervals of 256
n_bins = (max(code_lengths)) // 256

plt.hist(code_lengths, range=(0, max(code_lengths)), bins = n_bins)
plt.xlabel("Code Length")
plt.ylabel("Frequency")
plt.show()

# The plot shows that most codes examples are less than 256 characters long, with many in the 256-512 range
# Therefore the max_seq_length should probably be increased to 512 during the training phase to capture more information

# Inspect the dataset structure
if DEBUGGING:
    print("\n--- Dataset Structure ---")
    print(val_raw)


# Load pretrained model
model = load_model(PRETRAINED_MODEL)

# --- Main Execution ---
if __name__ == "__main__":
    if MODE == Mode.TRAIN:
        search_engine_train(model, train_params, train_set)
        # Also evaluate it
        search_engine_test(model, val_raw, val_set, display=DEBUGGING)
    elif MODE == Mode.DEMO:
        search_engine_demo(model, val_raw)
    elif MODE == Mode.TEST:
        search_engine_test(model, val_raw, val_set, display=DEBUGGING)