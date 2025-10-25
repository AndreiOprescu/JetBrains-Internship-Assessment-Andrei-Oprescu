import itertools
import json
from enum import Enum

import torch
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader
import pandas as pd

from Metrics import Metrics
from Utils import *
import matplotlib.pyplot as plt

class Mode(Enum):
    NONE = 0
    TRAIN = 1
    TEST = 2
    DEMO = 3

def load_model(pretrained_model = None):
    model = None
    try:
        if pretrained_model is not None:
            print("Loading default model")
            model = SentenceTransformer(pretrained_model, device=device)
        else:
            print(f"Loading fine-tuned model: {pretrained_model}")
            model = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have an internet connection and 'sentence-transformers' is installed.")
        exit()
    return model

# specify the train/val split for this dataset
splits = {'train': 'cosqa-train.json', 'validation': 'cosqa-dev.json'}

# get train and val CoSQA datasets
print("Loading CoSQA dataset...")
train_raw = pd.read_json("hf://datasets/gonglinyuan/CoSQA/" + splits["train"])
val_raw = pd.read_json("hf://datasets/gonglinyuan/CoSQA/" + splits["validation"])


code_lengths = [len(x) for x in train_raw["code"].values]
print("MIN LENGTH   |   MAX LENGTH: ", min(code_lengths), max(code_lengths))

# Get the number of bins we should group the counts in. Bins are in intervals of 256
n_bins = (max(code_lengths)) // 256

plt.hist(code_lengths, range=(0, max(code_lengths)), bins = n_bins)
plt.xlabel("Code Length")
plt.ylabel("Frequency")
plt.show()

# The plot shows that most codes examples are less than 256 characters long, with many in the 256-512 range
# Therefore the max_seq_length should probably be increased to 512 during the training phase to capture more information

# --- 1. Configuration ---

MODE = Mode.TRAIN
DEBUGGING = False
PRETRAINED_MODEL = None # Change to './cosqa-finetuned-model' for the finetuned model
RESULTS_LOG_FILE = "../result.txt"

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Using device: {device} ---")

train_params = {"batch_size": [128, 256], "input_length": [256, 512], "lr": [0.9, 0.95], "shuffle": [True], "epochs": [2, 4], "warmup": [50]}

# Load pretrained model
model = load_model(PRETRAINED_MODEL)

train_set = format_train_dataset(train_raw, display_data=DEBUGGING)
val_set = format_test_dataset(val_raw, display_data=DEBUGGING)

# Inspect the dataset structure
if DEBUGGING:
    print("\n--- Dataset Structure ---")
    print(val_raw)


def search_engine_demo(model):
    # On start, load documents and build the index
    index, all_documents, index_to_filename = create_search_index(val_raw, model)
    if index:
        print("\n--- Simple Code Search Engine Ready ---")
        print("Type your query, or 'q' to quit.")

        # Demonstrate the work of the search engine on test samples
        while True:
            query = input("\n> Search query: ")
            if query.lower() == 'q':
                break

            # Use the search "API"
            search_results = search(query, index, model, index_to_filename, all_documents, k=10)

            # Print top results
            print(f"\n--- Top {len(search_results)} results for '{query}' ---")
            if not search_results:
                print("No results found.")

            for res in search_results:
                print(f"\n[File: {res['filename']}]")
                print(res['content_snippet'])
                print("-" * 20)


def search_engine_test(model):
    # Run the model on the validation set and collect and calculate the evaluation metrics
    print("Testing the model on the validation data... ")
    index, all_documents, index_to_filename = create_search_index(val_raw, model)
    val_metrics = []

    for idx, query, code in val_set:
        print(f"IDX: {idx}")

        search_results = search(query, index, model, index_to_filename, all_documents, display=DEBUGGING)
        metrics = Metrics(search_results, idx)
        val_metrics.append(metrics)

        print("")

    # Calculate the average evaluation metrics
    recall_avg, mrr_avg, ndgc_avg = Metrics.get_averages(val_metrics)
    print(f"RECALL@10 AVERAGE: {recall_avg}")
    print(f"MRR@10 AVERAGE: {mrr_avg}")
    print(f"NDGC@10 AVERAGE: {ndgc_avg}")

    # Returns the metrics of the model
    return recall_avg, mrr_avg, ndgc_avg


def search_engine_train(params, model):
    # change the input length for the model (if the input is longer, it will be truncated)
    model.max_seq_length = params["input_length"]
    # Make the dataloader
    train_dataloader = DataLoader(train_set, shuffle=params["shuffle"], batch_size=params["batch_size"])

    # Loss function:
    #   "Pulls" the target code closer to the anchor
    #   "Pushes" the code examples from other anchors in the same batch away
    train_loss = losses.MultipleNegativesRankingLoss(model=model)

    # Fine-tune the model
    print("Starting fine-tuning process...")

    model.fit(train_objectives=[(train_dataloader, train_loss)],
              epochs=params["epochs"],  # Keep it to 1 epoch for a quick demo
              warmup_steps=params["warmup"],
              output_path=f'./cosqa-finetuned-model-{params.values()}',
              show_progress_bar=True,
              optimizer_params={'lr': params["lr"]})

def search_engine_grid_search(params):
    # Get the keys and the lists of values for the grid
    grid_keys = params.keys()
    grid_value_lists = params.values()

    print(f"Grid params to combine: {params}\n")

    # Clear the file at the start of the grid search
    with open(RESULTS_LOG_FILE, "w") as f:
        f.write("--- Grid Search Training Log ---\n")

    # Create all combinations
    total_runs = 1
    for combo_tuple in itertools.product(*grid_value_lists):
        print(f"--- Run {total_runs} ---")

        # Create a dict from the current grid combination
        current_grid_dict = dict(zip(grid_keys, combo_tuple))
        print(f"Current params:\n {current_grid_dict}\n")

        model = load_model()
        # Call the training function with this set of params
        search_engine_train(current_grid_dict, model)
        # Test the model and get metrics
        current_metrics = search_engine_test(model)

        # Log the params and metrics to a text file for later reference
        with open(RESULTS_LOG_FILE, "a") as f:
            f.write(f"--- Run {total_runs} Results ---\n")
            # Use json.dumps for a nice, readable dict format
            f.write(json.dumps(current_grid_dict, indent=2))
            f.write("\n")
            f.write(json.dumps(current_metrics, indent=2))
            f.write("\n\n")

        # Print the params
        total_runs += 1


# --- 4. Main Execution ---
if __name__ == "__main__":
    if MODE == Mode.TRAIN:
        search_engine_grid_search(train_params)
    elif MODE == Mode.DEMO:
        search_engine_demo(model)
    elif MODE == Mode.TEST:
        search_engine_test(model)