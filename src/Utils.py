import itertools
import json

import faiss
import numpy as np
from matplotlib import pyplot as plt
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.Metrics import Metrics
from src.config import DEBUGGING, device


def search_engine_train(model, train_params, train_set):
    print(f"Using device: {device}")

    # Prepare the DataLoader
    train_dataloader = DataLoader(train_set, shuffle=True, batch_size=train_params["batch_size"])
    train_dataloader.collate_fn = model.smart_batching_collate

    # Create loss function
    train_loss = losses.MultipleNegativesRankingLoss(model=model).to(device)
    # Create Optimizer
    optimizer = AdamW(model.parameters(), lr=train_params["lr"])

    # --- 4. Prepare Scheduler ---
    num_epochs = train_params["epochs"]
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=train_params["warmup"],
                                                num_training_steps=total_steps)
    # Create plot for the losses
    plt.title("Losses per Step")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Store the average losses
    average_losses = []

    # Store losses for line of best fit
    all_step_losses_combined = []

    # Set model to training mode
    print("Starting fine-tuning process...")
    model.train()

    epoch_steps = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch: {epoch}")
        all_step_losses_epoch = []

        step = 1
        for batch in train_dataloader:
            features, labels = batch

            # Move batch to device
            features = list(map(lambda batch_item: {k: v.to(device) for k, v in batch_item.items()}, features))
            labels = labels.to(device) if labels is not None else None

            optimizer.zero_grad()

            # Calculate losses and gradients
            loss_value = train_loss(features, labels)
            loss_value.backward()

            # Do a step for the optimizer and scheduler
            optimizer.step()
            scheduler.step()

            # Get the losses to plot later
            current_loss = loss_value.item()
            all_step_losses_epoch.append(current_loss)
            all_step_losses_combined.append(current_loss)

            # Count steps for logging
            if step % 10 == 0:
                print(f"Step: {step}")
            step += 1


        # Plot the losses for intermediate steps
        plt.plot(all_step_losses_epoch, label=f"Epoch: {epoch+1}")

        # Store the average loss for this epoch
        average_losses.append(sum(all_step_losses_epoch) / len(all_step_losses_epoch))

    # Plot the losses per epoch
    plt.legend()
    plt.show()

    print("Training complete.")

    # Save the model
    model.save(f'./cosqa-finetuned-model-{train_params.values()}')

    print(f"\nCaptured {len(average_losses)} step losses.")
    # print("First 10 losses:", all_step_losses[:10])

    # Plot average losses
    plt.plot(average_losses)
    plt.title("Training Loss per Epoch")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.show()


def search_engine_test(model, val_raw, val_set, display = False):
    # Run the model on the validation set and collect and calculate the evaluation metrics
    print("Testing the model on the validation data... ")
    index, all_documents, index_to_filename = create_search_index(val_raw, model)
    val_metrics = []

    for idx, query, code in val_set:
        print(f"IDX: {idx}")

        search_results = search(query, index, model, index_to_filename, all_documents, display=display)
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

def search_engine_demo(model, val_raw):
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


def load_model(device, pretrained_model = None):
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

def search(query_text, index, model, index_to_filename, all_documents, k=10, display=False):
    """
    Provides the "API" to search the index.

    Args:
        query_text (str): The natural language search query.
        k (int): The number of top results to return.

    :return:
        list: A list of search results.
    """
    if not index:
        print("Index is not initialized.")
        return []

    # 1. Generate an embedding for the query
    # Note: model.encode() expects a list, so we wrap the query in []
    query_embedding = model.encode([query_text])

    # 2. Search the index
    # D = distances (float), I = indices (int)
    # I will be a 2D numpy array, e.g., [[5, 2]] for k=2
    distances, indices = index.search(query_embedding, k)
    if display:
        display_distances_ranks(distances[0], indices[0])

    # 3. Format and return the results
    results = []
    for i in indices[0]:  # Iterate through the top-k indices
        filename = index_to_filename[i]
        content = all_documents[i]
        results.append({
            "filename": filename,
            "content_snippet": content[:512] + "..."  # Show a snippet
        })
    return results


def create_search_index(df, model):
    """
    Reads documents, generates embeddings, and builds a FAISS index.
    """
    documents = []
    # We need to map the index-position back to the filename
    index_to_filename = {}

    print(f"Loading and indexing {len(df)} documents...")

    # row structure: idx, doc, code, code_tokens, docstring_tokens, label
    for index, row in df.iterrows():
        code = row["code"]
        documents.append(code)
        index_to_filename[index] = row["idx"]

    if not documents:
        print("No documents were loaded. Exiting.")
        return None, None, None

    # Generate embeddings for all documents
    print("Generating embeddings...")
    doc_embeddings = model.encode(documents, show_progress_bar=True)

    # Get the dimensionality of the embeddings
    dimensions = doc_embeddings.shape[1]

    # Create a FAISS index: performs brute-force L2 (Euclidean) distance search.
    index = faiss.IndexFlatL2(dimensions)
    index.add(doc_embeddings)

    print(f"Index created successfully with {index.ntotal} documents.")
    return index, documents, index_to_filename


def format_train_dataset(df, display_data=False):
    dataset = []
    # row structure: idx, doc, code, code_tokens, docstring_tokens, label
    for index, row in df.iterrows():
        anchor = row["doc"]  # doc
        positive = row["code"]  # code

        # negative = positive
        # while negative != positive:
        #     negative = random.choice(row["code"])
        dataset.append(InputExample(texts=[anchor, positive]))

    if display_data:
        print(df)

    return dataset

def format_test_dataset(df, display_data=False):
    idx_query_code_groups = []
    # row structure: idx, doc, code, code_tokens, docstring_tokens, label
    for index, row in df.iterrows():
        idx = row["idx"]
        anchor = row["doc"]  # doc
        positive = row["code"]  # code

        idx_query_code_groups.append((idx, anchor, positive))

    if display_data:
        print(idx_query_code_groups)

    return idx_query_code_groups

def display_distances_ranks(distances, ranks):
    padding = max(len(str(item)) for item in distances) + 2

    # print all query IDX and their ranks
    for item_a, item_b in zip(ranks, distances):
        print(f"{str(item_a):<{padding}} {str(item_b)}")
