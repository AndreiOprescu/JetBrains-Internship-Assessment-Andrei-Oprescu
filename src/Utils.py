import faiss
from sentence_transformers import InputExample


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


# --- 2. Indexing Phase ---

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

        # negative = positive
        # while negative != positive:
        #     negative = random.choice(row["code"])
        idx_query_code_groups.append((idx, anchor, positive))

    if display_data:
        print(idx_query_code_groups)

    return idx_query_code_groups

def display_distances_ranks(distances, ranks):
    padding = max(len(str(item)) for item in distances) + 2

    # print all query IDX and their ranks
    for item_a, item_b in zip(ranks, distances):
        print(f"{str(item_a):<{padding}} {str(item_b)}")
