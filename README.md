# README: ML for Context in AI Assistant

This is the GitHub repository of Andrei Oprescu for the implementation of the JetBrains Internship task.
Below you can find commentaries on the tasks given, as well as instructions on how to run the search-engine model and API.

## Task 1: Embeddings-based search engine

For the implementation of the search engine, I used the all-MiniLM-L6-v2 sentence transformer from HuggingFace. 
To store and retrieve the embedding vectors, I used the FAISS index, which performs brute-force L2 distance retrieval.


## Task 2: Evaluation

The CoSQA dataset provided with the task showed strange values (https://huggingface.co/datasets/CoIR-Retrieval/cosqa), which were not representative of a search-engine task.
Therefore, I got another CoSQA dataset from HuggingFace that offered data fit for the task at hand:
https://huggingface.co/datasets/gonglinyuan/CoSQA/viewer/default/validation?views%5B%5D=validation

### Implementing Metrics

For the implementation of metrics, I created the Metrics class in Metrics.py. The class accepts the top k results returned by the model and the target result.
By calling recall_10(), mrr_10() and ndgc_10(), you can get the metrics for the given query result.

### Calculating Metrics

In the method "search_engine_test()", for each query in the validation set, the 3 metrics are calculated.
To get the average of each metric, the "get_averages(all_metrics)" method is used in the Metrics class, which returns the averages in a 3-tuple

For the untrained model, these are the following metrics on the validation set:

RECALL@10 AVERAGE: 0.9668874172185431
MRR@10 AVERAGE: 0.9668874172185431
NDGC@10 AVERAGE: 0.804717351117915

Since I only considered if the correct answer was in the top 10 (i.e. the other answers in the top 10 did not have a relevance score), the following calculations were performed.

*RECALL@10:* This now essentially becomes ACCURACY@10. 
If the answer is in the top 10, the metric is 1. 
Otherwise, 0.

*MRR@10:*
If target answer in top 10: return 1/rank
If target answer not in top 10: return 0

*NDGC@10:*
If target answer in top 10: return '1 / math.log(rank + 1, 2)'
If target answer not in top 10: return 0
