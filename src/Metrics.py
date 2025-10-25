import math

class Metrics:
    def __init__(self, results, target):
        self.results = results
        self.target = target

    @property
    def recall_10(self):
        """
        Performs the RECALL@10 accuracy metric calculation on the guesses of the model
        Since we consider only the target answer, we are essentially performing Accuracy@10

        :return:
        If target answer in top 10: return 1
        If target answer not in top 10: return 0
        """
        # Get answers from the results dict
        answers = [answer["filename"] for answer in self.results]

        return 1 if self.target in answers else 0

    @property
    def mrr_10(self):
        """
        Performs the MRR@10 accuracy metric calculation on the guesses of the model
        Since we consider only the target answer, it effectively becomes 1/rank

        :return:
        If target answer in top 10: return 1/rank
        If target answer not in top 10: return 0
        """
        # Get answers and rank of guess from the results dict
        answers = [answer["filename"] for answer in self.results]
        if self.target not in answers:
            return 0

        rank = answers.index(self.target) + 1
        return 1 ** (1 / rank)

    @property
    def ndcg_10(self):
        """
        Performs the NDCG@10 accuracy metric calculation on the guesses of the model
        NDCG = DCG / IDCG

        Since we consider only the target answer, it effectively becomes DCG

        :return:
        If target answer in top 10: return DCG
        If target answer not in top 10: return 0
        """

        # Get answers and rank of guess from the results dict
        answers = [answer["filename"] for answer in self.results]
        if self.target not in answers:
            return 0

        rank = answers.index(self.target) + 1
        ndcg = 1 / math.log(rank + 1, 2)

        return ndcg

    @staticmethod
    def get_averages(all_metrics):
        # Calculate the means of each metric
        recalls = [metric.recall_10 for metric in all_metrics]
        mrrs = [metric.mrr_10 for metric in all_metrics]
        ndgcs = [metric.ndcg_10 for metric in all_metrics]

        return sum(recalls)/len(recalls), sum(mrrs)/len(mrrs), sum(ndgcs)/len(ndgcs)

    def __str__(self):
        return f"RECALL@10 {self.recall_10} \nMRR@10 {self.mrr_10} \nNDCG@10 {self.ndcg_10}"