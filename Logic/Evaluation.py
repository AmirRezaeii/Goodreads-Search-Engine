import math
from typing import List


class Evaluation:
    def __init__(self, name: str):
        self.name = name

    def _validate(self, actual: List[List[str]], predicted: List[List[str]]):
        if len(actual) != len(predicted):
            raise ValueError("actual and predicted must have the same length")

    def calculate_precision(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro precision.
        """
        self._validate(actual, predicted)
        if len(actual) == 0:
            return 0.0
        
        precisions = []
        for i in range(len(actual)):
            if len(predicted[i]) == 0:
                precisions.append(0.0)
                continue
            
            relevant_retrieved = len(set(actual[i]) & set(predicted[i]))
            precision = relevant_retrieved / len(predicted[i])
            precisions.append(precision)
        
        return sum(precisions) / len(precisions)

    def calculate_recall(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates macro recall.
        """
        self._validate(actual, predicted)
        if len(actual) == 0:
            return 0.0
        
        recalls = []
        for i in range(len(actual)):
            if len(actual[i]) == 0:
                recalls.append(1.0 if len(predicted[i]) == 0 else 0.0)
                continue
            
            relevant_retrieved = len(set(actual[i]) & set(predicted[i]))
            recall = relevant_retrieved / len(actual[i])
            recalls.append(recall)
        
        return sum(recalls) / len(recalls)

    def calculate_F1(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates F1 score.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)

    def _average_precision_single(self, actual: List[str], predicted: List[str]) -> float:
        if len(predicted) == 0:
            return 0.0
        
        actual_set = set(actual)
        if len(actual_set) == 0:
            return 1.0 if len(predicted) == 0 else 0.0
        
        ap_sum = 0.0
        relevant_count = 0
        
        for i, doc in enumerate(predicted):
            if doc in actual_set:
                relevant_count += 1
                precision_at_k = relevant_count / (i + 1)
                ap_sum += precision_at_k
        
        return ap_sum / len(actual_set)

    def calculate_AP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean AP across all queries.
        """
        self._validate(actual, predicted)
        if len(actual) == 0:
            return 0.0
        
        ap_scores = []
        for i in range(len(actual)):
            ap = self._average_precision_single(actual[i], predicted[i])
            ap_scores.append(ap)
        
        return sum(ap_scores) / len(ap_scores)

    def calculate_MAP(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MAP.
        """
        return self.calculate_AP(actual, predicted)

    def _dcg_single(self, actual: List[str], predicted: List[str]) -> float:
        if len(predicted) == 0:
            return 0.0
        
        actual_set = set(actual)
        dcg = 0.0
        
        for i, doc in enumerate(predicted):
            if doc in actual_set:
                gain = 1
            else:
                gain = 0
            
            if i == 0:
                dcg += gain
            else:
                dcg += gain / math.log2(i + 1)
        
        return dcg

    def cacluate_DCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean DCG.
        """
        self._validate(actual, predicted)
        if len(actual) == 0:
            return 0.0
        
        dcg_scores = []
        for i in range(len(actual)):
            dcg = self._dcg_single(actual[i], predicted[i])
            dcg_scores.append(dcg)
        
        return sum(dcg_scores) / len(dcg_scores)

    def cacluate_NDCG(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates mean NDCG.
        """
        self._validate(actual, predicted)
        if len(actual) == 0:
            return 0.0
        
        ndcg_scores = []
        for i in range(len(actual)):
            dcg = self._dcg_single(actual[i], predicted[i])
            
            ideal_predicted = sorted(predicted[i], key=lambda x: 1 if x in set(actual[i]) else 0, reverse=True)
            idcg = self._dcg_single(actual[i], ideal_predicted)
            
            if idcg == 0:
                ndcg_scores.append(1.0 if dcg == 0 else 0.0)
            else:
                ndcg_scores.append(dcg / idcg)
        
        return sum(ndcg_scores) / len(ndcg_scores)

    def cacluate_RR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculate reciprocal rank.
        """
        self._validate(actual, predicted)
        if len(actual) == 0:
            return 0.0
        
        rr_scores = []
        for i in range(len(actual)):
            actual_set = set(actual[i])
            rr = 0.0
            
            for j, doc in enumerate(predicted[i]):
                if doc in actual_set:
                    rr = 1.0 / (j + 1)
                    break
            
            rr_scores.append(rr)
        
        return sum(rr_scores) / len(rr_scores)

    def cacluate_MRR(self, actual: List[List[str]], predicted: List[List[str]]) -> float:
        """
        Calculates MRR.
        """
        return self.cacluate_RR(actual, predicted)
        
    def print_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Prints the evaluation metrics.
        """
        print(f"name = {self.name}")
        print(f"Precision = {precision:.6f}")
        print(f"Recall = {recall:.6f}")
        print(f"F1 = {f1:.6f}")
        print(f"AP = {ap:.6f}")
        print(f"MAP = {map:.6f}")
        print(f"DCG = {dcg:.6f}")
        print(f"NDCG = {ndcg:.6f}")
        print(f"RR = {rr:.6f}")
        print(f"MRR = {mrr:.6f}")

    def log_evaluation(self, precision, recall, f1, ap, map, dcg, ndcg, rr, mrr):
        """
        Use Wandb to log the evaluation metrics.
        """
        try:
            import wandb
            if wandb.run is not None:
                wandb.log({
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'ap': ap,
                    'map': map,
                    'dcg': dcg,
                    'ndcg': ndcg,
                    'rr': rr,
                    'mrr': mrr,
                })
        except Exception:
            pass

    def calculate_evaluation(self, actual: List[List[str]], predicted: List[List[str]]):
        """
        Call all functions to calculate evaluation metrics.
        """
        precision = self.calculate_precision(actual, predicted)
        recall = self.calculate_recall(actual, predicted)
        f1 = self.calculate_F1(actual, predicted)
        ap = self.calculate_AP(actual, predicted)
        map_score = self.calculate_MAP(actual, predicted)
        dcg = self.cacluate_DCG(actual, predicted)
        ndcg = self.cacluate_NDCG(actual, predicted)
        rr = self.cacluate_RR(actual, predicted)
        mrr = self.cacluate_MRR(actual, predicted)
        
        self.print_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        self.log_evaluation(precision, recall, f1, ap, map_score, dcg, ndcg, rr, mrr)
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'ap': ap,
            'map': map_score,
            'dcg': dcg,
            'ndcg': ndcg,
            'rr': rr,
            'mrr': mrr
        }
