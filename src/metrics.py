import numpy as np
import pandas as pd


class MeanAveragePrecisionAtK():
    ##################################################################################################################
    #### EXTRACTED FROM : https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/average_precision.py ####
    ##################################################################################################################

    def __apk(self, actual, predicted, k=10):
        """
        Computes the average precision at k.

        This function computes the average prescision at k between two lists of
        items.

        Parameters
        ----------
        actual : list
                 A list of elements that are to be predicted (order doesn't matter)
        predicted : list
                    A list of predicted elements (order does matter)
        k : int, optional
            The maximum number of predicted elements

        Returns
        -------
        score : double
                The average precision at k over the input lists

        """
        if len(predicted) > k:
            predicted = predicted[:k]

        score = 0.0
        num_hits = 0.0

        for i, p in enumerate(predicted):
            if p in actual and p not in predicted[:i]:
                num_hits += 1.0
                score += num_hits / (i + 1.0)

        if not actual:
            return 0.0

        return score / min(len(actual), k)

    def compute(self, actual: np.array, predicted: np.array, k: int = 10):
        """
        Computes the mean average precision at k.

        This function computes the mean average prescision at k between two lists
        of lists of items.

        Parameters
        ----------
        actual : list
                 A list of lists of elements that are to be predicted
                 (order doesn't matter in the lists)
        predicted : list
                    A list of lists of predicted elements
                    (order matters in the lists)
        k : int, optional
            The maximum number of predicted elements

        Returns
        -------
        score : double
                The mean average precision at k over the input lists

        """
        return np.mean([self.__apk(a, p, k) for a, p in zip(actual, predicted)])


def compute_binary_mask_metrics(ground_truth, predicted):
    """
    Computes evaluation metrics (Precision, Recall, F1-score, IoU) for a single binary mask prediction
    compared to the ground truth.

    Parameters:
    -----------
    ground_truth : array-like
        Binary ground truth mask where True/1 represents the positive class.
    predicted : array-like
        Binary predicted mask where True/1 represents the positive class.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing Precision, Recall, F1-score, and IoU for the mask comparison.
    """
    ground_truth = np.asarray(ground_truth).astype(bool)
    predicted = np.asarray(predicted).astype(bool)

    # True Positives, False Positives, and False Negatives
    TP = np.sum((predicted == 1) & (ground_truth == 1))
    FP = np.sum((predicted == 1) & (ground_truth == 0))
    FN = np.sum((predicted == 0) & (ground_truth == 1))

    # Metrics calculations with zero-checking
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 0

    metrics_df = pd.DataFrame({
        'Precision': [precision],
        'Recall': [recall],
        'F1-score': [f1_score],
        'IoU': [iou]
    })

    return metrics_df


def compute_average_binary_mask_metrics(ground_truth_list, predicted_list):
    """
    Computes the average evaluation metrics across multiple binary mask predictions and ground truth pairs.

    Parameters:
    -----------
    ground_truth_list : list of array-like
        List of binary ground truth masks.
    predicted_list : list of array-like
        List of binary predicted masks.

    Returns:
    --------
    pd.DataFrame
        DataFrame containing the average Precision, Recall, F1-score, and IoU across all mask comparisons.
    """
    assert len(ground_truth_list) == len(predicted_list), "Lists must have the same length"

    metrics_list = [compute_binary_mask_metrics(gt, pred) for gt, pred in zip(ground_truth_list, predicted_list)]

    # Concatenate individual DataFrames and calculate averages
    all_metrics_df = pd.concat(metrics_list, ignore_index=True)
    avg_metrics_df = pd.DataFrame(all_metrics_df.mean()).T
    avg_metrics_df.index = ['Average']

    return avg_metrics_df
