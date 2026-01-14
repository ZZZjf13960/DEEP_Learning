from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

class Evaluator:
    """
    Class to calculate evaluation metrics.
    """
    def __init__(self):
        pass

    def calculate_metrics(self, y_true, y_pred):
        """
        Calculate accuracy and F1 score.

        Args:
            y_true (list or np.ndarray): True labels.
            y_pred (list or np.ndarray): Predicted labels.

        Returns:
            dict: Dictionary containing metrics.
        """
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='weighted')
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": acc,
            "f1_score": f1,
            "confusion_matrix": cm
        }
