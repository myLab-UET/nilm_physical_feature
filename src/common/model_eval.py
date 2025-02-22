from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
from itertools import combinations
from tqdm import tqdm
import pandas as pd

class ModelEvaluation:
    def cal_custom_f1_score_macro(self, y_true, y_pred):
        """
        Calculate the macro F1-score.

        Args:
            y_true (list or numpy array): True labels.
            y_pred (list or numpy array): Predicted labels.

        Returns:
            float: Macro F1-score.
        """
        # Initialize dictionaries to store true positive, false positive, and false negative counts
        tp_dict = {}
        fp_dict = {}
        fn_dict = {}

        # Calculate true positive, false positive, and false negative counts for each class
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                tp_dict[true] = tp_dict.get(true, 0) + 1
            else:
                fp_dict[pred] = fp_dict.get(pred, 0) + 1
                fn_dict[true] = fn_dict.get(true, 0) + 1

        # Calculate precision, recall, and F1-score for each class
        f1_scores = {}
        for cls in set(y_true):
            precision = tp_dict.get(cls, 0) / (tp_dict.get(cls, 0) + fp_dict.get(cls, 0) + 1e-9)
            recall = tp_dict.get(cls, 0) / (tp_dict.get(cls, 0) + fn_dict.get(cls, 0) + 1e-9)
            f1_scores[cls] = 2 * (precision * recall) / (precision + recall + 1e-9)

        # Calculate macro F1-score
        macro_f1 = sum(f1_scores.values()) / len(f1_scores)

        return macro_f1
    
    def cal_f1_score(self, y_true, y_pred):
        return f1_score(y_pred=y_pred, y_true=y_true, average='macro')
    
    def cal_precision_recall_macro(self, y_true, y_pred):
        """
        Calculate macro precision and macro recall.

        Args:
            y_true (list or numpy array): True labels.
            y_pred (list or numpy array): Predicted labels.

        Returns:
            float: Macro precision.
            float: Macro recall.
        """
        # Initialize dictionaries to store true positive, false positive, and false negative counts
        tp_dict = {}
        fp_dict = {}
        fn_dict = {}

        # Calculate true positive, false positive, and false negative counts for each class
        for true, pred in zip(y_true, y_pred):
            if true == pred:
                tp_dict[true] = tp_dict.get(true, 0) + 1
            else:
                fp_dict[pred] = fp_dict.get(pred, 0) + 1
                fn_dict[true] = fn_dict.get(true, 0) + 1

        # Calculate precision and recall for each class
        precision_scores = {}
        recall_scores = {}
        for cls in set(y_true):
            #1e-9 to advoid divine by 0
            precision = tp_dict.get(cls, 0) / (tp_dict.get(cls, 0) + fp_dict.get(cls, 0) + 1e-9)
            recall = tp_dict.get(cls, 0) / (tp_dict.get(cls, 0) + fn_dict.get(cls, 0) + 1e-9)
            precision_scores[cls] = precision
            recall_scores[cls] = recall

        # Calculate macro precision and macro recall
        macro_precision = sum(precision_scores.values()) / len(precision_scores)
        macro_recall = sum(recall_scores.values()) / len(recall_scores)

        return macro_precision, macro_recall
    
    def evaluate_and_print_results(self, y_pred, y_true, label_encoder: LabelEncoder, working_labels):
        # Tính toán độ chính xác
        accuracy = accuracy_score(y_true=y_true, y_pred=y_pred)
        # Tính toán F1-Score
        # f1_macro = f1_score(y_true=y_test, y_pred=y_pred, average='macro')
        f1_macro = self.f1_score_macro(y_true=y_true, y_pred=y_pred)
        #Precision and recall
        macro_precision = precision_score(y_true=y_test, y_pred=y_pred, average='macro')  
        macro_recall = recall_score(y_true=y_test, y_pred=y_pred, average='macro')
        # macro_precision, macro_recall = precision_recall_macro(y_true=y_true, y_pred=y_pred)
        #Revert transform
        y_test = label_encoder.inverse_transform(y_true)
        y_pred = label_encoder.inverse_transform(y_pred)
        # Print results
        print(classification_report(y_true=y_test, y_pred=y_pred, labels=working_labels, output_dict=True))
        print("[+] Evaluation results:")
        print(f"Accuracy: {accuracy}")
        print(f"F1-Score (Macro): {f1_macro}")
        print(f"Precision: {macro_precision}")
        print(f"Recall: {macro_recall}")
    
    def print_evaluation_result(self, model, X_test, y_test):
        y_pred_test = model.predict(X_test)
        print("[+] Evaluation results:")
        self.evaluate_and_print_results(y_pred=y_pred_test, y_test=y_test)
    
    def split_test_set_into_batches(self, X_test, y_test, batch_size, model):
        """
        Splits the test set (X_test and y_test) into batches of the specified size,
        and predicts using the provided model.

        Args:
            X_test (numpy.ndarray): The test features (input data).
            y_test (numpy.ndarray): The test labels (output data).
            batch_size (int): The desired batch size.
            model: Your trained machine learning model (e.g., RandomForestRegressor).

        Returns:
            list of tuple: A list of (y_pred_batch, y_test_batch) tuples.
        """
        num_samples = len(X_test)
        num_batches = num_samples // batch_size

        # Shuffle the test data indices for randomness
        indices = np.random.permutation(num_samples)

        # Split the indices into batches
        batch_indices = np.array_split(indices, num_batches)

        all_y_pred = np.array([])
        all_y_test = np.array([])

        # Create batches using the shuffled indices
        for batch_idx in tqdm(batch_indices, desc="Processing batches"):  # Use tqdm here
            X_test_batch = X_test[batch_idx]
            y_test_batch = y_test[batch_idx]
            y_pred_batch = model.predict(X_test_batch)
            all_y_pred = np.append(all_y_pred, y_pred_batch)
            all_y_test = np.append(all_y_test, y_test_batch)

        return all_y_pred, all_y_test
    
    def generate_combinations(self, arr):
        """
        Generates all combinations of elements from the input array.

        Args:
            arr (list): List of elements.
            n (int): Length of combinations.

        Returns:
            list: List of tuples representing combinations.
        """
        result = []
        for i in range(1, len(arr)+1):
            all_combinations = list(combinations(arr, i))
            combination_strings = [",".join(map(str, combo)) for combo in all_combinations]
            result.extend(combination_strings)
        return result 
    
    def cal_accuracy(self, y_true, y_pred):
        return accuracy_score(y_true=y_true, y_pred=y_pred)

    # Function to report most error class
    def report_most_error_each_class(self, y_true: np.array, y_pred: np.array, label_encoder: LabelEncoder) -> pd.DataFrame:
        """
        This function takes y_true, y_pred, and label encoder, for each class, it will find the class that has the most errors respective
        to that class, then return a dataframe with the class name, the number of errors, and the percentage of errors.
        Args:
            y_true (np.array): true labels
            y_pred (np.array): predicted labels
            label_encoder (LabelEncoder): label encoder

        Returns:
            DataFrame: with columns: class_name, f1_score, most_error_class, most_error_count, most_error_percentage
        """

        classes = label_encoder.classes_
        f1_scores = f1_score(y_true, y_pred, average=None)

        results = []

        for i, cls in enumerate(classes):
            indices = np.where(y_true == i)[0]
            y_true_i = y_true[indices]
            y_pred_i = y_pred[indices]
            misclassified = y_pred_i != y_true_i
            num_errors = np.sum(misclassified)

            if num_errors > 0:
                misclassified_labels = y_pred_i[misclassified]
                mis_class_counts = np.bincount(misclassified_labels.astype(np.int64), minlength=len(classes))
                most_error_class_idx = np.argmax(mis_class_counts)
                most_error_count = mis_class_counts[most_error_class_idx]
                most_error_percentage = (most_error_count / num_errors) * 100
                most_error_class_name = label_encoder.inverse_transform([most_error_class_idx])[0]
            else:
                most_error_class_name = None
                most_error_count = 0
                most_error_percentage = 0.0
            if len(indices) == 0:
                result = {
                    'class_name': cls,
                    'f1_score': f1_scores[i],
                    'accuracy': 0,
                    'number_of_samples': len(indices),
                    'most_error_class': None,
                    'most_error_count': 0,
                    'most_error_percentage_wrt_class': 0,
                    'most_error_percentage_wrt_total_error': 0,
                }
            else: 
                result = {
                    'class_name': cls,
                    'f1_score': f1_scores[i],
                    'accuracy': (len(indices) - num_errors) / len(indices),
                    'number_of_samples': len(indices),
                    'most_error_class': most_error_class_name,
                    'most_error_count': most_error_count,
                    'most_error_percentage_wrt_class': most_error_count / len(indices),
                    'most_error_percentage_wrt_total_error': most_error_percentage,
                }
            results.append(result)

        report_df = pd.DataFrame(results)
        return report_df