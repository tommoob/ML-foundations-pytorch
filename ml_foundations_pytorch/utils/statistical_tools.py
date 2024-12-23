import numpy as np
import torch
import matplotlib.pyplot as plt
from ml_foundations_pytorch.utils.file_utils import get_next_experiment_dir
import os


np.set_printoptions(threshold=np.inf)


def predict_with_confidence(model, test_loader, n_classes=10):
    """
    Run the model on the test set and get predictions and confidence scores.

    Args:
        model: Trained MNIST detection model.
        test_loader: DataLoader for the test dataset.

    Returns:
        true_labels: Ground truth labels.
        pred_probs: Predicted probabilities (confidence scores).
        pred_labels: Predicted labels.
    """
    model.eval()
    true_labels = []
    pred_probs = []
    pred_labels = []
    probabilities_list = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert to probabilities
            confidences, predictions = torch.max(probabilities, dim=1)
            true_labels.extend(labels.numpy())
            pred_probs.extend(confidences.numpy())
            pred_labels.extend(predictions.numpy())
            probabilities_list.append(probabilities)

    return (
        np.array(true_labels),
        np.array(pred_probs),
        np.array(pred_labels),
        np.vstack(probabilities_list),
    )


def compute_precision_recall_at_threshold(
    true_labels, pred_probs, pred_labels, threshold
):
    """
    Compute precision and recall for a given confidence threshold.

    Args:
        true_labels: Ground truth labels.
        pred_probs: Predicted probabilities.
        pred_labels: Predicted labels.
        threshold: Confidence threshold.

    Returns:
        precision: Precision at the given threshold.
        recall: Recall at the given threshold.
    """
    confident_indices = pred_probs >= threshold
    filtered_preds = pred_labels[confident_indices]
    filtered_truths = true_labels[confident_indices]

    tp = np.sum((filtered_preds == filtered_truths) & (filtered_preds == 1))
    fp = np.sum((filtered_preds != filtered_truths) & (filtered_preds == 1))
    fn = np.sum((filtered_truths == 1) & (filtered_preds != 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def precision_recall_curve(
    true_labels, pred_probs, pred_labels, probabilities, num_classes=10
):
    pr_data = {}

    for positive_class in range(num_classes):
        positive_class_probs = probabilities[:, positive_class]
        thresholds = np.sort(np.unique(positive_class_probs))[::-1]

        precision_values = []
        recall_values = []

        for threshold in thresholds:
            confident_predictions = (pred_labels == positive_class) & (
                positive_class_probs >= threshold
            )

            tp = np.sum(confident_predictions & (true_labels == positive_class))
            fp = np.sum(confident_predictions & (true_labels != positive_class))
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = (
                tp / np.sum(true_labels == positive_class)
                if np.sum(true_labels == positive_class) > 0
                else 0
            )

            precision_values.append(precision)
            recall_values.append(recall)

        pr_data[positive_class] = {
            "thresholds": thresholds,
            "precision": precision_values,
            "recall": recall_values,
        }
    return pr_data


def plot_precision_recall_curve(true_labels, pred_probs, pred_labels, probabilities):
    """
    Plot the precision-recall curve.

    Args:
        true_labels: Ground truth labels.
        pred_probs: Predicted probabilities.
    """
    pr_curve_save_path = "experiments/runs/evaluate"
    pr_exp_dir = get_next_experiment_dir(pr_curve_save_path)
    pr_data = precision_recall_curve(
        true_labels, pred_probs, pred_labels, probabilities
    )
    categories = list(pr_data.keys())
    grouped_categories = np.array_split(
        categories, len(categories) // 3 + (len(categories) % 3 > 0)
    )

    # Plot each group
    for i, group in enumerate(grouped_categories):
        plt.figure(figsize=(10, 6))
        for category in group:
            precision = pr_data[category]["precision"]
            recall = pr_data[category]["recall"]
            plt.plot(recall, precision, label=f"Category {category}")

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(
            f"Precision-Recall Curves for Categories {', '.join(map(str, group))}"
        )
        plt.legend()
        plt.grid()
        group_string = "_".join(map(str, group.flatten()))
        plt.savefig(os.path.join(pr_exp_dir, f"PR_curve_{group_string}.png"))

        plt.show()


def mnist_detection_statistics(model, test_loader, confidence_threshold):
    """
    Run MNIST detection and compute statistics.

    Args:
        model: Trained MNIST detection model.
        test_loader: DataLoader for the test dataset.
        confidence_threshold: Confidence threshold for precision/recall computation.
    """
    (true_labels, pred_probs, pred_labels, probabilities) = predict_with_confidence(
        model, test_loader
    )

    precision, recall = compute_precision_recall_at_threshold(
        true_labels, pred_probs, pred_labels, confidence_threshold
    )
    print(f"Precision at {confidence_threshold:.2f} confidence: {precision:.2f}")
    print(f"Recall at {confidence_threshold:.2f} confidence: {recall:.2f}")

    plot_precision_recall_curve(true_labels, pred_probs, pred_labels, probabilities)
