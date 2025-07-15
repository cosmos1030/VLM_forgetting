import os
import csv
from typing import List, Tuple
import matplotlib.pyplot as plt
from PIL import Image

def build_prompt(class_names: List[str]) -> str:
    classes_str = ", ".join(class_names)
    prompt = (f"What is the main object in this image? Choose one word from these options: {classes_str}. "
              "Answer in the form: 'The answer is [object name].'")
    print("Using prompt: " + prompt)
    return prompt


def save_first_batch(batch_preds: List[Tuple[Image.Image, str, str]],
                     output_image_path: str): # out_dir -> output_image_path
    """
    batch_preds: [(PIL.Image, ground_truth, prediction), ...]
    """
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    num_images = len(batch_preds)
    cols = min(num_images, 5)
    rows = (num_images + cols - 1) // cols
    rows = min(rows, 2)

    fig = plt.figure(figsize=(cols * 2.5, rows * 2.5))
    for idx, (img, gt, pred) in enumerate(batch_preds[:cols*rows]):
        plt.subplot(rows, cols, idx + 1)
        plt.imshow(img)
        plt.axis("off")
        color = "green" if gt == pred else "red"
        plt.title(f"GT: {gt}\nPred: {pred}", fontsize=8, color=color)
    plt.tight_layout(pad=0.5)
    plt.savefig(output_image_path, dpi=200)
    plt.close(fig)
    print(f"First batch visualizations saved to {output_image_path}")

def save_csv(total: int, correct: int,
             preds: List[Tuple[str, str, str]],
             accuracy_csv_path: str,
             predictions_csv_path: str):
    """
    save csv files containing accuracy, whole prediction results
    """
    os.makedirs(os.path.dirname(accuracy_csv_path), exist_ok=True)
    os.makedirs(os.path.dirname(predictions_csv_path), exist_ok=True)

    acc = 100 * correct / total if total > 0 else 0

    # save accuracy.csv
    with open(accuracy_csv_path, "w", newline="") as f:
        csv.writer(f).writerows([
            ["total_samples", "correct_predictions", "accuracy_percent"],
            [total, correct, f"{acc:.2f}"]
        ])
    print(f"Accuracy stats saved to {accuracy_csv_path}")

    # save predictions.csv
    with open(predictions_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ground_truth", "prediction", "generated_text"])
        for gt, pred, full_text in preds:
            w.writerow([gt, pred, full_text])
    print(f"All predictions saved to {predictions_csv_path}")
