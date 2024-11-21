import torch
from transformers import DonutProcessor
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import os
import pandas as pd
import json
from datasets import load_from_disk


def evaluate_donut_model(test_dataset_path, model_path, output_dir, base_model="naver-clova-ix/donut-base-finetuned-docvqa"):
    """
    Evaluate a Donut model on a given dataset and compute metrics.

    Args:
        test_dataset_path (str): Path to the test dataset in Huggingface format.
        model_path (str): Path to the trained Donut model.
        output_dir (str): Directory to save the evaluation results and metrics.
        base_model (str): Base model name for DonutProcessor (default: naver-clova-ix/donut-base-finetuned-docvqa).
    """

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the processor and model
    processor = DonutProcessor.from_pretrained(base_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.eval()

    # Load the dataset
    test_dataset = load_from_disk(test_dataset_path)["test"].select(range(2))
    # Initialize lists for predictions and ground truth
    predictions = []
    ground_truths = []

    # Evaluate the model on the test dataset
    for sample in tqdm(test_dataset):
        # Load image
        image = sample["image"].convert("RGB")
        # Get ground truth
        gt = json.loads(sample["ground_truth"])
        gt_json = gt["gt_parse"] if "gt_parse" in gt else gt["gt_parses"][0]

        # Preprocess the image
        pixel_values = processor(image, return_tensors="pt").pixel_values
        pixel_values = pixel_values.to(device)

        # Prepare the decoder input
        decoder_input_ids = torch.full(
            (1, 1),
            model.config.decoder_start_token_id,
            device=device,
        )

        # Perform inference
        outputs = model.generate(
            pixel_values,
            decoder_input_ids=decoder_input_ids,
            max_length=model.decoder.config.max_position_embeddings,
            early_stopping=True,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
            num_beams=1,
            bad_words_ids=[[processor.tokenizer.unk_token_id]],
        )

        # Decode and clean up the output
        seq = processor.batch_decode(outputs)[0]
        seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "").strip()
        pred_json = processor.token2json(seq)

        # Append ground truth and prediction
        pred_json['text_sequence'] = pred_json['text_sequence'].lstrip('<unk> ').strip()
        ground_truths.append(gt_json)
        predictions.append(pred_json)

    # Save predictions and ground truth
    results_df = pd.DataFrame({"ground_truth": ground_truths, "predictions": predictions})
    results_file = os.path.join(output_dir, "evaluation_results.json")
    results_df.to_json(results_file, orient="records", lines=True)
    print(f"Results saved to {results_file}")

    # Calculate metrics
    print("\nCalculating metrics...")

    # Flatten keys for classification metrics
    flat_ground_truths = ["|".join(sorted(gt.keys())) for gt in ground_truths]
    flat_predictions = ["|".join(sorted(pred.keys())) for pred in predictions]

    # Compute metrics using sklearn
    report = classification_report(flat_ground_truths, flat_predictions, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose()

    # Print metrics
    print("\nClassification Report:")
    print(metrics_df)

    # Save metrics
    metrics_file = os.path.join(output_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(metrics_file, index=True)
    print(f"Metrics saved to {metrics_file}")

    # Generate confusion matrix
    cm = confusion_matrix(flat_ground_truths, flat_predictions)
    labels = sorted(set(flat_ground_truths))

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.tight_layout()

    # Save confusion matrix as PNG
    confusion_matrix_file = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(confusion_matrix_file)
    print(f"Confusion matrix saved to {confusion_matrix_file}")
    plt.close()

    return metrics_df





if __name__ == "__main__":
    test_dataset_path = r"C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\dataset\advancedDatasetReady"
    model_path = r"C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\src\models\donut_finetuned.pth"
    #output_dir = r"C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\classificationModels\advancedClassification\eval_results"
    output_dir = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\temp'

    evaluate_donut_model(test_dataset_path, model_path, output_dir)
