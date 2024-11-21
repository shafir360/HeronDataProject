import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from datasets import Dataset
import evaluate
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import os


def eval_model_with_visualization(dataset_path, model_dir, output_dir):
    # Check if GPU is available
    if torch.cuda.is_available():
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        device = torch.device("cuda")
    else:
        print("No GPU available, using CPU instead.")
        device = torch.device("cpu")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the model, tokenizer, and label encoder
    model = BertForSequenceClassification.from_pretrained(model_dir)
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    with open(f"{model_dir}/label_encoder.json", "r") as f:
        label_encoder_mapping = json.load(f)
    label_decoder = {v: k for k, v in label_encoder_mapping.items()}

    model.to(device)

    # Load dataset
    data = pd.read_csv(dataset_path).sample(frac=1, random_state=42).reset_index(drop=True).tail(10000)

    # Preprocess text
    data['Data'] = data['Data'].str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True)

    # Tokenizer function
    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=32, return_tensors="pt")

    # Tokenize data
    encodings = tokenize_function(data['Data'].tolist())

    # Convert encodings into PyTorch tensors
    input_ids = encodings['input_ids']
    attention_mask = encodings['attention_mask']

    # Evaluate
    model.eval()
    predictions = []
    with torch.no_grad():
        for i in range(len(input_ids)):
            inputs = {
                'input_ids': input_ids[i].unsqueeze(0).to(device),
                'attention_mask': attention_mask[i].unsqueeze(0).to(device)
            }
            outputs = model(**inputs)
            logits = outputs.logits
            pred_class = torch.argmax(logits, dim=1).item()
            predictions.append(pred_class)

    # Decode predictions and compare with ground truth
    data['Predicted'] = [label_decoder[p] for p in predictions]
    if 'Label' in data.columns:
        data['Correct'] = data['Label'] == data['Predicted']
        
        # Compute classification metrics
        true_labels = [label_encoder_mapping[l] for l in data['Label']]
        pred_labels = [label_encoder_mapping[p] for p in data['Predicted']]

        # Filter labels to only include the ones present in the dataset
        unique_labels = sorted(set(true_labels) | set(pred_labels))
        target_names = [label_decoder[l] for l in unique_labels]

        report = classification_report(
            true_labels,
            pred_labels,
            labels=unique_labels,
            target_names=target_names,
            output_dict=True
        )
        metrics_df = pd.DataFrame(report).transpose()

        # Print metrics
        print("Evaluation Metrics:")
        print(metrics_df)

        # Save metrics to CSV
        metrics_csv_path = os.path.join(output_dir, "evaluation_metrics.csv")
        metrics_df.to_csv(metrics_csv_path, index=True)
        print(f"Metrics saved to {metrics_csv_path}")

        # Accuracy, Precision, Recall, F1-score summary
        overall_metrics = metrics_df.loc['accuracy':'weighted avg', ['precision', 'recall', 'f1-score']]
        print("\nSummary:")
        print(overall_metrics)

        # Create confusion matrix
        cm = confusion_matrix(true_labels, pred_labels, labels=unique_labels)
        labels = [label_decoder[i] for i in unique_labels]

        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.tight_layout()
        confusion_matrix_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(confusion_matrix_path)
        print(f"Confusion matrix saved to {confusion_matrix_path}")
        plt.close()

        # Class-wise accuracy
        class_wise_accuracy = (cm.diagonal() / cm.sum(axis=1)) * 100

        # Plot class-wise accuracy
        plt.figure(figsize=(10, 6))
        sns.barplot(x=labels, y=class_wise_accuracy)
        plt.title("Class-wise Accuracy")
        plt.xlabel("Classes")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)
        plt.tight_layout()
        class_accuracy_path = os.path.join(output_dir, "class_wise_accuracy.png")
        plt.savefig(class_accuracy_path)
        print(f"Class-wise accuracy plot saved to {class_accuracy_path}")
        plt.close()


if __name__ == '__main__':
    output_dir = r'classificationModels\simpleClassification\eval_results'
    dataset_path = r'classificationModels\simpleClassification\fileNameDataset\syntheticFilename'
    eval_model_with_visualization(dataset_path, r'classificationModels\simpleClassification\simpleModel', output_dir)
