import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset
import pandas as pd
import evaluate



if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("No GPU available, using CPU instead.")

def train_model(dataset_path, output_dir="./bert-word-classifier", num_epochs=3):
    print("Start of Train")
    # Load dataset
    data = pd.read_csv(dataset_path).sample(frac=1, random_state=42).reset_index(drop=True).head(70000)

    # Clean and normalize
    data['Data'] = data['Data'].str.lower().str.replace(r'[^a-z0-9 ]', ' ', regex=True)

    # Encode labels
    label_encoder = LabelEncoder()
    data['label_encoded'] = label_encoder.fit_transform(data['Label'])

    # Split into training and testing sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        data['Data'].tolist(),
        data['label_encoded'].tolist(),
        test_size=0.2,
        random_state=42
    )

    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenizer function
    def tokenize_function(texts):
        return tokenizer(texts, padding="max_length", truncation=True, max_length=32)

    # Tokenize the training and testing data
    train_encodings = tokenize_function(train_texts)
    test_encodings = tokenize_function(test_texts)

    # Create Dataset objects
    train_dataset = Dataset.from_dict({
        'input_ids': train_encodings['input_ids'],
        'attention_mask': train_encodings['attention_mask'],
        'labels': train_labels
    })

    test_dataset = Dataset.from_dict({
        'input_ids': test_encodings['input_ids'],
        'attention_mask': test_encodings['attention_mask'],
        'labels': test_labels
    })

    # Load pretrained model
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=len(label_encoder.classes_))

    # Load the accuracy metric
    metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = torch.argmax(torch.tensor(logits), dim=1)  # Find the most likely class
        return metric.compute(predictions=predictions.numpy(), references=labels)  # Compare with correct labels

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        dataloader_num_workers=4
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Save the model and tokenizer
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save the label encoder
    label_encoder_mapping = dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))
    pd.Series(label_encoder_mapping).to_json(f"{output_dir}/label_encoder.json")

    print(f"Model and tokenizer saved to {output_dir}")



if __name__ == '__main__':
    outputDir = 'classifierTraining/simpleModel'
    train_model(r'classifierTraining\fileNameDataset\syntheticFilename',outputDir)
