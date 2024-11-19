# Import necessary libraries
import torch
from transformers import AutoProcessor, UdopForConditionalGeneration
from PIL import Image
from torchvision import transforms
import pytesseract
from datasets import load_dataset, Dataset
from transformers import Trainer, TrainingArguments, DataCollatorWithPadding
from pdf2image import convert_from_path
import os

# Load the UDOP processor and model for multi-modal classification
model_name = "microsoft/udop-large"
processor = AutoProcessor.from_pretrained(model_name, apply_ocr=False)
model = UdopForConditionalGeneration.from_pretrained(model_name)

# Define label mapping for your classification task
label_mapping = {0: "drivingLicense", 1: "passport", 2: "bankStatements"}

# Function to load and preprocess image files
def preprocess_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Open and convert image to RGB format
    # Extract text from image using pytesseract (OCR)
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words = ocr_data['text']
    boxes = []
    for i in range(len(words)):
        if words[i].strip():
            (x, y, w, h) = (ocr_data['left'][i], ocr_data['top'][i], ocr_data['width'][i], ocr_data['height'][i])
            boxes.append([x, y, x + w, y + h])

    # Normalize bounding boxes to a 0-1000 scale
    width, height = image.size
    normalized_boxes = [
        [
            int(1000 * (bbox[0] / width)),
            int(1000 * (bbox[1] / height)),
            int(1000 * (bbox[2] / width)),
            int(1000 * (bbox[3] / height)),
        ]
        for bbox in boxes
    ]
    return image, words, normalized_boxes

# Function to load and preprocess PDF files
def preprocess_pdf(pdf_path):
    images, words_list, boxes_list = [], [], []
    pages = convert_from_path(pdf_path)
    for page in pages:
        image, words, boxes = preprocess_image(page)
        images.append(image)
        words_list.append(words)
        boxes_list.append(boxes)
    return images, words_list, boxes_list

# Load dataset (replace with your own dataset paths)
data_files = {"train": "train.json", "validation": "val.json"}
dataset = load_dataset("json", data_files=data_files)

# Preprocessing function for dataset entries
def preprocess_function(examples):
    # Assuming examples contain fields 'file_path' and 'label'
    images, words_list, boxes_list = [], [], []
    for file_path in examples['file_path']:
        if file_path.lower().endswith(".pdf"):
            # Handle PDF files
            pdf_images, pdf_words, pdf_boxes = preprocess_pdf(file_path)
            images.extend(pdf_images)
            words_list.extend(pdf_words)
            boxes_list.extend(pdf_boxes)
        else:
            # Handle image files (JPG, PNG, etc.)
            image, words, boxes = preprocess_image(file_path)
            images.append(image)
            words_list.append(words)
            boxes_list.append(boxes)
    encoding = processor(images, words_list, boxes=boxes_list, return_tensors="pt", padding=True)
    encoding["labels"] = examples["label"]
    return encoding

# Apply preprocessing to dataset
encoded_dataset = dataset.map(preprocess_function, batched=True)

# Set up training arguments
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
)

# Data collator for padding
collator = DataCollatorWithPadding(processor.tokenizer)

# Set up the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=encoded_dataset["train"],
    eval_dataset=encoded_dataset["validation"],
    tokenizer=processor.tokenizer,
    data_collator=collator,
)

# Train the model
trainer.train()

# Load an example image for inference
test_image_path = "your_image.jpg"
image, words, normalized_boxes = preprocess_image(test_image_path)
encoding = processor(image, words, boxes=normalized_boxes, return_tensors="pt")

# Perform the forward pass using the model to get predictions
outputs = model.generate(**encoding)

# Decode the output to get the predicted text
predicted_text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("Predicted Text:", predicted_text)

# Detailed comments explaining each step:
# 1. Import the required libraries including transformers, PIL for image processing, pytesseract for OCR, and torchvision for image transformations.
# 2. Load the UDOP processor and model for multi-modal classification.
# 3. Define the label mapping to convert class indices to human-readable labels.
# 4. Preprocess the dataset: load images, extract text and bounding boxes, normalize bounding boxes, and prepare input using the UDOP processor.
# 5. Use the Trainer API to fine-tune the model with your custom dataset.
# 6. Save training results and evaluate on validation data.
# 7. Load an image for inference, preprocess it, and generate predictions.

# Note:
# - Replace 'train.json' and 'val.json' with the paths to your own dataset files. These files should contain 'file_path' and 'label' fields.
# - You need to install pytesseract and tesseract-OCR. You can install pytesseract using 'pip install pytesseract'.
# - Tesseract-OCR is an external tool and must be installed separately. You can find installation instructions at: https://github.com/tesseract-ocr/tesseract
# - Make sure to have the correct path to your image file.
# - Fine-tuning a multi-modal model like UDOP requires substantial computational resources.
