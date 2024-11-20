import torch
from transformers import BertTokenizer, BertForSequenceClassification
import json





def simpleInference(word, model_dir):
    # Load model and tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_dir)
    model = BertForSequenceClassification.from_pretrained(model_dir)

    # Load label encoder mapping
    with open(f"{model_dir}/label_encoder.json", "r") as f:
        label_encoder_mapping = json.load(f)
    inverse_label_encoder = {v: k for k, v in label_encoder_mapping.items()}

    # Clean and normalize the input word
    word = word.lower().replace(r'[^a-z0-9 ]', ' ')

    # Tokenize the word
    inputs = tokenizer(word, return_tensors="pt", truncation=True, padding="max_length", max_length=32)

    # Get model predictions
    outputs = model(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1)

    # Convert prediction back to label
    return inverse_label_encoder[int(prediction)]

test = False
if test:
    dir = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\flask\models\simpleModel'
    test_word = "61NsUWXqelL"
    predicted_class = simpleInference(test_word,dir)
    print(f"Predicted class for '{test_word}': {predicted_class}")
    print(predicted_class)
    print(type(predicted_class))
