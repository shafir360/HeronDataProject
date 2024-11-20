import torch
from transformers import DonutProcessor
from PIL import Image
import re
import base64
from io import BytesIO
import json

def model_fn(model_dir):
    """
    Load the model and processor from the specified directory.
    """
    model_path = f"{model_dir}/model_final_0.pth"
    processor = DonutProcessor.from_pretrained(f"{model_dir}/processor")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return processor, model

def input_fn(request_body, content_type):
    """
    Process the input data received by the endpoint.
    """
    if content_type == 'application/json':
        input_data = json.loads(request_body)
        if 'base64_image' in input_data:
            image_data = input_data['base64_image']
            image = Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            return image
        else:
            raise ValueError("Missing 'base64_image' in input data")
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def predict_fn(input_data, model):
    """
    Generate predictions using the processed input data and the loaded model.
    """
    processor, model = model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pixel_values = processor(input_data, return_tensors="pt").pixel_values.to(device)
    decoder_input_ids = torch.full((1, 1), model.config.decoder_start_token_id, device=device)

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
        return_dict_in_generate=True,
        output_scores=True,
    )

    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()
    return processor.token2json(seq)

def output_fn(prediction, accept):
    """
    Format the prediction output to be returned by the endpoint.
    """
    if accept == 'application/json':
        return json.dumps(prediction), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
