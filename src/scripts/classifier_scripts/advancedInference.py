from transformers import VisionEncoderDecoderModel, DonutProcessor
import torch
import os
from PIL import Image
import re


def advancedInference(image,model_path,processor_path):
    # Load the processor and model
    
    processor = DonutProcessor.from_pretrained(processor_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(model_path,map_location=torch.device(device))
    model.to(device)


    
    pixel_values = processor(image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)

    # Prepare the decoder input
    decoder_input_ids = torch.full((1, 1), model.config.decoder_start_token_id, device=device)

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
        return_dict_in_generate=True,
        output_scores=True,
    )

    # Decode and clean up the output
    seq = processor.batch_decode(outputs.sequences)[0]
    seq = seq.replace(processor.tokenizer.eos_token, "").replace(processor.tokenizer.pad_token, "")
    seq = re.sub(r"<.*?>", "", seq, count=1).strip()

    # Convert the sequence to JSON if applicable
    json_output = processor.token2json(seq)

    return json_output