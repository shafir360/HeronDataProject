import os
import sys
from PIL import Image, UnidentifiedImageError
import pytest

# Add the parent directory to sys.path
file_path = os.path.abspath(__file__)  # Full path to the current file
parent_dir = os.path.dirname(os.path.dirname(file_path))  # One directory up
sys.path.append(parent_dir)

from src.scripts.classifier_scripts.advancedInference import advancedInference

# Paths to model and processor
model_path = r'src\models\donut_finetuned.pth'
processor_path = r'src\models\donut_processor'

model_path = os.path.join(parent_dir, model_path)
processor_path = os.path.join(parent_dir, processor_path)

def path2image(file_path):
    try:
        image = Image.open(file_path).convert("RGB")
        return image
    except UnidentifiedImageError:
        raise ValueError(f"Invalid image file: {file_path}")

@pytest.mark.parametrize(
    ('image_path', 'expected'),
    [
        (r'testing\testData\balance_sheets.png', 'balance_sheets'),
        (r'testing\testData\bank_statement.png', 'bank_statement'),
        (r'testing\testData\cash_flow.png', 'cash_flow'),
        (r'testing\testData\drivingLicense.jpg', 'driving_license'),
        (r'testing\testData\income_statement_train_331.png', 'income_statement'),
        (r'testing\testData\invoice_186.png', 'invoice'),
        (r'testing\testData\passport.jpg', 'passport'),
    ]
)


def test_classes(image_path, expected):
    assert advancedInference(path2image(image_path), model_path, processor_path)['text_sequence'] == expected
