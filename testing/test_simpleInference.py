import os
import sys
from PIL import Image, UnidentifiedImageError
import pytest

# Add the parent directory to sys.path
file_path = os.path.abspath(__file__)  # Full path to the current file
parent_dir = os.path.dirname(os.path.dirname(file_path))  # One directory up
sys.path.append(parent_dir)

from src.scripts.classifier_scripts.simpleInference import simpleInference

# Paths to model and processor
model_path = r'src\models\simpleModel'
model_path = os.path.join(parent_dir, model_path)



@pytest.mark.parametrize(
    ('word', 'expected'),
    [
        ('_Bankcheques2022_2', 'BankCheques'),
        ('BnkSatements', 'BankStatements'),
        ('Cotcracts', 'Contracts'),
        ('EducationalCertificates', 'EducationalCertificates'),
        ('passpsortids', 'IDProofs'),
        ('InsuranceDocuments-2022-12', 'InsuranceDocuments'),
    ]
) 



def test_classify_filename_with_typos(word, expected):
    assert simpleInference(word,model_path) == expected
