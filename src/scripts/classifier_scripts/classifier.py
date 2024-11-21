# classifier.py

import os
from PIL import Image

# Import your inference functions
from scripts.classifier_scripts.advancedInference import advancedInference
from scripts.classifier_scripts.simpleInference import simpleInference

# Define allowed extensions for different file types
allowedExtensions = {
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'pdf': ['.pdf'],
}

def isAllowed(filename, allowedExtensions):
    """
    Check if the file has an allowed extension.
    """
    extension = '.' + filename.split('.')[-1].lower()
    for ext_list in allowedExtensions.values():
        if extension in ext_list:
            return True
    return False

def findFileType(file_path):
    """
    Get the file extension of the given file path.
    """
    return os.path.splitext(file_path)[1].lower()

def classifier(input_data, option, model_path, processor_path=None):
    """
    Main classifier function that routes the input to the appropriate inference function.
    
    Parameters:
    - input_data: File path (for advanced) or text input (for simple)
    - option: 'simple' or 'advanced'
    - model_path: Path to the model
    - processor_path: Path to the processor (only for advanced)
    
    Returns:
    - Classification result as a string
    """
    if option == 'simple':
        # Handle Simple Classification (Text Input)
        text_input = input_data.strip()
        if text_input == '':
            return 'Empty text input provided.'
        try:
            # Call the simpleInference function with the text input
            res = simpleInference(text_input, model_path)
            return res
        except Exception as e:
            return f'Error during simple classification: {str(e)}'
    
    elif option == 'advanced':
        # Handle Advanced Classification (File Upload)
        file_path = input_data
        if not os.path.exists(file_path):
            return 'File does not exist.'
        
        fileType = findFileType(file_path)
        
        # Check if the file type is allowed
        if fileType in allowedExtensions['image'] + allowedExtensions['pdf']:
            try:
                # Open and convert the image to RGB
                image = Image.open(file_path).convert("RGB")
            except Exception as e:
                return f'Error opening image: {str(e)}'
            
            try:
                # Call the advancedInference function with the image
                res = advancedInference(image, model_path, processor_path)
                # Assuming advancedInference returns a dictionary with 'text_sequence'
                return res.get('text_sequence', 'No text sequence found in the result.')
            except Exception as e:
                return f'Error during advanced classification: {str(e)}'
        else:
            return 'Unsupported file type for advanced classification.'
    
    else:
        return 'Incorrect option provided to classifier.'


test = False
if test:
    # Paths for testing
    a = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\classifierTraining\simpleModel'
    b = 'Sample text input for simple classifier.'
    path = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\flask\static\uploads\61NsUWXqelL.jpg'
    print('Simple Classification Result:', classifier(b, 'simple', a))
    print('Advanced Classification Result:', classifier(path, 'advanced', model_path_advanced, processor_path))
