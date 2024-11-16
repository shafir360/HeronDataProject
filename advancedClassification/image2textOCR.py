from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import cv2
import numpy as np
import pytesseract

# Path to the tesseract executable (needed for Windows users)
# Uncomment and modify the following line if you're on Windows
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image_path):
    
    # Open an image file
    with Image.open(image_path) as img:
        # Convert to a numpy array for OpenCV processing
        img = np.array(img)
        # Convert image to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply bilateral filtering to reduce noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        # Convert back to PIL Image for further processing
        img = Image.fromarray(filtered)
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.0)
        # Sharpen the image
        img = img.filter(ImageFilter.SHARPEN)
        # Automatically adjust contrast
        img = ImageOps.autocontrast(img)
        
        return img

def detect_text_from_image(image_path):
    
    try:
        # Preprocess the image
        img = preprocess_image(image_path)
        # Use Tesseract to do OCR on the image
        text = pytesseract.image_to_string(img, config='--psm 6')
        return text
    except Exception as e:
        return f"An error occurred: {e}"




def Image2TextOCR(image_path):
    return detect_text_from_image(image_path)


print(Image2TextOCR(r'files\drivers_licence_2.jpg'))