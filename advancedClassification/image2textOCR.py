# Step 1: Install required libraries
# Run these commands in your terminal to install the required dependencies:
# pip install pytesseract pillow opencv-python-headless numpy

# Import necessary libraries
from PIL import Image
import pytesseract
import cv2
import numpy as np
import os

def enhance_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply GaussianBlur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to improve contrast
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Apply morphological operations to improve character boundaries
    kernel = np.ones((2, 2), np.uint8)
    processed_image = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Deskew the image
    coords = np.column_stack(np.where(processed_image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = processed_image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed_image = cv2.warpAffine(processed_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return deskewed_image

def perform_ocr(image_path):
   
    try:
       
        enhanced_image = enhance_image(image_path)


        pil_image = Image.fromarray(enhanced_image)

      
        text = pytesseract.image_to_string(pil_image, config='--psm 6')

        return text
    except Exception as e:
        return f"Error: {str(e)}"

def Image2TextOCR(image_path):
    
    #image_path = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\files\drivers_license_1.jpg'

    if not os.path.exists(image_path):
        print(f"Error: File '{image_path}' not found.")
        return

    # Perform OCR and print the result
    recognized_text = perform_ocr(image_path)
    
    return recognized_text


recognized_text = Image2TextOCR(r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\files\drivers_license_1.jpg')


print("Recognized Text:")
print(recognized_text)



