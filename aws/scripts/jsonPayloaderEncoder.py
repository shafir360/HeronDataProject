import base64
import pyperclip
import json

# Encode an image into base64
image_path = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\passport-Uk-Specimen-highlighted.webp'
with open(image_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode("utf-8")

# Prepare JSON payload
payload = {"base64_image": base64_image}
print(type(base64_image))
print('*************************************************************')
print(payload)
print('*************************************************************')
pyperclip.copy(json.dumps(payload, indent=4))