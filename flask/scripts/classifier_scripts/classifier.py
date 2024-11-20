import os
from PIL import Image

#from  advancedInference import advancedInference
#from simpleInference import simpleInference

from scripts.classifier_scripts.advancedInference import advancedInference
from scripts.classifier_scripts.simpleInference import simpleInference


allowedExtensions = {
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'pdf': ['.pdf'],
}



def isAllowed(filename,allowedExtensions):
    extension = '.' + filename.split('.')[-1].lower()
    for ext_list in allowedExtensions.values():
        if extension in ext_list:
            return True
    return False

def findFileType(file_path):
    return os.path.splitext(file_path)[1]


def classifier(file_path,option,model_path,processor_path= None):
    fileType = findFileType(file_path)

    if option == 'simple':
        file_name = os.path.basename(file_path)
        if isAllowed(file_name,allowedExtensions):
            file_without_extension = os.path.splitext(file_name)[0]
            res = simpleInference(file_without_extension,model_path)
            return res 
        else:
           return 'Not accepted extension type'

        return
        
    elif option == 'advanced':

        if fileType in allowedExtensions['image']:
            image = Image.open(file_path).convert("RGB")
        elif fileType in allowedExtensions['pdf']:
            image = Image.open(file_path).convert("RGB")
        else:
            return 'Not Supported File Type'

        res = advancedInference(image,model_path,processor_path)['text_sequence']
        return res
    else:
        return 'Incorrect Option Given'


    
        
    return fileType

test = False
if test:
    a = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\classifierTraining\simpleModel'
    b ='e'
    path = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\flask\static\uploads\61NsUWXqelL.jpg'
    print('here', classifier(path,'simple',a))