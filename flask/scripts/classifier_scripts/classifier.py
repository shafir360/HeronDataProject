import os
from PIL import Image
#from  advancedInference import advancedInference
from scripts.classifier_scripts.advancedInference import advancedInference


#def_model_path = r''
#processor_path = r'C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\aws\awsContainer\processor'

allowedExtensions = {
    'image': ['.jpg', '.jpeg', '.png', '.bmp', '.webp'],
    'pdf': ['.pdf'],
}

def findFileType(file_path):
    return os.path.splitext(file_path)[1]


def classifier(file_path,option,model_path,processor_path):
    fileType = findFileType(file_path)

    if option == 'simple':
        print('simple')
        return 'simple'
        
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



#print(classifier(r'flask\scripts\temp\61NsUWXqelL.jpg','advanced'))