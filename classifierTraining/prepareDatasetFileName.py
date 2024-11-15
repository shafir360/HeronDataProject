import os 
import shutil
import pandas as pd
import glob


#Prepare synthetic filename dataset


parent_folder = r"generators\filenameGen\genResult"
outFolder = 'fileNameDataset'


# Get the directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

outFolder = os.path.join(script_dir, outFolder)

classes = os.listdir(parent_folder)

# Check if the folder exists
if os.path.exists(outFolder):
    # Remove the folder and its contents
    shutil.rmtree(outFolder)
    print(f"Deleted folder: {outFolder}")

# Create an empty folder
os.makedirs(outFolder)


#extract all filenames 
data_arrays =[]
for class_ in classes:
    all_lines = []
    folder_path = os.path.join(parent_folder, class_)
    for file_path in glob.glob(os.path.join(folder_path, "*.txt")):
        with open(file_path, "r") as file:
            # Read all lines from the current file and extend the list
            all_lines.extend(file.readlines())

        
    all_lines = [line.strip() for line in all_lines]
    data_arrays.append(all_lines)




# Flatten data and labels using list comprehensions
df = pd.DataFrame({
    "Data": [item for array in data_arrays for item in array],
    "Label": [label for label, array in zip(classes, data_arrays) for _ in array]
})


#print(df.to_string(index=False))


output_path = os.path.join(outFolder,'syntheticFilename')

df.to_csv(output_path, index=False)


