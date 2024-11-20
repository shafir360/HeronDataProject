import tarfile
import os

def create_tar_gz(source_dir, output_file):
    """
    Creates a .tar.gz archive of the specified directory.

    Args:
        source_dir (str): The directory to compress.
        output_file (str): The path to save the .tar.gz file.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"Source directory '{source_dir}' does not exist.")
    
    with tarfile.open(output_file, "w:gz") as tar:
        tar.add(source_dir, arcname=".")
    print(f"Archive created at: {output_file}")

# Example usage
if __name__ == "__main__":
    # Source directory to archive
    source_directory = r"C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\aws\awsContainer"
    
    # Destination of the .tar.gz file

    output_file_path = r"C:\Users\Shafir R\Documents\code\herondata\HeronDataProject\aws"
    outPut = os.path.join(output_file_path,"model.tar.gz" ) 
    
    create_tar_gz(source_directory, outPut)
