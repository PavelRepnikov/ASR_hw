import gdown
import os

def download_file(drive_url, output_path):
    """
    Downloads a file from Google Drive.

    Parameters:
        drive_url (str): The shareable link or file ID of the Google Drive file.
        output_path (str): The local file path where the file should be saved.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    try:
        gdown.download(drive_url, output_path, quiet=False)
        print(f"File saved to {output_path}")
    except Exception as e:
        print(f"Failed to download file: {e}")

def download_files(file_list):
    """
    Downloads multiple files from Google Drive.

    Parameters:
        file_list (list): A list of tuples (drive_url, output_path).
    """
    for drive_url, output_path in file_list:
        download_file(drive_url, output_path)

if __name__ == "__main__":
    files_to_download = [
        ("https://drive.google.com/uc?id=1yyPfX0TpaeTLvKQVJsblJcyxTquBWEM_", "saved/model_best.pth"),
        ("https://drive.google.com/uc?id=1_81btPmePUKCu5OH5a3dtsRykkuC6r1i", "saved/config.yaml") 
    ]

    download_files(files_to_download)
