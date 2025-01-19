import gdown
import os

def download_file(drive_url, output_path):
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    try:
        gdown.download(drive_url, output_path, quiet=False, fuzzy=True)
        print(f"File saved to: {output_path}")
    except Exception as e:
        print(f"Failed to download file from {drive_url} to {output_path}. Error: {e}")

def download_files(file_list):
    for drive_url, output_path in file_list:
        download_file(drive_url, output_path)



if __name__ == "__main__":
    files_to_download = [
        ("https://drive.google.com/file/d/1m-imvy-GeMxtd37Aj9QvnGjrP-sid-zj/view?usp=sharing", "librispeech-vocab.txt"),
        ("https://drive.google.com/file/d/1DwYzL9BT0YkVDmLOgKfKQ0_-bra0q-uD/view?usp=sharing", "3-gram.pruned.3e-7.arpa"),
        ("https://drive.google.com/file/d/1jyajnE-8ksgJOh1i4qT_fcxLEYNJSpd8/view?usp=sharing", "checkpoint-epoch200.pth")

    ]

    download_files(files_to_download)
