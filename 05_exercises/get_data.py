import requests
import zipfile
import os
from pathlib import Path

# Setup dataset path
data_path = Path('data/')
image_path = data_path / 'pizza_steak_sushi'

# Check if the dataset is already exist, if not, download it
if image_path.is_dir():
    print(f'Path: {image_path} is existing.')
else:
    print(f'Path: {image_path} not exist, creating...')
    image_path.mkdir(parents=True, exist_ok=True)

    # Download dataset
    with open(image_path / 'pizza_steak_sushi.zip', 'wb') as f:
        print('Download dataset...')
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        f.write(request.content)

    # Unzip dataset
    with zipfile.ZipFile(image_path / 'pizza_steak_sushi.zip', 'r') as f:
        print('Unzipping dataset...')
        f.extractall(image_path)

    # Remove the zip file
    os.remove(path=image_path / 'pizza_steak_sushi.zip')
