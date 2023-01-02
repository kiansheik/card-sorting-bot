import json
import os
import random
import time
import urllib.request

import cv2
import numpy as np
import pandas as pd
from PIL import Image


def load_annotations(json_path):
    with open(json_path) as f:
        a = json.load(f)
    annotations = [
        x
        for x in a
        if pd.to_datetime(x["updated_at"])
        >= pd.Timestamp("2022-12-24 21:14:36.599512+0000", tz="UTC")
    ]
    return annotations


def download_image(url, file_path, overwrite=False):
    # Open the URL and download the image data
    if not overwrite and os.path.exists(file_path):
        print(f"{file_path} already downloaded...")
        return False
    with urllib.request.urlopen(url) as url_handle:
        image_data = url_handle.read()

    # Write the image data to the specified file
    with open(file_path, "wb") as f:
        f.write(image_data)
    return True


with open("/Users/kiansheik/Downloads/oracle-cards-20221222220256.json") as f:
    all_cards = json.load(f)


annotations = load_annotations(
    "/Users/kiansheik/code/card-sorting-bot/vision/annotations/project-3-at-2022-12-25-00-09-f9c60779.json"
)
annotated_ids = {
    x["data"]["image"].split("/")[-1].split(".")[0]: x for x in annotations
}

for card in all_cards:
    if card["id"] in annotated_ids:
        try:
            dl = download_image(
                card["image_uris"]["border_crop"], f"data/{card['id']}.jpg"
            )
            if dl:
                time.sleep(1)
        except Exception as e:
            print(f"failed: ({e})")