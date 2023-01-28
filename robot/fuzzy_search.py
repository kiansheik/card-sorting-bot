# %%
import json
import os
from io import BytesIO
from tempfile import NamedTemporaryFile

import cv2
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from cairosvg import svg2png
from PIL import Image

with open(
    "/Users/kiansheik/code/mask_rcnn/assets/augmented_data/default-cards-20230121220708.json",
    encoding="utf8",
) as f:
    cards = json.load(f)

cards = pd.DataFrame(cards)


def get_scryfall(url):
    headers = {
        "User-Agent": "my-app/0.0.1",
        "Accept": "application/json;q=0.9,*/*;q=0.8",
    }

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.content
    else:
        # handle error
        print(response.status_code)
        return None


url = "https://api.scryfall.com/sets"
resp = get_scryfall(url)

# Check that the request was successful
if resp:
    set_data = json.loads(resp)["data"]
    set_data = {x["code"]: x for x in set_data}


def get_unique_set_names(group):
    return set(group["set"])


def get_closest_match(input_string, dictionary):
    input_string = input_string.lower().strip()
    # If the input string is a key in the dictionary, return the value
    if input_string in dictionary:
        print(f"Found exact match: {input_string}")
        return set([set_data[x]["icon_svg_uri"] for x in dictionary[input_string]])

    # Otherwise, initialize the closest match and best ratio to 0
    closest_match = None
    best_ratio = 0

    # Iterate through the dictionary
    for key in dictionary:
        # Calculate the ratio of similarity between the input string and the key
        ratio = Levenshtein.ratio(input_string, key)

        # If the ratio is greater than the best ratio so far, update the closest match and best ratio
        if ratio > best_ratio:
            closest_match = key
            best_ratio = ratio

    print(f"guessing with ratio of {ratio}: {closest_match}")
    # Return the value for the closest matching key
    return set([set_data[x]["icon_svg_uri"] for x in dictionary[closest_match]])


set_names = {
    x.lower(): y
    for x, y in cards.groupby("name").apply(get_unique_set_names).to_dict().items()
}
expanded_names = {
    x.lower(): y
    for x, y in cards.groupby("name").apply(get_unique_set_names).to_dict().items()
}
for name, sets in set_names.items():
    if " // " in name:
        sp = name.split(" // ")
        expanded_names[sp[0]] = set_names.get(sp[0], set()).union(set_names[name])
        expanded_names[sp[1]] = set_names.get(sp[1], set()).union(set_names[name])


def load_set_image(url, width=None, height=None):
    print(url)
    border_size = int(width / 45.0)
    svg_data = get_scryfall(url)
    png_data = svg2png(
        bytestring=svg_data,
        output_width=width - border_size * 2,
        output_height=height - border_size * 2,
    )
    cv_img = cv2.imdecode(np.frombuffer(png_data, np.uint8), cv2.IMREAD_UNCHANGED)
    # Add border padding to the image so the edges aren't touching
    border_color = (255, 255, 255)  # white
    cv_img = cv2.copyMakeBorder(
        cv_img,
        border_size,
        border_size,
        border_size,
        border_size,
        cv2.BORDER_CONSTANT,
        value=border_color,
    )
    return cv_img


def print_set_images(card_name):
    for img_url in get_closest_match(card_name.lower(), expanded_names):
        fig, ax = plt.subplots()
        img = load_set_image(img_url, width=500)
        ax.imshow(img)
        # Show the image
        plt.show()


def get_set_images_by_name(card_name, shape=(400, None)):
    imgs = []
    for img_url in get_closest_match(card_name.lower(), expanded_names):
        img = load_set_image(img_url, width=shape[1], height=shape[0])
        imgs.append((img, img_url))
    return imgs


def match_card_name(input_string):
    input_string = input_string.lower().strip()
    # If the input string is a key in the dictionary, return the value
    if input_string in expanded_names:
        print(f"Found exact match: {input_string}")
        return set([set_data[x]["icon_svg_uri"] for x in expanded_names[input_string]])

    # Otherwise, initialize the closest match and best ratio to 0
    closest_match = None
    best_ratio = 0

    # Iterate through the expanded_names
    for key in expanded_names:
        # Calculate the ratio of similarity between the input string and the key
        ratio = Levenshtein.ratio(input_string, key)

        # If the ratio is greater than the best ratio so far, update the closest match and best ratio
        if ratio > best_ratio:
            closest_match = key
            best_ratio = ratio

    return closest_match


def get_best_match(image, card_name):
    image_list = get_set_images_by_name(card_name, shape=image.shape)
    # Initialize the ORB feature detector and descriptor
    orb = cv2.ORB_create()

    # Detect and compute ORB features for the input image
    kp1, desc1 = orb.detectAndCompute(image, None)
    # Initialize a dictionary to store the SSIM scores for each image in the list
    scores = {}

    # Iterate through the list of images and compare each one to the input image
    for img, img_url in image_list:
        # Detect and compute ORB features for the current image
        kp2, desc2 = orb.detectAndCompute(img, None)

        # Initialize the brute-force matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match the ORB features
        matches = bf.match(desc1, desc2)

        # Sort the matches in ascending order of distance
        matches = sorted(matches, key=lambda x: x.distance)
        # If there are no matches, assign a high score to the image
        if len(matches) == 0:
            score = float("inf")
        else:
            # Calculate the average distance between the matched features
            score = sum(m.distance for m in matches) / len(matches)
        scores[score] = (img, img_url)

    # Sort the scores in descending order
    sorted_scores = sorted(scores.keys())

    # Return the image with the highest SSIM score
    return scores[sorted_scores[0]]


# %%

if __name__ == "__main__":
    print_set_images("cypt rat")
