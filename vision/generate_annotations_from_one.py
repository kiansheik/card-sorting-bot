import copy
import json
import os


def get_all_files(path):
    files = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            files.append(filepath)
    return files


with open(
    "/Users/kiansheik/Downloads/project-1-at-2022-12-24-20-44-84c84855.json"
) as f:
    annotations = json.load(f)


tot = []
for i, file in enumerate(
    get_all_files("/Users/kiansheik/code/card-sorting-bot/vision/data")
):
    cp = copy.deepcopy(annotations[0])
    cp["data"][
        "image"
    ] = f'/data/local-files/?d=code/card-sorting-bot/vision/data/{file.split("/")[-1]}'
    cp["id"] = i + 1
    cp["inner_id"] = i + 1
    tot.append(cp)

with open("annotations.json", "w") as f:
    json.dump(tot, f)
