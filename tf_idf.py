import json
import os
from collections import defaultdict

import numpy as np

COMMON_PATH = os.getenv('COMMON_PATH')
IMAGE_PATH = "/opt/LSC_DATA/"
map2deeplab = json.load(open(f"{COMMON_PATH}/map2deeplap.json"))
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))
simples = json.load(open(f"{COMMON_PATH}/simples.json"))


def create(features_dict, chosen_attr="sum_prob"):
    tf = {}
    idf = defaultdict(lambda: 0)
    for image, feature in features_dict.items():
        tf[image] = {}
        for label in feature:
            idf[label] += int(feature[label]["area"] > 10000)
            tf[image][label] = 1 + np.log(feature[label][chosen_attr])

    tf_idf = {}
    for image in tf:
        tf_idf[image] = {}
        for word in tf[image]:
            if idf[word]:
                tf_idf[image][word] = tf[image][word] * np.log(len(tf) / idf[word])
            else:
                tf_idf[image][word] = 0

    # json.dump(tf_idf, open(f"tf_idf.json", "w"))
    return idf, tf_idf


def to_deeplab(word):
    print(word)
    for kw in map2deeplab:
        if word in map2deeplab[kw][1]:
            yield deeplab2simple[kw]


def query(keywords, tf_idf, idf):
    """Assume that all the keywords are valid"""
    score = {}
    keywords = ([w for kw in keywords for w in set(to_deeplab(kw.strip(', ')))])
    for image in tf_idf:
        score[image] = 0
        for kw in keywords:
            if kw in tf_idf[image]:
                score[image] += tf_idf[image][kw]

    return [(i, s) for (i, s) in sorted(score.items(), key=lambda x: x[1], reverse=True) if s > 0][:10]


if __name__ == "__main__":
    features_dict = json.load(open(f"{COMMON_PATH}/area_deeplab_features.json"))
    idf, tf_idf = create(features_dict, "area")
    while True:
        keywords = input("Keywords: ")
        if '_' in keywords:
            for image in features_dict:
                if keywords in image:
                    print(features_dict[image])
        else:
            for image, score in query(keywords.split(), tf_idf, idf):
                print(image)
                print(features_dict[image])
