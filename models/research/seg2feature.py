import json
import os

import cv2
import numpy as np
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("date")


COMMON_PATH = os.getenv('COMMON_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')
DATE = parser.parse_args().date
VIS_PATH = os.getenv('LSC_PATH') + '/deeplab/vis/' + DATE

map2deeplap = json.load(open(f"{COMMON_PATH}/map2deeplap.json"))
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))
simples = json.load(open(f"{COMMON_PATH}/simples.json"))
LABEL_NAMES = np.asarray([
    "none", "wall", "building;edifice", "sky", "floor;flooring", "tree", "ceiling", "road;route", "bed",
    "windowpane;window", "grass", "cabinet", "sidewalk;pavement", "person;individual;someone;somebody;mortal;soul",
    "earth;ground", "door;double;door", "table", "mountain;mount", "plant;flora;plant;life",
    "curtain;drape;drapery;mantle;pall", "chair", "car;auto;automobile;machine;motorcar", "water", "painting;picture",
    "sofa;couch;lounge", "shelf", "house", "sea", "mirror", "rug;carpet;carpeting", "field", "armchair", "seat",
    "fence;fencing", "desk", "rock;stone", "wardrobe;closet;press", "lamp", "bathtub;bathing;tub;bath;tub",
    "railing;rail", "cushion", "base;pedestal;stand", "box", "column;pillar", "signboard;sign",
    "chest;of;drawers;chest;bureau;dresser", "counter", "sand", "sink", "skyscraper", "fireplace;hearth;open;fireplace",
    "refrigerator;icebox", "grandstand;covered;stand", "path", "stairs;steps", "runway",
    "case;display;case;showcase;vitrine", "pool;table;billiard;table;snooker;table", "pillow", "screen;door;screen",
    "stairway;staircase", "river", "bridge;span", "bookcase", "blind;screen", "coffee;table;cocktail;table",
    "toilet;can;commode;crapper;pot;potty;stool;throne", "flower", "book", "hill", "bench", "countertop",
    "stove;kitchen;stove;range;kitchen;range;cooking;stove", "palm;palm;tree", "kitchen;island",
    "computer;computing;machine;computing;device;data;processor;electronic;computer;information;processing;system",
    "swivel;chair", "boat", "bar", "arcade;machine", "hovel;hut;hutch;shack;shanty",
    "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle", "towel",
    "light;light;source", "truck;motortruck", "tower", "chandelier;pendant;pendent", "awning;sunshade;sunblind",
    "streetlight;street;lamp", "booth;cubicle;stall;kiosk",
    "television;television;receiver;television;set;tv;tv;set;idiot;box;boob;tube;telly;goggle;box",
    "airplane;aeroplane;plane", "dirt;track", "apparel;wearing;apparel;dress;clothes", "pole", "land;ground;soil",
    "bannister;banister;balustrade;balusters;handrail", "escalator;moving;staircase;moving;stairway",
    "ottoman;pouf;pouffe;puff;hassock", "bottle", "buffet;counter;sideboard", "poster;posting;placard;notice;bill;card",
    "stage", "van", "ship", "fountain", "conveyer;belt;conveyor;belt;conveyer;conveyor;transporter", "canopy",
    "washer;automatic;washer;washing;machine", "plaything;toy", "swimming;pool;swimming;bath;natatorium", "stool",
    "barrel;cask", "basket;handbasket", "waterfall;falls", "tent;collapsible;shelter", "bag", "minibike;motorbike",
    "cradle", "oven", "ball", "food;solid;food", "step;stair", "tank;storage;tank",
    "trade;name;brand;name;brand;marque", "microwave;microwave;oven", "pot;flowerpot",
    "animal;animate;being;beast;brute;creature;fauna", "bicycle;bike;wheel;cycle", "lake",
    "dishwasher;dish;washer;dishwashing;machine", "screen;silver;screen;projection;screen", "blanket;cover",
    "sculpture", "hood;exhaust;hood", "sconce", "vase", "traffic;light;traffic;signal;stoplight", "tray",
    "ashcan;trash;can;garbage;can;wastebin;ash;bin;ash-bin;ashbin;dustbin;trash;barrel;trash;bin", "fan",
    "pier;wharf;wharfage;dock", "crt;screen", "plate", "monitor;monitoring;device", "bulletin;board;notice;board",
    "shower", "radiator", "glass;drinking;glass", "clock", "flag"
])
ind2LABEL_NAMES = dict([(label, i) for i, label in enumerate(LABEL_NAMES)])
simple_indices = dict([(simple, i) for i, simple in enumerate(simples)])
ind2simple = np.asarray(list(simples.keys()))
concept_dict = json.load(open(f"{COMMON_PATH}/concept_bb_dict.json"))


def seg2feature(seg_map):
    features = {}
    for i in range(len(ind2simple)):
        if i > 0:
            all_pos = seg_map == i
            area = np.sum(all_pos.flatten())
            if area > 500:
                ind = np.nonzero(all_pos.any(axis=0))[0]  # indices of non empty columns
                left, right = (ind[0], ind[-1]) if len(ind) else (0, -1)
                ind = np.nonzero(all_pos.any(axis=1))[0]  # indices of non empty rows
                top, bottom = (ind[0], ind[-1]) if len(ind) else (0, -1)
                center = (right - left) // 2, (bottom - top) // 2
                position_weight = 480 - np.sqrt((center[0] - 256) ** 2 + (center[1] - 256) ** 2)
                h, w = 513, 513
                position = []
                if top > int(h * 0.7):
                    position.append("bottom")
                if bottom < int(h * 0.3):
                    position.append("top")

                if left > int(w * 0.7):
                    position.append("right")
                if right < int(w * 0.3):
                    position.append("left")
                if int(w * 0.3) < (left - right) // 2 < int(w * 0.7) and \
                        int(h * 0.3) < (bottom - top) // 2 < int(h * 0.7):
                    position.append("middle")

                features[ind2simple[i]] = {"area": int(area),
                                           "bb": [int(top), int(left), int(bottom), int(right)],
                                           "position": position,
                                           "position_weight": position_weight}
    return features


if os.path.isfile(f"{COMMON_PATH}/area_deeplab_features.json"):
    features = json.load(open(f"{COMMON_PATH}/area_deeplab_features.json"))
else:
    features = {}
for image in tqdm(sorted(os.listdir(VIS_PATH))):
    original_image = f"{DATE}/{image.split('.')[0]}.jpg"
    if original_image not in features:
        segmap = cv2.imread(f"{VIS_PATH}/{image}", 0)
        features[original_image] = seg2feature(segmap)
        json.dump(features, open(f"{COMMON_PATH}/area_deeplab_features.json", 'w'))
