import json
import numpy as np
import cv2
import os
from tqdm import tqdm
COMMON_PATH = os.getenv('COMMON_PATH')
IMAGE_PATH = os.getenv('IMAGE_PATH')

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
simple_indices = dict([(simple, i) for i, simple in enumerate(simples)])
deeplab2simple = json.load(open(f"{COMMON_PATH}/deeplab2simple.json"))

m = 0
for mode in ['training', 'validation']:
    os.system(f'mkdir ADE20K/ADEChallengeData2016/annotations/{mode}2')
    for file in tqdm(sorted(os.listdir(f'ADE20K/ADEChallengeData2016/annotations/{mode}'))):
        if os.path.isfile(f"ADE20K/ADEChallengeData2016/annotations/{mode}2/{file}"):
            continue
        img = cv2.imread(f"ADE20K/ADEChallengeData2016/annotations/{mode}/{file}",
                         cv2.IMREAD_GRAYSCALE)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                label = LABEL_NAMES[img[i, j]]
                img[i, j] = simple_indices[deeplab2simple[label]]
                m = max(m, img[i, j])
        cv2.imwrite(f"ADE20K/ADEChallengeData2016/annotations/{mode}2/{file}", img)

print("Number of classes:", m)
