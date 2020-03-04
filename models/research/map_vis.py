from PIL import Image
from PIL import ImageEnhance
from collections import defaultdict, Counter
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import json
import os
import cv2


COMMON_PATH = os.getenv('COMMON_PATH')
IMAGE_PATH = "/opt/LSC_DATA/"
DATE = "2016-09-03"
VIS_PATH = '/home/tlduyen/LSC2020/deeplab/vis/' + DATE

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


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.

    Returns:
      A Colormap for visualizing segmentation results.
    """
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)

    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= ((ind >> channel) & 1) << shift
        ind >>= 3

    return colormap


def label_to_color_image(label):
    """Adds color defined by the dataset colormap to the label.

    Args:
      label: A 2D array with integer type, storing the segmentation label.

    Returns:
      result: A 2D array with floating type. The element of the array
        is the color indexed by the corresponding element in the input label
        to the PASCAL color map.

    Raises:
      ValueError: If label is not of rank 2 or its value is larger than color
        map maximum entry.
    """
    if label.ndim != 2:
        raise ValueError('Expect 2-D input label')

    colormap = create_pascal_label_colormap()

    if np.max(label) >= len(colormap):
        raise ValueError('label value too large.')

    return colormap[label]

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

def vis_segmentation(name, image, seg_map):
    """Visualizes input image, segmentation map and overlay view."""
    plt.figure(figsize=(25, 15))
    grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

    plt.subplot(grid_spec[0])
    plt.imshow(image)
    plt.axis('off')
    plt.title('input image')

    plt.subplot(grid_spec[1])
    seg_image = label_to_color_image(seg_map).astype(np.uint8)
    plt.imshow(seg_image)
    plt.axis('off')
    plt.title('segmentation map')

    plt.subplot(grid_spec[2])
    plt.imshow(image)
    plt.imshow(seg_image, alpha=0.7)
    plt.axis('off')
    plt.title('segmentation overlay')

    unique_labels = np.unique(seg_map)
    ax = plt.subplot(grid_spec[3])
    plt.imshow(
        FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
    ax.yaxis.tick_right()
    plt.yticks(range(len(unique_labels)), ind2simple[unique_labels])
    plt.xticks([], [])
    ax.tick_params(width=0.0)
    plt.grid('off')
    plt.savefig(f"/home/tlduyen/LSC2020/deeplab/doll_res/{name}.png")

def dilate(segmap, c_win=5, r_win=5):
    h, w = segmap.shape[:2]
    for i in range(h):
        xmin = max(0, i - r_win)
        xmax = min(h, i + r_win + 1)
        for j in range(w):
            ymin = max(0, j - c_win)
            ymax = min(w, j + c_win + 1)
            box = Counter(segmap[xmin:xmax, ymin:ymax].flatten())
            segmap[i, j] = box.most_common(1)[0][0]
    return segmap

for image in sorted(os.listdir(VIS_PATH)):
    original_image = image.split('.')[0] + '.jpg'
    original_image = Image.open(f"{IMAGE_PATH}/{DATE}/{original_image}")
    segmap = cv2.imread(f"{VIS_PATH}/{image}", 0)
    segmap = cv2.resize(dilate(segmap), (1024, 768), interpolation=cv2.INTER_NEAREST)
    vis_segmentation(image.split('.')[0], original_image, segmap)