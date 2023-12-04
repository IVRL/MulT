import os
from argparse import Namespace
from functools import reduce

import numpy as np
import seaborn as sns
import torch
from PIL import Image
from PIL import ImageFile
from matplotlib import pyplot as plt
from mmcv.utils import print_log
from mmseg.core import mean_iou
from torchvision import transforms
from tqdm import tqdm

from .builder import DATASETS
from .calculate_error import compute_errors_NYU
from .custom import CustomDataset
from .transform_list import CenterCrop, RandomCropNumpy, EnhancedCompose, RandomColor, RandomHorizontalFlip, \
    ArrayToTensorNumpy

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _is_pil_image(img):
    return isinstance(img, Image.Image)


def _get_colormap(n):
    def bitget(byteval, idx):
        return (byteval & (1 << idx)) != 0

    cmap = np.zeros((n, 3), dtype='uint8')
    for i in range(n):
        r = g = b = 0
        c = i
        for j in range(8):
            r = r | (bitget(c, 0) << 7 - j)
            g = g | (bitget(c, 1) << 7 - j)
            b = b | (bitget(c, 2) << 7 - j)
            c = c >> 3

        cmap[i] = np.array([r, g, b])

    return cmap


def save_indexed_png(filepath, label, colormap):
    # note that OpenCV is not able to handle indexed pngs correctly.
    img = Image.fromarray(np.asarray(label, dtype='uint8'))
    img.putpalette(list(np.asarray(colormap, dtype='uint8').flatten()))
    img.save(filepath)


@DATASETS.register_module()
class DepthDataset(CustomDataset):
    def __init__(self, args, train=True, return_filename=False, test_mode=False):
        # assert train != test_mode
        args = Namespace(**args)
        self.use_dense_depth = args.use_dense_depth
        if train is True:
            if args.dataset == 'KITTI':
                self.datafile = args.trainfile_kitti
                self.angle_range = (-1, 1)
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.trainfile_nyu
                self.angle_range = (-2.5, 2.5)
                self.depth_scale = 1000.0
                args.height = 464
                args.width = 464
        else:
            if args.dataset == 'KITTI':
                self.datafile = args.testfile_kitti
                self.depth_scale = 256.0
            elif args.dataset == 'NYU':
                self.datafile = args.testfile_nyu
                self.depth_scale = 1000.0
                args.height = 464
                args.width = 464
        self.train = train
        self.transform = Transformer(args)
        self.args = args
        self.return_filename = return_filename
        with open(self.datafile, 'r') as f:
            self.fileset = f.readlines()
        self.fileset = sorted(self.fileset)

        self.mapping_894_to_40 = np.concatenate([[0], [40, 40, 3, 22, 5, 40, 12, 38, 40, 40, 2, 39, 40, 40, 26, 40,
                                                       24, 40, 7, 40, 1, 40, 40, 34, 38, 29, 40, 8, 40, 40, 40, 40,
                                                       38, 40, 40, 14, 40, 38, 40, 40, 40, 15, 39, 40, 30, 40, 40, 39,
                                                       40, 39, 38, 40, 38, 40, 37, 40, 38, 38, 9, 40, 40, 38, 40, 11,
                                                       38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 13,
                                                       40, 40, 6, 40, 23, 40, 39, 10, 16, 40, 40, 40, 40, 38, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 38, 40, 39, 40, 40, 40, 40, 39, 38,
                                                       40, 40, 40, 40, 40, 40, 18, 40, 40, 19, 28, 33, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 38, 27, 36, 40, 40, 40, 40, 21, 40, 20, 35,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 4, 32, 40, 40,
                                                       39, 40, 39, 40, 40, 40, 40, 40, 17, 40, 40, 25, 40, 39, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 39, 38, 38, 40, 40, 39, 40, 39, 40, 38, 39,
                                                       38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 38, 40, 40,
                                                       38, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38,
                                                       40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 40, 39, 40, 40, 40, 38, 40, 40, 39, 40, 40, 38, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 31, 40, 40, 40, 40, 40,
                                                       40, 40, 38, 40, 40, 38, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 38, 40, 39, 40, 40, 39, 40, 40, 40, 38, 40, 40, 40, 40, 40,
                                                       40, 40, 40, 38, 39, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 38, 39, 40, 40, 40, 40, 40, 40, 40,
                                                       39, 40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 38, 40, 39, 40, 40,
                                                       40, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 40, 40, 39, 39,
                                                       40, 40, 40, 40, 38, 40, 40, 38, 39, 39, 40, 39, 40, 39, 38, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 38, 40, 39, 40,
                                                       40, 40, 40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 39, 39, 40, 40,
                                                       38, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 40, 40,
                                                       40, 40, 39, 40, 40, 40, 40, 40, 39, 40, 40, 39, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 38, 40, 40, 40,
                                                       40, 40, 40, 40, 39, 38, 39, 40, 38, 39, 40, 39, 40, 39, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 38, 40, 40, 39,
                                                       40, 40, 40, 39, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39,
                                                       38, 40, 40, 38, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 38, 40, 40, 38, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       38, 38, 38, 40, 40, 40, 38, 40, 40, 40, 38, 38, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 38, 40, 38, 39, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 39, 40, 39, 40, 40, 40, 40, 38, 38, 40, 40, 40, 38, 40, 40,
                                                       40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 39,
                                                       40, 40, 39, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 39, 39, 40,
                                                       40, 40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 40,
                                                       40, 40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40,
                                                       40, 38, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40, 38, 40, 39, 40,
                                                       40, 40, 40, 38, 40, 40, 40, 40, 40, 38, 40, 40, 40, 40, 40, 40,
                                                       40, 39, 40, 40, 40, 40, 40, 40, 40, 40, 40, 39, 40, 40]])
        if self.args.use_seg:
            if self.args.use_40:
                self.CLASSES = ['void', 'wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window',
                                'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser',
                                'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator',
                                'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person',
                                'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure',
                                'otherfurniture', 'otherprop']
                self.CLASS_COLORS = np.array(_get_colormap(1 + 40).tolist(), dtype='uint8')
            else:
                self.CLASSES = ['void', 'book', 'bottle', 'cabinet', 'ceiling', 'chair', 'cone', 'counter',
                                'dishwasher', 'faucet', 'fire extinguisher', 'floor', 'garbage bin', 'microwave',
                                'paper towel dispenser', 'paper', 'pot', 'refridgerator', 'stove burner', 'table',
                                'unknown', 'wall', 'bowl', 'magnet', 'sink', 'air vent', 'box', 'door knob', 'door',
                                'scissor', 'tape dispenser', 'telephone cord', 'telephone', 'track light', 'cork board',
                                'cup', 'desk', 'laptop', 'air duct', 'basket', 'camera', 'pipe', 'shelves',
                                'stacked chairs', 'styrofoam object', 'whiteboard', 'computer', 'keyboard', 'ladder',
                                'monitor', 'stand', 'bar', 'motion camera', 'projector screen', 'speaker', 'bag',
                                'clock', 'green screen', 'mantel', 'window', 'ball', 'hole puncher', 'light',
                                'manilla envelope', 'picture', 'mail shelf', 'printer', 'stapler', 'fax machine',
                                'folder', 'jar', 'magazine', 'ruler', 'cable modem', 'fan', 'file', 'hand sanitizer',
                                'paper rack', 'vase', 'air conditioner', 'blinds', 'flower', 'plant', 'sofa', 'stereo',
                                'books', 'exit sign', 'room divider', 'bookshelf', 'curtain', 'projector', 'modem',
                                'wire', 'water purifier', 'column', 'hooks', 'hanging hooks', 'pen',
                                'electrical outlet', 'doll', 'eraser', 'pencil holder', 'water carboy', 'mouse',
                                'cable rack', 'wire rack', 'flipboard', 'map', 'paper cutter', 'tape', 'thermostat',
                                'heater', 'circuit breaker box', 'paper towel', 'stamp', 'duster', 'poster case',
                                'whiteboard marker', 'ethernet jack', 'pillow', 'hair brush', 'makeup brush', 'mirror',
                                'shower curtain', 'toilet', 'toiletries bag', 'toothbrush holder', 'toothbrush',
                                'toothpaste', 'platter', 'rug', 'squeeze tube', 'shower cap', 'soap', 'towel rod',
                                'towel', 'bathtub', 'candle', 'tissue box', 'toilet paper', 'container', 'clothes',
                                'electric toothbrush', 'floor mat', 'lamp', 'drum', 'flower pot', 'banana',
                                'candlestick', 'shoe', 'stool', 'urn', 'earplugs', 'mailshelf', 'placemat',
                                'excercise ball', 'alarm clock', 'bed', 'night stand', 'deoderant', 'headphones',
                                'headboard', 'basketball hoop', 'foot rest', 'laundry basket', 'sock', 'football',
                                'mens suit', 'cable box', 'dresser', 'dvd player', 'shaver', 'television',
                                'contact lens solution bottle', 'drawer', 'remote control', 'cologne', 'stuffed animal',
                                'lint roller', 'tray', 'lock', 'purse', 'toy bottle', 'crate', 'vasoline',
                                'gift wrapping roll', 'wall decoration', 'hookah', 'radio', 'bicycle', 'pen box',
                                'mask', 'shorts', 'hat', 'hockey glove', 'hockey stick', 'vuvuzela', 'dvd',
                                'chessboard', 'suitcase', 'calculator', 'flashcard', 'staple remover', 'umbrella',
                                'bench', 'yoga mat', 'backpack', 'cd', 'sign', 'hangers', 'notebook', 'hanger',
                                'security camera', 'folders', 'clothing hanger', 'stairs', 'glass rack', 'saucer',
                                'tag', 'dolly', 'machine', 'trolly', 'shopping baskets', 'gate', 'bookrack',
                                'blackboard', 'coffee bag', 'coffee packet', 'hot water heater', 'muffins',
                                'napkin dispenser', 'plaque', 'plastic tub', 'plate', 'coffee machine', 'napkin holder',
                                'radiator', 'coffee grinder', 'oven', 'plant pot', 'scarf', 'spice rack', 'stove',
                                'tea kettle', 'napkin', 'bag of chips', 'bread', 'cutting board', 'dish brush',
                                'serving spoon', 'sponge', 'toaster', 'cooking pan', 'kitchen items', 'ladel',
                                'spatula', 'spice stand', 'trivet', 'knife rack', 'knife', 'baking dish',
                                'dish scrubber', 'drying rack', 'vessel', 'kichen towel', 'tin foil', 'kitchen utensil',
                                'utensil', 'blender', 'garbage bag', 'sink protector', 'box of ziplock bags',
                                'spice bottle', 'pitcher', 'pizza box', 'toaster oven', 'step stool',
                                'vegetable peeler', 'washing machine', 'can opener', 'can of food',
                                'paper towel holder', 'spoon stand', 'spoon', 'wooden kitchen utensils', 'bag of flour',
                                'fruit', 'sheet of metal', 'waffle maker', 'cake', 'cell phone', 'tv stand',
                                'tablecloth', 'wine glass', 'sculpture', 'wall stand', 'iphone', 'coke bottle', 'piano',
                                'wine rack', 'guitar', 'light switch', 'shirts in hanger', 'router', 'glass pot',
                                'cart', 'vacuum cleaner', 'bin', 'coins', 'hand sculpture', 'ipod', 'jersey', 'blanket',
                                'ironing board', 'pen stand', 'mens tie', 'glass baking dish', 'utensils', 'frying pan',
                                'shopping cart', 'plastic bowl', 'wooden container', 'onion', 'potato', 'jacket',
                                'dvds', 'surge protector', 'tumbler', 'broom', 'can', 'crock pot', 'person',
                                'salt shaker', 'wine bottle', 'apple', 'eye glasses', 'menorah', 'bicycle helmet',
                                'fire alarm', 'water fountain', 'humidifier', 'necklace', 'chandelier', 'barrel',
                                'chest', 'decanter', 'wooden utensils', 'globe', 'sheets', 'fork', 'napkin ring',
                                'gift wrapping', 'bed sheets', 'spot light', 'lighting track', 'cannister',
                                'coffee table', 'mortar and pestle', 'stack of plates', 'ottoman', 'server',
                                'salt container', 'utensil container', 'phone jack', 'switchbox', 'casserole dish',
                                'oven handle', 'whisk', 'dish cover', 'electric mixer', 'decorative platter',
                                'drawer handle', 'fireplace', 'stroller', 'bookend', 'table runner', 'typewriter',
                                'ashtray', 'key', 'suit jacket', 'range hood', 'cleaning wipes', 'six pack of beer',
                                'decorative plate', 'watch', 'balloon', 'ipad', 'coaster', 'whiteboard eraser', 'toy',
                                'toys basket', 'toy truck', 'classroom board', 'chart stand', 'picture of fish',
                                'plastic box', 'pencil', 'carton', 'walkie talkie', 'binder', 'coat hanger',
                                'filing shelves', 'plastic crate', 'plastic rack', 'plastic tray', 'flag',
                                'poster board', 'lunch bag', 'board', 'leg of a girl', 'file holder', 'chart',
                                'glass pane', 'cardboard tube', 'bassinet', 'toy car', 'toy shelf', 'toy bin',
                                'toys shelf', 'educational display', 'placard', 'soft toy group', 'soft toy',
                                'toy cube', 'toy cylinder', 'toy rectangle', 'toy triangle', 'bucket', 'chalkboard',
                                'game table', 'storage shelvesbooks', 'toy cuboid', 'toy tree', 'wooden toy', 'toy box',
                                'toy phone', 'toy sink', 'toyhouse', 'notecards', 'toy trucks',
                                'wall hand sanitizer dispenser', 'cap stand', 'music stereo', 'toys rack',
                                'display board', 'lid of jar', 'stacked bins  boxes', 'stacked plastic racks',
                                'storage rack', 'roll of paper towels', 'cables', 'power surge', 'cardboard sheet',
                                'banister', 'show piece', 'pepper shaker', 'kitchen island', 'excercise equipment',
                                'treadmill', 'ornamental plant', 'piano bench', 'sheet music', 'grandfather clock',
                                'iron grill', 'pen holder', 'toy doll', 'globe stand', 'telescope', 'magazine holder',
                                'file container', 'paper holder', 'flower box', 'pyramid', 'desk mat', 'cordless phone',
                                'desk drawer', 'envelope', 'window frame', 'id card', 'file stand', 'paper weight',
                                'toy plane', 'money', 'papers', 'comforter', 'crib', 'doll house', 'toy chair',
                                'toy sofa', 'plastic chair', 'toy house', 'child carrier', 'cloth bag', 'cradle',
                                'baby chair', 'chart roll', 'toys box', 'railing', 'clothing dryer', 'clothing washer',
                                'laundry detergent jug', 'clothing detergent', 'bottle of soap', 'box of paper',
                                'trolley', 'hand sanitizer dispenser', 'soap holder', 'water dispenser', 'photo',
                                'water cooler', 'foosball table', 'crayon', 'hoola hoop', 'horse toy',
                                'plastic toy container', 'pool table', 'game system', 'pool sticks', 'console system',
                                'video game', 'pool ball', 'trampoline', 'tricycle', 'wii', 'furniture', 'alarm',
                                'toy table', 'ornamental item', 'copper vessel', 'stick', 'car', 'mezuza',
                                'toy cash register', 'lid', 'paper bundle', 'business cards', 'clipboard',
                                'flatbed scanner', 'paper tray', 'mouse pad', 'display case', 'tree sculpture',
                                'basketball', 'fiberglass case', 'framed certificate', 'cordless telephone', 'shofar',
                                'trophy', 'cleaner', 'cloth drying stand', 'electric box', 'furnace', 'piece of wood',
                                'wooden pillar', 'drying stand', 'cane', 'clothing drying rack', 'iron box',
                                'excercise machine', 'sheet', 'rope', 'sticks', 'wooden planks', 'toilet plunger',
                                'bar of soap', 'toilet bowl brush', 'light bulb', 'drain', 'faucet handle',
                                'nailclipper', 'shaving cream', 'rolled carpet', 'clothing iron', 'window cover',
                                'charger and wire', 'quilt', 'mattress', 'hair dryer', 'stones', 'pepper grinder',
                                'cat cage', 'dish rack', 'curtain rod', 'calendar', 'head phones', 'cd disc',
                                'head phone', 'usb drive', 'water heater', 'pan', 'tuna cans', 'baby gate',
                                'spoon sets', 'cans of cat food', 'cat', 'flower basket', 'fruit platter', 'grapefruit',
                                'kiwi', 'hand blender', 'knobs', 'vessels', 'cell phone charger', 'wire basket',
                                'tub of tupperware', 'candelabra', 'litter box', 'shovel', 'cat bed', 'door way',
                                'belt', 'surge protect', 'glass', 'console controller', 'shoe rack', 'door frame',
                                'computer disk', 'briefcase', 'mail tray', 'file pad', 'letter stand',
                                'plastic cup of coffee', 'glass box', 'ping pong ball', 'ping pong racket',
                                'ping pong table', 'tennis racket', 'ping pong racquet', 'xbox',
                                'electric toothbrush base', 'toilet brush', 'toiletries', 'razor',
                                'bottle of contact lens solution', 'contact lens case', 'cream', 'glass container',
                                'container of skin cream', 'soap dish', 'scale', 'soap stand', 'cactus',
                                'door  window  reflection', 'ceramic frog', 'incense candle', 'storage space',
                                'door lock', 'toilet paper holder', 'tissue', 'personal care liquid', 'shower head',
                                'shower knob', 'knob', 'cream tube', 'perfume box', 'perfume', 'back scrubber',
                                'door facing trimreflection', 'doorreflection', 'light switchreflection',
                                'medicine tube', 'wallet', 'soap tray', 'door curtain', 'shower pipe',
                                'face wash cream', 'flashlight', 'shower base', 'window shelf', 'shower hose',
                                'toothpaste holder', 'soap box', 'incense holder', 'conch shell',
                                'roll of toilet paper', 'shower tube', 'bottle of listerine',
                                'bottle of hand wash liquid', 'tea pot', 'lazy susan', 'avocado', 'fruit stand',
                                'fruitplate', 'oil container', 'package of water', 'bottle of liquid', 'door way arch',
                                'jug', 'bulb', 'bagel', 'bag of bagels', 'banana peel', 'bag of oreo', 'flask',
                                'collander', 'brick', 'torch', 'dog bowl', 'wooden plank', 'eggs', 'grill', 'dog',
                                'chimney', 'dog cage', 'orange plastic cap', 'glass set', 'vessel set', 'mellon',
                                'aluminium foil', 'orange', 'peach', 'tea coaster', 'butterfly sculpture', 'corkscrew',
                                'heating tray', 'food processor', 'corn', 'squash', 'watermellon', 'vegetables',
                                'celery', 'glass dish', 'hot dogs', 'plastic dish', 'vegetable', 'sticker', 'chapstick',
                                'sifter', 'fruit basket', 'glove', 'measuring cup', 'water filter', 'wine accessory',
                                'dishes', 'file box', 'ornamental pot', 'dog toy', 'salt and pepper',
                                'electrical kettle', 'kitchen container plastic', 'pineapple', 'suger jar', 'steamer',
                                'charger', 'mug holder', 'orange juicer', 'juicer', 'bag of hot dog buns',
                                'hamburger bun', 'mug hanger', 'bottle of ketchup', 'toy kitchen',
                                'food wrapped on a tray', 'kitchen utensils', 'oven mitt', 'bottle of comet',
                                'wooden utensil', 'decorative dish', 'handle', 'label', 'flask set',
                                'cooking pot cover', 'tupperware', 'garlic', 'tissue roll', 'lemon', 'wine',
                                'decorative bottle', 'wire tray', 'tea cannister', 'clothing hamper', 'guitar case',
                                'wardrobe', 'boomerang', 'button', 'karate belts', 'medal', 'window seat', 'window box',
                                'necklace holder', 'beeper', 'webcam', 'fish tank', 'luggage', 'life jacket',
                                'shoelace', 'pen cup', 'eyeball plastic ball', 'toy pyramid', 'model boat',
                                'certificate', 'puppy toy', 'wire board', 'quill', 'canister', 'toy boat', 'antenna',
                                'bean bag', 'lint comb', 'travel bag', 'wall divider', 'toy chest', 'headband',
                                'luggage rack', 'bunk bed', 'lego', 'yarmulka', 'package of bedroom sheets',
                                'bedding package', 'comb', 'dollar bill', 'pig', 'storage bin', 'storage chest',
                                'slide', 'playpen', 'electronic drumset', 'ipod dock', 'microphone', 'music keyboard',
                                'music stand', 'microphone stand', 'album', 'kinect', 'inkwell', 'baseball',
                                'decorative bowl', 'book holder', 'toy horse', 'desser', 'toy apple', 'toy dog',
                                'scenary', 'drawer knob', 'shoe hanger', 'tent', 'figurine', 'soccer ball',
                                'hand weight', 'magic 8ball', 'bottle of perfume', 'sleeping bag', 'decoration item',
                                'envelopes', 'trinket', 'hand fan', 'sculpture of the chrysler building',
                                'sculpture of the eiffel tower', 'sculpture of the empire state building', 'jeans',
                                'garage door', 'case', 'rags', 'decorative item', 'toy stroller', 'shelf frame',
                                'cat house', 'can of beer', 'dog bed', 'lamp shade', 'bracelet',
                                'reflection of window shutters', 'decorative egg', 'indoor fountain', 'photo album',
                                'decorative candle', 'walkietalkie', 'serving dish', 'floor trim',
                                'mini display platform', 'american flag', 'vhs tapes', 'throw', 'newspapers', 'mantle',
                                'package of bottled water', 'serving platter', 'display platter', 'centerpiece',
                                'tea box', 'gold piece', 'wreathe', 'lectern', 'hammer', 'matchbox', 'pepper',
                                'yellow pepper', 'duck', 'eggplant', 'glass ware', 'sewing machine', 'rolled up rug',
                                'doily', 'coffee pot', 'torah']
                self.CLASS_COLORS = np.array(_get_colormap(1 + 894).tolist(), dtype='uint8')

    def __getitem__(self, index):
        divided_file = self.fileset[index].split()
        if self.args.dataset == 'KITTI':
            date = divided_file[0].split('/')[0] + '/'

        # Opening image files.   rgb: input color image, gt: sparse depth map
        rgb_file = self.args.data_path + '/' + divided_file[0]
        rgb = Image.open(rgb_file)
        gt = False
        gt_dense = False
        gt_seg = False
        if (self.train is False):
            divided_file_ = divided_file[0].split('/')
            if self.args.dataset == 'KITTI':
                filename = divided_file_[1] + '_' + divided_file_[4]
            else:
                filename = divided_file_[1] + '_' + divided_file_[2][-9:]

            if self.args.dataset == 'KITTI':
                # Considering missing gt in Eigen split
                if divided_file[1] != 'None':
                    gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                    gt = Image.open(gt_file)
                    if self.use_dense_depth is True:
                        gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
                        gt_dense = Image.open(gt_dense_file)
                else:
                    pass
            elif self.args.dataset == 'NYU':
                gt_file = self.args.data_path + '/' + divided_file[1]
                gt = Image.open(gt_file)
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
                    gt_dense = Image.open(gt_dense_file)
                if self.args.use_seg:
                    gt_seg_file = divided_file[0][:-13] + 'labels/labels' + divided_file[0][-10:-4] + '.png'
                    gt_seg_file = self.args.data_path + '/' + gt_seg_file
                    gt_seg = Image.open(gt_seg_file)
        else:
            angle = np.random.uniform(self.angle_range[0], self.angle_range[1])
            if self.args.dataset == 'KITTI':
                gt_file = self.args.data_path + '/data_depth_annotated/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/data_depth_annotated/' + divided_file[2]
            elif self.args.dataset == 'NYU':
                gt_file = self.args.data_path + '/' + divided_file[1]
                if self.use_dense_depth is True:
                    gt_dense_file = self.args.data_path + '/' + divided_file[2]
                if self.args.use_seg:
                    gt_seg_file = divided_file[0][:-13] + 'labels/labels' + divided_file[0][-10:-4] + '.png'
                    gt_seg_file = self.args.data_path + '/' + gt_seg_file

            gt = Image.open(gt_file)
            rgb = rgb.rotate(angle, resample=Image.BILINEAR)
            gt = gt.rotate(angle, resample=Image.NEAREST)
            if self.use_dense_depth is True:
                gt_dense = Image.open(gt_dense_file)
                gt_dense = gt_dense.rotate(angle, resample=Image.NEAREST)
            if self.args.use_seg:
                gt_seg = Image.open(gt_seg_file)
                gt_seg = gt_seg.rotate(angle, resample=Image.NEAREST)

        # cropping in size that can be divided by 16
        if self.args.dataset == 'KITTI':
            h = rgb.height
            w = rgb.width
            bound_left = (w - 1216) // 2
            bound_right = bound_left + 1216
            bound_top = h - 352
            bound_bottom = bound_top + 352
        elif self.args.dataset == 'NYU':
            bound_left = 43
            bound_right = 608
            bound_top = 8
            bound_bottom = 472
        # crop and normalize 0 to 1 ==>  rgb range:(0,1),  depth range: (0, max_depth)

        rgb = rgb.crop((bound_left, bound_top, bound_right, bound_bottom))

        rgb = np.asarray(rgb, dtype=np.float32) / 255.0

        if _is_pil_image(gt):
            gt = gt.crop((bound_left, bound_top, bound_right, bound_bottom))
            gt = (np.asarray(gt, dtype=np.float32)) / self.depth_scale
            gt = np.expand_dims(gt, axis=2)
            gt = np.clip(gt, 0, self.args.max_depth)
        if self.use_dense_depth is True:
            if _is_pil_image(gt_dense):
                gt_dense = gt_dense.crop((bound_left, bound_top, bound_right, bound_bottom))
                gt_dense = (np.asarray(gt_dense, dtype=np.float32)) / self.depth_scale
                gt_dense = np.expand_dims(gt_dense, axis=2)
                gt_dense = np.clip(gt_dense, 0, self.args.max_depth)
                gt_dense = gt_dense * (gt.max() / gt_dense.max())
        if self.args.use_seg:
            if _is_pil_image(gt_seg):
                gt_seg = gt_seg.crop((bound_left, bound_top, bound_right, bound_bottom))
                gt_seg = np.asarray(gt_seg, dtype=np.int_)
                gt_seg = np.expand_dims(gt_seg, axis=2)

        rgb, gt, gt_dense, gt_seg = self.transform([rgb] + [gt] + [gt_dense] + [gt_seg], self.train)

        if self.args.use_seg:
            if self.args.use_40:
                gt_seg = self.mapping_894_to_40[gt_seg]
            if self.args.use_dense_depth:
                gt_dense = (gt_seg, gt_dense)
            else:
                gt_dense = gt_seg

        if self.return_filename is True:
            return rgb, gt, gt_dense, filename
        elif self.args.use_sparse:
            return rgb, gt, gt_dense
        else:
            if self.train:
                return dict(img=rgb, gt_semantic_seg=gt_dense, img_metas=rgb)
            else:
                return dict(img=(rgb, gt_dense), img_metas=(rgb, gt_dense))

    def __len__(self):
        return len(self.fileset)

    def evaluate(self, results, metric='mIoU', logger=None, **kwargs):
        if self.args.use_seg:
            output_seg, output_depth, gt_seg, gt_depth = [], [], [], []
            if self.args.use_dense_depth:
                for output, gt in results:
                    output_seg.append(output[0])
                    output_depth.append(output[1].cuda())
                    gt_seg.append(gt[0].cpu().numpy())
                    gt_depth.append(gt[1].cuda())

                output_depth = torch.cat(output_depth, 0)
                gt_depth = torch.cat(gt_depth, 0)
                err_result = compute_errors_NYU(gt_depth, output_depth, crop=False)  # TODO: pass dataset
                print(err_result)
            else:
                for output, gt in results:
                    output_seg.append(output)
                    gt_seg.append(gt.cpu().numpy())

                err_result = {}

            if not isinstance(metric, str):
                assert len(metric) == 1
                metric = metric[0]
            allowed_metrics = ['mIoU']
            if metric not in allowed_metrics:
                raise KeyError('metric {} is not supported'.format(metric))

            eval_results = {}
            if self.CLASSES is None:
                num_classes = len(
                    reduce(np.union1d, [np.unique(_) for _ in gt_seg]))
            else:
                num_classes = len(self.CLASSES)

            all_acc, acc, iou = mean_iou(
                output_seg, gt_seg, num_classes, ignore_index=self.args.ignore_index)
            summary_str = ''
            summary_str += 'Summary:\n'
            line_format = '{:<15} {:>10} {:>10} {:>10}\n'
            summary_str += line_format.format('Scope', 'mIoU', 'mAcc', 'aAcc')

            iou_str = '{:.2f}'.format(np.nanmean(iou) * 100)
            acc_str = '{:.2f}'.format(np.nanmean(acc) * 100)
            all_acc_str = '{:.2f}'.format(all_acc * 100)
            summary_str += line_format.format('global', iou_str, acc_str,
                                              all_acc_str)

            summary_str += 'per class results:\n'

            line_format = '{:<15} {:>10} {:>10}\n'
            summary_str += line_format.format('Class', 'IoU', 'Acc')
            if self.CLASSES is None:
                class_names = tuple(range(num_classes))
            else:
                class_names = self.CLASSES
            for i in range(num_classes):
                iou_str = '{:.2f}'.format(iou[i] * 100)
                acc_str = '{:.2f}'.format(acc[i] * 100)
                summary_str += line_format.format(class_names[i], iou_str, acc_str)

            print_log(summary_str, logger)

            eval_results['mIoU'] = np.nanmean(iou)
            eval_results['mAcc'] = np.nanmean(acc)
            eval_results['aAcc'] = all_acc

            err_result.update(eval_results)
            return err_result
        else:
            output_depth, gt_data = [], []
            for od, gd in results:
                output_depth.append(od.cuda())
                gt_data.append(gd.cuda())
            output_depth = torch.cat(output_depth, 0)
            gt_data = torch.cat(gt_data, 0)
            err_result = compute_errors_NYU(gt_data, output_depth, crop=False)  # TODO: pass dataset
            print(err_result)
            return err_result

    def format_results(self, results, **kwargs):
        print('use_seg:         ', self.args.use_seg)
        print('use_dense_depth: ', self.args.use_dense_depth)
        if self.args.use_seg:
            output_seg, output_depth, gt_seg, gt_depth = [], [], [], []
            if self.args.use_dense_depth:
                cmap = plt.cm.turbo_r
                diff_cmap = sns.color_palette("vlag", as_cmap=True)
                if not os.path.exists(os.path.join('images', 'output_depth_cmap_turbor')):
                    os.makedirs(os.path.join('images', 'output_depth_cmap_turbor'))
                if not os.path.exists(os.path.join('images', 'dense_gt_cmap_turbor')):
                    os.makedirs(os.path.join('images', 'dense_gt_cmap_turbor'))
                if not os.path.exists(os.path.join('images', 'cdiff_output_depth')):
                    os.makedirs(os.path.join('images', 'cdiff_output_depth'))
                if not os.path.exists(os.path.join('images', 'diff_output_depth')):
                    os.makedirs(os.path.join('images', 'diff_output_depth'))

                for output, gt in results:
                    output_seg.append(output[0])
                    output_depth.append(output[1].cuda())
                    gt_seg.append(gt[0].cpu().numpy())
                    gt_depth.append(gt[1].cuda())

                for i in tqdm(list(range(len(self)))):
                    final_depth = output_depth[i]
                    gt_data = gt_depth[i]
                    gt_dense = gt_data

                    img_H = gt_data.shape[2]
                    img_W = gt_data.shape[3]

                    d_min = gt_dense.min()
                    d_max = gt_dense.max()

                    d_min = d_min.cpu().detach().numpy().astype(np.float32)
                    d_max = d_max.cpu().detach().numpy().astype(np.float32)
                    # TODO: save contours
                    divided_file = self.fileset[i].split()
                    divided_file_ = divided_file[0].split('/')
                    filename = divided_file_[1] + '_' + divided_file_[2][-9:]

                    gt_ = np.squeeze(gt_dense.cpu().numpy().astype(np.float32))
                    out_ = np.squeeze(final_depth.cpu().numpy().astype(np.float32))
                    gt_ = ((gt_ - d_min) / (d_max - d_min))
                    out_ = ((out_ - d_min) / (d_max - d_min))
                    diff = out_ - gt_
                    cdiff = diff / 2 + 0.5
                    gt_ = cmap(gt_) * 255
                    out_ = cmap(out_) * 255
                    diff_ = plt.cm.Greys(np.abs(diff)) * 255
                    cdiff_ = diff_cmap(cdiff) * 255
                    Image.fromarray(gt_.astype('uint8')).save(
                        os.path.join('images', 'dense_gt_cmap_turbor', 'gt_dense_cmap_turbor_' + filename)[
                        :-4] + '.png')
                    Image.fromarray(out_.astype('uint8')).save(
                        os.path.join('images', 'output_depth_cmap_turbor', 'cmap_turbor_' + filename)[:-4] + '.png')
                    Image.fromarray(diff_.astype('uint8')).save(
                        os.path.join('images', 'diff_output_depth', 'diff_' + filename)[:-4] + '.png')
                    Image.fromarray(cdiff_.astype('uint8')).save(
                        os.path.join('images', 'cdiff_output_depth', 'cdiff_' + filename)[:-4] + '.png')
            else:
                for output, gt in results:
                    output_seg.append(output)
                    gt_seg.append(gt.cpu().numpy())

            for idx in tqdm(list(range(len(self)))):
                divided_file = self.fileset[idx].split()
                divided_file_ = divided_file[0].split('/')
                basename = divided_file_[1] + '_' + divided_file_[2][-9:-4]

                imgfile_prefix = './images/output_seg'
                if not os.path.exists(imgfile_prefix):
                    os.makedirs(imgfile_prefix)
                result = output_seg[idx][0, 0, :, :]

                # save seg_pred
                # print(result.shape)
                png_filename = os.path.join(imgfile_prefix, f'out_seg_{basename}.png')
                # result = result + 1

                # output = Image.fromarray(result.astype(np.uint8))
                # output.save(png_filename)
                # colored label image
                # (indexed png8 with color palette)
                save_indexed_png(png_filename, result, self.CLASS_COLORS)

                imgfile_prefix = './images/gt_seg'
                if not os.path.exists(imgfile_prefix):
                    os.makedirs(imgfile_prefix)
                result = gt_seg[idx][0, 0, :, :]

                # save seg_gt
                # print(result.shape)
                png_filename = os.path.join(imgfile_prefix, f'gt_seg_{basename}.png')
                # result = result + 1

                # output = Image.fromarray(result.astype(np.uint8))
                # output.save(png_filename)
                # colored label image
                # (indexed png8 with color palette)
                save_indexed_png(png_filename, result, self.CLASS_COLORS)
        else:
            cmap = plt.cm.turbo_r
            diff_cmap = sns.color_palette("vlag", as_cmap=True)
            if not os.path.exists(os.path.join('images', 'output_depth_cmap_turbor')):
                os.makedirs(os.path.join('images', 'output_depth_cmap_turbor'))
            if not os.path.exists(os.path.join('images', 'dense_gt_cmap_turbor')):
                os.makedirs(os.path.join('images', 'dense_gt_cmap_turbor'))
            if not os.path.exists(os.path.join('images', 'cdiff_output_depth')):
                os.makedirs(os.path.join('images', 'cdiff_output_depth'))
            if not os.path.exists(os.path.join('images', 'diff_output_depth')):
                os.makedirs(os.path.join('images', 'diff_output_depth'))

            output_depth, gt_depth = [], []
            for od, gd in results:
                output_depth.append(od.cuda())
                gt_depth.append(gd.cuda())
            for i in tqdm(list(range(len(self)))):
                final_depth = output_depth[i]
                gt_data = gt_depth[i]
                gt_dense = gt_data

                img_H = gt_data.shape[2]
                img_W = gt_data.shape[3]

                d_min = gt_dense.min()
                d_max = gt_dense.max()

                d_min = d_min.cpu().detach().numpy().astype(np.float32)
                d_max = d_max.cpu().detach().numpy().astype(np.float32)
                # TODO: save contours
                divided_file = self.fileset[i].split()
                divided_file_ = divided_file[0].split('/')
                filename = divided_file_[1] + '_' + divided_file_[2][-9:]

                gt_ = np.squeeze(gt_dense.cpu().numpy().astype(np.float32))
                out_ = np.squeeze(final_depth.cpu().numpy().astype(np.float32))
                gt_ = ((gt_ - d_min) / (d_max - d_min))
                out_ = ((out_ - d_min) / (d_max - d_min))
                diff = out_ - gt_
                cdiff = diff / 2 + 0.5
                gt_ = cmap(gt_) * 255
                out_ = cmap(out_) * 255
                diff_ = plt.cm.Greys(np.abs(diff)) * 255
                cdiff_ = diff_cmap(cdiff) * 255
                Image.fromarray(gt_.astype('uint8')).save(
                    os.path.join('images', 'dense_gt_cmap_turbor', 'gt_dense_cmap_turbor_' + filename)[:-4] + '.png')
                Image.fromarray(out_.astype('uint8')).save(
                    os.path.join('images', 'output_depth_cmap_turbor', 'cmap_turbor_' + filename)[:-4] + '.png')
                Image.fromarray(diff_.astype('uint8')).save(
                    os.path.join('images', 'diff_output_depth', 'diff_' + filename)[:-4] + '.png')
                Image.fromarray(cdiff_.astype('uint8')).save(
                    os.path.join('images', 'cdiff_output_depth', 'cdiff_' + filename)[:-4] + '.png')


class Transformer(object):
    def __init__(self, args):
        if args.dataset == 'KITTI':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.9, 1.1)), None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
            self.test_transform = EnhancedCompose([
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None]
            ])
        elif args.dataset == 'NYU':
            self.train_transform = EnhancedCompose([
                RandomCropNumpy((args.height, args.width)),
                RandomHorizontalFlip(),
                [RandomColor(multiplier_range=(0.8, 1.2), brightness_mult_range=(0.75, 1.25)), None, None, None],
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None, None]
            ])
            self.test_transform = EnhancedCompose([
                CenterCrop((args.height, args.width)),
                ArrayToTensorNumpy(),
                [transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), None, None, None]
            ])

    def __call__(self, images, train=True):
        if train is True:
            return self.train_transform(images)
        else:
            return self.test_transform(images)
