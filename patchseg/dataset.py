import os
from PIL import Image
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms

from patchseg.transforms import (
    LargestCenterCrop,
    ReduceAnnotation,
    CocoLabelAssignment,
)


class CocoPatchDataset(Dataset):
    """Dataset to fetch and pre-process COCO images and segmentation masks."""

    def __init__(
        self,
        image_width: int,
        patch_size: int,
        split: str,
        image_mean: Tuple[float],
        image_std: Tuple[float],
        label_reduction: str,
        class_set: str = "all",
        dataset_path: str = "coco",
    ) -> None:
        # Verify arguments.
        assert split in ["train", "val", "test"]
        assert label_reduction in ["majority", "union", "global-union"]
        self.image_width = image_width
        self.patch_size = patch_size
        self.split = split
        self.label_reduction = label_reduction
        self.class_set = class_set

        # Find image filenames.
        image_dir = os.path.join(dataset_path, "images", f"{split}2017")
        image_filenames = sorted(os.listdir(image_dir))

        # Find annotation filenames.
        annotation_dir = os.path.join(dataset_path, "annotations", f"{split}2017")
        annotation_filenames = sorted(os.listdir(annotation_dir))

        # Verify that filenames match.
        assert len(image_filenames) == len(annotation_filenames)
        for image_file, annotation_file in zip(image_filenames, annotation_filenames):
            assert image_file == annotation_file.split(".")[0] + ".jpg"
        self.image_filenames = [os.path.join(image_dir, filename) for filename in image_filenames]
        self.annotation_filenames = [os.path.join(annotation_dir, filename) for filename in annotation_filenames]

        # Set class indices.
        if class_set == "all":
            self.num_classes = 183
        elif class_set == "things":
            self.num_classes = 92
        else:
            raise ValueError(f"Invalid class set: {class_set}")
        self.index_to_class = {index: coco_index_to_class[str(index)] for index in range(self.num_classes)}

        # Prepare image transforms.
        self.image_transform = transforms.Compose(
            [
                LargestCenterCrop(),
                transforms.Resize((image_width, image_width)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=torch.tensor(image_mean), std=torch.tensor(image_std)),
            ]
        )

        # Prepare annotation transforms.
        self.annotation_transform = transforms.Compose(
            [
                LargestCenterCrop(),
                transforms.Resize((image_width, image_width), interpolation=Image.NEAREST),
                transforms.ToImage(),
                transforms.ToDtype(torch.long),
                CocoLabelAssignment(class_set),
                ReduceAnnotation(patch_size, self.num_classes, label_reduction),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare image.
        image_file = self.image_filenames[index]
        x = Image.open(image_file).convert("RGB")
        x = self.image_transform(x)

        # Prepare annotation.
        annotation_file = self.annotation_filenames[index]
        y = Image.open(annotation_file)
        y = self.annotation_transform(y)

        return x, y


class Ade20kPatchDataset(Dataset):
    """Dataset to fetch and pre-process Ade20k images and segmentation masks."""

    def __init__(
        self,
        image_width: int,
        patch_size: int,
        split: str,
        image_mean: Tuple[float],
        image_std: Tuple[float],
        label_reduction: str,
        dataset_path: str = "ade20k",
    ) -> None:
        # Verify arguments.
        assert split in ["train", "val"]
        assert label_reduction in ["majority", "union", "global-union"]
        self.image_width = image_width
        self.patch_size = patch_size
        self.split = split
        self.label_reduction = label_reduction

        # Find image filenames.
        split = {"train": "training", "val": "validation"}[split]
        image_dir = os.path.join(dataset_path, "images", split)
        image_filenames = sorted(os.listdir(image_dir))

        # Find annotation filenames.
        annotation_dir = os.path.join(dataset_path, "annotations", split)
        annotation_filenames = sorted(os.listdir(annotation_dir))

        # Verify that filenames match.
        assert len(image_filenames) == len(annotation_filenames)
        for image_file, annotation_file in zip(image_filenames, annotation_filenames):
            assert image_file == annotation_file.split(".")[0] + ".jpg"
        self.image_filenames = [os.path.join(image_dir, filename) for filename in image_filenames]
        self.annotation_filenames = [os.path.join(annotation_dir, filename) for filename in annotation_filenames]

        # Set class indices.
        self.num_classes = 151
        self.index_to_class = {index: ade20k_index_to_class[str(index)] for index in range(self.num_classes)}

        # Prepare image transforms.
        self.image_transform = transforms.Compose(
            [
                LargestCenterCrop(),
                transforms.Resize((image_width, image_width)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(mean=torch.tensor(image_mean), std=torch.tensor(image_std)),
            ]
        )

        # Prepare annotation transforms.
        self.annotation_transform = transforms.Compose(
            [
                LargestCenterCrop(),
                transforms.Resize((image_width, image_width), interpolation=Image.NEAREST),
                transforms.ToImage(),
                transforms.ToDtype(torch.long),
                ReduceAnnotation(patch_size, self.num_classes, label_reduction),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_filenames)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Prepare image.
        image_file = self.image_filenames[index]
        x = Image.open(image_file).convert("RGB")
        x = self.image_transform(x)

        # Prepare annotation.
        annotation_file = self.annotation_filenames[index]
        y = Image.open(annotation_file)
        y = self.annotation_transform(y)

        return x, y


# 91 `things` classes (1-91), 91 `stuff` classes (92-182) and 1 class 'unlabeled' (0).
coco_index_to_class = {
    "0": "unlabeled",
    "1": "person",
    "2": "bicycle",
    "3": "car",
    "4": "motorcycle",
    "5": "airplane",
    "6": "bus",
    "7": "train",
    "8": "truck",
    "9": "boat",
    "10": "traffic light",
    "11": "fire hydrant",
    "12": "street sign",
    "13": "stop sign",
    "14": "parking meter",
    "15": "bench",
    "16": "bird",
    "17": "cat",
    "18": "dog",
    "19": "horse",
    "20": "sheep",
    "21": "cow",
    "22": "elephant",
    "23": "bear",
    "24": "zebra",
    "25": "giraffe",
    "26": "hat",
    "27": "backpack",
    "28": "umbrella",
    "29": "shoe",
    "30": "eye glasses",
    "31": "handbag",
    "32": "tie",
    "33": "suitcase",
    "34": "frisbee",
    "35": "skis",
    "36": "snowboard",
    "37": "sports ball",
    "38": "kite",
    "39": "baseball bat",
    "40": "baseball glove",
    "41": "skateboard",
    "42": "surfboard",
    "43": "tennis racket",
    "44": "bottle",
    "45": "plate",
    "46": "wine glass",
    "47": "cup",
    "48": "fork",
    "49": "knife",
    "50": "spoon",
    "51": "bowl",
    "52": "banana",
    "53": "apple",
    "54": "sandwich",
    "55": "orange",
    "56": "broccoli",
    "57": "carrot",
    "58": "hot dog",
    "59": "pizza",
    "60": "donut",
    "61": "cake",
    "62": "chair",
    "63": "couch",
    "64": "potted plant",
    "65": "bed",
    "66": "mirror",
    "67": "dining table",
    "68": "window",
    "69": "desk",
    "70": "toilet",
    "71": "door",
    "72": "tv",
    "73": "laptop",
    "74": "mouse",
    "75": "remote",
    "76": "keyboard",
    "77": "cell phone",
    "78": "microwave",
    "79": "oven",
    "80": "toaster",
    "81": "sink",
    "82": "refrigerator",
    "83": "blender",
    "84": "book",
    "85": "clock",
    "86": "vase",
    "87": "scissors",
    "88": "teddy bear",
    "89": "hair drier",
    "90": "toothbrush",
    "91": "hair brush",
    "92": "banner",
    "93": "blanket",
    "94": "branch",
    "95": "bridge",
    "96": "building-other",
    "97": "bush",
    "98": "cabinet",
    "99": "cage",
    "100": "cardboard",
    "101": "carpet",
    "102": "ceiling-other",
    "103": "ceiling-tile",
    "104": "cloth",
    "105": "clothes",
    "106": "clouds",
    "107": "counter",
    "108": "cupboard",
    "109": "curtain",
    "110": "desk-stuff",
    "111": "dirt",
    "112": "door-stuff",
    "113": "fence",
    "114": "floor-marble",
    "115": "floor-other",
    "116": "floor-stone",
    "117": "floor-tile",
    "118": "floor-wood",
    "119": "flower",
    "120": "fog",
    "121": "food-other",
    "122": "fruit",
    "123": "furniture-other",
    "124": "grass",
    "125": "gravel",
    "126": "ground-other",
    "127": "hill",
    "128": "house",
    "129": "leaves",
    "130": "light",
    "131": "mat",
    "132": "metal",
    "133": "mirror-stuff",
    "134": "moss",
    "135": "mountain",
    "136": "mud",
    "137": "napkin",
    "138": "net",
    "139": "paper",
    "140": "pavement",
    "141": "pillow",
    "142": "plant-other",
    "143": "plastic",
    "144": "platform",
    "145": "playingfield",
    "146": "railing",
    "147": "railroad",
    "148": "river",
    "149": "road",
    "150": "rock",
    "151": "roof",
    "152": "rug",
    "153": "salad",
    "154": "sand",
    "155": "sea",
    "156": "shelf",
    "157": "sky-other",
    "158": "skyscraper",
    "159": "snow",
    "160": "solid-other",
    "161": "stairs",
    "162": "stone",
    "163": "straw",
    "164": "structural-other",
    "165": "table",
    "166": "tent",
    "167": "textile-other",
    "168": "towel",
    "169": "tree",
    "170": "vegetable",
    "171": "wall-brick",
    "172": "wall-concrete",
    "173": "wall-other",
    "174": "wall-panel",
    "175": "wall-stone",
    "176": "wall-tile",
    "177": "wall-wood",
    "178": "water-other",
    "179": "waterdrops",
    "180": "window-blind",
    "181": "window-other",
    "182": "wood",
}


ade20k_index_to_class = {
    "0": "unlabeled",
    "1": "wall",
    "2": "building, edifice",
    "3": "sky",
    "4": "floor, flooring",
    "5": "tree",
    "6": "ceiling",
    "7": "road, route",
    "8": "bed",
    "9": "windowpane, window",
    "10": "grass",
    "11": "cabinet",
    "12": "sidewalk, pavement",
    "13": "person, individual, someone, somebody, mortal, soul",
    "14": "earth, ground",
    "15": "door, double door",
    "16": "table",
    "17": "mountain, mount",
    "18": "plant, flora, plant life",
    "19": "curtain, drape, drapery, mantle, pall",
    "20": "chair",
    "21": "car, auto, automobile, machine, motorcar",
    "22": "water",
    "23": "painting, picture",
    "24": "sofa, couch, lounge",
    "25": "shelf",
    "26": "house",
    "27": "sea",
    "28": "mirror",
    "29": "rug, carpet, carpeting",
    "30": "field",
    "31": "armchair",
    "32": "seat",
    "33": "fence, fencing",
    "34": "desk",
    "35": "rock, stone",
    "36": "wardrobe, closet, press",
    "37": "lamp",
    "38": "bathtub, bathing tub, bath, tub",
    "39": "railing, rail",
    "40": "cushion",
    "41": "base, pedestal, stand",
    "42": "box",
    "43": "column, pillar",
    "44": "signboard, sign",
    "45": "chest of drawers, chest, bureau, dresser",
    "46": "counter",
    "47": "sand",
    "48": "sink",
    "49": "skyscraper",
    "50": "fireplace, hearth, open fireplace",
    "51": "refrigerator, icebox",
    "52": "grandstand, covered stand",
    "53": "path",
    "54": "stairs, steps",
    "55": "runway",
    "56": "case, display case, showcase, vitrine",
    "57": "pool table, billiard table, snooker table",
    "58": "pillow",
    "59": "screen door, screen",
    "60": "stairway, staircase",
    "61": "river",
    "62": "bridge, span",
    "63": "bookcase",
    "64": "blind, screen",
    "65": "coffee table, cocktail table",
    "66": "toilet, can, commode, crapper, pot, potty, stool, throne",
    "67": "flower",
    "68": "book",
    "69": "hill",
    "70": "bench",
    "71": "countertop",
    "72": "stove, kitchen stove, range, kitchen range, cooking stove",
    "73": "palm, palm tree",
    "74": "kitchen island",
    "75": "computer, computing machine, computing device, data processor, electronic computer, information processing system",
    "76": "swivel chair",
    "77": "boat",
    "78": "bar",
    "79": "arcade machine",
    "80": "hovel, hut, hutch, shack, shanty",
    "81": "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle",
    "82": "towel",
    "83": "light, light source",
    "84": "truck, motortruck",
    "85": "tower",
    "86": "chandelier, pendant, pendent",
    "87": "awning, sunshade, sunblind",
    "88": "streetlight, street lamp",
    "89": "booth, cubicle, stall, kiosk",
    "90": "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box",
    "91": "airplane, aeroplane, plane",
    "92": "dirt track",
    "93": "apparel, wearing apparel, dress, clothes",
    "94": "pole",
    "95": "land, ground, soil",
    "96": "bannister, banister, balustrade, balusters, handrail",
    "97": "escalator, moving staircase, moving stairway",
    "98": "ottoman, pouf, pouffe, puff, hassock",
    "99": "bottle",
    "100": "buffet, counter, sideboard",
    "101": "poster, posting, placard, notice, bill, card",
    "102": "stage",
    "103": "van",
    "104": "ship",
    "105": "fountain",
    "106": "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
    "107": "canopy",
    "108": "washer, automatic washer, washing machine",
    "109": "plaything, toy",
    "110": "swimming pool, swimming bath, natatorium",
    "111": "stool",
    "112": "barrel, cask",
    "113": "basket, handbasket",
    "114": "waterfall, falls",
    "115": "tent, collapsible shelter",
    "116": "bag",
    "117": "minibike, motorbike",
    "118": "cradle",
    "119": "oven",
    "120": "ball",
    "121": "food, solid food",
    "122": "step, stair",
    "123": "tank, storage tank",
    "124": "trade name, brand name, brand, marque",
    "125": "microwave, microwave oven",
    "126": "pot, flowerpot",
    "127": "animal, animate being, beast, brute, creature, fauna",
    "128": "bicycle, bike, wheel, cycle",
    "129": "lake",
    "130": "dishwasher, dish washer, dishwashing machine",
    "131": "screen, silver screen, projection screen",
    "132": "blanket, cover",
    "133": "sculpture",
    "134": "hood, exhaust hood",
    "135": "sconce",
    "136": "vase",
    "137": "traffic light, traffic signal, stoplight",
    "138": "tray",
    "139": "ashcan, trash can, garbage can, wastebin, ash bin, ash-bin, ashbin, dustbin, trash barrel, trash bin",
    "140": "fan",
    "141": "pier, wharf, wharfage, dock",
    "142": "crt screen",
    "143": "plate",
    "144": "monitor, monitoring device",
    "145": "bulletin board, notice board",
    "146": "shower",
    "147": "radiator",
    "148": "glass, drinking glass",
    "149": "clock",
    "150": "flag",
}
