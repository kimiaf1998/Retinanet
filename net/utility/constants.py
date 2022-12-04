DATASET_PATH = 'C:\\Users\\Mohammad\\Downloads\\cocodataset2014\\data'
MODEL_PATH = 'C:\\Users\\Mohammad\\Downloads\\coco_resnet_50_map_0_335_state_dict.pt'

DATASET_MODEL = 'train2014'

TRAIN_EPOCH = 30
RESNET_DEP = 50

RESNET_MODELS_URL = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
}

TRAIN_SAMPLES_QTY = 5000
VALIDATION_SAMPLES_QTY = 2500

FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2

PYRAMID_LEVELS = [3, 4, 5, 6, 7]
RATIOS = [0.5, 1, 2]
SCALES = [2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]

NUMBER_OF_CLASSES = 80