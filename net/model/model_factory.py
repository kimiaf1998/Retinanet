import torch.utils.model_zoo as model_zoo

from net.model.bottleneck import Bottleneck
from net.model.retinanet import RetinaNet
from net.utility.constants import RESNET_MODELS_URL


def create_resnet(number_of_layers, num_classes, pretrained=False, **kwargs):
    if number_of_layers == 50:
        shape = [3, 4, 6, 3]
        url = RESNET_MODELS_URL['resnet50']
    elif number_of_layers == 101:
        shape = [3, 4, 23, 3]
        url = RESNET_MODELS_URL['resnet101']
    else:
        raise Exception('Wrong number of layers - RESNET')

    model = RetinaNet(num_classes, Bottleneck, shape, **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(url, model_dir='../model'), strict=False)
    return model
