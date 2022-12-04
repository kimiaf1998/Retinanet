import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from net.dataset.data_loader import CocoDataset, collater
from net.dataset.evaluator import evaluate
from net.model.model_factory import create_resnet
from net.utility.constants import MODEL_PATH
from net.utility.utils import Normalizer, Resizer, get_parallel_retinanet_from

if __name__ == '__main__':
    dataset_val = CocoDataset(transforms.Compose([Normalizer(), Resizer()]))
    val_indices = torch.randperm(len(dataset_val))[:250]
    dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater,
                                sampler=SubsetRandomSampler(val_indices))
    retinanet = create_resnet(50, num_classes=dataset_val.num_classes(), pretrained=True)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet.load_state_dict(torch.load(MODEL_PATH))
    retinanet = get_parallel_retinanet_from(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.module.freeze_bn()
    evaluate(dataloader_val, retinanet)