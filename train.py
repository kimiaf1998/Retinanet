import collections


import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision import transforms

from net.dataset.data_loader import CocoDataset, collater, AspectRatioBasedSampler
from net.dataset.evaluator import evaluate
from net.model.model_factory import create_resnet
from net.utility.constants import RESNET_DEP, TRAIN_EPOCH, VALIDATION_SAMPLES_QTY, TRAIN_SAMPLES_QTY
from net.utility.utils import Normalizer, Augmenter, Resizer, get_parallel_retinanet_from, log_epoch

if __name__ == '__main__':

    if RESNET_DEP != 50 and RESNET_DEP != 101:
        print("The depth of ResNet is wrong!")
        exit(0)

    dataset_train = CocoDataset(transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
    dataset_val = CocoDataset(transforms.Compose([Normalizer(), Resizer()]))
    train_indices = torch.randperm(len(dataset_train))[:TRAIN_SAMPLES_QTY]
    val_indices = torch.randperm(len(dataset_val))[:VALIDATION_SAMPLES_QTY]
    sampler = AspectRatioBasedSampler(dataset_train, 2)
    dataloader_train = DataLoader(dataset_train, num_workers=3, collate_fn=collater,
                                  sampler=SubsetRandomSampler(train_indices))

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, 1)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater,
                                    sampler=SubsetRandomSampler(val_indices))

    retinanet = create_resnet(RESNET_DEP, num_classes=dataset_train.num_classes(), pretrained=True)

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()

    retinanet = get_parallel_retinanet_from(retinanet)

    retinanet.training = True
    optimizer = optim.Adam(retinanet.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)
    loss_hist = collections.deque(maxlen=500)
    retinanet.train()
    retinanet.module.freeze_bn()

    print('# training images: ', len(dataloader_train))

    for epoch in range(TRAIN_EPOCH):

        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for iter_num, data in enumerate(dataloader_train):
            try:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)
                optimizer.step()
                loss_hist.append(float(loss))
                epoch_loss.append(float(loss))
                log_epoch(epoch + 1, iter_num, float(classification_loss), float(regression_loss), np.mean(loss_hist))

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue

        scheduler.step(np.mean(epoch_loss))
        torch.save(retinanet.module.state_dict(), 'retinanet_model_epoch_{}.pt'.format(epoch))

    retinanet.eval()
    torch.save(retinanet.module.state_dict(), 'retinanet_model.pt')
    evaluate(dataloader_val, retinanet)
