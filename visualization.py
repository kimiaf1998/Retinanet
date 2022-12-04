import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from net.dataset.data_loader import CocoDataset, collater, AspectRatioBasedSampler
from net.model.model_factory import create_resnet
from net.utility.constants import MODEL_PATH
from net.utility.utils import Normalizer, Resizer, Denormalizer, get_parallel_retinanet_from, \
	get_coordinates_from_bounding_box, draw_caption, draw_bounding_box

if __name__ == '__main__':

	dataset_val = CocoDataset(transforms.Compose([Normalizer(), Resizer()]))
	sampler_val = AspectRatioBasedSampler(dataset_val, 1)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)
	retinanet = torch.load(MODEL_PATH)

	if torch.cuda.is_available():
		retinanet = create_resnet(50, num_classes=dataset_val.num_classes(), )
		retinanet.load_state_dict(torch.load(MODEL_PATH))
		retinanet = retinanet.to('cuda')

	retinanet = get_parallel_retinanet_from(retinanet)

	retinanet.eval()
	denormalize = Denormalizer()

	for idx, item in enumerate(dataloader_val):
		with torch.no_grad():
			st = time.time()
			scores, classification, transformed_anchors = retinanet(
				item['img'].cuda().float()) if torch.cuda.is_available() else retinanet(item['img'].float())
			print('Elapsed time: {}'.format(time.time() - st))
			idxs = np.where(scores.cpu() > 0.5)

			img = np.array(255 * denormalize(item['img'][0, :, :, :])).copy()
			img[img < 0] = 0
			img[img > 255] = 255
			img = np.transpose(img, (1, 2, 0))
			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1, y1, x2, y2 = get_coordinates_from_bounding_box(bbox)

				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)
				draw_bounding_box(img, (x1, y1), (x2, y2))
				print('Detected object: ', label_name)

			cv2.imshow('img', img)
			cv2.waitKey(0)
