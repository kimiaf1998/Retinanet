import json

import torch
from pycocotools.cocoeval import COCOeval
from net.utility.constants import DATASET_MODEL


def evaluate(dataset, model, threshold=0.05):
    model.eval()

    with torch.no_grad():
        results = []
        image_ids = []

        for iter_num, data in enumerate(dataset):
            scale = data['scale'][0]
            if torch.cuda.is_available():
                scores, labels, boxes = model(data['img'].squeeze().cuda().float().unsqueeze(dim=0))
            else:
                scores, labels, boxes = model(data['img'].squeeze().float().unsqueeze(dim=0))
            scores = scores.cpu()
            labels = labels.cpu()
            boxes = boxes.cpu()
            boxes /= scale
            if boxes.shape[0] > 0:
                # change to standard COCO model (x, y, w, h)
                boxes[:, 2] -= boxes[:, 0]
                boxes[:, 3] -= boxes[:, 1]

                for box_id in range(boxes.shape[0]):
                    score = float(scores[box_id])
                    label = int(labels[box_id])
                    box = boxes[box_id, :]
                    if score < threshold:
                        break

                    image_result = {
                        'image_id': dataset.dataset.image_ids[iter_num],
                        'category_id': dataset.dataset.label_to_coco_label(label),
                        'score': float(score),
                        'bbox': box.tolist(),
                    }
                    results.append(image_result)

            image_ids.append(dataset.dataset.image_ids[iter_num])
            print('{}/{}'.format(iter_num, len(dataset)), end='\r')

        if not len(results):
            return

        json.dump(results, open('{}_bbox_results.json'.format(DATASET_MODEL), 'w'), indent=4)
        coco_true = dataset.dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(DATASET_MODEL))
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        model.train()
        return
