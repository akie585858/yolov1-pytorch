from collections import defaultdict
from utils.predict import get_result_boxes, voc_eval, VOC_CLASSES
import torch

class mAp():
    def __init__(self):
        self.target =  defaultdict(list)
        self.preds = defaultdict(list)

        self.count = 0

    def update(self, y, y_hat):
        batch_size = len(y)
        for b_idx in range(batch_size):
            yi = y[b_idx].cpu()
            y_hati = y_hat[b_idx].cpu()
            
            assert(yi.shape == torch.Size([7, 7, 30]))
            assert(y_hati.shape == torch.Size([7, 7, 30]))

            target_boxes = get_result_boxes(None, yi, 448)
            pred_boxes = get_result_boxes(None, y_hati, 448)

            idx = self.count * batch_size + b_idx
            for box in target_boxes:
                self.target[(idx,box[2])].append([box[0][0],box[0][1],box[1][0],box[1][1]])
            for box in pred_boxes:
                self.preds[box[2]].append([idx,box[3],box[0][0],box[0][1],box[1][0],box[1][1]])

        self.count += 1

    def close(self):
        self.count = 0
        mAp = voc_eval(self.preds, self.target,VOC_CLASSES=VOC_CLASSES)
        self.target =  defaultdict(list)
        self.preds = defaultdict(list)

        return mAp

    def __call__(self, y, y_hat):
        self.update(y, y_hat)