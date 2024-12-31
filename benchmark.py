import os
from thop import profile, clever_format
import torch

def yolov1_mark(mark_path:str, net):
    temp_path = os.path.join(mark_path, 'last.tmp')
    with open(temp_path, 'r') as f:
        _ = int(f.readline())
        best_val = float(f.readline())

    test_tensor = torch.rand((1, 3, 448, 448))
    map50 = best_val
    flops, params = clever_format(profile(net, (test_tensor,)), '%.3f')
    name = mark_path.split('/')[-1]

    with open('benchmark.txt', 'a') as f:
        f.write(f'\n{name}: mAp50:{map50:.3f} flops:{flops} params size:{params}')

    

if __name__ == '__main__':
    from model.Resnet import resnet50
    mark_path = 'result/ResNet50_yolov1'
    yolov1_mark(mark_path, resnet50())

