import numpy as np

VOC_CLASSES = (    # always index 0
    'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair',
    'cow', 'diningtable', 'dog', 'horse',
    'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor')
Color = [[0, 0, 0],
                    [128, 0, 0],
                    [0, 128, 0],
                    [128, 128, 0],
                    [0, 0, 128],
                    [128, 0, 128],
                    [0, 128, 128],
                    [128, 128, 128],
                    [64, 0, 0],
                    [192, 0, 0],
                    [64, 128, 0],
                    [192, 128, 0],
                    [64, 0, 128],
                    [192, 0, 128],
                    [64, 128, 128],
                    [192, 128, 128],
                    [0, 64, 0],
                    [128, 64, 0],
                    [0, 192, 0],
                    [128, 192, 0],
                    [0, 64, 128]]

def voc_ap(rec,prec,use_07_metric=False):
    if use_07_metric:
        # 11 point metric
        ap = 0.
        for t in np.arange(0.,1.1,0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec>=t])
            ap = ap + p/11.

    else:
        # correct ap caculation
        mrec = np.concatenate(([0.],rec,[1.]))
        mpre = np.concatenate(([0.],prec,[0.]))

        for i in range(mpre.size -1, 0, -1):
            mpre[i-1] = np.maximum(mpre[i-1],mpre[i])

        i = np.where(mrec[1:] != mrec[:-1])[0]

        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

    return ap

def voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES,threshold=0.5,use_07_metric=False,):
    '''
    preds {'cat':[[image_id,confidence,x1,y1,x2,y2],...],'dog':[[],...]}
    target {(image_id,class):[[],]}
    '''
    aps = []
    for _,class_ in enumerate(VOC_CLASSES):
        # 获取该类别的预测框
        pred = preds[class_] #[[image_id,confidence,x1,y1,x2,y2],...]
        if len(pred) == 0: #如果这个类别一个都没有检测到的异常情况
            ap = -1
            print('---class {} ap {}---'.format(class_,ap))
            aps += [ap]
            break
        
        # 获取预测框所在的图片id列表
        image_ids = [x[0] for x in pred]

        # 按置信度排序
        confidence = np.array([float(x[1]) for x in pred])
        BB = np.array([x[2:] for x in pred])
        sorted_ind = np.argsort(-confidence)
        BB = BB[sorted_ind, :]  #坐标信息
        image_ids = [image_ids[x] for x in sorted_ind] #图片id

        #统计这个类别的正样本
        npos = 0.
        for (key1,key2) in target:
            if key2 == class_:
                npos += len(target[(key1,key2)])
        nd = len(image_ids) #预测positve个数

        tp = np.zeros(nd)
        fp = np.zeros(nd)
        for d,image_id in enumerate(image_ids):
            bb = BB[d] #预测框
            if (image_id,class_) in target:
                BBGT = target[(image_id,class_)]

                for bbgt in BBGT:
                    # 计算iou
                    ixmin = np.maximum(bbgt[0], bb[0])
                    iymin = np.maximum(bbgt[1], bb[1])
                    ixmax = np.minimum(bbgt[2], bb[2])
                    iymax = np.minimum(bbgt[3], bb[3])
                    iw = np.maximum(ixmax - ixmin + 1., 0.)
                    ih = np.maximum(iymax - iymin + 1., 0.)
                    inters = iw * ih

                    union = (bb[2]-bb[0]+1.)*(bb[3]-bb[1]+1.) + (bbgt[2]-bbgt[0]+1.)*(bbgt[3]-bbgt[1]+1.) - inters
                    if union == 0:
                        print(bb,bbgt)
                    
                    overlaps = inters/union #iou
                    # 根据iou阈值判断预测框是否正确匹配
                    if overlaps > threshold:
                        tp[d] = 1
                        BBGT.remove(bbgt) #这个框已经匹配到了，不能再匹配
                        if len(BBGT) == 0:
                            del target[(image_id,class_)] #删除没有box的键值
                        break
                fp[d] = 1-tp[d]
            else:
                fp[d] = 1 # 该图像不存在该类别或已被高置信度框匹配完了
        fp = np.cumsum(fp)
        tp = np.cumsum(tp)
        rec = tp/float(npos)
        prec = tp/np.maximum(tp + fp, np.finfo(np.float64).eps) #获取float64数型信息后获取该类型最小非负值
        
        # 计算ap
        ap = voc_ap(rec, prec, use_07_metric)
        print('---class {} ap {}---'.format(class_,ap))
        aps += [ap]
    print('---map {}---'.format(np.mean(aps)))

if __name__ == '__main__':
    from utils.predict import *
    from collections import defaultdict
    import tqdm
    from dataset.voc2012 import yoloDataset
    from model.Resnet import resnet34

    target =  defaultdict(list)
    preds = defaultdict(list)

    # 加载模型
    model = resnet34()
    model.load_state_dict(torch.load('result/ResNet34_yolov1/last.pt'))
    model = model.eval()
    model = model.cuda()

    dataset = yoloDataset(False, 448)
    with tqdm.trange(len(dataset)) as tbar:
        for idx, (img, y) in enumerate(dataset):
            img = img.cuda()
            pred = model(img.unsqueeze(0))
            pred = pred.squeeze()

            target_boxes = get_result_boxes(img, y.cpu(), 448)
            pred_boxes = get_result_boxes(img, pred.cpu(), 448)
            
            for box in target_boxes:
                target[(idx,box[2])].append([box[0][0],box[0][1],box[1][0],box[1][1]])
            for box in pred_boxes:
                preds[box[2]].append([idx,box[3],box[0][0],box[0][1],box[1][0],box[1][1]])
            tbar.update()
    print('---start evaluate---')
    voc_eval(preds,target,VOC_CLASSES=VOC_CLASSES)
    