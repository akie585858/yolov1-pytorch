import torch
import matplotlib.pyplot as plt


if __name__ == '__main__':
    data = torch.load('result/ResNet50_yolov1/train_result.pt')
    test_data = torch.load('result/ResNet50_yolov1/test_result.pt')

    mAp = [d['mAp50'] for d in test_data]
    print(max(mAp))

    show_attr = 'loss'
    test_rate = 5
    loss = [d[show_attr] for d in data]
    test_loss = [d[show_attr] for d in test_data]
    test_x = torch.arange(1, len(loss)+1, test_rate)
    
    plt.title(f'{show_attr} curve')
    plt.plot(loss, 'o-', linewidth=2, label='train_loss')
    plt.plot(test_x.tolist(), test_loss, 'o-', linewidth=2, label='test_loss')
    plt.plot(test_x.tolist(), mAp, '--', linewidth=2, label='mAp50')
    plt.legend()
    plt.grid()
    plt.show()
