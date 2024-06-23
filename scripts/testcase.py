import os
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# 假设MVTV类定义在mvtv_dataset.py文件中
from datasets.aligned_dataset import LLVIP

def visualize_sample(data, index=0):
    """Visualize a sample from the dataset."""
    infrared = data[0][index].permute(1, 2, 0).numpy()
    visible = data[1][index].permute(1, 2, 0).numpy()

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(infrared)
    axs[0].set_title('Infrared Image')
    axs[0].axis('off')

    axs[1].imshow(visible)
    axs[1].set_title('Visible Image')
    axs[1].axis('off')

    plt.show()

def main():
    # 设置数据集根目录
    dataroot = '/mnt/c/Users/dyf/Downloads/LLVIP'

    # 创建训练和验证数据集
    train_dataset = LLVIP(dataroot, train=True, img_size=128, random_crop=True, random_flip=True)
    val_dataset = LLVIP(dataroot, train=False, img_size=128, random_crop=False, random_flip=False)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=2)

    # 打印数据集大小
    print(f'Training set size: {len(train_dataset)}')
    print(f'Validation set size: {len(val_dataset)}')

    # 取出一个批次的数据
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    # 打印图像大小和路径
    print(f'Infrared image shape: {train_batch[0].shape}')
    print(f'Visible image shape: {train_batch[1].shape}')

    # 可视化训练集的一个样本
    visualize_sample(train_batch)

if __name__ == "__main__":
    main()
