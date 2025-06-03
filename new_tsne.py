import torch
import matplotlib.pyplot as plt
import os,argparse
import numpy as np
from util.data_loader import get_data_loaders
from sklearn.manifold import TSNE
from random import random

from model_create.hope import MISAWithGating

from config import get_args_parser

import warnings

warnings.filterwarnings("ignore")

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 初始网络模型
def return_model(args):
    """Return the model based on the arguments."""
    if args.model_name == 'hope': #fusion
        model = MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    return model


# 初始化数据集
def return_data(args):
    """Return the dataset path based on the arguments."""
    if args.no_noise:
        return f'ori_dataSet/{args.data_type}'
    return f'noise_dataSet/{args.noise_path}/{args.data_type}'

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()
    
    random_name = str(random())
    random_seed = 123
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    
    # -------------------------加载模型------------------------#
    model = return_model(args)
    model.load_state_dict(torch.load(
        f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}/{args.data_type}_{args.model_name}.pkl'))

    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()

    data_set = return_data(args)

    train_loader, test_loader,val_loader = get_data_loaders(data_set, args.batch_size, train_ratio=0.8,  random_seed=random_seed, num_workers=8)
    # 假设 model 是你训练好的模型，test_data_loader 是待测试的数据集的 DataLoader

    model.eval()  # 切换到评估模式
    
    if args.data_type == "Cut_ESC_10":
        CLASS_MAPPING = ['chainsaw', 'clock_tick', 'crackling_fire', 'crying_baby', 'dog', 'helicopter', 'rain', 'rooster', 'sea_waves', 'sneezing']
    elif args.data_type == "Cut_deepShip":
        CLASS_MAPPING = ['Cargo', 'Passengership', 'Tanker', 'Tug']
    elif args.data_type == "Cut_ShipEar":
        CLASS_MAPPING = ["0", "1", "2", "3", "4"]
    elif args.data_type == "Cut_ESC_50":
        CLASS_MAPPING = ["airplane", "breathing", "brushing_teeth", "can_opening", "car_horn", "cat", 'chainsaw', 'chirping_birds', 'church_bells', 'clapping',
                         'clock_alarm', 'clock_tick', 'coughing', 'cow', 'crackling_fire', 'crickets', 'crow', 'crying_baby', 'dog', 'door_wood_creaks',
                         'door_wood_knock', 'drinking_sipping', 'engine', 'fireworks', 'footsteps', 'frog', 'glass_breaking', 'hand_saw', 'helicopter', 'hen',
                         'insects', 'keyboard_typing', 'laughing', 'mouse_click', 'pig', 'pouring_water', 'rain', 'rooster', 'sea_waves', 'sheep',
                         'siren', 'sneezing', 'snoring', 'thunderstorm', 'toilet_flush', 'train', 'vacuum_cleaner', 'washing_machine', 'water_drops', 'wind']
    elif args.data_type == "Cut_whale":
        CLASS_MAPPING = ['BowheadWhale', 'FalseKillerWhale', 'NarWhale', 'WhiteWhale']
    else:
        CLASS_MAPPING = None

    features = []
    labels = []

    with torch.no_grad():  # 不需要梯度计算
        for inputs, label in test_loader:
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            # labels = torch.LongTensor(labels.numpy()).to(device)
            output = model(inputs)  # 获取模型的输出或特征
            features.append(output.cpu().numpy())  # 将特征从 GPU 移到 CPU 并转为 NumPy
            labels.append(label.cpu().numpy())

    features = np.concatenate(features)  # 合并所有批次的特征
    labels = np.concatenate(labels)  # 合并所有批次的标签

    # 使用 t-SNE 进行降维
    tsne = TSNE(n_components=2, random_state=42)  # n_components=2 表示降维到2D
    reduced_features = tsne.fit_transform(features)
    
    if args.data_type == "ESC_50":
        # 假设我们选择展示的类别索引（比如0, 1, 2,... 9）
        selected_classes = ['water_drops', 'rooster', 'cow', 'dog', 'crying_baby']
        # 转换成标签索引（假设 CLASS_MAPPING 已定义）
        selected_classes = [CLASS_MAPPING.index(c) for c in selected_classes]

        # 筛选出要显示的特征和标签
        selected_features = []
        selected_labels = []

        for i in range(len(labels)):
            if labels[i] in selected_classes:
                selected_features.append(reduced_features[i])
                selected_labels.append(labels[i])

        selected_features = np.array(selected_features)
        selected_labels = np.array(selected_labels)

        # 绘制 t-SNE 可视化
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(selected_features[:, 0], selected_features[:, 1], c=selected_labels,
                              cmap='viridis')
        plt.colorbar(scatter)

        # 为每个类别添加标签
        label_centers = []
        for label in selected_classes:
            indices = np.where(selected_labels == label)[0]
            center = np.mean(selected_features[indices], axis=0)
            label_centers.append(center)

        # 添加类别中心标签
        for label, center in zip(selected_classes, label_centers):
            plt.text(center[0], center[1], str(label), fontsize=16, color='red', ha='center', va='center')

        plt.title(f't-SNE Visualization of 10 Selected Classes under {args.noise_level}',fontsize=22)
        plt.xlabel('t-SNE 1',fontsize=20)
        plt.ylabel('t-SNE 2',fontsize=20)
        plt.tight_layout()
        plt.savefig(f'{args.noise_level}/{args.data_type}/{args.model_name}/tsne_visualize_5_classes.pdf', dpi=300)
        plt.tight_layout()
        plt.close()

    else:
        # 计算每个类别的中心点
        unique_labels = np.unique(labels)
        label_centers = []
        for label in unique_labels:
            # 找到当前类别的所有点
            indices = np.where(labels == label)[0]
            # 计算当前类别的中心点
            center = np.mean(reduced_features[indices], axis=0)
            label_centers.append(center)

        # 绘制 t-SNE 可视化
        plt.figure(figsize=(10, 6))
        # scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis', s=10)
        scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=labels, cmap='viridis')
        plt.colorbar(scatter)

        # 在每个类别的中心点添加文本标签
        for label, center in zip(unique_labels, label_centers):
            plt.text(center[0], center[1], str(label), fontsize=16, color='red', ha='center', va='center')

        plt.title(f't-SNE Visualization of Test Data Features under {args.noise_level} dB',fontsize=22)
        plt.xlabel('t-SNE 1',fontsize=20)
        plt.ylabel('t-SNE 2',fontsize=20)
        # plt.show()
        plt.savefig(f'{args.noise_level}/{args.data_type}/{args.model_name}/tsne_visualize.pdf')
    