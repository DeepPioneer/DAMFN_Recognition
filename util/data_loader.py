# import numpy as np
# import torch
# import glob,os
# from sklearn.model_selection import KFold
# from torch.utils.data import Dataset, DataLoader, Subset
# from collections import Counter

# class SoundDataset(Dataset):
#     def __init__(self, preprocessed_path):
#         self.audio_paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
#         # print(self.audio_paths)
#         self.labels = [int(os.path.basename(x).split('_')[-1].replace('.npz', '')) for x in self.audio_paths]

#     def __getitem__(self, index):
#         npz_path = self.audio_paths[index]
#         data = np.load(npz_path)
#         waveform = data['waveform']
#         label = data['label']
#         soundData = torch.tensor(waveform)
#         # 提取文件名（没有扩展名）
#         # file_name = os.path.splitext(os.path.basename(npz_path))[0]
#         # return soundData, label,file_name
#         return soundData, label

#     def __len__(self):
#         return len(self.audio_paths)

# def split_dataset(dataset, train_ratio=0.8,random_seed=123):
#     np.random.seed(random_seed)  # 设置随机种子
#     total_size = len(dataset)
#     indices = list(range(total_size))
#     np.random.shuffle(indices)

#     train_split = int(np.floor(train_ratio * total_size))

#     train_indices = indices[:train_split]

#     test_indices = indices[train_split:]

#     train_set = Subset(dataset, train_indices)
#     test_set = Subset(dataset, test_indices)

#     return train_set, test_set

# def get_data_loaders(preprocessed_path, batch_size, train_ratio=0.8, random_seed=123, num_workers=4):
#     dataset = SoundDataset(preprocessed_path)
#     train_set, test_set = split_dataset(dataset, train_ratio, random_seed)
    
#     train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
#     test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
#     return train_loader, test_loader

from torch.utils.data import Dataset, Subset, DataLoader
import numpy as np
import os
import glob
import torch

class SoundDataset(Dataset):
    def __init__(self, preprocessed_path):
        self.audio_paths = glob.glob(os.path.join(preprocessed_path, '*.npz'))
        self.labels = [int(os.path.basename(x).split('_')[-1].replace('.npz', '')) for x in self.audio_paths]

    def __getitem__(self, index):
        npz_path = self.audio_paths[index]
        data = np.load(npz_path)
        waveform = data['waveform']
        label = data['label']
        soundData = torch.tensor(waveform)
        return soundData, label

    def __len__(self):
        return len(self.audio_paths)

def split_dataset(dataset, train_ratio=0.8, random_seed=123):
    np.random.seed(random_seed)  # 设置随机种子
    total_size = len(dataset)
    indices = list(range(total_size))
    np.random.shuffle(indices)

    train_split = int(np.floor(train_ratio * total_size))
    train_indices = indices[:train_split]
    test_indices = indices[train_split:]

    train_set = Subset(dataset, train_indices)
    test_set = Subset(dataset, test_indices)

    return train_set, test_set

def get_data_loaders(preprocessed_path, batch_size, train_ratio=0.8, random_seed=123, num_workers=4):
    # 加载数据集
    dataset = SoundDataset(preprocessed_path)
    # 第一次划分：训练+验证集（80%）和测试集（20%）
    train_val_set, test_set = split_dataset(dataset, train_ratio, random_seed)
    # 第二次划分：训练集（60%）和验证集（20%）
    train_set, val_set = split_dataset(train_val_set, train_ratio=0.75)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, test_loader, val_loader
