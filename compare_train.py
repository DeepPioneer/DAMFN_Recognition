import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import time
import numpy as np
import argparse
from random import random
import torch.optim as optim
import copy
import pandas as pd
from util.data_loader import get_data_loaders
from torch.utils.data import random_split

# 音频模型
from model_method.TSNA import TSLANet
from model_method.wave_transformer import restv2_tiny
from model_method.Resnet1D import Res1dNet18

# 谱图模型
from model_method.SimPFs import SimPFs_model
from model_method.wave_VIT import wavevit_s
from model_method.mpvit import mpvit_tiny
from model_method.Resnet2D import Res2dNet18
from model_method.VIT import vit_base_patch16_224
from model_method.ConvNet_model import ConvNeXt

# 音频+谱图模型
from model_method.joint_model import Wavegram_Logmel

# 消融实验
from ablation_model.abation_design import MISAWithGating

from config import get_args_parser

# Set random seeds for reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(model, optimizer, loss_fn, train_loader, val_loader, n_epoch, train_path):
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all, train_acc_all = [], []
    val_loss_all, val_acc_all = [], []

    for epoch in range(1, n_epoch + 1):
        model.train()
        running_loss, running_correct, train_num = 0, 0, 0

        # 训练阶段
        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels.numpy()).to(device)

            output = model(inputs).to(device)
            
            loss = loss_fn(output, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, pred = torch.max(output.data, 1)
            running_loss += loss.item() * inputs.size(0)
            running_correct += torch.sum(pred == labels)
            train_num += inputs.size(0)

        train_loss = running_loss / train_num
        train_acc = 100.0 * running_correct.double().item() / train_num

        # 验证阶段
        model.eval()
        val_loss, val_corrects, val_num = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
                labels = torch.LongTensor(labels.numpy()).to(device)

                output = model(inputs).to(device)
                
                loss = loss_fn(output, labels)

                _, pred = torch.max(output.data, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(pred == labels)
                val_num += inputs.size(0)

        val_loss = val_loss / val_num
        val_acc = 100.0 * val_corrects.double().item() / val_num

        # 记录日志
        elapse = time.time() - start
        log_message = (
            f'Epoch: {epoch}/{n_epoch} lr: {optimizer.param_groups[0]["lr"]:.4g} '
            f'samples: {len(train_loader.dataset)} TrainLoss: {train_loss:.3f} TrainAcc: {train_acc:.2f}% '
            f'ValLoss: {val_loss:.3f} ValAcc: {val_acc:.2f}%'
        )
        print(log_message)

        with open(train_path, "a") as file:
            file.write(log_message + "\n")

        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}'
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f'{save_path}/{args.data_type}_{args.model_name}.pkl')

    with open(train_path, "a") as file:
        file.write(f"Model train time: {elapse:.1f}s\n")

    model.load_state_dict(best_model_wts)
    train_process = pd.DataFrame({
        "epoch": range(n_epoch),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })
    return model, train_process

def main_fold(model, train_loader, val_loader,train_path, args):
    """Main function for training the model."""
    print('------------------------- Train Start --------------------------------')

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    # exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)
    model, train_process = train(model, optimizer, loss_fn, train_loader, val_loader, args.n_epoch, train_path)
    return model, train_process

def return_model(args):
    """Return the model based on the arguments."""
    # 音频模型
    if args.model_name == 'TSNA':
        model = TSLANet(num_classes=args.num_classes).to(device)
    
    elif args.model_name == 'WTS':
        model = restv2_tiny(num_classes=args.num_classes).to(device)
    
    elif args.model_name == 'Resnet1D':
        model = Res1dNet18(num_classes=args.num_classes).to(device)
    
    #谱图模型
    elif args.model_name == 'SimPFs':
        model = SimPFs_model(num_classes=args.num_classes).to(device)
        
    elif args.model_name == 'wavevit':
        model = wavevit_s().to(device)
    elif args.model_name == 'mpvit':
        model = mpvit_tiny().to(device)
    elif args.model_name == 'Resnet2D':
        model = Res2dNet18(sample_rate=args.sample_rate, window_size=args.window_size,
                                hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                                num_classes=args.num_classes).to(device)
    elif args.model_name == 'VIT':
        model = vit_base_patch16_224(num_classes=args.num_classes).to(device)
    elif args.model_name == 'convnet':
        model = ConvNeXt(num_classes=args.num_classes).to(device)
    
    # 时域+谱图
    elif args.model_name == 'joint':
        model = Wavegram_Logmel(sample_rate=args.sample_rate, window_size=args.window_size,
                                hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
                                num_classes=args.num_classes).to(device)
        
    # 消融实验
    elif args.model_name == 'onlya':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    
    elif args.model_name == 'onlyv':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)

    elif args.model_name == 'private':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    
    elif args.model_name == 'shared':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    
    elif args.model_name == 'wofusion':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
        
    elif args.model_name == 'woTrans':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
        
    elif args.model_name == 'woMoE':
        model =  MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
        
    return model

def return_data(args):
    """Return the dataset path based on the arguments."""
    if args.no_noise:
        return f'ori_dataSet/{args.data_type}'
    return f'noise_dataSet/{args.noise_path}/{args.data_type}'

if __name__ == "__main__":
    # Set random seeds for reproducibility
    random_name = str(random())
    random_seed = 123
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    # Parse arguments
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()

    # Load dataset
    data_set = return_data(args)
    train_loader, test_loader,val_loader = get_data_loaders(data_set, args.batch_size, train_ratio=0.8, random_seed=random_seed, num_workers=8)
    print(f"----------------------------- Load data: {args.data_type} -----------------------------")

    # Initialize model
    model = return_model(args)
    os.makedirs(f'{args.noise_level}/{args.data_type}/{args.model_name}', exist_ok=True)
    train_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/loss_accuracy.txt"

    # Train the model
    print(f"Train: {len(train_loader.dataset)} samples")
    model_ft, train_process = main_fold(model, train_loader, val_loader,train_path, args)

    # Visualize training process
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss", markersize=5)
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss", markersize=5)
    plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc", markersize=5)
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc", markersize=5)
    plt.xlabel("epoch")
    plt.ylabel("acc")
    plt.legend()

    # Save training results
    PATH_fig = os.path.join(f"{args.noise_level}/{args.data_type}/{args.model_name}" + '.pdf')
    plt.savefig(PATH_fig)