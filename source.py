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
from model_create.my_model import MISAWithGating, SpectrogramModel

# 音频模型
from model_method.TSNA import TSLANet
from model_method.wave_transformer import restv2_tiny

# 谱图模型
from model_method.SimPFs import SimPFs_model
from model_method.wave_VIT import wavevit_s
from model_method.mpvit import mpvit_tiny

from config import get_args_parser
from util.loss_function import CMD, DiffLoss, DistillationLoss

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

# Loss functions
loss_diff = DiffLoss()
loss_cmd = CMD()
distillation_loss = DistillationLoss()

def get_cmd_loss(model):
    """Compute the CMD loss between shared states."""
    loss = loss_cmd(model.utt_shared_a, model.utt_shared_v, 5)
    return loss

def get_diff_loss(model):
    """Compute the Diff loss between private and shared states."""
    shared_a = model.utt_shared_a
    shared_v = model.utt_shared_v
    private_v = model.utt_private_v
    private_a = model.utt_private_a

    # Between private and shared
    loss = loss_diff(private_a, shared_a)
    loss += loss_diff(private_v, shared_v)
    # Across privates
    loss += loss_diff(private_a, private_v)

    return loss

def train(model, optimizer, loss_fn, train_loader, n_epoch, train_path, teacher_model):
    """Train the model."""
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    batch_num = len(train_loader)
    train_batch_num = round(batch_num * 0.75)
    print(f"Total batches: {batch_num}, Training batches: {train_batch_num}")

    train_loss_all, train_acc_all = [], []
    val_loss_all, val_acc_all = [], []

    for epoch in range(1, n_epoch + 1):
        # exp_lr_scheduler.step()

        running_loss, running_correct = 0, 0
        train_num, val_loss, val_corrects, val_num = 0, 0, 0, 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels.numpy()).to(device)

            if i < train_batch_num:
                model.train()
                output = model(inputs).to(device)

                cls_loss = loss_fn(output, labels)
                diff_loss = get_diff_loss(model)
                cmd_loss = get_cmd_loss(model)
                teacher_output = teacher_model(inputs).to(device)
                dist_loss = distillation_loss(output, teacher_output, labels)
                loss = cls_loss + 0.2 * diff_loss + 0.2 * cmd_loss + 0.2 * dist_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                _, pred = torch.max(output.data, 1)
                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(pred == labels)
                train_num += inputs.size(0)
            else:
                model.eval()
                output = model(inputs).to(device)

                cls_loss = loss_fn(output, labels)
                diff_loss = get_diff_loss(model)
                cmd_loss = get_cmd_loss(model)
                teacher_output = teacher_model(inputs)
                dist_loss = distillation_loss(output, teacher_output, labels)
                loss = cls_loss + 0.2 * diff_loss + 0.2 * cmd_loss + 0.2 * dist_loss

                _, pred = torch.max(output.data, 1)
                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(pred == labels)
                val_num += inputs.size(0)

        train_loss = running_loss / train_num
        train_acc = 100.0 * running_correct.double().item() / train_num
        val_loss = val_loss / val_num
        val_acc = 100.0 * val_corrects.double().item() / val_num

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

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]
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

def main_fold(model, train_loader, train_path, args):
    """Main function for training the model."""
    print('------------------------- Train Start --------------------------------')
    teacher_model = SpectrogramModel(
        sample_rate=args.sample_rate, window_size=args.window_size,
        hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
        classes_num=args.num_classes
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)
    # exp_lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40], gamma=0.1)

    model, train_process = train(model, optimizer, loss_fn, train_loader, args.n_epoch, train_path, teacher_model)
    return model, train_process

def return_model(args):
    """Return the model based on the arguments."""
    if args.model_name == 'fusion':
        model = MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    
    elif args.model_name == 'TSNA':
        model = TSLANet(num_classes=args.num_classes).to(device)
    
    elif args.model_name == 'WTS':
        model = restv2_tiny(num_classes=args.num_classes).to(device)
    
    #谱图模型
    elif args.model_name == 'SimPFs':
        model = SimPFs_model(num_classes=args.num_classes).to(device)
        
    elif args.model_name == 'wavevit':
        model = wavevit_s().to(device)
    elif args.model_name == 'mpvit':
        model = mpvit_tiny().to(device)
        
    return model

def return_data(args):
    """Return the dataset path based on the arguments."""
    if args.no_noise:
        if args.data_type == 'ESC_10':
            return r'ori_dataSet/Cut_ESC_10'
        elif args.data_type == 'Cut_ShipEar':
            return r"ori_dataSet/Cut_ShipEar"
        elif args.data_type == 'Cut_deepShip':
            return r"ori_dataSet/Cut_deepShip"
        elif args.data_type == 'ESC_50':
            return r"ori_dataSet/Cut_ESC_50"
        else:
            return None
    else:
        if args.data_type == 'ESC_10':
            return f'dataSet/{args.noise_path}/Cut_ESC_10'
        if args.data_type == 'ESC_50':
            return f'dataSet/{args.noise_path}/Cut_ESC_50'
        elif args.data_type == 'Cut_ShipEar':
            return f"dataSet/{args.noise_path}/Cut_ShipEar"
        elif args.data_type == 'Cut_deepShip':
            return f"dataSet/{args.noise_path}/Cut_deepShip"
        else:
            return None

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
    train_loader, test_loader = get_data_loaders(data_set, args.batch_size, train_ratio=0.8, random_seed=random_seed, num_workers=8)
    print(f"----------------------------- Load data: {args.data_type} -----------------------------")

    # Initialize model
    model = return_model(args)
    os.makedirs(f'{args.noise_level}/{args.data_type}/{args.model_name}', exist_ok=True)
    train_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/loss_accuracy.txt"

    # Train the model
    print(f"Train: {len(train_loader.dataset)} samples")
    model_ft, train_process = main_fold(model, train_loader, train_path, args)

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