import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import copy
import pandas as pd
import argparse
from random import random
from util.data_loader import get_data_loaders
from model_create.hope import MISAWithGating
from model_method.TSNA import TSLANet
from model_method.wave_transformer import restv2_tiny
from model_method.SimPFs import SimPFs_model
from model_method.wave_VIT import wavevit_s
from model_method.mpvit import mpvit_tiny
from config import get_args_parser
from util.loss_function import CMD, DiffLoss

# Set random seeds for reproducibility
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
np.random.seed(123)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore")

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Loss functions
loss_diff = DiffLoss()
loss_cmd = CMD()

class DynamicLossWeights(nn.Module):
    """Class for dynamically adjusting loss weights based on uncertainty."""
    def __init__(self, num_losses=3):
        super(DynamicLossWeights, self).__init__()
        self.log_vars = nn.Parameter(torch.zeros(num_losses))  # Log variance for each loss term

    def forward(self, losses):
        """Apply dynamic weighting to each loss."""
        weighted_losses = []
        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])  # Precision (inverse of variance)
            weighted_loss = precision * loss + self.log_vars[i]  # Weighted loss with bias term
            weighted_losses.append(weighted_loss)
        return sum(weighted_losses)  # Sum of all weighted losses

def get_cmd_loss(model, loss_cmd):
    """Compute the CMD loss between shared states."""
    return loss_cmd(model.utt_shared_a, model.utt_shared_v, 5)

def get_diff_loss(model, loss_diff):
    """Compute the Diff loss between private and shared states."""
    shared_a, shared_v = model.utt_shared_a, model.utt_shared_v
    private_v, private_a = model.utt_private_v, model.utt_private_a

    loss = loss_diff(private_a, shared_a) + loss_diff(private_v, shared_v)
    loss += loss_diff(private_a, private_v)
    return loss

def train(args,model, optimizer, loss_fn, train_loader, val_loader, n_epoch, train_path, loss_weights):
    """Training loop with dynamic loss weighting."""
    start = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss_all, train_acc_all = [], []
    val_loss_all, val_acc_all = [], []

    for epoch in range(1, n_epoch + 1):
        model.train()
        running_loss, running_correct, train_num = 0, 0, 0

        # Training phase
        for inputs, labels in train_loader:
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels.numpy()).to(device)

            # output = model(inputs)
                # Forward pass with distillation
            output = model(inputs)
            cls_loss = loss_fn(output, labels)
            diff_loss = get_diff_loss(model, loss_diff)
            cmd_loss = get_cmd_loss(model, loss_cmd)
            # print(f"[DEBUG] cls_loss: {cls_loss.item()}, diff_loss: {diff_loss.item()}, cmd_loss: {cmd_loss.item()}")
            # Apply dynamic loss weights
            losses = [cls_loss, diff_loss, cmd_loss]
            total_loss = loss_weights(losses)
            # total_loss = cls_loss + 0.2 * diff_loss + 0.2 * cmd_loss 

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            _, pred = torch.max(output.data, 1)
            running_loss += total_loss.item() * inputs.size(0)
            running_correct += torch.sum(pred == labels)
            train_num += inputs.size(0)

        train_loss = running_loss / train_num
        train_acc = 100.0 * running_correct.double().item() / train_num

        # Validation phase
        model.eval()
        val_loss, val_corrects, val_num = 0, 0, 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
                labels = torch.LongTensor(labels.numpy()).to(device)

                # output = model(inputs)
                output = model(inputs)
                cls_loss = loss_fn(output, labels)
                diff_loss = get_diff_loss(model, loss_diff)
                cmd_loss = get_cmd_loss(model, loss_cmd)
                
                losses = [cls_loss, diff_loss, cmd_loss]
                total_loss = loss_weights(losses)
                # total_loss = cls_loss + 0.2 * diff_loss + 0.2 * cmd_loss

                _, pred = torch.max(output.data, 1)
                val_loss += total_loss.item() * inputs.size(0)
                val_corrects += torch.sum(pred == labels)
                val_num += inputs.size(0)

        val_loss = val_loss / val_num
        val_acc = 100.0 * val_corrects.double().item() / val_num

        # Log results
        log_message = f'Epoch: {epoch}/{n_epoch} TrainLoss: {train_loss:.3f} TrainAcc: {train_acc:.2f}% ValLoss: {val_loss:.3f} ValAcc: {val_acc:.2f}%'
        print(log_message)

        with open(train_path, "a") as file:
            file.write(log_message + "\n")

        train_loss_all.append(train_loss)
        train_acc_all.append(train_acc)
        val_loss_all.append(val_loss)
        val_acc_all.append(val_acc)

        # Save the best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            save_path = f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}'
            os.makedirs(save_path, exist_ok=True)
            torch.save(model.state_dict(), f'{save_path}/{args.data_type}_{args.model_name}.pkl')
    # scheduler.step()  # <-- 加在这里
    model.load_state_dict(best_model_wts)
    elapse = time.time() - start
    with open(train_path, "a") as file:
        file.write(f"Model train time: {elapse:.1f}s\n")

    # Create DataFrame for training process
    train_process = pd.DataFrame({
        "epoch": range(n_epoch),
        "train_loss_all": train_loss_all,
        "val_loss_all": val_loss_all,
        "train_acc_all": train_acc_all,
        "val_acc_all": val_acc_all
    })
    return model, train_process

def return_model(args):
    """Return the model based on the arguments."""
    if args.model_name == 'fusion':
        return MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    # Add more model options if needed
    return None

def return_data(args):
    """Return the dataset path based on the arguments."""
    if args.no_noise:
        return f'ori_dataSet/{args.data_type}'
    return f'noise_dataSet/{args.noise_path}/{args.data_type}'

def visualize_training_process(train_process, args):
    """Visualize training and validation loss/accuracy."""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_process.epoch, train_process.train_loss_all, "ro-", label="Train loss", markersize=5)
    plt.plot(train_process.epoch, train_process.val_loss_all, "bs-", label="Val loss", markersize=5)
    plt.xlabel("epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_process.epoch, train_process.train_acc_all, "ro-", label="Train acc", markersize=5)
    plt.plot(train_process.epoch, train_process.val_acc_all, "bs-", label="Val acc", markersize=5)
    plt.xlabel("epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.savefig(f'{args.noise_level}/{args.data_type}/{args.model_name}_training_process.pdf')

def main():
    """Main function to train the model."""
    random_seed = 123
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    np.random.seed(random_seed)

    # Parse arguments
    parser = argparse.ArgumentParser('Model Training', parents=[get_args_parser()])
    args = parser.parse_args()

    # Load dataset and initialize model
    data_set = return_data(args)
    train_loader, val_loader, _ = get_data_loaders(data_set, args.batch_size, train_ratio=0.8, random_seed=random_seed, num_workers=8)

    # Initialize model and loss function
    model = return_model(args)
    loss_fn = nn.CrossEntropyLoss()
    loss_weights = DynamicLossWeights(num_losses=3)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0005)

    os.makedirs(f'{args.noise_level}/{args.data_type}/{args.model_name}', exist_ok=True)
    train_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/loss_accuracy.txt"

    # Train the model
    model, train_process = train(args,model, optimizer, loss_fn, train_loader, val_loader, args.n_epoch, train_path, loss_weights)

    # Visualize the training process
    visualize_training_process(train_process, args)

if __name__ == "__main__":
    main()
