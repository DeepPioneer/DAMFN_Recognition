from sklearn.metrics import roc_curve, cohen_kappa_score, auc, f1_score, confusion_matrix
import torch
import matplotlib.pyplot as plt
import os, time, argparse
import numpy as np
from random import random

# 设计的模型
from model_create.hope import MISAWithGating
# 消融实验
# 消融实验
# from ablation_model.abation_design import MISAWithGating
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

from util.data_loader import get_data_loaders
from config import get_args_parser
import seaborn as sns  # 用于混淆矩阵可视化
import warnings

warnings.filterwarnings("ignore")

# Set device
os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# 可视化 Mel 频谱图
def imshow_mel(audio, title=None):
    npimg = audio.numpy()
    plt.imshow(npimg, origin='lower', aspect='auto')
    if title:
        plt.title(title)

# 可视化模型预测结果
def visualize_model(model, class_mapping, test_loader, num_waveplot=6):
    was_training = model.training  # 记录模型当前的训练模式
    model.eval()
    audio_so_far = 0
    fig = plt.figure()
    for step, (inputs, labels) in enumerate(test_loader):
        inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
        labels = torch.LongTensor(labels.numpy()).to(device)
        outputs = model(inputs).to(device)
        preds = torch.max(outputs, 1)[1]
        for i in range(inputs.size(0)):
            audio_so_far += 1
            plt.subplot(2, 3, audio_so_far)
            color = 'blue' if class_mapping[preds[i].cpu()] == class_mapping[labels[i].cpu()] else 'red'
            plt.title('predict: {}, expected: {}'.format(class_mapping[preds[i].cpu()], class_mapping[labels[i].cpu()]), color=color)
            imshow_mel(inputs.cpu().data[i])
            plt.subplots_adjust(left=0.125, bottom=0.1, right=0.9, top=0.9, wspace=0.2, hspace=0.5)
            plt.tight_layout()
            plt.axis('off')
            if audio_so_far == num_waveplot:
                model.train(mode=was_training)
                return
    model.train(mode=was_training)

# 测试模型
def test(model, test_loader, test_path, args,CLASS_MAPPING):
    model.eval()
    start = time.time()
    test_loss = 0.0
    total = 0
    correct = 0
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(torch.float32).unsqueeze(1).to(device)
            labels = torch.LongTensor(labels.numpy()).to(device)

            output = model(inputs)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(output.cpu().numpy())

    test_acc = 100. * correct / total

    # 计算 F1-score
    f1 = f1_score(all_labels, all_preds, average='weighted')  # 加权 F1-score

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    # 绘制混淆矩阵 
    plt.figure(figsize=(14, 12))  # ESC 14, 12
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_MAPPING, yticklabels=CLASS_MAPPING,
               annot_kws={"size": 14})  # y轴刻度标签大小)
    plt.xlabel('Predicted',fontsize=22)
    plt.ylabel('True',fontsize=22)
    plt.title('Confusion Matrix',fontsize=24)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    confusion_matrix_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/confusion_matrix.pdf"
    plt.savefig(confusion_matrix_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()

    # 计算 ROC 曲线和 AUC
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_preds = np.array(all_preds)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(args.num_classes):
        fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_probs[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # 绘制 ROC 曲线
    if args.data_type == "Cut_ESC_50":
        # 计算每类的样本数量
        label_counts = np.bincount(all_labels)
        top_classes = np.argsort(label_counts)[-5:]  # 取出现次数最多的5类

        plt.figure(figsize=(8, 6))
        for i in top_classes:
            plt.plot(fpr[i], tpr[i], label=f'{CLASS_MAPPING[i]} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate',fontsize=20)
        plt.ylabel('True Positive Rate',fontsize=20)
        plt.title('Top-5 Classes ROC Curve',fontsize=22)
        plt.legend(loc="lower right")
        # plt.tight_layout()
        plt.savefig(f"{args.noise_level}/{args.data_type}/{args.model_name}/ROC_top5.pdf", bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
    else:
        plt.figure(figsize=(8, 6))
        for i in range(args.num_classes):
            plt.plot(fpr[i], tpr[i], label=f'Class {i} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate',fontsize=20)
        plt.ylabel('True Positive Rate',fontsize=20)
        plt.title('Receiver Operating Characteristic',fontsize=22)
        plt.legend(loc="lower right")
        ROC_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/ROC.pdf"
        plt.savefig(ROC_path, bbox_inches='tight', pad_inches=0, dpi=300)
        plt.close()
        
    # 计算 Cohen's Kappa
    kappa = cohen_kappa_score(all_labels, all_preds)

    # 输出结果
    log_message = (
        f'Test set: Test acc: {test_acc:.2f}%, F1-score: {f1:.2f}, Kappa: {kappa:.2f}')

    print(log_message)

    with open(test_path, "a") as file:
        file.write(log_message + "\n")

    return test_acc, f1, kappa


def return_model(args):
    """Return the model based on the arguments."""
    if args.model_name == 'hope': #fusion
        model = MISAWithGating(
            sample_rate=args.sample_rate, window_size=args.window_size,
            hop_size=args.hop_size, mel_bins=args.mel_bins, fmin=args.fmin, fmax=args.fmax,
            classes_num=args.num_classes
        ).to(device)
    
    # 音频波形
    elif args.model_name == 'TSNA':
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
  
    # 加载模型
    model = return_model(args)
    model.load_state_dict(torch.load(
        f'{args.noise_level}_model_save/{args.data_type}/{args.model_name}/{args.data_type}_{args.model_name}.pkl'))

    # 加载数据集
    data_set = return_data(args)
    if not os.path.exists('{}/{}/{}'.format(args.noise_level, args.data_type, args.model_name)):
        os.makedirs('{}/{}/{}'.format(args.noise_level, args.data_type, args.model_name))

    test_path = f"{args.noise_level}/{args.data_type}/{args.model_name}/test_accuracy.txt"
    
    train_loader, test_loader,val_loader = get_data_loaders(data_set, args.batch_size, train_ratio=0.8,  random_seed=random_seed, num_workers=8)
    # train_loader, test_loader = get_data_loaders(data_set, args.batch_size, num_workers=8)
    # 类别映射
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


    # 测试模型
    test_acc, f1, kappa = test(model, test_loader, test_path, args,CLASS_MAPPING)

    # 可视化模型预测结果
    plt.figure(figsize=(14, 8))
    visualize_model(model, CLASS_MAPPING, test_loader)
    plt.savefig(f'{args.noise_level}/{args.data_type}/{args.model_name}/test_visualize.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()