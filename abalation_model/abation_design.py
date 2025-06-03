import torch, os
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import Spectrogram, LogmelFilterBank
from thop.profile import profile
from torchlibrosa.augmentation import SpecAugmentation
from timm.models.layers import DropPath, trunc_normal_
from config import get_args_parser
parser = get_args_parser()
args = parser.parse_args()

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_layer(layer):
    """Initialize a Linear or Convolutional layer. """
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)

def init_bn(bn):
    """Initialize a Batchnorm layer. """
    bn.bias.data.fill_(0.)
    bn.weight.data.fill_(1.)

class ExpertMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ExpertMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.mlp(x)

class MoEBlock(nn.Module):
    def __init__(self, input_dim, num_experts=4, hidden_dim=128, top_k=1):
        super(MoEBlock, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.experts = nn.ModuleList([ExpertMLP(input_dim, hidden_dim) for _ in range(num_experts)])
        self.router = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        logits = self.router(x)  # [B, num_experts]
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)  # [B, k]

        # Softmax over top-k logits
        topk_weights = F.softmax(topk_vals, dim=-1)  # [B, k]

        B, D = x.shape
        output = torch.zeros(B, D, device=x.device)

        for k_i in range(self.top_k):
            idx = topk_idx[:, k_i]  # [B]
            mask = F.one_hot(idx, num_classes=self.num_experts).float()  # [B, num_experts]

            for expert_id in range(self.num_experts):
                expert_mask = mask[:, expert_id].bool()  # [B]
                if expert_mask.any():
                    selected_x = x[expert_mask]  # [b_i, D]
                    expert_output = self.experts[expert_id](selected_x)  # [b_i, D]

                    weighted = topk_weights[expert_mask, k_i].unsqueeze(1) * expert_output  # [b_i, D]
                    output[expert_mask] += weighted

        return output

class AttentionProjection(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionProjection, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, shared_a, shared_v):
        # 拼接音频和视觉特征，输入注意力网络
        fusion_input = torch.cat([shared_a, shared_v], dim=-1)
        attention_weights = self.attention(fusion_input)

        # 用注意力权重平衡两个特征
        refined_a = attention_weights * shared_a + (1 - attention_weights) * shared_v
        refined_v = (1 - attention_weights) * shared_a + attention_weights * shared_v

        return refined_a, refined_v

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):

        super(ConvBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.conv2 = nn.Conv2d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x


class ConvPreWavBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvPreWavBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1,
                               padding=1, bias=False)

        self.conv2 = nn.Conv1d(in_channels=out_channels,
                               out_channels=out_channels,
                               kernel_size=3, stride=1, dilation=2,
                               padding=2, bias=False)

        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_bn(self.bn1)
        init_bn(self.bn2)

    def forward(self, input, pool_size):
        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, kernel_size=pool_size)

        return x

class DynamicGateFusion(nn.Module):
    def __init__(self, feature_dim):
        super(DynamicGateFusion, self).__init__()
        self.feature_dim = feature_dim

        # 模态内融合门控网络
        self.gate_audio = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        self.gate_visual = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

        # 模态间融合门控网络
        self.cross_modal_gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, a1, a2, v1, v2):
        # 模态内融合
        audio_gate_weight = self.gate_audio(torch.cat([a1, a2], dim=-1))
        visual_gate_weight = self.gate_visual(torch.cat([v1, v2], dim=-1))

        fused_audio = audio_gate_weight * a1 + (1 - audio_gate_weight) * a2
        fused_visual = visual_gate_weight * v1 + (1 - visual_gate_weight) * v2

        # 模态间融合
        cross_modal_weight = self.cross_modal_gate(torch.cat([fused_audio, fused_visual], dim=-1))
        fused_feature = cross_modal_weight * fused_audio + (1 - cross_modal_weight) * fused_visual

        return fused_feature


# Modifying MISA Model to incorporate self-adaptive gating mechanism
class MISAWithGating(nn.Module):
    def __init__(self, sample_rate, window_size, hop_size, mel_bins, fmin,
                 fmax, classes_num, hidden_size=128):
        super(MISAWithGating, self).__init__()

        # Initial layers
        self.pre_conv0 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=11, stride=5, padding=5, bias=False)
        self.pre_bn0 = nn.BatchNorm1d(64)
        self.pre_block1 = ConvPreWavBlock(64, 64)
        self.pre_block2 = ConvPreWavBlock(64, 128)
        self.pre_block3 = ConvPreWavBlock(128, 128)

        # Spectrogram extractor
        self.spectrogram_extractor = Spectrogram(n_fft=window_size, hop_length=hop_size,
                                                 win_length=window_size, window='hann', center=True,
                                                 pad_mode='reflect', freeze_parameters=True)

        # Logmel feature extractor
        self.logmel_extractor = LogmelFilterBank(sr=sample_rate, n_fft=window_size, n_mels=mel_bins,
                                                 fmin=fmin, fmax=fmax, ref=1.0, amin=1e-10, top_db=None,
                                                 freeze_parameters=True)
        # Spec augmenter
        self.spec_augmenter = SpecAugmentation(time_drop_width=16, time_stripes_num=2,
                                               freq_drop_width=4, freq_stripes_num=2)

        # self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Private-shared components
        self.private_t = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Sigmoid()
        )
        self.private_v = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Sigmoid()
        )
        
        self.shared = nn.Sequential(
            nn.Linear(in_features=hidden_size, out_features=hidden_size),
            nn.Sigmoid()
        )
            
        # Replace shared with MoE experts
        # self.shared = MoEBlock(input_dim=hidden_size, num_experts=6, hidden_dim=128)
        # self.moe_visual = MoEBlock(input_dim=hidden_size, num_experts=4, hidden_dim=128)

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(in_features=hidden_size * 5, out_features=hidden_size * 3),
            nn.ReLU(),
            nn.Linear(in_features=hidden_size * 3, out_features=classes_num)
        )

        # Gating Mechanism for dynamic fusion
        self.gating_module = DynamicGateFusion(hidden_size)

        # Transformer Encoder (for sequential information)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        # 替换正交投影为注意力投影
        self.attention_projection = AttentionProjection(hidden_size)

    def alignment(self, input):
        # audio features
        # print("input",input.shape)
        a = self.pre_bn0(self.pre_conv0(input[:, None, :]))
        # print("a",a.shape) # 32, 64, 3200]
        a = self.pre_block1(a, pool_size=4)
        a = self.pre_block2(a, pool_size=4)
        a = self.pre_block3(a, pool_size=4)
        a = self.avg_pool(a).squeeze(-1)

        # Spectrogram features
        v = self.spectrogram_extractor(input)  # (batch_size, 1, time_steps, freq_bins)
        v = self.logmel_extractor(v)  # (batch_size, 1, time_steps, mel_bins)
        v = self.conv_block1(v, pool_size=(2, 2), pool_type='avg')
        v = self.conv_block2(v)
        v = F.dropout(v, p=0.2, training=self.training)

        b, c, h, w = v.size()

        a = a.view([b, c])
        v = self.max_pool(v).view([b, c])

        # Private-shared components
        self.utt_private_a = self.private_t(a)
        self.utt_private_v = self.private_v(v)

        self.utt_shared_a = self.shared(a)
        self.utt_shared_v = self.shared(v)
        
        # 用注意力投影
        shared_a_refined, shared_v_refined = self.attention_projection(self.utt_shared_a, self.utt_shared_v)

        # 进入动态融合模块
        fused_av = self.gating_module(shared_a_refined, self.utt_private_a, shared_v_refined, self.utt_private_v)

        if args.model_name == "onlya":
            out = torch.cat([self.utt_private_a, self.utt_shared_a], dim=-1)
            # print(out.shape)
            # h = self.transformer_encoder(out)
            # h = torch.cat((h[0], h[1]), dim=1)
        elif args.model_name == "onlyv":
            out = torch.cat([self.utt_private_v, self.utt_shared_v], dim=-1)
            # print(out.shape)
            # h = self.transformer_encoder(out)
            # h = torch.cat((h[0], h[1]), dim=1)
        elif args.model_name == "private":
            out = torch.cat([self.utt_private_a, self.utt_private_v], dim=-1)
            # h = self.transformer_encoder(out)
            # h = torch.cat((h[0], h[1]), dim=1)
        elif args.model_name == "shared":
            out = torch.cat([self.utt_shared_a, self.utt_shared_v], dim=-1)
            # h = self.transformer_encoder(out)
            # h = torch.cat((h[0], h[1]), dim=1)
        elif args.model_name == "wofusion":
            out = torch.cat([self.utt_private_v, self.utt_private_a, self.utt_shared_a, self.utt_shared_v], dim=-1)
            # h = self.transformer_encoder(out)
            # h = torch.cat((h[0], h[1], h[2], h[3]), dim=1)
        elif args.model_name == "woTrans":
            out = torch.cat([self.utt_private_v, self.utt_private_a, self.utt_shared_a, self.utt_shared_v,fused_av], dim=-1)
            # h = self.transformer_encoder(out)
            # h = torch.cat((h[0], h[1], h[2], h[3]), dim=1) 
        elif args.model_name == "woMoE":
            out = torch.stack([self.utt_private_v, self.utt_private_a, self.utt_shared_a, self.utt_shared_v,fused_av], dim=0)
            h = self.transformer_encoder(out)
            out = torch.cat((h[0], h[1], h[2], h[3],h[4]), dim=1) 
        # Final fusion layer
        o = self.fusion(out)
        return o

    def forward(self, input):
        input = input.squeeze(1)
        return self.alignment(input)


if __name__ == "__main__":
    data = torch.rand(32, 1, 16000)
    model = MISAWithGating(16000, 1024, 320, 64, 10, 6000, 2)
    output = model(data)
    print(output.shape)
    total_ops, total_params = profile(model, (data,), verbose=False)
    flops, params = profile(model, inputs=(data,))
    print("FLOPs=", str(flops / 1e9) + '{}'.format("G"))
    print("params=", str(params / 1e6) + '{}'.format("M"))

