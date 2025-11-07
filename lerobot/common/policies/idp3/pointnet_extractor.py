# Credit: https://github.com/YanjieZe/Improved-3D-Diffusion-Policy
import logging
from typing import Dict, List, Type
import numpy as np
import torch
import torch.nn as nn
logger = logging.getLogger(__name__)

def shuffle_point_numpy(point_cloud):
    """
        使用 NumPy 对点云进行随机打乱。
        这确保模型对点的输入顺序不敏感。

        :param point_cloud: NumPy 数组，形状为 (B, N, C)，其中 B 是批量大小，N 是点数，C 是特征维度。
        :return: 打乱顺序后的点云数组。
    """
    B, N, C = point_cloud.shape
    # 生成一个 0 到 N-1 的随机排列索引
    indices = np.random.permutation(N)
    # 根据索引重新排列点
    return point_cloud[:, indices]

def pad_point_numpy(point_cloud, num_points):
    """
        使用 NumPy 将点云填充到指定的点数。
        如果当前点数小于目标点数，则用零向量填充。

        :param point_cloud: NumPy 数组，形状为 (B, N, C)。
        :param num_points: 目标点数。
        :return: 填充后的点云数组。
    """
    B, N, C = point_cloud.shape
    if num_points > N:
        num_pad = num_points - N
        # 创建零向量用于填充
        pad_points = np.zeros((B, num_pad, C))
        # 将填充点与原始点云拼接
        point_cloud = np.concatenate([point_cloud, pad_points], axis=1)
        # 填充后打乱，避免填充的零点总在末尾
        point_cloud = shuffle_point_numpy(point_cloud)
    return point_cloud

def uniform_sampling_numpy(point_cloud, num_points):
    """
        使用 NumPy 对点云进行均匀采样，以达到指定的点数。
        如果点数不足，则进行填充。如果点数过多，则进行随机下采样。

        :param point_cloud: NumPy 数组，形状为 (B, N, C)。
        :param num_points: 目标点数。
        :return: 采样/填充后的点云数组。
    """
    B, N, C = point_cloud.shape
    # padd if num_points > N
    if num_points > N:
        return pad_point_numpy(point_cloud, num_points)

    # 随机下采样
    indices = np.random.permutation(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points

def shuffle_point_torch(point_cloud):
    """
        使用 PyTorch 对点云进行随机打乱。
        功能同 shuffle_point_numpy，但作用于 PyTorch 张量。

        :param point_cloud: PyTorch 张量，形状为 (B, N, C)。
        :return: 打乱顺序后的点云张量。
    """
    B, N, C = point_cloud.shape
    indices = torch.randperm(N)
    return point_cloud[:, indices]


def pad_point_torch(point_cloud, num_points):
    """
        使用 PyTorch 将点云填充到指定的点数。
        功能同 pad_point_numpy，但作用于 PyTorch 张量。

        :param point_cloud: PyTorch 张量，形状为 (B, N, C)。
        :param num_points: 目标点数。
        :return: 填充后的点云张量。
    """
    B, N, C = point_cloud.shape
    device = point_cloud.device
    if num_points > N:
        num_pad = num_points - N
        pad_points = torch.zeros(B, num_pad, C).to(device)
        point_cloud = torch.cat([point_cloud, pad_points], dim=1)
        point_cloud = shuffle_point_torch(point_cloud)
    return point_cloud

def uniform_sampling_torch(point_cloud, num_points):
    """
        使用 PyTorch 对点云进行均匀采样，以达到指定的点数。
        功能同 uniform_sampling_numpy，但作用于 PyTorch 张量。

        :param point_cloud: PyTorch 张量，形状为 (B, N, C)。
        :param num_points: 目标点数。
        :return: 采样/填充后的点云张量。
    """
    B, N, C = point_cloud.shape
    device = point_cloud.device
    # 如果点数正好，直接返回
    if num_points == N:
        return point_cloud
    # 如果点数不足，进行填充
    if num_points > N:
        return pad_point_torch(point_cloud, num_points)

    # random sampling
    indices = torch.randperm(N)[:num_points]
    sampled_points = point_cloud[:, indices]
    return sampled_points


def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
) -> List[nn.Module]:
    """
        创建一个多层感知机（MLP）的模块列表。

        :param input_dim: 输入向量的维度。
        :param output_dim: 输出向量的维度。
        :param net_arch: 神经网络的架构，表示每层神经元的数量。
        :param activation_fn: 每层之后使用的激活函数。
        :param squash_output: 是否在输出层后使用 Tanh 激活函数进行压缩。
        :return: 一个包含 MLP 所有层的 nn.Module 列表。
    """
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())

    return modules


def meanpool(x, dim=-1, keepdim=False):
    out = x.mean(dim=dim, keepdim=keepdim)
    return out

def maxpool(x, dim=-1, keepdim=False):
    out = x.max(dim=dim, keepdim=keepdim).values
    return out

class MultiStagePointNetEncoder(nn.Module):
    """
        一个多阶段的 PointNet 编码器，用于从点云中提取全局特征。
        在每个阶段，它都结合了局部点特征和全局上下文特征。
    """
    def __init__(self, h_dim=128, out_channels=128, num_layers=4, **kwargs):
        super().__init__()
        self.h_dim = h_dim
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.act = nn.LeakyReLU(negative_slope=0.0, inplace=False)
        # 输入卷积层，将点云从 3 维（XYZ）映射到 h_dim 维
        self.conv_in = nn.Conv1d(3, h_dim, kernel_size=1)
        # 存储每个阶段的局部和全局处理层
        self.layers, self.global_layers = nn.ModuleList(), nn.ModuleList()
        for i in range(self.num_layers):
            # 局部特征提取层
            self.layers.append(nn.Conv1d(h_dim, h_dim, kernel_size=1))
            # 全局特征融合层，输入维度为 2*h_dim (局部特征+全局特征)
            self.global_layers.append(nn.Conv1d(h_dim * 2, h_dim, kernel_size=1))

        # 输出卷积层，将所有阶段的特征拼接后进行处理
        self.conv_out = nn.Conv1d(h_dim * self.num_layers, out_channels, kernel_size=1)

    def forward(self, x):
        """
        前向传播。
        :param x: 输入点云张量，形状为 [B, N, 3]。
        :return: 全局特征向量，形状为 [B, out_channels]。
        """
        x = x.transpose(1, 2)  # [B, N, 3] --> [B, 3, N]
        y = self.act(self.conv_in(x))
        feat_list = []
        for i in range(self.num_layers):
            # 1. 提取局部特征
            y = self.act(self.layers[i](y))
            # 2. 计算当前阶段的全局特征（通过最大池化）
            y_global = y.max(-1, keepdim=True).values
            # 3. 将全局特征广播并与每个点的局部特征拼接
            y = torch.cat([y, y_global.expand_as(y)], dim=1)
            # 4. 融合局部和全局特征
            y = self.act(self.global_layers[i](y))
            feat_list.append(y)
        # cat all features
        x = torch.cat(feat_list, dim=1)
        x = self.conv_out(x)
        x_global = x.max(-1).values

        return x_global

class StateEncoder(nn.Module):
    def __init__(self, observation_space: Dict, state_mlp_size=(64, 64), state_mlp_activation_fn=nn.ReLU):
        super().__init__()
        self.state_key = "full_state"
        self.state_shape = observation_space[self.state_key]
        logger.debug(f"[StateEncoder] state shape: {self.state_shape}")
        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]
        output_dim = state_mlp_size[-1]

        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )
        logger.debug(f"[StateEncoder] output dim: {output_dim}")
        self.output_dim = output_dim


    def output_shape(self):
        return self.output_dim

    def forward(self, observations: Dict) -> torch.Tensor:
        state = observations[self.state_key]
        state_feat = self.state_mlp(state)
        return state_feat

class IDP3Encoder(nn.Module):  # noqa: N801
    def __init__(
        self,
        observation_space: Dict,
        state_mlp_size=(64, 64),
        state_mlp_activation_fn=nn.ReLU,
        pointcloud_encoder_cfg=None,
        use_pc_color=False,
        pointnet_type="dp3_encoder",
        point_downsample=True,
    ):
        # 输入: observations (包含 observation.state 和 observation.pointcloud 两部分)
        #
        # ┌──────────────────────────────────────────┐
        # │              IDP3Encoder                 │
        # └──────────────────────────────────────────┘
        #                    │
        #       ┌────────────┴─────────────┐
        #       │                          │
        # observation.pointcloud     observation.state
        #       │                          │
        #  (B, N, C)                  (B, state_dim)
        #       │                          │
        #  [如需要] 下采样                   │
        #  uniform_sampling_torch          │
        #       │                          │
        #       ▼                          ▼
        # ┌─────────────────┐      ┌──────────────────────┐
        # │ MultiStagePoint │      │    MLP (state_mlp)   │
        # │ Net Encoder     │      └──────────────────────┘
        # │ (extractor)     │
        # └─────────────────┘
        #       │                          │
        #   pn_feat                  state_feat
        # (B, out_channels)           (B, output_dim)
        #       └───────────┬────────────┘
        #                   ▼
        #         torch.cat([pn_feat, state_feat], dim=-1)
        #                   │
        #                   ▼
        #           final_feat (B, n_output_channels)

        super().__init__()
        self.state_key = "observation.state"
        self.point_cloud_key = "observation.pointcloud"
        self.n_output_channels = pointcloud_encoder_cfg.out_channels
        self.point_cloud_shape = observation_space[self.point_cloud_key]
        self.state_shape = observation_space[self.state_key]

        self.num_points = pointcloud_encoder_cfg.num_points  # 4096

        logger.debug(f"[IDP3Encoder] point cloud shape: {self.point_cloud_shape}")
        logger.debug(f"[IDP3Encoder] state shape: {self.state_shape}")

        self.use_pc_color = use_pc_color
        self.pointnet_type = pointnet_type

        self.downsample = point_downsample
        if self.downsample:
            self.point_preprocess = uniform_sampling_torch
        else:
            self.point_preprocess = nn.Identity()

        if pointnet_type == "multi_stage_pointnet":
            self.extractor = MultiStagePointNetEncoder(out_channels=pointcloud_encoder_cfg.out_channels)
        else:
            raise NotImplementedError(f"pointnet_type: {pointnet_type}")

        if len(state_mlp_size) == 0:
            raise RuntimeError("State mlp size is empty")
        elif len(state_mlp_size) == 1:
            net_arch = []
        else:
            net_arch = state_mlp_size[:-1]

        output_dim = state_mlp_size[-1]
        self.n_output_channels += output_dim
        self.state_mlp = nn.Sequential(
            *create_mlp(self.state_shape[0], output_dim, net_arch, state_mlp_activation_fn)
        )
        logger.debug(f"[DP3Encoder] output dim: {self.n_output_channels}")

    def forward(self, observations: Dict) -> torch.Tensor:
        points = observations[self.point_cloud_key]
        assert len(points.shape) == 3, f"point cloud shape: {points.shape}, length should be 3"

        # points = torch.transpose(points, 1, 2)   # B * 3 * N
        # points: B * 3 * (N + sum(Ni))
        if self.downsample:
            points = self.point_preprocess(points, self.num_points)

        pn_feat = self.extractor(points)  # B * out_channel

        state = observations[self.state_key]
        state_feat = self.state_mlp(state)  # B * 64
        final_feat = torch.cat([pn_feat, state_feat], dim=-1)
        return final_feat

    def output_shape(self):
        return self.n_output_channels
