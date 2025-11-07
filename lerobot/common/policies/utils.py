#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from torch import nn
import numpy as np
from scipy.signal import butter, filtfilt


def populate_queues(queues, batch):
    for key in batch:
        # Ignore keys not in the queues already (leaving the responsibility to the caller to make sure the
        # queues have the keys they want).
        if key not in queues:
            continue
        if len(queues[key]) != queues[key].maxlen:
            # initialize by copying the first observation several times until the queue is full
            while len(queues[key]) != queues[key].maxlen:
                queues[key].append(batch[key])
        else:
            # add latest observation to the queue
            queues[key].append(batch[key])
    return queues


def get_device_from_parameters(module: nn.Module) -> torch.device:
    """Get a module's device by checking one of its parameters.

    Note: assumes that all parameters have the same device
    """
    return next(iter(module.parameters())).device


def get_dtype_from_parameters(module: nn.Module) -> torch.dtype:
    """Get a module's parameter dtype by checking one of its parameters.

    Note: assumes that all parameters have the same dtype.
    """
    return next(iter(module.parameters())).dtype


def get_output_shape(module: nn.Module, input_shape: tuple) -> tuple:
    """
    Calculates the output shape of a PyTorch module given an input shape.

    Args:
        module (nn.Module): a PyTorch module
        input_shape (tuple): A tuple representing the input shape, e.g., (batch_size, channels, height, width)

    Returns:
        tuple: The output shape of the module.
    """
    dummy_input = torch.zeros(size=input_shape)
    with torch.inference_mode():
        output = module(dummy_input)
    return tuple(output.shape)

def butterworth_lowpass_filter(
    data: np.ndarray, cutoff_freq: float = 1, sampling_freq: float = 30.0, order=2
) -> np.ndarray:
    """
    Applies a low-pass Butterworth filter to the input data.

    Parameters:
        data (np.array): Input data array.
        cutoff (float): Cutoff frequency of the filter (Hz). Smoother for lower values.
        fs (float): Sampling frequency of the data (Hz).
        order (int): Order of the filter. Higher order may introduce phase distortions.

    Returns:
        filtered_data (np.array): Filtered data array with same shape as data.
    """
    nyquist = 0.5 * sampling_freq
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(order, normal_cutoff, btype="low", analog=False)

    # apply the filter along axis 0
    filtered_data = filtfilt(b, a, data, axis=0)
    return filtered_data


def smoothen_actions(actions: torch.Tensor) -> torch.Tensor:
    """
    Smoothens the provided action sequence tensor
    Args:
        actions (torch.Tensor): actions from policy
    """
    if not isinstance(actions, torch.Tensor):
        raise ValueError(f"Invalid input type for actions {type(actions)}. Expected torch.Tensor!")

    if len(actions.shape) == 3 and not actions.shape[0] == 1:
        raise NotImplementedError("Batch processing not implemented!!")

    actions_np = actions.squeeze(0).cpu().numpy()
    # apply the low-pass filter
    filtered_actions_np = butterworth_lowpass_filter(actions_np.copy())
    # disable filtering for the gripper joint
    filtered_actions_np[:, -1] = actions_np[:, -1]

    return torch.from_numpy(filtered_actions_np.copy()).unsqueeze(0).to(actions.device,dtype=torch.float32)

# class KalmanFilter:
#     """
#     简单线性卡尔曼滤波器（状态维度 = 动作维度）。
#     假设状态转移 F = I，观测矩阵 H = I。
#     process_var: 过程噪声方差（Q = process_var * I）
#     meas_var: 观测噪声方差（R = meas_var * I）
#     """
#     def __init__(self, dim: int, process_var: float = 1e-4, meas_var: float = 1e-2, device: str = "cpu"):
#         self.dim = dim
#         self.device = device
#         self.F = torch.eye(dim, device=device)
#         self.H = torch.eye(dim, device=device)
#         self.Q = torch.eye(dim, device=device) * process_var
#         self.R = torch.eye(dim, device=device) * meas_var
#         self.x = torch.zeros(dim, device=device)  # 初始状态估计
#         self.P = torch.eye(dim, device=device)    # 初始协方差
#
#     def predict(self):
#         # x_{k|k-1} = F x_{k-1|k-1}
#         self.x = self.F @ self.x
#         # P_{k|k-1} = F P_{k-1|k-1} F^T + Q
#         self.P = self.F @ self.P @ self.F.t() + self.Q
#
#     def update(self, z: torch.Tensor) -> torch.Tensor:
#         """
#         更新并返回滤波后的状态估计（维度为 dim）。
#         z: 观测值，形状 (dim,) 或者 (1, dim)
#         """
#         z = z.to(self.device).reshape(self.dim)
#         self.predict()
#         y = z - (self.H @ self.x)                       # 预留差
#         S = self.H @ self.P @ self.H.t() + self.R       # 创新协方差
#         K = self.P @ self.H.t() @ torch.linalg.inv(S)   # 卡尔曼增益
#         self.x = self.x + K @ y                         # 更新状态
#         I = torch.eye(self.dim, device=self.device)
#         self.P = (I - K @ self.H) @ self.P              # 更新协方差
#         return self.x.clone()

# 使用示例（与当前项目集成）
# 假设在 ACTPolicy.__init__ 中：
# self.kf = KalmanFilter(action_dim, process_var=1e-4, meas_var=1e-2, device='cuda')

# 在 select_action 中获取原始 action（形状例如 (1, action_dim) 或 (action_dim,)）后：
# raw_action = ...  # torch.Tensor
# filtered = self.kf.update(raw_action.squeeze(0))  # 返回 (action_dim,)
# 返回与原始形状一致的 tensor，例如：
# return filtered.unsqueeze(0)


class KalmanFilter:
    """
    支持批量的线性卡尔曼滤波器。
    - 状态维度 = action_dim (dim)
    - 支持 batch_size 个并行滤波器
    - 假设 F = I, H = I；Q = process_var * I，R = meas_var * I
    Args:
        dim (int): 动作维度
        batch_size (int): 并行滤波器数量
        process_var (float): 过程噪声方差
        meas_var (float): 观测噪声方差
        device (str|torch.device): 运行设备
        dtype (torch.dtype): 数据类型
    """
    def __init__(
        self,
        dim: int,
        batch_size: int = 1,
        process_var: float = 1e-4,
        meas_var: float = 1e-3,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
    ):
        self.dim = dim
        self.batch = batch_size
        self.device = torch.device(device)
        self.dtype = dtype

        eye = torch.eye(dim, device=self.device, dtype=self.dtype)
        # shape: (batch, dim, dim)
        self.F = eye.unsqueeze(0).repeat(self.batch, 1, 1)
        self.H = eye.unsqueeze(0).repeat(self.batch, 1, 1)
        self.Q = eye.unsqueeze(0).repeat(self.batch, 1, 1) * process_var
        self.R = eye.unsqueeze(0).repeat(self.batch, 1, 1) * meas_var

        # state x shape: (batch, dim)
        self.x = torch.zeros(self.batch, dim, device=self.device, dtype=self.dtype)
        # covariance P shape: (batch, dim, dim)
        self.P = eye.unsqueeze(0).repeat(self.batch, 1, 1)

    def predict(self):
        # x_{k|k-1} = F x_{k-1|k-1}
        self.x = torch.matmul(self.F, self.x.unsqueeze(-1)).squeeze(-1)
        # P_{k|k-1} = F P F^T + Q
        self.P = torch.matmul(self.F, torch.matmul(self.P, self.F.transpose(1, 2))) + self.Q

    def update(self, z: torch.Tensor) -> torch.Tensor:
        """
        输入 z 可以是:
          - (dim,)         -> 视为单个观测，复制到所有 batch
          - (1, dim)       -> 若 batch>1，则复制；若 batch==1 则直接使用
          - (batch, dim)   -> 与 batch 对齐
        返回:
          - filtered states 形状为 (batch, dim)
        """
        z = torch.as_tensor(z, device=self.device, dtype=self.dtype)

        # 规范化形状到 (batch, dim)
        if z.dim() == 1:
            z = z.unsqueeze(0).repeat(self.batch, 1)
        elif z.dim() == 2:
            if z.shape[0] == 1 and self.batch > 1:
                z = z.repeat(self.batch, 1)
            elif z.shape[0] != self.batch:
                raise ValueError(f"观测 batch 大小 {z.shape[0]} 不匹配滤波器 batch {self.batch}")

        self.predict()

        # innovation y = z - H x
        y = z - torch.matmul(self.H, self.x.unsqueeze(-1)).squeeze(-1)
        # S = H P H^T + R
        S = torch.matmul(self.H, torch.matmul(self.P, self.H.transpose(1, 2))) + self.R
        # K = P H^T S^{-1}
        S_inv = torch.linalg.inv(S)
        K = torch.matmul(self.P, torch.matmul(self.H.transpose(1, 2), S_inv))
        # 更新状态 (使用 matmul 支持 2D/3D 广播)
        self.x = self.x + torch.matmul(K, y.unsqueeze(-1)).squeeze(-1)
        # 更新协方差
        I = torch.eye(self.dim, device=self.device, dtype=self.dtype).unsqueeze(0).repeat(self.batch, 1, 1)
        self.P = torch.matmul((I - torch.matmul(K, self.H)), self.P)
        return self.x.clone()
