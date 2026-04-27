import os
import glob
import torch
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset

class RadarMultiLabelNumpyDataset(Dataset):
    def __init__(self, root_dir, target_size=(224, 224)):
        """
        Args:
            root_dir (string): 数据集根目录路径 (包含所有子文件夹的目录)
            target_size (tuple): 网络期望的输入尺寸
        """
        self.root_dir = root_dir
        self.target_size = target_size

        # 1. 定义6种基础干扰类型 (顺序决定了输出向量每一位的含义)
        self.base_classes = ['CSJ', 'DFJ', 'ISRJ', 'NAMJ', 'NFMJ', 'RGPO']

        # 2. 扫描所有 .npy 文件并生成多标签
        self.samples = []  # 存储 (npy_path, label_vector)
        self._load_data()

    def _load_data(self):
        if not os.path.exists(self.root_dir):
            raise RuntimeError(f"目录不存在: {self.root_dir}")

        # 获取所有子文件夹
        folder_names = [d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))]
        print(f"检测到的文件夹: {folder_names}")

        for folder in folder_names:
            folder_path = os.path.join(self.root_dir, folder)
            # 扫描 .npy 文件
            npy_paths = glob.glob(os.path.join(folder_path, '*.npy'))

            # 生成多热编码标签
            label = torch.zeros(len(self.base_classes), dtype=torch.float32)
            for idx, cls_name in enumerate(self.base_classes):
                if cls_name in folder:
                    label[idx] = 1.0

            for npy_path in npy_paths:
                self.samples.append((npy_path, label))

        print(f"数据加载完成，共找到 {len(self.samples)} 个样本。")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]

        # 1. 读取 numpy 矩阵，形状预期为 (3, H, W)
        try:
            data_matrix = np.load(file_path)
            tensor_data = torch.from_numpy(data_matrix).float()
        except Exception as e:
            print(f"无法读取文件: {file_path}, error: {e}")
            return None, None

        # 2. 分通道独立归一化 (保持实部虚部的正负号)
        amp = tensor_data[0] 
        re = tensor_data[1]  
        im = tensor_data[2]  

        amp = 20 * torch.log10(amp + 1e-6)
        amp = (amp - amp.min()) / (amp.max() - amp.min() + 1e-8)

        re = re / (torch.max(torch.abs(re)) + 1e-8)
        im = im / (torch.max(torch.abs(im)) + 1e-8)

        normalized_data = torch.stack([amp, re, im], dim=0)

        # 3. 插值缩放至 224x224
        normalized_data = normalized_data.unsqueeze(0)
        resized_data = F.interpolate(normalized_data, size=self.target_size, mode='bilinear', align_corners=False)
        resized_data = resized_data.squeeze(0)

        return resized_data, label

if __name__ == '__main__':
    # 测试路径，替换为你真实的 a_r_i 数据集路径
    test_root_dir = r"D:\code\radar_classify\data\train\Dataset\a_r_i"
    
    if os.path.exists(test_root_dir):
        dataset = RadarMultiLabelNumpyDataset(root_dir=test_root_dir)
        print("\n--- 标签测试 ---")
        print(f"标签对应关系: {dataset.base_classes}")

        if len(dataset) > 0:
            sample_indices = [0, len(dataset) // 2, len(dataset) - 1]
            for i in sample_indices:
                data, lab = dataset[i]
                path = dataset.samples[i][0]
                folder_name = os.path.basename(os.path.dirname(path))
                print(f"文件夹: {folder_name} -> 标签: {lab.numpy()} | 数据形状: {data.shape}")