import torch
import pytest

# 导入你自己编写的模型类
from model import PureResNet18MultiLabel

def test_cnn_model_forward():
    """测试雷达多标签分类模型的前向传播和输出维度是否符合预期"""
    
    # 1. 模拟输入数据
    # 根据你的 dataset.py，输入会被插值到 224x224，且包含 amp, re, im 三个通道
    # 我们假设 Batch Size 为 4
    dummy_input = torch.randn(4, 3, 224, 224)
    
    # 2. 实例化你自定义的模型
    # dataset.py 中定义了 6 种基础干扰类型，因此 num_classes=6
    # 在 CI 测试中我们设 pretrained=False，不需要去下载 ImageNet 权重，能让测试跑得更快
    model = PureResNet18MultiLabel(num_classes=6, pretrained=False)
    model.eval() 
    
    # 3. 执行前向传播运算
    with torch.no_grad():
        output = model(dummy_input)
    
    # 4. 关键断言：验证输出张量的形状必须是 (Batch Size, 6)
    assert output.shape == (4, 6), f"模型输出张量维度错误，期望 (4, 6)，实际拿到 {output.shape}"

def test_model_device_compatibility():
    """测试模型是否能顺利在 CPU 环境下初始化（CI服务器默认只有CPU）"""
    try:
        model = PureResNet18MultiLabel(num_classes=6, pretrained=False)
        device = torch.device("cpu")
        model.to(device)
        assert True
    except Exception as e:
        pytest.fail(f"模型在 CPU 环境下初始化失败: {e}")