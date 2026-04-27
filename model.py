import torch
import torch.nn as nn
from torchvision import models

class PureResNet18MultiLabel(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        """
        纯粹的 ResNet-18 多标签分类模型
        没有任何注意力机制附加模块。
        """
        super(PureResNet18MultiLabel, self).__init__()
        
        # 1. 加载官方 ResNet-18
        if pretrained:
            self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            print("🚀 已加载 ImageNet 预训练权重。")
        else:
            self.resnet = models.resnet18(weights=None)
            
        # 2. 修改全连接层 (FC) 适配多标签输出
        num_ftrs = self.resnet.fc.in_features
        # 注意这里直接输出 logits，不要加 Sigmoid，因为训练脚本里的 BCEWithLogitsLoss 会自动处理
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# --- 测试代码 ---
if __name__ == '__main__':
    # 模拟输入 [Batch=2, Channels=3, Height=224, Width=224]
    dummy_input = torch.randn(2, 3, 224, 224)
    model = PureResNet18MultiLabel(num_classes=6)
    
    output = model(dummy_input)
    print(f"输入形状: {dummy_input.shape}")
    print(f"输出形状: {output.shape} (预期应当是 [2, 6])")