import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score, hamming_loss, multilabel_confusion_matrix
from tqdm import tqdm  # 进度条库

# 导入我们刚刚编写的最新模块
from dataset import RadarMultiLabelNumpyDataset
from model import PureResNet18MultiLabel

# --- 配置 ---
CLASSES = ['CSJ', 'DFJ', 'ISRJ', 'NAMJ', 'NFMJ', 'RGPO']
num_classes = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'best_model.pth'

# 测试集路径 (请替换为你真实的包含6种干扰文件夹的路径)
TEST_DATA_DIR = r"D:\code\radar_classify\data\test\Dataset_test\a_r_i_test"


def load_model_and_thresholds():
    print(f"正在加载模型: {MODEL_PATH} ...")
    # 加载我们新的纯净版 ResNet18 模型，这里不用预训练权重，因为只是用来装载本地参数
    model = PureResNet18MultiLabel(num_classes=num_classes, pretrained=False)
    
    # 加载 checkpoint
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    # 读取训练时保存的自适应阈值
    if 'best_thresholds' in checkpoint:
        thresholds = checkpoint['best_thresholds']
        print(f"✅ 加载自适应阈值: {['%.2f' % t for t in thresholds]}")
    else:
        thresholds = [0.5] * num_classes
        print("⚠️ 未找到阈值，使用默认 0.5")

    return model, thresholds


def plot_confusion_matrices(y_true, y_pred, classes):
    """
    绘制多标签混淆矩阵：为每一类画一个 2x2 的矩阵
    优化：改成 2 行 3 列的网格布局，更美观
    """
    mcm = multilabel_confusion_matrix(y_true, y_pred)

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()  # 展平以便遍历

    for i, (matrix, class_name) in enumerate(zip(mcm, classes)):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Predicted 0', 'Predicted 1'],
                    yticklabels=['Actual 0', 'Actual 1'],
                    annot_kws={"size": 14}) # 字体放大点看得更清楚

        axes[i].set_title(f"Class: {class_name}", fontsize=14, fontweight='bold')
        axes[i].set_ylabel('True Label', fontsize=12)
        axes[i].set_xlabel('Predicted Label', fontsize=12)

    plt.tight_layout()
    plt.savefig('confusion_matrices.png', dpi=300) # 提高保存图片的分辨率
    print("\n📊 多标签混淆矩阵已保存为: confusion_matrices.png (2x3布局)")
    # plt.show() # 如果在服务器上跑，保持注释


def evaluate():
    # 1. 准备数据
    print("\n>>> 正在扫描并加载测试集数据...")
    # 注意：这里直接传入 root_dir，不需要任何 transforms 了
    dataset = RadarMultiLabelNumpyDataset(root_dir=TEST_DATA_DIR, target_size=(224, 224))
    
    # 推荐在 Windows 下将 num_workers 设为 0，防止进程堵塞
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    # 2. 加载模型
    model, thresholds = load_model_and_thresholds()

    # 3. 批量预测
    all_preds = []
    all_targets = []

    print(f"开始评估 {len(dataset)} 个样本...")

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(DEVICE)

            # 模型输出 Logits
            logits = model(images)
            # 转为概率 (因为是多标签)
            probs = torch.sigmoid(logits)

            # 收集结果
            all_preds.append(probs.cpu().numpy())
            all_targets.append(labels.numpy())

    # 拼接所有批次
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)

    # 4. 应用自适应阈值进行二值化
    all_preds_bin = np.zeros_like(all_preds)
    for i in range(num_classes):
        all_preds_bin[:, i] = (all_preds[:, i] > thresholds[i]).astype(int)

    # --- 5. 计算并打印指标 ---
    print("\n" + "=" * 50)
    print("📢 最终评估报告 (Final Evaluation Report)")
    print("=" * 50)

    # 指标 1: Exact Match Ratio (完全匹配率)
    # 最严格指标：必须所有的干扰是否存在都判断正确
    exact_acc = accuracy_score(all_targets, all_preds_bin)
    print(f"✅ Exact Match Accuracy (完全匹配率): {exact_acc:.4f}")

    # 指标 2: Hamming Loss
    # 容错率：平均预测错了多少个标签
    h_loss = hamming_loss(all_targets, all_preds_bin)
    print(f"📉 Hamming Loss (汉明损失): {h_loss:.4f}")

    print("-" * 50)
    print("📋 分类别详细报告:")

    # 指标 3: 详细分类报告 (Precision, Recall, F1)
    print(classification_report(all_targets, all_preds_bin, target_names=CLASSES, digits=4))

    # 6. 绘制混淆矩阵
    plot_confusion_matrices(all_targets, all_preds_bin, CLASSES)

if __name__ == '__main__':
    evaluate()
