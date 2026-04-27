import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import f1_score, accuracy_score

# 导入我们的新模块
from dataset import RadarMultiLabelNumpyDataset
from model import PureResNet18MultiLabel

# --- 配置参数 ---
BATCH_SIZE = 32
LEARNING_RATE = 1e-4  
NUM_EPOCHS = 40
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VAL_SPLIT = 0.2  

def find_optimal_thresholds(y_true, y_probs):
    best_thresholds = []
    num_classes = y_true.shape[1]
    thresholds_candidates = np.arange(0.1, 0.95, 0.05)

    print("\n>>> 正在搜索最佳阈值 (Adaptive Threshold Search)...")
    for i in range(num_classes):
        y_t = y_true[:, i]
        y_p = y_probs[:, i]

        best_f1 = -1
        best_thresh = 0.5

        for t in thresholds_candidates:
            y_pred_bin = (y_p > t).astype(int)
            score = f1_score(y_t, y_pred_bin, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresh = t

        best_thresholds.append(float(best_thresh))
    return best_thresholds

def train():
    # 1. 数据准备 (注意替换路径)
    data_dir = r"D:\code\radar_classify\data\train\Dataset\a_r_i"

    # 直接使用自定义 Dataset，不需要传 transform
    full_dataset = RadarMultiLabelNumpyDataset(root_dir=data_dir, target_size=(224, 224))

    val_size = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    
    # 固定随机种子
    generator = torch.Generator().manual_seed(42)
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    print(f"训练集大小: {len(train_dataset)}, 验证集大小: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 2. 模型初始化
    model = PureResNet18MultiLabel(num_classes=6, pretrained=True).to(DEVICE)
    
    # 3. 损失函数与优化器
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_val_accuracy = 0.0

    print(">>> 开始训练...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- 验证循环 ---
        model.eval()
        val_targets = []
        val_probs = []
        val_loss = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())
                val_targets.append(labels.cpu().numpy())

        val_probs = np.vstack(val_probs)
        val_targets = np.vstack(val_targets)
        avg_val_loss = val_loss / len(val_loader)

        # 寻找本轮最佳阈值
        current_thresholds = find_optimal_thresholds(val_targets, val_probs)

        # 计算 Exact Match 准确率
        val_preds_bin = np.zeros_like(val_probs)
        for i in range(6):
            val_preds_bin[:, i] = (val_probs[:, i] > current_thresholds[i]).astype(int)

        val_acc = accuracy_score(val_targets, val_preds_bin)

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
              f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
              f"Val Acc (Exact Match): {val_acc:.4f}")
        print(f"  -> 本轮最佳阈值: {['%.2f' % t for t in current_thresholds]}")

        # 保存模型
        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_thresholds': current_thresholds,
                'best_acc': best_val_accuracy
            }, 'best_model.pth')
            print("  -> [*] 模型已保存 (准确率提升!)")

    print("\n训练结束！最佳验证集准确率: {:.4f}".format(best_val_accuracy))

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()
    train()