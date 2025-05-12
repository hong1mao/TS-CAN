import os

import numpy as np
import random
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from data.ubfc_rppg_dataset import UBFCrPPGDataset
from model.TS_CAN import TSCAN
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt


def main():
    # 加载配置文件
    with open("./config/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 设置随机种子
    seed = 100
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 创建数据集和数据加载器
    train_dataset = UBFCrPPGDataset(data_path=config['train']['data']['data_path'],
                                    cached_path=config['train']['data']['cached_path'],
                                    file_list_path=config['train']['data']['file_list_path'],
                                    split_ratio=config['train']['data']['split_ratio'],
                                    chunk_length=config['train']['data']['chunk_length'],
                                    preprocess=config['train']['data']['preprocess'],
                                    re_size=config['model']['re_size'],
                                    larger_box_coef=config['train']['data']['larger_box_coef'],
                                    backend=config['train']['data']['backend'])

    val_dataset = UBFCrPPGDataset(data_path=config['val']['data']['data_path'],
                                  cached_path=config['val']['data']['cached_path'],
                                  file_list_path=config['val']['data']['file_list_path'],
                                  split_ratio=config['val']['data']['split_ratio'],
                                  chunk_length=config['val']['data']['chunk_length'],
                                  preprocess=config['val']['data']['preprocess'],
                                  re_size=config['model']['re_size'],
                                  larger_box_coef=config['val']['data']['larger_box_coef'],
                                  backend=config['val']['data']['backend'])

    train_loader = DataLoader(train_dataset,
                              batch_size=config['training']['batch_size'],
                              shuffle=config['training']['shuffle'],
                              num_workers=config['training']['num_workers'])

    val_loader = DataLoader(val_dataset,
                            batch_size=config['training']['batch_size'],
                            shuffle=False,
                            num_workers=config['training']['num_workers'])

    # 设置设备
    if torch.cuda.is_available():
        print("✔ Using CUDA")
    else:
        print("❌ Using CPU")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型
    frame_depth = config['model']['frame_depth']
    model = TSCAN(frame_depth=frame_depth, img_size=config['model']['re_size'])
    model = model.to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=config['training']['optimizer']['lr'], weight_decay=0)

    # 添加 OneCycleLR 调度器
    num_train_batches = len(train_loader)
    scheduler = lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config['training']['optimizer']['lr'],
        epochs=config['training']['num_epochs'],
        steps_per_epoch=num_train_batches
    )

    # 训练循环
    num_epochs = config['training']['num_epochs']

    # 创建 run 目录
    run_dir = "./run"
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    # 生成训练结果编号
    run_id = len(os.listdir(run_dir)) + 1
    run_path = os.path.join(run_dir, f"run_{run_id}")
    os.makedirs(run_path)
    best_model_path = os.path.join(run_path, "tscan_best.pth")
    best_val_loss = float('inf')

    # 记录训练和验证损失
    train_losses = []
    val_losses = []
    lrs = []

    # 训练循环
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        progress_bar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] - Training", total=len(train_loader))
        for i, batch in enumerate(progress_bar):
            inputs, labels = batch[0], batch[1]
            inputs = inputs.to(device)
            N, D, C, H, W = inputs.shape
            inputs = inputs.view(N * D, C, H, W)  # 调整为 (B*T, C*2, H, W)
            labels = labels.to(device)
            labels = labels.view(-1, 1)  # 调整为 (B*T, 1)

            inputs = inputs[:(N * D) // frame_depth * frame_depth]
            labels = labels[:(N * D) // frame_depth * frame_depth]

            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)  # shape: (batch_size, 1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()
            scheduler.step()  # 每个 batch 后更新学习率

            train_loss += loss.item()

            # 更新进度条描述（显示当前 loss）
            current_lr = optimizer.param_groups[0]['lr']
            lrs.append(current_lr)
            progress_bar.set_postfix(loss=loss.item(), lr=current_lr)

        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}")

        # 验证阶段
        model.eval()
        val_loss = 0.0
        progress_bar_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}] - Validation",
                                total=len(val_loader))
        with torch.no_grad():
            for i, batch in enumerate(progress_bar_val):
                inputs, labels = batch[0], batch[1]
                inputs = inputs.to(device)
                N, D, C, H, W = inputs.shape
                inputs = inputs.view(N * D, C, H, W)
                labels = labels.to(device).view(-1, 1)

                inputs = inputs[:(N * D) // frame_depth * frame_depth]
                labels = labels[:(N * D) // frame_depth * frame_depth]

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                progress_bar_val.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")

        # 保存最优模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"✅ Best model saved with Val Loss: {best_val_loss:.4f}")

        # 记录损失
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

    print("Training finished.")

    # 保存训练结果
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.legend()
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(run_path, "loss_curve.png"))
    plt.close()

    # 绘制学习率曲线
    plt.figure(figsize=(8, 5))
    plt.plot(lrs)
    plt.title("Learning Rate Schedule")
    plt.xlabel("Step")
    plt.ylabel("Learning Rate")
    plt.show()
    plt.savefig(os.path.join(run_path, "lr_curve.png"))
    plt.close()


if __name__ == "__main__":
    main()
