import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
from data.ubfc_rppg_dataset import UBFCrPPGDataset
from model.TS_CAN import TSCAN


def test():
    # 加载配置文件
    with open("./config/train.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 创建测试集
    test_dataset = UBFCrPPGDataset(
        data_path=config['test']['data']['data_path'],
        cached_path=config['test']['data']['cached_path'],
        file_list_path=config['test']['data']['file_list_path'],
        split_ratio=config['test']['data']['split_ratio'],
        chunk_length=config['test']['data']['chunk_length'],
        preprocess=config['test']['data']['preprocess'],
    )
    test_loader = DataLoader(test_dataset,
                             batch_size=config['training']['batch_size'],
                             shuffle=False,
                             num_workers=0)

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化模型并加载最佳权重
    frame_depth = config['model']['frame_depth']
    model = TSCAN(frame_depth=frame_depth).to(device)
    best_model_path = "run/run_1/tscan_best.pth"
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    criterion = nn.MSELoss()
    test_loss = 0.0

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            N, D, C, H, W = inputs.shape
            inputs = inputs.view(N * D, C, H, W)
            labels = labels.to(device).view(-1, 1)

            inputs = inputs[:(N * D) // frame_depth * frame_depth]
            labels = labels[:(N * D) // frame_depth * frame_depth]

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

    avg_test_loss = test_loss / len(test_loader)
    print(f"Test Loss: {avg_test_loss:.4f}")

if __name__ == "__main__":
    test()