from scipy import io
import os
import numpy as np
import torch
import pandas as pd
from openpyxl import load_workbook
from fault_diag_utils import *

# 确保每个片段的长度大于或等于网络需求的最小长度
MIN_LENGTH = 1024  # 实际计算的最小长度
SEGMENT_LENGTH = max(MIN_LENGTH, 1024)  # 选择一个适当的片段长度，确保 >= MIN_LENGTH

if __name__ == "__main__":
    # 读取测试数据
    PATH = "data"
    files = os.listdir(PATH)
    for file in files:
        if "csv" not in file:
            continue
        test_data = pd.read_csv(PATH + "/" + file).values.squeeze()  # 移除单维度的轴
        test_data = torch.tensor(test_data, dtype=torch.float32)

        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = torch.load("cnn_net.pth").to(DEVICE)
        print(f"DEVICE:{DEVICE}")

        net.eval()
        predict = []

        for i in range(0, len(test_data), SEGMENT_LENGTH):
            segment = test_data[i:i + SEGMENT_LENGTH]
            if len(segment) < MIN_LENGTH:
                break  # 如果片段长度小于最小长度，跳出循环
            segment = segment.unsqueeze(0).unsqueeze(0).to(DEVICE)  # 添加批次和通道维度
            y = net(segment)
            output = y.max(1, keepdim=True)[1]
            predict.extend([get_label(o.item()) for o in output])

        df = {
            "index": [i for i in range(len(predict))],
            "label": predict
        }
        df = pd.DataFrame(df)
        results_name = os.path.splitext(os.path.basename(PATH + "/" + file))[0]
        df.to_excel("text_" + results_name + ".xlsx", index=False)
