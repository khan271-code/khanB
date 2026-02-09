#!/usr/bin/env python3
import os
import sys
import shutil
import time
import math
import numpy as np
import torch
import torchvision
import cv2
import wget
import pathlib
import tqdm
import scipy.signal
import warnings

# 忽略 torchvision 的版本警告
warnings.filterwarnings("ignore", category=UserWarning)

# 重要：将 Matplotlib 设置为非交互模式 (Agg)
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

# 确保 echonet 在 python 路径中
sys.path.append("..") 
import echonet

# ==========================================
# 1. 配置 / 路径设置
# ==========================================
destinationFolder = "/hpc/khan271/project_2025/dynamic/test_hk/output"
videosFolder = "/hpc/khan271/project_2025/dynamic/test_hk/AVI"
DestinationForWeights = "/hpc/khan271/project_2025/dynamic/EchoNetDynamic-Weights"

pathlib.Path(destinationFolder).mkdir(parents=True, exist_ok=True)
pathlib.Path(DestinationForWeights).mkdir(parents=True, exist_ok=True)

# ==========================================
# 2. 下载模型权重
# ==========================================
print(f"正在检查权重文件位置: {DestinationForWeights}")

segmentationWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/deeplabv3_resnet50_random.pt'
ejectionFractionWeightsURL = 'https://github.com/douyang/EchoNetDynamic/releases/download/v1.0.0/r2plus1d_18_32_2_pretrained.pt'

seg_weights_path = os.path.join(DestinationForWeights, os.path.basename(segmentationWeightsURL))
ef_weights_path = os.path.join(DestinationForWeights, os.path.basename(ejectionFractionWeightsURL))

if not os.path.exists(seg_weights_path):
    print(f"正在下载分割权重到 {seg_weights_path} ...")
    wget.download(segmentationWeightsURL, out=DestinationForWeights)
else:
    print("分割权重文件已存在。")

if not os.path.exists(ef_weights_path):
    print(f"正在下载 EF 权重到 {ef_weights_path} ...")
    wget.download(ejectionFractionWeightsURL, out=DestinationForWeights)
else:
    print("EF 权重文件已存在。")

# ==========================================
# 3. 初始化并运行射血分数 (EF) 模型
# ==========================================
print("--- 正在初始化 EF 模型 ---")

frames = 32
period = 1 
batch_size = 20 

model = torchvision.models.video.r2plus1d_18(weights=None) 
model.fc = torch.nn.Linear(model.fc.in_features, 1)

print(f"从 {ef_weights_path} 加载权重")

if torch.cuda.is_available():
    print("CUDA 可用，使用 GPU 进行计算。")
    device = torch.device("cuda")
    model = torch.nn.DataParallel(model)
    model.to(device)
    # 修复：weights_only=False
    checkpoint = torch.load(ef_weights_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("CUDA 不可用，使用 CPU 进行计算。")
    device = torch.device("cpu")
    # 修复：weights_only=False
    checkpoint = torch.load(ef_weights_path, map_location="cpu", weights_only=False)
    state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict_cpu)

output_csv = os.path.join(destinationFolder, "cedars_ef_output.csv")

# 修复：删除了 crops="all"
ds = echonet.datasets.Echo(split="external_test", external_test_location=videosFolder)
print(f"找到 {len(ds.fnames)} 个文件待处理。")

mean, std = echonet.utils.get_mean_and_std(ds)

kwargs = {
    "target_type": "EF",
    "mean": mean,
    "std": std,
    "length": frames,
    "period": period,
}

# 修复：删除了 crops="all"
ds = echonet.datasets.Echo(split="external_test", external_test_location=videosFolder, **kwargs)

test_dataloader = torch.utils.data.DataLoader(
    ds, batch_size=1, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda")
)

print("正在运行 EF 预测...")

# 修复：将 "test" 改为 False
loss, yhat, y = echonet.utils.video.run_epoch(model, test_dataloader, False, None, device, save_all=True)

with open(output_csv, "w") as g:
    for (filename, pred) in zip(ds.fnames, yhat):
        for (i, p) in enumerate(pred):
            g.write("{},{},{:.4f}\n".format(filename, i, p))
print(f"EF 结果已保存至 {output_csv}")

# ==========================================
# 4. 初始化并运行分割模型 (Segmentation)
# ==========================================
print("--- 正在运行分割任务 ---")

torch.cuda.empty_cache()

def collate_fn(x):
    x, f = zip(*x)
    i = list(map(lambda t: t.shape[1], x))
    x = torch.as_tensor(np.swapaxes(np.concatenate(x, 1), 0, 1))
    return x, f, i

dataloader = torch.utils.data.DataLoader(
    echonet.datasets.Echo(
        split="external_test", 
        external_test_location=videosFolder, 
        target_type=["Filename"], 
        length=None, 
        period=1, 
        mean=mean, 
        std=std
    ),
    batch_size=10, 
    num_workers=4, 
    shuffle=False, 
    pin_memory=(device.type == "cuda"), 
    collate_fn=collate_fn
)

if not all([os.path.isfile(os.path.join(destinationFolder, "labels", os.path.splitext(f)[0] + ".npy")) for f in dataloader.dataset.fnames]):
    pathlib.Path(os.path.join(destinationFolder, "labels")).mkdir(parents=True, exist_ok=True)
    block = 1024 
    
    print("正在加载分割模型...")
    model_seg = torchvision.models.segmentation.deeplabv3_resnet50(weights=None, aux_loss=False)
    model_seg.classifier = torchvision.models.segmentation.deeplabv3.DeepLabHead(2048, 1)
    
    if torch.cuda.is_available():
        model_seg = torch.nn.DataParallel(model_seg)
        model_seg.to(device)
        # 修复：weights_only=False
        checkpoint = torch.load(seg_weights_path, weights_only=False)
        try:
            model_seg.load_state_dict(checkpoint['state_dict'])
        except KeyError:
             model_seg.load_state_dict(checkpoint)
    else:
        # 修复：weights_only=False
        checkpoint = torch.load(seg_weights_path, map_location="cpu", weights_only=False)
        model_seg.load_state_dict(checkpoint)

    model_seg.eval()

    print("正在生成分割标签 (.npy)...")
    with torch.no_grad():
        for (x, f, i) in tqdm.tqdm(dataloader):
            x = x.to(device)
            y = np.concatenate([
                model_seg(x[i:(i + block), :, :, :])["out"].detach().cpu().numpy() 
                for i in range(0, x.shape[0], block)
            ]).astype(np.float16)
            
            start = 0
            for (filename, offset) in zip(f, i):
                np.save(os.path.join(destinationFolder, "labels", os.path.splitext(filename)[0]), y[start:(start + offset), 0, :, :])
                start += offset

# ==========================================
# 5. 可视化与心室大小计算
# ==========================================
print("--- 正在生成可视化结果 ---")

# 修复：移除了不支持的 'segmentation' 参数
dataloader = torch.utils.data.DataLoader(
    echonet.datasets.Echo(
        split="external_test", 
        external_test_location=videosFolder, 
        target_type=["Filename"], 
        length=None, 
        period=1
        # mean 和 std 在这里省略，保持原始像素值用于可视化可能更好，或者之后手动反归一化
        # 这里为了保持一致性，我们不传 mean/std，让它返回原始处理过的图像
    ),
    batch_size=1, 
    num_workers=8, 
    shuffle=False, 
    pin_memory=False
)

if not all(os.path.isfile(os.path.join(destinationFolder, "videos", f)) for f in dataloader.dataset.fnames):
    pathlib.Path(os.path.join(destinationFolder, "videos")).mkdir(parents=True, exist_ok=True)
    pathlib.Path(os.path.join(destinationFolder, "size")).mkdir(parents=True, exist_ok=True)
    
    with open(os.path.join(destinationFolder, "size.csv"), "w") as g:
        g.write("Filename,Frame,Size,ComputerSmall\n")
        
        for (x, filename) in tqdm.tqdm(dataloader):
            x = x.numpy()
            for i in range(len(filename)):
                # x 形状: (1, 3, Frames, H, W)
                img = x[i, :, :, :, :].copy()
                
                # 修复：手动加载对应的 .npy 分割文件
                mask_path = os.path.join(destinationFolder, "labels", os.path.splitext(filename[i])[0] + ".npy")
                if os.path.exists(mask_path):
                    logit = np.load(mask_path) # 形状通常为 (Frames, H, W)
                else:
                    print(f"Warning: Mask not found for {filename[i]}")
                    logit = np.zeros((img.shape[1], img.shape[2], img.shape[3]))

                # 将图像转为灰度 (复制红色通道到其他通道)
                img[1, :, :, :] = img[0, :, :, :]
                img[2, :, :, :] = img[0, :, :, :]
                
                # 拼接图像用于显示：左边原图，右边加掩膜
                img = np.concatenate((img, img), 3)
                
                # 在右半部分的红色通道上叠加掩膜 (logit > 0)
                # img[0] 是红色通道，img[..., 112:] 是右半部分
                # 注意：这里假设图像宽度是 112，拼接后是 224。右半部分起始索引为 112。
                img[0, :, :, 112:] = np.maximum(255. * (logit > 0), img[0, :, :, 112:])
                
                # 添加黑色背景用于绘制曲线 (扩展宽度)
                img = np.concatenate((img, np.zeros_like(img)), 2)
                
                size = (logit > 0).sum(2).sum(1)
                
                try:
                    trim_min = sorted(size)[round(len(size) ** 0.05)]
                    trim_max = sorted(size)[round(len(size) ** 0.95)]
                    trim_range = trim_max - trim_min
                    peaks = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
                except Exception as e:
                    # print(f"计算峰值出错 {filename[i]}: {e}") # 减少刷屏
                    peaks = set()

                for (frame_idx, s_val) in enumerate(size):
                    g.write("{},{},{},{}\n".format(filename[i], frame_idx, s_val, 1 if frame_idx in peaks else 0))

                # 绘制图表
                fig = plt.figure(figsize=(size.shape[0] / 50 * 1.5, 3))
                plt.scatter(np.arange(size.shape[0]) / 50, size, s=1)
                ylim = plt.ylim()
                for p in peaks:
                    plt.plot(np.array([p, p]) / 50, ylim, linewidth=1)
                plt.ylim(ylim)
                plt.title(os.path.splitext(filename[i])[0])
                plt.xlabel("Seconds")
                plt.ylabel("Size (pixels)")
                plt.tight_layout()
                plt.savefig(os.path.join(destinationFolder, "size", os.path.splitext(filename[i])[0] + ".pdf"))
                plt.close(fig)

                # 处理视频：添加时间轴进度条
                size -= size.min()
                if size.max() != 0:
                    size = size / size.max()
                size = 1 - size
                
                for (frame_idx, y_val) in enumerate(size):
                    # 绘制游标
                    img[:, :, int(round(115 + 100 * y_val)), int(round(frame_idx / len(size) * 200 + 10))] = 255.
                    
                    interval = np.array([-3, -2, -1, 0, 1, 2, 3])
                    for a in interval:
                        for b in interval:
                            # 绘制点
                            y_pos = int(round(115 + 100 * y_val)) + a
                            x_pos = int(round(frame_idx / len(size) * 200 + 10)) + b
                            
                            # 边界检查
                            if 0 <= y_pos < img.shape[2] and 0 <= x_pos < img.shape[3]:
                                img[:, frame_idx, y_pos, x_pos] = 255.
                    
                    # 高亮显示峰值帧
                    if frame_idx in peaks:
                        x_base = int(round(frame_idx / len(size) * 200 + 10))
                        # 确保不越界
                        x_end = min(x_base + 25, img.shape[3])
                        x_start_rect = min(x_base, img.shape[3])
                        img[:, :, 200:225, x_start_rect:x_end] = 255.
                
                echonet.utils.savevideo(os.path.join(destinationFolder, "videos", filename[i]), img.astype(np.uint8), 50)

print("所有处理已完成。")