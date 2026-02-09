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
    checkpoint = torch.load(ef_weights_path, weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
else:
    print("CUDA 不可用，使用 CPU 进行计算。")
    device = torch.device("cpu")
    checkpoint = torch.load(ef_weights_path, map_location="cpu", weights_only=False)
    state_dict_cpu = {k[7:]: v for (k, v) in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict_cpu)

output_csv = os.path.join(destinationFolder, "cedars_ef_output.csv")

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

ds = echonet.datasets.Echo(split="external_test", external_test_location=videosFolder, **kwargs)

test_dataloader = torch.utils.data.DataLoader(
    ds, batch_size=1, num_workers=5, shuffle=True, pin_memory=(device.type == "cuda")
)

print("正在运行 EF 预测...")
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
        checkpoint = torch.load(seg_weights_path, weights_only=False)
        try:
            model_seg.load_state_dict(checkpoint['state_dict'])
        except KeyError:
             model_seg.load_state_dict(checkpoint)
    else:
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
# 5. 可视化与保存 ED/ES 关键帧
# ==========================================
print("--- 正在生成可视化结果并保存 ED/ES 关键帧 ---")

# 1. 创建 ED 和 ES 的保存文件夹
ed_folder = os.path.join(destinationFolder, "ed")
es_folder = os.path.join(destinationFolder, "es")
pathlib.Path(ed_folder).mkdir(parents=True, exist_ok=True)
pathlib.Path(es_folder).mkdir(parents=True, exist_ok=True)

dataloader = torch.utils.data.DataLoader(
    echonet.datasets.Echo(
        split="external_test", 
        external_test_location=videosFolder, 
        target_type=["Filename"], 
        length=None, 
        period=1
    ),
    batch_size=1, 
    num_workers=8, 
    shuffle=False, 
    pin_memory=False
)

# 注意：这里我们移除了 check，因为即便 video 已经存在，我们现在也需要运行它来提取 ED/ES 图片
# if not all(os.path.isfile(os.path.join(destinationFolder, "videos", f)) for f in dataloader.dataset.fnames):

pathlib.Path(os.path.join(destinationFolder, "videos")).mkdir(parents=True, exist_ok=True)
pathlib.Path(os.path.join(destinationFolder, "size")).mkdir(parents=True, exist_ok=True)

with open(os.path.join(destinationFolder, "size.csv"), "w") as g:
    g.write("Filename,Frame,Size,ComputerSmall\n")
    
    for (x, filename) in tqdm.tqdm(dataloader):
        x = x.numpy()
        for i in range(len(filename)):
            # x 形状: (1, 3, Frames, H, W)
            # 这里的 img 包含了原始像素信息 (通常是3通道, 虽然是灰度图)
            img = x[i, :, :, :, :].copy()
            
            # 手动加载 .npy 分割掩膜
            mask_path = os.path.join(destinationFolder, "labels", os.path.splitext(filename[i])[0] + ".npy")
            if os.path.exists(mask_path):
                logit = np.load(mask_path) # 形状: (Frames, H, W)
            else:
                logit = np.zeros((img.shape[1], img.shape[2], img.shape[3]))

            # 计算心室大小
            size = (logit > 0).sum(2).sum(1)
            
            # ==========================================
            # 新增功能: 提取并保存 ED 和 ES 帧
            # ==========================================
            if len(size) > 0:
                # ED (舒张末期) = 面积最大值索引
                ed_frame_idx = np.argmax(size)
                # ES (收缩末期) = 面积最小值索引
                es_frame_idx = np.argmin(size)
                
                # 提取对应的图像帧 (通道0即可，因为超声通常是灰度的)
                # 形状从 (3, Frames, H, W) 取出 (H, W)
                img_ed_raw = x[i, 0, ed_frame_idx, :, :]
                img_es_raw = x[i, 0, es_frame_idx, :, :]
                
                # 确保图像数据在 0-255 之间并转为 uint8 格式以便保存
                # 如果数据已经是 0-255，这步也没问题；如果是 0-1 或标准化过的，这步会归一化到可视范围
                def norm_to_uint8(im):
                    if im.max() == im.min(): return im.astype(np.uint8)
                    return ((im - im.min()) / (im.max() - im.min()) * 255).astype(np.uint8)

                img_ed_save = norm_to_uint8(img_ed_raw)
                img_es_save = norm_to_uint8(img_es_raw)

                # 保存图片
                # 文件名格式: 原文件名_ED.jpg / 原文件名_ES.jpg
                base_name = os.path.splitext(filename[i])[0]
                cv2.imwrite(os.path.join(ed_folder, f"{base_name}_ED.jpg"), img_ed_save)
                cv2.imwrite(os.path.join(es_folder, f"{base_name}_ES.jpg"), img_es_save)
            # ==========================================

            # 后续逻辑保持不变：生成视频和图表
            
            # 归一化/复制通道用于显示
            img[1, :, :, :] = img[0, :, :, :]
            img[2, :, :, :] = img[0, :, :, :]
            
            img = np.concatenate((img, img), 3)
            img[0, :, :, 112:] = np.maximum(255. * (logit > 0), img[0, :, :, 112:])
            img = np.concatenate((img, np.zeros_like(img)), 2)
            
            try:
                trim_min = sorted(size)[round(len(size) ** 0.05)]
                trim_max = sorted(size)[round(len(size) ** 0.95)]
                trim_range = trim_max - trim_min
                peaks = set(scipy.signal.find_peaks(-size, distance=20, prominence=(0.50 * trim_range))[0])
            except Exception as e:
                peaks = set()

            for (frame_idx, s_val) in enumerate(size):
                g.write("{},{},{},{}\n".format(filename[i], frame_idx, s_val, 1 if frame_idx in peaks else 0))

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

            size -= size.min()
            if size.max() != 0:
                size = size / size.max()
            size = 1 - size
            
            for (frame_idx, y_val) in enumerate(size):
                img[:, :, int(round(115 + 100 * y_val)), int(round(frame_idx / len(size) * 200 + 10))] = 255.
                interval = np.array([-3, -2, -1, 0, 1, 2, 3])
                for a in interval:
                    for b in interval:
                        y_pos = int(round(115 + 100 * y_val)) + a
                        x_pos = int(round(frame_idx / len(size) * 200 + 10)) + b
                        if 0 <= y_pos < img.shape[2] and 0 <= x_pos < img.shape[3]:
                            img[:, frame_idx, y_pos, x_pos] = 255.
                if frame_idx in peaks:
                    x_base = int(round(frame_idx / len(size) * 200 + 10))
                    x_end = min(x_base + 25, img.shape[3])
                    x_start_rect = min(x_base, img.shape[3])
                    img[:, :, 200:225, x_start_rect:x_end] = 255.
            
            echonet.utils.savevideo(os.path.join(destinationFolder, "videos", filename[i]), img.astype(np.uint8), 50)

print("所有处理已完成。")