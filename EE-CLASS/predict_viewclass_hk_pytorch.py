# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import pydicom
import shutil
import sys
import csv
import time
import warnings
import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F
import tensorflow as tf 

# 屏蔽无关警告
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# ================= 配置区域 =================
base_project_dir = "/hpc/khan271/project_2025/echo_class/echocv"
dicom_input_dir = os.path.join(base_project_dir, "test_hk")
model_name_str = "view_23_e5_class_11-Mar-2018"
tf_checkpoint_path = os.path.join(base_project_dir, 'models', model_name_str)
viewclasses_file = os.path.join(base_project_dir, "viewclasses_" + model_name_str + ".txt")
output_txt_dir = os.path.join(dicom_input_dir, "txt")
temp_image_root = os.path.join(base_project_dir, "temp_images")

# GPU 设置
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# ===========================================

# ---------------- 1. 模型定义 ----------------
class EchoVGG(nn.Module):
    def __init__(self, num_classes):
        super(EchoVGG, self).__init__()
        self.vgg = models.vgg16(weights=None)
        num_ftrs = self.vgg.classifier[6].in_features
        self.vgg.classifier[6] = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.vgg(x)

# ---------------- 2. 加载 TF 权重 ----------------
def load_tf_weights_into_pytorch(pytorch_model, tf_ckpt_path):
    print(f"[Loader] Reading TensorFlow weights from: {tf_ckpt_path}")
    try:
        reader = tf.train.load_checkpoint(tf_ckpt_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
    except Exception as e:
        print(f"Error reading TF checkpoint: {e}")
        sys.exit(1)

    layer_mapping = [
        ('network/conv1_1', 'vgg.features.0'),
        ('network/conv1_2', 'vgg.features.2'),
        ('network/conv2_1', 'vgg.features.5'),
        ('network/conv2_2', 'vgg.features.7'),
        ('network/conv3_1', 'vgg.features.10'),
        ('network/conv3_2', 'vgg.features.12'),
        ('network/conv3_3', 'vgg.features.14'),
        ('network/conv4_1', 'vgg.features.17'),
        ('network/conv4_2', 'vgg.features.19'),
        ('network/conv4_3', 'vgg.features.21'),
        ('network/conv5_1', 'vgg.features.24'),
        ('network/conv5_2', 'vgg.features.26'),
        ('network/conv5_3', 'vgg.features.28'),
        ('network/fc6', 'vgg.classifier.0'),
        ('network/fc7', 'vgg.classifier.3'),
        ('network/fc8', 'vgg.classifier.6'),
    ]

    new_state_dict = pytorch_model.state_dict()
    loaded_count = 0

    for tf_prefix, pt_prefix in layer_mapping:
        tf_w_name = tf_prefix + '/W'  
        pt_w_name = pt_prefix + '.weight'
        if tf_w_name not in var_to_shape_map: tf_w_name = tf_prefix + '/weights'

        if tf_w_name in var_to_shape_map:
            w = reader.get_tensor(tf_w_name)
            if len(w.shape) == 4: w = np.transpose(w, (3, 2, 0, 1))
            elif len(w.shape) == 2: w = np.transpose(w, (1, 0))
            new_state_dict[pt_w_name] = torch.from_numpy(w)
            loaded_count += 1
        
        tf_b_name = tf_prefix + '/b'
        pt_b_name = pt_prefix + '.bias'
        if tf_b_name not in var_to_shape_map: tf_b_name = tf_prefix + '/biases'
            
        if tf_b_name in var_to_shape_map:
            b = reader.get_tensor(tf_b_name)
            new_state_dict[pt_b_name] = torch.from_numpy(b)

    pytorch_model.load_state_dict(new_state_dict)
    print(f"[Loader] Transferred {loaded_count} layers.")
    return pytorch_model

# ---------------- 3. DICOM 提取 (核心修改：加入高对比度归一化) ----------------
def extract_imgs_from_dicom(dicom_files, out_directory):
    if not os.path.exists(out_directory): os.makedirs(out_directory)
    jpg_to_dicom_map = {}
    total_files = len(dicom_files)
    print(f"Extracting {total_files} DICOMs with high contrast grayscale...")
    
    for idx, filepath in enumerate(dicom_files):
        if idx % 10 == 0:
            sys.stdout.write(f"\r[Pre-process] {idx}/{total_files}")
            sys.stdout.flush()
        
        filename = os.path.basename(filepath)
        try:
            ds = pydicom.dcmread(filepath)
            is_color = False
            if 'PhotometricInterpretation' in ds:
                if ds.PhotometricInterpretation in ['RGB', 'YBR_FULL', 'YBR_FULL_422']:
                    is_color = True

            imgs_to_save = []
            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                count = min(10, ds.NumberOfFrames)
                for n in range(count):
                    # 注意：这里取出的 pixel_array 可能是原始的高位深数据（如16bit）
                    imgs_to_save.append((ds.pixel_array[n], f"{filename}_{n}.jpg"))
            else:
                imgs_to_save.append((ds.pixel_array, f"{filename}.jpg"))
            
            for img_arr, save_name in imgs_to_save:
                # --- 步骤1: 确保转为单通道灰度图 ---
                if img_arr.ndim == 3:
                    if is_color:
                        if ds.PhotometricInterpretation.startswith('YBR'):
                            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_YCrCb2BGR)
                        else:
                            # 假设是RGB
                            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
                    # 强制转为灰度
                    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                
                # --- 步骤2: [核心修改] 高对比度归一化 (Min-Max Normalization) ---
                # 此时 img_arr 是 2D 灰度数组，但数值范围可能很窄（例如 50-150），导致看起来灰蒙蒙。
                # 我们要把它拉伸到 0-255。
                
                img_float = img_arr.astype(np.float32)
                min_val = img_float.min()
                max_val = img_float.max()

                # 防止除以零（如果图像全是纯黑或纯色）
                if max_val - min_val > 1e-5:
                    # 拉伸公式：(像素值 - 最小值) / (最大值 - 最小值) * 255
                    normalized_img = (img_float - min_val) / (max_val - min_val) * 255.0
                else:
                    normalized_img = img_float - min_val # 简单的平移，或者全黑

                # 转回 8-bit 无符号整数，准备保存为 JPG
                final_gray_img = normalized_img.astype(np.uint8)
                
                # --- 步骤3: 缩放并保存 ---
                img_resized = cv2.resize(final_gray_img, (224, 224))
                
                out_path = os.path.join(out_directory, save_name)
                # cv2.imwrite 接收 uint8 的单通道数组，会保存为完美的黑白灰JPG
                cv2.imwrite(out_path, img_resized, [cv2.IMWRITE_JPEG_QUALITY, 98])
                jpg_to_dicom_map[save_name] = filepath
                
        except Exception as e:
            # print(f"Error processing {filename}: {e}")
            continue
    print("\nExtraction complete.")
    return jpg_to_dicom_map

# ---------------- 4. 推理主循环 ----------------
def classify_workflow(image_dir, views, model):
    predictions = {}
    model.eval()
    model.to(device)
    
    img_files = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]
    print(f"Starting inference on {len(img_files)} frames...")
    
    BATCH_SIZE = 32
    with torch.no_grad():
        for i in range(0, len(img_files), BATCH_SIZE):
            batch_files = img_files[i:i+BATCH_SIZE]
            batch_tensors = []
            valid_batch_names = []
            
            for fname in batch_files:
                fpath = os.path.join(image_dir, fname)
                # cv2.imread 读取黑白JPG也会默认复制成3通道BGR，符合模型输入
                img = cv2.imread(fpath) 
                if img is None: continue
                
                t = torch.from_numpy(img).float() 
                t = t.permute(2, 0, 1) 
                
                batch_tensors.append(t)
                valid_batch_names.append(fname)
            
            if not batch_tensors: continue
            
            input_tensor = torch.stack(batch_tensors).to(device)
            
            # VGG 预处理: 减去均值
            bgr_mean = torch.tensor([103.939, 116.779, 123.68]).view(1, 3, 1, 1).to(device)
            input_tensor = input_tensor - bgr_mean
            
            outputs = model(input_tensor)
            probs = F.softmax(outputs, dim=1)
            probs_np = probs.cpu().numpy()
            
            if i == 0:
                # 这里的Mean可能会变小，因为我们拉大了对比度，黑色区域更多了
                print(f"[DEBUG Input] Min: {input_tensor.min():.2f}, Max: {input_tensor.max():.2f}, Mean: {input_tensor.mean():.2f}")

            for idx, name in enumerate(valid_batch_names):
                predictions[name] = np.around(probs_np[idx], decimals=4)
                
            if i % 100 == 0:
                sys.stdout.write(f"\r[Inference] {i}/{len(img_files)}")
                sys.stdout.flush()
                
    print("\nInference complete.")
    return predictions

# ---------------- 主函数 ----------------
def main():
    if not os.path.exists(output_txt_dir): os.makedirs(output_txt_dir)
    
    if os.path.exists(temp_image_root):
        shutil.rmtree(temp_image_root) 
    os.makedirs(temp_image_root)

    # 使用新的文件夹名提示这是高对比度版
    temp_dir = os.path.join(temp_image_root, "run_high_contrast_gray") 
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)

    try:
        with open(viewclasses_file, 'r') as f:
            views = [line.strip() for line in f.readlines() if line.strip()]
    except:
        views = [f"class_{i}" for i in range(23)]
    
    print(f"Classes detected: {len(views)}")

    model = EchoVGG(num_classes=len(views))
    model = load_tf_weights_into_pytorch(model, tf_checkpoint_path)

    all_dcm = []
    for r, d, f in os.walk(dicom_input_dir):
        for file in f:
            if file.lower().endswith('.dcm'):
                all_dcm.append(os.path.join(r, file))
    
    if not all_dcm:
        print("No dicom files found.")
        return

    jpg_map = extract_imgs_from_dicom(all_dcm, temp_dir)
    raw_preds = classify_workflow(temp_dir, views, model)

    dicom_probs = {}
    for jpg, prob in raw_preds.items():
        if jpg in jpg_map:
            dcm = jpg_map[jpg]
            if dcm not in dicom_probs: dicom_probs[dcm] = []
            dicom_probs[dcm].append(prob)

    out_csv = os.path.join(output_txt_dir, f"{model_name_str}_results_HC_GRAY.csv")
    print(f"Writing results to {out_csv}...")
    
    header = ["study", "image"] + [f"prob_{v}" for v in views] + ["Predicted_Class"]
    
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for dcm_path in sorted(dicom_probs.keys()):
            if not dicom_probs[dcm_path]: continue
            avg_prob = np.mean(dicom_probs[dcm_path], axis=0)
            pred_idx = np.argmax(avg_prob) 
            pred_label = views[pred_idx]   
            
            study = os.path.basename(os.path.dirname(dcm_path))
            fname = os.path.basename(dcm_path)
            row = [study, fname] + [f"{p:.4f}" for p in avg_prob] + [pred_label]
            writer.writerow(row)
            
    print(f"\nAll Done. High contrast gray images are saved in: {temp_dir}")
    print("Please check the images, they should look like standard sharp black & white echocardiograms now.")

if __name__ == '__main__':
    main()