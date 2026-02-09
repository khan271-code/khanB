# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import cv2
import os
import pydicom as dicom
from shutil import rmtree
# from scipy.misc import imread  <-- 已弃用, 不再需要
import time
import sys
import csv # (为了与您之前的请求保持一致，导入csv)

# --- 修复 ImportError ---
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, 'funcs'))
sys.path.append(os.path.join(script_dir, 'nets'))
# -------------------------

import vgg as network
from echoanalysis_tools import output_imgdict

# ---------------- 配置参数 ----------------
# 顶层 DICOM 目录
base_dir = "/hpc/khan271/project_2025/echo_class/echocv/test_hk"

# 模型名称（models 文件夹下）
model_name_str = "view_23_e5_class_11-Mar-2018"
model_path = os.path.join(script_dir, 'models', model_name_str)

# 输出 TXT 文件夹
output_txt_dir = "/hpc/khan271/project_2025/echo_class/echocv/test_hk/txt"
if not os.path.exists(output_txt_dir):
    os.makedirs(output_txt_dir)

# 临时 image 文件夹
temp_image_root = "/hpc/khan271/project_2025/echo_class/echocv/temp_images"
if os.path.exists(temp_image_root):
    rmtree(temp_image_root)
os.makedirs(temp_image_root)

# GPU 设置
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# ---------------- 加载 view classes ----------------
viewclasses_file = os.path.join(script_dir, "viewclasses_" + model_name_str + ".txt")
try:
    with open(viewclasses_file) as f:
        views = [line.strip() for line in f.readlines()]
except IOError:
    print("Error: Cannot find view classes file: {}".format(viewclasses_file))
    sys.exit(1)

feature_dim = 1
label_dim = len(views)

# ---------------- DICOM -> JPG (已修复) ----------------
def extract_imgs_from_dicom(dicom_files, out_directory):
    """
    dicom_files: list of DICOM文件完整路径
    out_directory: JPG输出文件夹
    (已修复YBR色彩空间问题)
    """
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    jpg_to_dicom_map = {} # (为了与您之前的请求保持一致，保留此功能)

    for filepath in dicom_files:
        filename = os.path.basename(filepath)
        try:
            ds = dicom.read_file(filepath)
            
            # 检查是否为 YBR 色彩空间 (导致发黄的原因)
            is_ybr = False
            if 'PhotometricInterpretation' in ds and ds.PhotometricInterpretation.startswith('YBR'):
                is_ybr = True

            if hasattr(ds, 'NumberOfFrames') and ds.NumberOfFrames > 1:
                num_frames = ds.NumberOfFrames
                for n in range(min(10, num_frames)):
                    img = ds.pixel_array[n]
                    
                    # --- 修复开始 ---
                    # 如果图像是3通道 (彩色的)
                    if img.ndim == 3:
                        if is_ybr:
                            # 如果是YBR，先转 BGR, 再转 灰度
                            img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                        else:
                            # 如果是 BGR/RGB, 直接转 灰度
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    # --- 修复结束 ---
                    
                    outfile_name = "{}_{}.jpg".format(filename, n)
                    outfile_path = os.path.join(out_directory, outfile_name)
                    cv2.imwrite(outfile_path, cv2.resize(img, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 95])
                    jpg_to_dicom_map[outfile_name] = filepath
            else:
                img = ds.pixel_array
                
                # --- 修复开始 (同样应用于单帧图像) ---
                if img.ndim == 3:
                    if is_ybr:
                        img = cv2.cvtColor(img, cv2.COLOR_YCrCb2BGR)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # --- 修复结束 ---
                
                outfile_name = "{}.jpg".format(filename)
                outfile_path = os.path.join(out_directory, outfile_name)
                cv2.imwrite(outfile_path, cv2.resize(img, (224, 224)), [cv2.IMWRITE_JPEG_QUALITY, 95])
                jpg_to_dicom_map[outfile_name] = filepath
        except Exception as e:
            print("Error reading {}: {}".format(filename, e))
            
    return jpg_to_dicom_map # (返回映射)

# ---------------- 分类函数 (已修复) ----------------
def classify(directory, feature_dim, label_dim, model_path):
    imagedict = {}
    predictions = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith("jpg"):
            
            # --- 修复：使用 cv2.imread 替换 scipy.misc.imread ---
            image_path = os.path.join(directory, filename)
            # 以灰度模式读取，这与原先的 flatten=True 效果一致
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
            
            if image is None:
                print("Warning: Could not read image {}, skipping.".format(image_path))
                continue
            # --- 修复结束 ---
            
            imagedict[filename] = [image.reshape((224,224,1))]

    tf.reset_default_graph()
    sess = tf.Session()
    model = network.Network(0.0, 0.0, feature_dim, label_dim, False)
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    for filename in imagedict:
        predictions[filename] = np.around(model.probabilities(sess, imagedict[filename]), decimals=3)
    
    return predictions

# ---------------- 主函数 (已修改为输出CSV和TXT) ----------------
def main():
    # 递归遍历 base_dir 下所有 .dcm 文件
    all_dcm_files = []
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower().endswith(".dcm"):
                all_dcm_files.append(os.path.join(root, file))

    print("Found {} DICOM files.".format(len(all_dcm_files)))
    if len(all_dcm_files) == 0:
        print("No DICOM files found, exiting.")
        return

    # 临时 image 文件夹
    temp_image_directory = os.path.join(temp_image_root, "current_run")
    if os.path.exists(temp_image_directory):
        rmtree(temp_image_directory)
    os.makedirs(temp_image_directory)

    start_time = time.time()

    # 提取图片
    print("Extracting images from DICOM files...")
    # 接收返回的映射
    jpg_to_dicom_map = extract_imgs_from_dicom(all_dcm_files, temp_image_directory)

    # 分类
    print("Classifying images...")
    predictions = classify(temp_image_directory, feature_dim, label_dim, model_path)

    # 聚合每个 DICOM 文件的预测
    predictprobdict = {}
    for image in predictions.keys(): # 'image' 是 JPG 基础文件名
        if image not in jpg_to_dicom_map:
            print("Warning: Skipping image with no DICOM mapping: {}".format(image))
            continue
        full_dicom_path = jpg_to_dicom_map[image] # 这是完整的路径

        if full_dicom_path not in predictprobdict:
            predictprobdict[full_dicom_path] = []
        predictprobdict[full_dicom_path].append(predictions[image][0])

    # --- 写入 CSV 和 TXT (根据您的新请求修改) ---
    
    # 1. 定义基础文件名
    base_out_filename = os.path.join(output_txt_dir, "{}_all_probabilities".format(model_name_str))
    csv_out_filename = base_out_filename + ".csv"
    txt_out_filename = base_out_filename + ".txt"
    
    print("Writing predictions to {} and {}...".format(csv_out_filename, txt_out_filename))
    
    # 2. 准备表头 (根据要求修改)
    header = ["study", "image"] # <-- 修改
    for v in views:
        header.append("prob_" + v)
        
    # 3. 准备所有数据行 (一次性准备，便于写入两个文件)
    all_rows = []
    for full_path in sorted(predictprobdict.keys()):
        predictprobmean = np.mean(predictprobdict[full_path], axis=0)
        
        # --- 拆分路径和文件名 ---
        study_path = os.path.dirname(full_path)
        image_name = os.path.basename(full_path)
        # ------------------------
        
        row_data = [study_path, image_name] # <-- 修改
        for i in predictprobmean:
            row_data.append(str(i))
        all_rows.append(row_data)

    # 4. 写入 CSV (兼容 Python 2.7)
    try:
        with open(csv_out_filename, 'wb') as f_csv:
            csv_writer = csv.writer(f_csv, delimiter=',')
            csv_writer.writerow(header)
            for row in all_rows:
                csv_writer.writerow(row)
    except Exception as e:
        print("Error writing CSV file: {}".format(e))

    # 5. 写入 TXT (使用制表符 '\t' 分隔)
    try:
        with open(txt_out_filename, 'wb') as f_txt:
            # 同样使用 csv.writer，但分隔符改为制表符
            txt_writer = csv.writer(f_txt, delimiter='\t')
            txt_writer.writerow(header)
            for row in all_rows:
                txt_writer.writerow(row)
    except Exception as e:
        print("Error writing TXT file: {}".format(e))
    # --- 修改结束 ---

    end_time = time.time()
    print("Processed {} DICOM files in {:.2f} seconds.".format(len(predictprobdict.keys()), end_time - start_time))

    # 清理临时文件夹

    print("Cleaning up temporary image directory...")
    rmtree(temp_image_directory)
    print("Done.")

if __name__ == '__main__':
    main()