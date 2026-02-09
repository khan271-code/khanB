# -*- coding: utf-8 -*-
"""
ConvertDICOMToAVI.py
Author: David Ouyang
Description: Recursively convert DICOM files in a folder to AVI videos of fixed size.
"""

import os
import numpy as np
import pydicom as dicom
import cv2

# ------------------ User Config ------------------
input_folder = "/hpc/khan271/project_2025/dynamic/test_hk/p10002221"       # 输入 DICOM 文件根目录
destination_folder = "/hpc/khan271/project_2025/dynamic/test_hk/AVI" # 输出 AVI 文件目录
crop_size = (112, 112)                   # 输出视频尺寸
# -------------------------------------------------

# 自动创建输出目录
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

def mask(output):
    """Mask pixels outside the scanning sector."""
    dimension = output.shape[0]
    m1, m2 = np.meshgrid(np.arange(dimension), np.arange(dimension))
    mask_array = ((m1 + m2) > int(dimension / 2) + int(dimension / 10))
    mask_array *= ((m1 - m2) < int(dimension / 2) + int(dimension / 10))
    mask_array = np.reshape(mask_array, (dimension, dimension)).astype(np.int8)
    masked_image = cv2.bitwise_and(output, output, mask=mask_array)
    return masked_image

def make_video(file_path, destination_folder):
    """Convert a single DICOM file to AVI."""
    try:
        file_name = os.path.basename(file_path)
        video_filename = os.path.join(destination_folder, file_name + '.avi')

        if not os.path.exists(video_filename):
            dataset = dicom.dcmread(file_path, force=True)
            testarray = dataset.pixel_array

            # Crop zero-padding from first frame
            frame0 = testarray[0]
            mean_line = np.mean(np.mean(frame0, axis=1), axis=1)
            y_crop = np.where(mean_line < 1)[0][0]
            testarray = testarray[:, y_crop:, :, :]

            # Center crop to square
            bias = int(abs(testarray.shape[2] - testarray.shape[1]) / 2)
            if bias > 0:
                if testarray.shape[1] < testarray.shape[2]:
                    testarray = testarray[:, :, bias:-bias, :]
                else:
                    testarray = testarray[:, bias:-bias, :, :]

            frames, height, width, channels = testarray.shape

            # Get FPS from DICOM
            fps = 30
            try:
                fps = dataset[(0x18, 0x40)].value
            except:
                print("FPS not found, defaulting to 30")

            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            out = cv2.VideoWriter(video_filename, fourcc, fps, crop_size)

            for i in range(frames):
                output_a = testarray[i, :, :, 0]
                small_output = output_a[int(height / 10):(height - int(height / 10)),
                                        int(height / 10):(height - int(height / 10))]

                output_resized = cv2.resize(small_output, crop_size, interpolation=cv2.INTER_CUBIC)
                final_output = mask(output_resized)
                final_output = cv2.merge([final_output, final_output, final_output])
                out.write(final_output)

            out.release()
            print(f"Processed: {file_name}")
        else:
            print(f"{file_name} already processed")
    except Exception as e:
        print(f"Failed processing {file_path}: {e}")
    return 0

# ------------------ Main ------------------
count = 0

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.lower().endswith(".dcm"):  # 只处理 DICOM 文件
            count += 1
            file_path = os.path.join(root, file)
            print(count, file_path)
            make_video(file_path, destination_folder)

print(f"Total files processed: {count}")
