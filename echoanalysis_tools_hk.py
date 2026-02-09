# -*- coding: utf-8 -*-
import sys
import os
import dicom 
import time
import numpy as np
from scipy.misc import imresize
import cv2
from PIL import Image
import io

# --- 辅助函数：尝试从 dataset 中提取标签 ---

def computehr_dicom(ds):
    try:
        if (0x0018, 0x1088) in ds:
            hr = ds[0x0018, 0x1088].value
            print("heart rate found: ", hr)
            return hr
    except Exception as e:
        pass
    return "None"

def computexy_dicom(ds):
    rows = int(ds.Rows)
    cols = int(ds.Columns)
    return rows, cols

def computebsa_dicom(ds):
    try:
        h = float(ds.PatientSize) 
        w = float(ds.PatientWeight) 
        return 0.20247 * (h**0.725) * (w**0.425)
    except Exception:
        return 0.0

def computedeltaxy_dicom(ds):
    xlist = []
    ylist = []
    try:
        if (0x0018, 0x602c) in ds:
            dx = ds[0x0018, 0x602c].value
            if np.abs(dx) > 0.012:
                xlist.append(np.abs(dx))
        if (0x0018, 0x602e) in ds:
            dy = ds[0x0018, 0x602e].value
            if np.abs(dy) > 0.012:
                ylist.append(np.abs(dy))
        
        if not xlist: return 0.0, 0.0
        return np.min(xlist), np.min(ylist)
    except:
        return 0.0, 0.0

def remove_periphery(imgs):
    imgs_ret = []
    for img in imgs:
        image = img.astype('uint8').copy()
        image[image > 0 ] = 255
        image = cv2.bilateralFilter(image, 11, 17, 17)
        thresh = cv2.threshold(image, 200, 255, cv2.THRESH_BINARY)[1]
        
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(cnts) == 2:
            contours = cnts[0]
        elif len(cnts) == 3:
            contours = cnts[1]
        else:
            contours = []

        areas = []
        for i in range(0, len(contours)):
            areas.append(cv2.contourArea(contours[i]))

        if len(areas) == 0:
            imgs_ret.append(img)
        else:
            select = np.argmax(areas)
            roi_corners_clean = []
            roi_corners = np.array(contours[select], dtype = np.int32)
            for i in roi_corners:
                roi_corners_clean.append(i[0])
            hull = cv2.convexHull(np.array([roi_corners_clean], dtype = np.int32))
            mask = np.zeros(image.shape, dtype=np.uint8)
            mask = cv2.fillConvexPoly(mask, hull, 1)
            imgs_ret.append(img*mask)
    return np.array(imgs_ret)

def computeft_dicom(ds):
    defaultframerate = 30.0
    try:
        if (0x0018, 0x1063) in ds:
            return float(ds[0x0018, 0x1063].value)
        elif (0x0018, 0x0040) in ds:
            rate = float(ds[0x0018, 0x0040].value)
            if rate > 0:
                return 1000.0 / rate
        
        if (0x7fdf, 0x1074) in ds: 
             rate = float(ds[0x7fdf, 0x1074].value)
             return 1000.0 / rate
             
    except Exception as e:
        print("Error computing FT:", e)
    
    return 1000.0 / defaultframerate

def ybr2gray(y, u, v):
    y = y.astype(float)
    u = u.astype(float)
    v = v.astype(float)
    r = y + 1.402 * (v - 128)
    g = y - 0.34414 * (u - 128) - 0.71414 * (v - 128)
    b = y + 1.772 * (u - 128)
    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b)
    return np.array(gray, dtype="uint8")

# --- 核心逻辑修复：手动解压函数 ---

def decompress_pixel_data(ds):
    """
    如果 pydicom 无法读取 compressed pixel data，
    尝试使用 PIL 手动处理 Sequence 中的 bytes。
    """
    try:
        import dicom.encaps
    except ImportError:
        pass # 旧版可能没有这个模块，我们手动处理

    pixel_data = ds.PixelData
    frames = []
    
    # 检查是否为多帧序列 (Encapsulated Format)
    # 在旧版 dicom 中，PixelData 可能是一个 list (Sequence) 或者 string
    
    # 尝试解析封装数据
    # Encapsulated pixel data 包含一个 Offset Table (第一项) 和后续的 Fragments
    
    # 如果是字符串（单帧或未解析的序列），直接尝试读取
    if isinstance(pixel_data, str) or isinstance(pixel_data, bytes):
        try:
            img = Image.open(io.BytesIO(pixel_data))
            frames.append(np.array(img))
        except Exception:
            # 可能是 raw sequence bytes，需要更复杂的解析，
            # 但旧版 dicom 通常会把 PixelData 读作 string
            pass
            
    # 如果是 List 或 Sequence (通常 pydicom 会这样处理 compressed data)
    # 注意：ds.PixelData 在某些版本直接返回二进制串，需要手动分割
    # 但在 'NotImplementedError' 的情况下，ds.PixelData 通常是原始的 Encapsulated data
    
    # 简单的 Hack：针对常见的 JPEG 压缩 DICOM
    # 我们可以尝试利用 pydicom 的私有函数，或者简单地尝试解析
    
    # 重新读取 PixelData 为 Sequence
    # 由于我们没有 gdcm，我们尝试直接把 ds.PixelData 当做 bytes 序列处理
    # 这是一个针对 Python 2.7 + Old Dicom 的强力尝试：
    
    raw_data = ds.PixelData
    
    # 如果 PIL 可以直接打开整个数据块（某些单帧情况）
    try:
        img = Image.open(io.BytesIO(raw_data))
        return np.array(img)[np.newaxis, ...] # 增加一维作为 frames
    except:
        pass

    # 多帧处理：这在没有 gdcm 的情况下很难完美，
    # 但通常 Echo 数据的每个 Fragment 都是一个 JPEG 图像。
    # 我们需要找到 JPEG 的头 (0xFFD8) 和尾 (0xFFD9)
    
    images = []
    start_tokens = [b'\xff\xd8']
    end_tokens = [b'\xff\xd9']
    
    # 在 raw_data 中搜索所有的 JPEG 流
    # 这是一个低效但无需 gdcm 的方法
    
    current_pos = 0
    while True:
        start_pos = raw_data.find(b'\xff\xd8', current_pos)
        if start_pos == -1:
            break
        
        end_pos = raw_data.find(b'\xff\xd9', start_pos)
        if end_pos == -1:
            break
            
        # 提取 potential jpeg
        jpg_data = raw_data[start_pos:end_pos+2]
        try:
            img = Image.open(io.BytesIO(jpg_data))
            img_arr = np.array(img)
            # 简单的校验，确保尺寸匹配
            if img_arr.shape[0] == ds.Rows and img_arr.shape[1] == ds.Columns:
                images.append(img_arr)
            
            current_pos = end_pos + 2
        except Exception:
            current_pos = start_pos + 2 # 尝试下一个
            continue
            
    if len(images) > 0:
        return np.array(images)
    
    return None

def output_imgdict(ds):
    '''
    converts dicom dataset to numpy arrays dictionary
    '''
    try:
        # 1. 尝试直接读取 (Uncompressed)
        try:
            raw_pixel_data = ds.pixel_array
        except NotImplementedError:
            # 2. 如果报错 (Compressed)，尝试手动解压
            # print("Trying manual decompression...")
            raw_pixel_data = decompress_pixel_data(ds)
            
        if raw_pixel_data is None:
            raise ValueError("Failed to extract pixel data (Compression not supported by PIL)")

        imgdict = {}
        nrow = int(ds.Rows)
        ncol = int(ds.Columns)
        
        shape = raw_pixel_data.shape
        
        # 归一化形状处理
        # 目标是将所有数据转为 list of frames，每个 frame 是 (Row, Col) 的灰度
        
        frames_list = []
        
        # Case A: (Frames, Rows, Cols, 3) - RGB/YBR
        if len(shape) == 4:
            for i in range(shape[0]):
                frame = raw_pixel_data[i]
                # 转灰度
                if shape[3] == 3:
                    # 检查是否需要 YBR 转换 (根据 PhotometricInterpretation)
                    pi = ds.get("PhotometricInterpretation", "RGB")
                    if "YBR" in pi:
                        # 假设是 YBR_FULL
                        gray = ybr2gray(frame[:,:,0], frame[:,:,1], frame[:,:,2])
                    else:
                        # RGB
                        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                else:
                    gray = frame[:,:,0] # 这种格式很少见，取第一通道
                frames_list.append(gray)
                
        # Case B: (Frames, 3, Rows, Cols) - Planar Config
        elif len(shape) == 4 and shape[1] == 3: 
             for i in range(shape[0]):
                frame = raw_pixel_data[i]
                pi = ds.get("PhotometricInterpretation", "RGB")
                if "YBR" in pi:
                    gray = ybr2gray(frame[0], frame[1], frame[2])
                else:
                    # Planar RGB to Gray
                    # manual conversion
                    r, g, b = frame[0].astype(float), frame[1].astype(float), frame[2].astype(float)
                    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                    gray = gray.astype('uint8')
                frames_list.append(gray)

        # Case C: (Frames, Rows, Cols) - Grayscale or (Rows, Cols, 3) Single Frame Color
        elif len(shape) == 3:
            # 区分是 多帧灰度 还是 单帧彩色
            if shape[2] == 3 and shape[0] == nrow: # 极可能是单帧彩色 (Rows, Cols, 3)
                # Single frame color
                frame = raw_pixel_data
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                frames_list.append(gray)
            else:
                # Multi-frame grayscale
                for i in range(shape[0]):
                    frames_list.append(raw_pixel_data[i])
                    
        # Case D: (Rows, Cols) - Single Frame Grayscale
        elif len(shape) == 2:
             frames_list.append(raw_pixel_data)
             
        # 保存到 imgdict 并 Resize
        for i, gray_frame in enumerate(frames_list):
            # 隐私遮挡
            gray_frame[0:int(nrow / 10), 0:int(ncol)] = 0
            gray_frame = np.clip(gray_frame, 0, 255)
            imgdict[i] = imresize(gray_frame, (nrow, ncol))
            
        return imgdict
        
    except Exception as e:
        # print("Error in output_imgdict: {}".format(e))
        return None

def create_mask(imgs):
    from scipy.ndimage.filters import gaussian_filter
    if len(imgs) < 2: return None
    diffs = []
    for i in range(len(imgs) - 1):
        temp = np.abs(imgs[i] - imgs[i + 1])
        temp = gaussian_filter(temp, 10)
        temp[temp <= 50] = 0
        temp[temp > 50] = 1
        diffs.append(temp)
    if not diffs: return None
    diff = np.mean(np.array(diffs), axis=0)
    diff[diff >= 0.5] = 1
    diff[diff < 0.5] = 0
    return diff

def create_imgdict_from_dicom(directory, filename):
    """
    Reads DICOM file directly using pydicom.
    """
    targetfile = os.path.join(directory, filename)
    if not os.path.exists(targetfile): return None
    try:
        ds = dicom.read_file(targetfile, force=True)
        if 'PixelData' not in ds: return None
        imgdict = output_imgdict(ds)
        return imgdict
    except Exception as e:
        print("Error reading DICOM file {}: {}".format(filename, e))
        return None