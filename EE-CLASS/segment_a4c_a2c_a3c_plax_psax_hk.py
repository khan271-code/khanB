# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import os
# --- GPU Configuration ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
# -------------------------
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from util import *
from scipy.misc import imresize
from skimage.color import rgb2gray, gray2rgb

# 假设你将上面的工具代码保存为了 echoanalysis_tools_hk.py
# 如果保存为 echoanalysis_tools.py，请修改下行导入
from echoanalysis_tools_hk import create_imgdict_from_dicom

import numpy as np
import shutil 
import sys 

class Unet(object):
    def __init__(self, mean, weight_decay, learning_rate, label_dim, maxout = False):
        self.x_train = tf.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_train = tf.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_test = tf.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.unet(self.x_train, mean)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.output, labels = self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.unet(self.x_test, mean, keep_prob = 1.0, reuse = True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
    
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob = 0.5, reuse = None):
        width = 1
        weight_decay = 1e-12
        label_dim = self.label_dim
        with tf.variable_scope('vgg', reuse=reuse):
            input = input - mean
            pool_ = lambda x: max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding = 'SAME', relu = True, filter_size = 3: conv(x, filter_size, output_depth, 1, weight_decay, name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name: deconv(x, 2, output_depth, 2, weight_decay, name=name)
            fc_ = lambda x, features, name, relu = True: fc(x, features, weight_decay, name, relu)
            
            conv_1_1 = conv_(input, int(64*width), 'conv1_1')
            conv_1_2 = conv_(conv_1_1, int(64*width), 'conv1_2')
            
            pool_1 = pool_(conv_1_2)

            conv_2_1 = conv_(pool_1, int(128*width), 'conv2_1')
            conv_2_2 = conv_(conv_2_1, int(128*width), 'conv2_2')
            
            pool_2 = pool_(conv_2_2)

            conv_3_1 = conv_(pool_2, int(256*width), 'conv3_1')
            conv_3_2 = conv_(conv_3_1, int(256*width), 'conv3_2')

            pool_3 = pool_(conv_3_2)

            conv_4_1 = conv_(pool_3, int(512*width), 'conv4_1')
            conv_4_2 = conv_(conv_4_1, int(512*width), 'conv4_2')

            pool_4 = pool_(conv_4_2)

            conv_5_1 = conv_(pool_4, int(1024*width), 'conv5_1')
            conv_5_2 = conv_(conv_5_1, int(1024*width), 'conv5_2')

            pool_5 = pool_(conv_5_2)

            conv_6_1 = tf.nn.dropout(conv_(pool_5, int(2048*width), 'conv6_1'), keep_prob)
            conv_6_2 = tf.nn.dropout(conv_(conv_6_1, int(2048*width), 'conv6_2'), keep_prob)
            
            up_7 = tf.concat([deconv_(conv_6_2, int(1024*width), 'up7'), conv_5_2], 3)
            
            conv_7_1 = conv_(up_7, int(1024*width), 'conv7_1')
            conv_7_2 = conv_(conv_7_1, int(1024*width), 'conv7_2')
            
            up_8 = tf.concat([deconv_(conv_7_2, int(512*width), 'up8'), conv_4_2], 3)
            
            conv_8_1 = conv_(up_8, int(512*width), 'conv8_1')
            conv_8_2 = conv_(conv_8_1, int(512*width), 'conv8_2')
            
            up_9 = tf.concat([deconv_(conv_8_2, int(256*width), 'up9'), conv_3_2], 3)
            
            conv_9_1 = conv_(up_9,int(256*width), 'conv9_1')
            conv_9_2 = conv_(conv_9_1, int(256*width), 'conv9_2')

            up_10 = tf.concat([deconv_(conv_9_2, int(128*width), 'up10'), conv_2_2], 3)
            
            conv_10_1 = conv_(up_10, int(128*width), 'conv10_1')
            conv_10_2 = conv_(conv_10_1, int(128*width), 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, int(64*width), 'up11'), conv_1_2], 3)
            
            conv_11_1 = conv_(up_11, int(64*width), 'conv11_1')
            conv_11_2 = conv_(conv_11_1, int(64*width), 'conv11_2')
            
            conv_12 = conv_(conv_11_2, label_dim, 'conv12_2', filter_size = 1, relu = False)
            return conv_12

def extract_areas(segs):
    areas = []
    for seg in segs:
        area = len(np.where(seg > 0)[0])
        areas.append(area)
    return areas

def segmentChamber(videofile, dicomdir, view, base_output_dir):
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    
    # 根据视图选择 Label Dimension
    label_map = {
        "a4c": 6, "a2c": 4, "a3c": 4, "psax": 4, "plax": 7
    }
    
    if view not in label_map:
        print("Unknown view:", view)
        return 0
        
    label_dim = label_map[view]
    
    # 构建 Graph 和 Session
    g = tf.Graph()
    with g.as_default():
        sess = tf.Session()
        model = Unet(mean, weight_decay, learning_rate, label_dim, maxout=maxout)
        sess.run(tf.local_variables_initializer())
        saver = tf.train.Saver()
        
        # 模型路径映射
        model_paths = {
            "a4c": '/hpc/khan271/project_2025/echo_class/echocv/models/a4c_45_20_all_model.ckpt-9000',
            "a2c": '/hpc/khan271/project_2025/echo_class/echocv/models/a2c_45_20_all_model.ckpt-10600',
            "a3c": '/hpc/khan271/project_2025/echo_class/echocv/models/a3c_45_20_all_model.ckpt-10500',
            "psax": '/hpc/khan271/project_2025/echo_class/echocv/models/psax_45_20_all_model.ckpt-9300',
            "plax": '/hpc/khan271/project_2025/echo_class/echocv/models/plax_45_20_all_model.ckpt-9600'
        }
        
        if view in model_paths:
            try:
                saver.restore(sess, model_paths[view])
            except Exception as e:
                print("Error loading model for {}: {}".format(view, e))
                return 0

    outpath = os.path.join(base_output_dir, view)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    # --- 调用无 GDCM 的图像提取 ---
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    
    if framedict is None or len(framedict) == 0:
        print("警告: 无法从 {} 提取图像帧 (可能文件损坏或无法解码)。".format(videofile))
        return 0

    images, orig_images = extract_images(framedict)
    
    if len(images) == 0:
        print("警告: 提取的图像列表为空。")
        return 0

    all_lv_segs = []

    # 执行预测
    if view == "a4c":
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 4, 1)
        all_lv_segs = a4c_lv_segs
        np.save(outpath + '/' + videofile + '_lv', np.array(a4c_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(a4c_la_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lvo', np.array(a4c_lvo_segs).astype('uint8'))
    elif view == "a2c":
        a2c_lv_segs, a2c_la_segs, a2c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 3, 1)
        all_lv_segs = a2c_lv_segs
        np.save(outpath + '/' + videofile + '_lv', np.array(a2c_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(a2c_la_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lvo', np.array(a2c_lvo_segs).astype('uint8'))
    elif view == "psax":
        psax_lv_segs, psax_lvo_segs, psax_rv_segs, preds = extract_segs(images, orig_images, model, sess, 2, 1, 3)
        all_lv_segs = psax_lv_segs
        np.save(outpath + '/' + videofile + '_lv', np.array(psax_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lvo', np.array(psax_lvo_segs).astype('uint8'))
    elif view == "a3c":
        a3c_lv_segs, a3c_la_segs, a3c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 3, 1)
        all_lv_segs = a3c_lv_segs
        np.save(outpath + '/' + videofile + '_lvo', np.array(a3c_lvo_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_lv', np.array(a3c_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(a3c_la_segs).astype('uint8'))
    elif view == "plax":
        plax_lv_segs, plax_la_segs, plax_ao_segs, preds = extract_segs(images, orig_images, model, sess, 1, 5, 3)
        all_lv_segs = plax_lv_segs
        np.save(outpath + '/' + videofile + '_lv', np.array(plax_lv_segs).astype('uint8'))
        np.save(outpath + '/' + videofile + '_la', np.array(plax_la_segs).astype('uint8'))

    # 查找并保存 ED 帧
    try:
        if all_lv_segs and len(orig_images) > 0:
            lv_areas = extract_areas(all_lv_segs)
            if lv_areas:
                ed_frame_index = np.argmax(lv_areas)
                # 边界检查
                if ed_frame_index < len(orig_images):
                    ed_frame_image = orig_images[ed_frame_index]
                    ed_image_pil = Image.fromarray(ed_frame_image.astype('uint8'))
                    save_path = os.path.join(outpath, "{}_ED_frame.png".format(videofile))
                    ed_image_pil.save(save_path)
                    print("成功保存 ED 帧到: {}".format(save_path))
                else:
                    print("错误: ED 帧索引超出图像数组范围。")
            else:
                print("警告: 无法计算 {} 的 LV 面积。".format(videofile))
        else:
            print("警告: 未找到 {} 的 LV 分割或原始图像。".format(videofile))
    except Exception as e:
        print("保存 {} 的 ED 帧时出错: {}".format(videofile, e))

    sess.close()
    return 1

def create_seg(output, label):
    output = output.copy()
    output[output != label] = -1
    output[output == label] = 1
    output[output == -1] = 0
    return output

def extract_images(framedict):
    images = []
    orig_images = []
    # framedict 是 {index: image_array}，需要排序确保顺序
    keys = sorted(framedict.keys())
    for key in keys:
        img = framedict[key]
        # 确保是灰度
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_gray = rgb2gray(img)
        else:
            img_gray = img
            
        image = np.zeros((384,384))
        image[:,:] = imresize(img_gray, (384,384,1)) # resize 会自动归一化或转换类型，需注意
        images.append(image)
        orig_images.append(img)
        
    images = np.array(images).reshape((len(images), 384,384,1))
    return images, orig_images

def extract_segs(images, orig_images, model, sess, lv_label, la_label, lvo_label):
    segs = []
    # 处理单帧
    if len(images) > 0:
        preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    else:
        preds = []
        
    label_all = range(1, 8)
    label_good = [lv_label, la_label, lvo_label]
    
    # Masking logic for first frame preview (preds)
    if len(preds) > 0:
        for i in label_all:
            if not i in label_good:
                preds[preds == i] = 0
            
    # Batch prediction logic could be added here for speed, but keeping original loop
    for i in range(len(images)):
        seg = np.argmax(model.predict(sess, images[i:i+1])[0,:,:,:], 2)
        segs.append(seg)
        
    lv_segs = []
    lvo_segs = []
    la_segs = []
    for seg in segs:
        la_seg = create_seg(seg, la_label)
        lvo_seg = create_seg(seg, lvo_label)
        lv_seg = create_seg(seg, lv_label)
        lv_segs.append(lv_seg)
        lvo_segs.append(lvo_seg)
        la_segs.append(la_seg)
    return lv_segs, la_segs, lvo_segs, preds

def main():
    # --- 用户配置 ---
    dicom_directory = "/hpc/khan271/project_2025/echo_class/echocv/test_hk/class/a2c" 
    output_directory = "/hpc/khan271/project_2025/echo_class/echocv/test_hk/class/a2c"
    view_to_use = "a2c"
    # ---------------

    valid_views = ["a4c", "a2c", "a3c", "psax", "plax"]
    if view_to_use not in valid_views:
        print("错误: 指定的视图类型 '{}' 无效。".format(view_to_use))
        sys.exit(1)

    print("--- 正在扫描 '{}' ---".format(dicom_directory))
    print("--- 视图: '{}' ---".format(view_to_use))

    processed_count = 0
    
    for root, dirs, files in os.walk(dicom_directory):
        for filename in files:
            if not filename.lower().endswith(".dcm"):
                continue

            dicom_file_directory = root
            full_path = os.path.join(root, filename)
            
            print("\n" + "="*40)
            print("Processing: {}".format(full_path))
            
            try:
                res = segmentChamber(filename, dicom_file_directory, view_to_use, output_directory)
                if res == 1:
                    processed_count += 1
            except Exception as e:
                print("!! Error processing {}: {}".format(filename, e))
                import traceback
                traceback.print_exc()

    print("=" * 40)
    print("Done. Processed {} files.".format(processed_count))

if __name__ == '__main__':
    main()