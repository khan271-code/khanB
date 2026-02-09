# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
from scipy.misc import imresize
from skimage.color import rgb2gray, gray2rgb
import os       # 添加 os 模块用于路径操作
import shutil   # 添加 shutil 模块
import numpy as np # 确保 numpy 已导入

# --- 新增的导入 ---
import pydicom
import pydicom.data

# 尝试导入解码器，以提供更友好的提示
try:
    import pylibjpeg
    import imagecodecs
except ImportError:
    print("="*80)
    print("警告：缺少 pylibjpeg 或 imagecodecs 库。")
    print("如果 DICOM 文件是压缩格式（如 JPEG2000, RLE, JPEG），读取将会失败。")
    print("请运行: pip install \"pydicom[all]\" 来安装所有依赖。")
    print("="*80)


# ======================================================================
# 来自 util.py 的 TensorFlow 1.x 辅助函数 (已内联)
# *** 已修复以匹配 checkpoint 变量命名空间 (conv/W, deconv/kernel) ***
# ======================================================================

def conv(x, k, c_out, s, wd, name, padding='SAME', relu=True):
    """
    创建与 'name/W' 和 'name/b' 命名约定匹配的卷积层
    (此函数现在是正确的，请勿修改)
    """
    c_in = x.get_shape().as_list()[-1]
    
    with tf.variable_scope(name):
        if hasattr(tf.contrib, 'layers'):
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.glorot_uniform_initializer()

        w_shape = [k, k, c_in, c_out]
        b_shape = [c_out]
        
        w = tf.get_variable('W', w_shape, initializer=initializer)
        b = tf.get_variable('b', b_shape, initializer=tf.constant_initializer(0.0))

        if wd is not None:
            tf.add_to_collection('losses', tf.nn.l2_loss(w) * wd)
            
        stride = [1, s, s, 1]
        out = tf.nn.conv2d(x, w, stride, padding) + b
        
        if relu:
            return tf.nn.relu(out)
        else:
            return out

def deconv(x, k, c_out, s, wd, name):
    """
    *** 已修复 ***
    创建与 'name/kernel' 和 'name/bias' 命名约定匹配的 *反*卷积层
    """
    c_in = x.get_shape().as_list()[-1]
    
    with tf.variable_scope(name):
        if hasattr(tf.contrib, 'layers'):
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.glorot_uniform_initializer()

        # 反卷积的权重形状 [k, k, c_out, c_in]
        w_shape = [k, k, c_out, c_in]
        b_shape = [c_out]
        
        # *** 关键修复 ***
        # 变量名从 'W'/'b' 更改为 'kernel'/'bias'
        w = tf.get_variable('kernel', w_shape, initializer=initializer)
        b = tf.get_variable('bias', b_shape, initializer=tf.constant_initializer(0.0))
        
        if wd is not None:
            tf.add_to_collection('losses', tf.nn.l2_loss(w) * wd)
            
        stride = [1, s, s, 1]
        
        # 计算输出形状
        x_shape = tf.shape(x)
        batch_size = x_shape[0]
        height = x_shape[1] * s
        width = x_shape[2] * s
        
        output_shape = tf.stack([batch_size, height, width, c_out])
        
        return tf.nn.conv2d_transpose(x, w, output_shape, stride, 'SAME') + b

def max_pool(x, k, s):
    # max_pool 不创建变量，所以保持不变
    return tf.nn.max_pool(x, [1, k, k, 1], [1, s, s, 1], 'SAME')

def fc(x, c_out, wd, name, relu=True):
    """
    全连接层 (未使用，但为保持完整性而修复)
    """
    shape = x.get_shape().as_list()
    dim = np.prod(shape[1:]) 
    x_flat = tf.reshape(x, [-1, dim])
    
    c_in = dim
    
    with tf.variable_scope(name):
        if hasattr(tf.contrib, 'layers'):
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.glorot_uniform_initializer()
            
        w_shape = [c_in, c_out]
        b_shape = [c_out]

        # 假设 fc 层也使用 'W'/'b' (如果未使用则无关紧要)
        w = tf.get_variable('W', w_shape, initializer=initializer)
        b = tf.get_variable('b', b_shape, initializer=tf.constant_initializer(0.0))
        
        if wd is not None:
            tf.add_to_collection('losses', tf.nn.l2_loss(w) * wd)
            
        out = tf.matmul(x_flat, w) + b
        
        if relu:
            return tf.nn.relu(out)
        else:
            return out

# ======================================================================
# 来自 echoanalysis_tools.py 的 DICOM 读取器 (已内联并使用 pydicom)
# ======================================================================

def create_imgdict_from_dicom(dicomdir, videofile):
    """
    使用 pydicom 纯 Python 方式读取 DICOM 文件（包括压缩格式）。
    """
    full_path = os.path.join(dicomdir, videofile)
    framedict = {}
    
    try:
        ds = pydicom.dcmread(full_path)
    except Exception as e:
        print("错误：无法读取 DICOM 文件: {}".format(full_path))
        print("详细错误: {}".format(e))
        return {} # 返回空字典

    try:
        pixel_data = ds.pixel_array
        # .pixel_array 会自动调用 pylibjpeg/imagecodecs 进行解压
    except Exception as e:
        print("错误：无法从 DICOM 提取像素数据: {}".format(full_path))
        print("这通常是因为缺少解码器 (pylibjpeg/imagecodecs)。")
        print("详细错误: {}".format(e))
        return {}

    # 归一化和格式转换
    def process_frame(frame):
        # 1. 将数据转换为 float32 以进行归一化
        frame = frame.astype(np.float32)
        
        # 2. 归一化到 0.0 - 1.0
        min_val = frame.min()
        max_val = frame.max()
        if max_val > min_val:
            frame = (frame - min_val) / (max_val - min_val)
        else:
            frame = np.zeros_like(frame) # 避免除以零
            
        # 3. 转换为 0-255, uint8
        frame = (frame * 255).astype(np.uint8)
        
        # 4. 确保是 RGB（因为 extract_images 总是调用 rgb2gray）
        if frame.ndim == 2:
            frame = gray2rgb(frame) # from skimage.color
        
        return frame

    # --- 处理不同类型的 DICOM ---

    # 案例 1: 多帧（Cine-loop），例如 (num_frames, rows, cols)
    if pixel_data.ndim == 3:
        num_frames = pixel_data.shape[0]
        for i in range(num_frames):
            framedict[i] = process_frame(pixel_data[i])
            
    # 案例 2: 多帧 RGB, 例如 (num_frames, rows, cols, 3)
    elif pixel_data.ndim == 4 and pixel_data.shape[-1] == 3:
        num_frames = pixel_data.shape[0]
        for i in range(num_frames):
            framedict[i] = pixel_data[i] 
            
    # 案例 3: 单帧灰度, 例如 (rows, cols)
    elif pixel_data.ndim == 2:
        framedict[0] = process_frame(pixel_data)
        
    # 案例 4: 单帧 RGB, 例如 (rows, cols, 3)
    elif pixel_data.ndim == 3 and pixel_data.shape[-1] == 3:
        framedict[0] = pixel_data # 假设是 uint8
        
    else:
        print("警告：无法处理的像素数据维度 {}，文件: {}".format(pixel_data.shape, full_path))

    return framedict


# ======================================================================
#  配置区域
# ---
# ======================================================================
CONFIG = {
    "model_dir": "./models",
    "output_dir": "./segment",
    "main": {
        "view_probabilities_file": "/hpc/khan271/project_2025/echo_class/echocv/test_hk/txt/view_23_e5_class_11-Mar-2018_all_probabilities.txt",
        "view_classes_file": "/hpc/khan271/project_2025/echo_class/echocv/viewclasses_view_23_e5_class_11-Mar-2018.txt",
        "dicom_base_dir": "dicomsample"
    },
    "models": {
        "a4c": "a4c_45_20_all_model.ckpt-9000",
        "a2c": "a2c_45_20_all_model.ckpt-10600",
        "a3c": "a3c_45_20_all_model.ckpt-10500",
        "psax": "psax_45_20_all_model.ckpt-9300",
        "plax": "plax_45_20_all_model.ckpt-9600"
    }
}
# ======================================================================


#purpose:

class Unet(object):
    def __init__(self, mean, weight_decay, learning_rate, label_dim, maxout=False):
        self.x_train = tf.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_train = tf.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.x_test = tf.placeholder(tf.float32, [None, 384, 384, 1])
        self.y_test = tf.placeholder(tf.float32, [None, 384, 384, label_dim])
        self.label_dim = label_dim
        self.weight_decay = weight_decay
        self.learning_rate = learning_rate
        self.maxout = maxout

        self.output = self.unet(self.x_train, mean)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.output, labels=self.y_train))
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.pred = self.unet(self.x_test, mean, keep_prob=1.0, reuse=True)
        self.loss_summary = tf.summary.scalar('loss', self.loss)
    
    def fit_batch(self, sess, x_train, y_train):
        _, loss, loss_summary = sess.run((self.opt, self.loss, self.loss_summary), feed_dict={self.x_train: x_train, self.y_train: y_train})
        return loss, loss_summary
    
    def predict(self, sess, x):
        prediction = sess.run((self.pred), feed_dict={self.x_test: x})
        return prediction

    def unet(self, input, mean, keep_prob=0.5, reuse=None):
        width = 1
        weight_decay = 1e-12
        label_dim = self.label_dim
        
        with tf.variable_scope('vgg', reuse=reuse):
            input = input - mean
            
            # conv_ 将调用我们 'W'/'b' 版本的辅助函数
            # deconv_ 将调用我们 'kernel'/'bias' 版本的辅助函数
            pool_ = lambda x: max_pool(x, 2, 2)
            conv_ = lambda x, output_depth, name, padding='SAME', relu=True, filter_size=3: conv(x, filter_size, output_depth, 1, weight_decay, name=name, padding=padding, relu=relu)
            deconv_ = lambda x, output_depth, name: deconv(x, 2, output_depth, 2, weight_decay, name=name)
            
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
            
            # deconv_ 将 'up7' 作为 name 传入
            # 我们的新函数将在 'vgg/up7' 范围内查找 'kernel' 和 'bias'
            up_7 = tf.concat([deconv_(conv_6_2, int(1024*width), 'up7'), conv_5_2], 3)
            
            conv_7_1 = conv_(up_7, int(1024*width), 'conv7_1')
            conv_7_2 = conv_(conv_7_1, int(1024*width), 'conv7_2')
            
            up_8 = tf.concat([deconv_(conv_7_2, int(512*width), 'up8'), conv_4_2], 3)
            
            conv_8_1 = conv_(up_8, int(512*width), 'conv8_1')
            conv_8_2 = conv_(conv_8_1, int(512*width), 'conv8_2')
            
            up_9 = tf.concat([deconv_(conv_8_2, int(256*width), 'up9'), conv_3_2], 3)
            
            conv_9_1 = conv_(up_9,int(256*width), 'conv9_1')
            conv_9_2 = conv_(conv_9_1, int(256*width), 'conv9_2')

            # deconv_ 将 'up10' 作为 name 传入
            up_10 = tf.concat([deconv_(conv_9_2, int(128*width), 'up10'), conv_2_2], 3)
            
            conv_10_1 = conv_(up_10, int(128*width), 'conv10_1')
            conv_10_2 = conv_(conv_10_1, int(128*width), 'conv10_2')

            up_11 = tf.concat([deconv_(conv_10_2, int(64*width), 'up11'), conv_1_2], 3)
            
            conv_11_1 = conv_(up_11, int(64*width), 'conv11_1')
            conv_11_2 = conv_(conv_11_1, int(64*width), 'conv11_2')
            
            conv_12 = conv_(conv_11_2, label_dim, 'conv12_2', filter_size=1, relu=False)
            return conv_12

def segmentChamber(videofile, dicomdir, view):
    mean = 24
    weight_decay = 1e-12
    learning_rate = 1e-4
    maxout = False
    sesses = []
    models = []
    
    gpu_options = tf.GPUOptions(allow_growth=True)
    session_config = tf.ConfigProto(gpu_options=gpu_options)

    if view == "a4c":
        g_1 = tf.Graph()
        with g_1.as_default():
            label_dim = 6 #a4c
            sess1 = tf.Session(config=session_config) 
            model1 = Unet(mean, weight_decay, learning_rate, label_dim , maxout=maxout)
            sess1.run(tf.local_variables_initializer())
            sess = sess1
            model = model1
        with g_1.as_default():
            saver = tf.train.Saver()
            model_path = os.path.join(CONFIG["model_dir"], CONFIG["models"]["a4c"])
            print("正在从 {} 恢复 A4C 模型...".format(model_path))
            saver.restore(sess1, model_path)
    elif view == "a2c":
        g_2 = tf.Graph()
        with g_2.as_default():
            label_dim = 4 
            sess2 = tf.Session(config=session_config) 
            model2 = Unet(mean, weight_decay, learning_rate, label_dim , maxout=maxout)
            sess2.run(tf.local_variables_initializer())
            sess = sess2
            model = model2
        with g_2.as_default():
            saver = tf.train.Saver()
            model_path = os.path.join(CONFIG["model_dir"], CONFIG["models"]["a2c"])
            print("正在从 {} 恢复 A2C 模型...".format(model_path))
            saver.restore(sess2, model_path)
    elif view == "a3c":
        g_3 = tf.Graph()
        with g_3.as_default():
            label_dim = 4 
            sess3 = tf.Session(config=session_config) 
            model3 = Unet(mean, weight_decay, learning_rate, label_dim , maxout=maxout)
            sess3.run(tf.local_variables_initializer())
            sess = sess3
            model = model3
        with g_3.as_default():
            saver = tf.train.Saver()
            model_path = os.path.join(CONFIG["model_dir"], CONFIG["models"]["a3c"])
            print("正在从 {} 恢复 A3C 模型...".format(model_path))
            saver.restore(sess3, model_path)
    elif view == "psax":
        g_4 = tf.Graph()
        with g_4.as_default():
            label_dim = 4 
            sess4 = tf.Session(config=session_config) 
            model4 = Unet(mean, weight_decay, learning_rate, label_dim , maxout=maxout)
            sess4.run(tf.local_variables_initializer())
            sess = sess4
            model = model4
        with g_4.as_default():
            saver = tf.train.Saver()
            model_path = os.path.join(CONFIG["model_dir"], CONFIG["models"]["psax"])
            print("正在从 {} 恢复 PSAX 模型...".format(model_path))
            saver.restore(sess4, model_path)
    elif view == "plax":
        g_5 = tf.Graph()
        with g_5.as_default():
            label_dim = 7 
            sess5 = tf.Session(config=session_config) 
            model5 = Unet(mean, weight_decay, learning_rate, label_dim , maxout=maxout)
            sess5.run(tf.local_variables_initializer())
            sess = sess5
            model = model5
        with g_5.as_default():
            saver = tf.train.Saver()
            model_path = os.path.join(CONFIG["model_dir"], CONFIG["models"]["plax"])
            print("正在从 {} 恢复 PLAX 模型...".format(model_path))
            saver.restore(sess5, model_path)
            
    outpath = os.path.join(CONFIG["output_dir"], view)
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    
    framedict = create_imgdict_from_dicom(dicomdir, videofile)
    
    if not framedict:
        print("跳过文件 {}，因为 DICOM 读取失败。".format(videofile))
        return 0 
        
    images, orig_images = extract_images(framedict)
    if view == "a4c":
        a4c_lv_segs, a4c_la_segs, a4c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 4, 1)
        np.save(os.path.join(outpath, videofile + '_lv'), np.array(a4c_lv_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_la'), np.array(a4c_la_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_lvo'), np.array(a4c_lvo_segs).astype('uint8'))
    elif view == "a2c":
        a2c_lv_segs, a2c_la_segs, a2c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 3, 1)
        np.save(os.path.join(outpath, videofile + '_lv'), np.array(a2c_lv_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_la'), np.array(a2c_la_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_lvo'), np.array(a2c_lvo_segs).astype('uint8'))
    elif view == "psax":
        psax_lv_segs, psax_lvo_segs, psax_rv_segs, preds = extract_segs(images, orig_images, model, sess, 2, 1, 3) 
        np.save(os.path.join(outpath, videofile + '_lv'), np.array(psax_lv_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_lvo'), np.array(psax_lvo_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_rv'), np.array(psax_rv_segs).astype('uint8')) 
    elif view == "a3c":
        a3c_lv_segs, a3c_la_segs, a3c_lvo_segs, preds = extract_segs(images, orig_images, model, sess, 2, 3, 1)
        np.save(os.path.join(outpath, videofile + '_lvo'), np.array(a3c_lvo_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_lv'), np.array(a3c_lv_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_la'), np.array(a3c_la_segs).astype('uint8'))
    elif view == "plax":
        plax_lv_segs, plax_la_segs, plax_ao_segs, preds = extract_segs(images, orig_images, model, sess, 1, 5, 3) 
        np.save(os.path.join(outpath, videofile + '_lv'), np.array(plax_lv_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_la'), np.array(plax_la_segs).astype('uint8'))
        np.save(os.path.join(outpath, videofile + '_ao'), np.array(plax_ao_segs).astype('uint8')) 
    j = 0
    nrow = orig_images[0].shape[0]
    ncol = orig_images[0].shape[1]
    print("原始图像尺寸: {} x {}".format(nrow, ncol))
    
    seg_file = os.path.join(outpath, '{}_{}_segmentation.png'.format(videofile, j))
    orig_file = os.path.join(outpath, '{}_{}_originalimage.png'.format(videofile, j))
    overlay_file = os.path.join(outpath, '{}_{}_overlay.png'.format(videofile, j))
    
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(imresize(preds, (nrow,ncol)))
    plt.savefig(seg_file)
    plt.close() 
    plt.figure(figsize=(5, 5))
    plt.axis('off')
    plt.imshow(orig_images[0])
    plt.savefig(orig_file)
    plt.close() 
    background = Image.open(orig_file)
    overlay = Image.open(seg_file)
    background = background.convert("RGBA")
    overlay = overlay.convert("RGBA")
    outImage = Image.blend(background, overlay, 0.5)
    outImage.save(overlay_file, "PNG")
    
    print("处理完成: {}, 关闭会话。".format(videofile))
    sess.close()
    
    return 1

def segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax):
    for full_path in viewlist_a4c:
        dicomdir, videofile = os.path.split(full_path)
        print("Processing A4C: {} in {}".format(videofile, dicomdir))
        segmentChamber(videofile, dicomdir, "a4c")
        
    for full_path in viewlist_a2c:
        dicomdir, videofile = os.path.split(full_path)
        print("Processing A2C: {} in {}".format(videofile, dicomdir))
        segmentChamber(videofile, dicomdir, "a2c")
        
    for full_path in viewlist_psax:
        dicomdir, videofile = os.path.split(full_path)
        print("Processing PSAX: {} in {}".format(videofile, dicomdir))
        segmentChamber(videofile, dicomdir, "psax")
        
    for full_path in viewlist_plax:
        dicomdir, videofile = os.path.split(full_path)
        print("Processing PLAX: {} in {}".format(videofile, dicomdir))
        segmentChamber(videofile, dicomdir, "plax")
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
    
    sorted_keys = sorted(framedict.keys())
    
    for key in sorted_keys:
        image = np.zeros((384,384))
        image[:,:] = imresize(rgb2gray(framedict[key]), (384,384,1))
        images.append(image)
        orig_images.append(framedict[key])
        
    images = np.array(images).reshape((len(images), 384,384,1))
    return images, orig_images

def extract_segs(images, orig_images, model, sess, lv_label, la_label, lvo_label):
    segs = []
    preds = np.argmax(model.predict(sess, images[0:1])[0,:,:,:], 2)
    label_all = range(1, 8)
    label_good = [lv_label, la_label, lvo_label]
    for i in label_all:
        if not i in label_good:
            preds[preds == i] = 0
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
    viewfile = CONFIG["main"]["view_probabilities_file"]
    view_classes_filepath = CONFIG["main"]["view_classes_file"]

    viewlist_a2c = []
    viewlist_a3c = []
    viewlist_a4c = []
    viewlist_plax = []
    viewlist_psax = []
    
    try:
        infile = open(view_classes_filepath)
        infile_lines = infile.readlines()
        infile.close() 
    except IOError as e:
        print("错误：无法打开 view classes 文件: {}".format(view_classes_filepath))
        print(e)
        return
        
    infile_lines = [i.rstrip() for i in infile_lines]

    viewdict = {}

    for i in range(len(infile_lines)):
        viewdict[infile_lines[i]] = i + 2
        
    probthresh = 0.5 

    try:
        infile = open(viewfile)
        infile_lines = infile.readlines()
        infile.close() 
    except IOError as e:
        print("错误：无法打开 view probabilities 文件: {}".format(viewfile))
        print(e)
        return
        
    infile_lines = [i.rstrip() for i in infile_lines]
    infile_lines = [i.split('\t') for i in infile_lines]

    for i in infile_lines[1:]:
        dicomdir_from_file = i[0]
        filename = i[1]
        
        full_video_path = os.path.join(dicomdir_from_file, filename)
        
        try:
            if eval(i[viewdict['psax_pap']]) > probthresh:
                viewlist_psax.append(full_video_path)
            elif eval(i[viewdict['a4c']]) > probthresh:
                viewlist_a4c.append(full_video_path)
            elif eval(i[viewdict['a2c']]) > probthresh:
                viewlist_a2c.append(full_video_path)
            elif eval(i[viewdict['a3c']]) > probthresh:
                viewlist_a3c.append(full_video_path)
            elif eval(i[viewdict['plax_plax']]) > probthresh:
                viewlist_plax.append(full_video_path)
        except (IndexError, KeyError, SyntaxError, NameError) as e:
            print("警告：解析行 {} 时出错 ({}): {}".format(i, e, full_video_path))
            continue
            
    print("A2C 视图文件: ", viewlist_a2c)
    print("A4C 视图文件: ", viewlist_a4c)
    print("A3C 视图文件: ", viewlist_a3c)
    print("PSAX 视图文件: ", viewlist_psax)
    print("PLAX 视图文件: ", viewlist_plax)
    
    segmentstudy(viewlist_a2c, viewlist_a4c, viewlist_psax, viewlist_plax)
    
if __name__ == '__main__':
    main()