import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, stft
import pywt
import os

# --- 1. 信号预处理 (滤波) ---

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=5):
    """
    对信号进行带通滤波。
    :param data: 信号数据 (n_samples, n_leads)
    :param lowcut: 低截止频率
    :param highcut: 高截止频率
    :param fs: 采样率
    :param order: 滤波器阶数
    :return: 滤波后的信号
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def notch_filter(data, notch_freq=50, fs=500, Q=30):
    """
    对信号进行陷波滤波 (去除工频干扰)。
    :param data: 信号数据 (n_samples, n_leads)
    :param notch_freq: 陷波频率 (例如 50Hz 或 60Hz)
    :param fs: 采样率
    :param Q: 品质因数
    :return: 滤波后的信号
    """
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, Q)
    y = filtfilt(b, a, data, axis=0)
    return y

def preprocess_signals(signals, fs=500):
    """
    应用陷波和带通滤波器。
    """
    # 处理NaN值 (如果有)
    signals = np.nan_to_num(signals)
    
    # 1. 陷波滤波
    signals_notched = notch_filter(signals, notch_freq=60, fs=fs) # BIDMC 在美国，工频为60Hz
    
    # 2. 带通滤波
    signals_filtered = bandpass_filter(signals_notched, lowcut=0.5, highcut=40, fs=fs)
    
    return signals_filtered


# --- 2. 信号转图像 (方法B) ---

def create_spectrogram_image(signals, fs, lead_names, output_path):
    """
    方法 B (选项 1): 生成并保存12导联的谱图 (STFT)
    
    将12个导联的谱图绘制在一个 4x3 的网格中。
    """
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten() # 将 4x3 网格转换为 1D 数组
    
    for i in range(12):
        lead_signal = signals[:, i]
        
        # 计算 STFT
        f, t, Zxx = stft(lead_signal, fs=fs, nperseg=128) # nperseg 可调
        
        # 绘制谱图 (使用分贝)
        # 加上一个很小的数 1e-12 以避免 log(0)
        ax = axes[i]
        ax.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-12), shading='gouraud', cmap='viridis')
        ax.set_title(f'Lead {lead_names[i]} (Spectrogram)')
        
        # 为了给GAN使用，关闭所有标签和坐标轴
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
        # ax.axis('off') # <-- 如果用于GAN，取消注释此行
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150) # dpi可调
    plt.close(fig)
    print(f"Spectrogram image saved to: {output_path}")

def create_scalogram_image(signals, fs, lead_names, output_path):
    """
    方法 B (选项 2): 生成并保存12导联的尺度图 (CWT) - (推荐)
    
    将12个导联的尺度图绘制在一个 4x3 的网格中。
    """
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()
    
    # 定义时间和尺度
    time_array = np.arange(signals.shape[0]) / fs
    scales = np.arange(1, 128) # 可调的尺度范围
    wavelet = 'morl' # Morlet小波，非常适用于ECG
    
    for i in range(12):
        lead_signal = signals[:, i]
        
        # 计算 CWT
        coefficients, frequencies = pywt.cwt(lead_signal, scales, wavelet, sampling_period=1/fs)
        
        # 绘制尺度图
        ax = axes[i]
        ax.pcolormesh(time_array, frequencies, np.abs(coefficients), cmap='viridis')
        ax.set_title(f'Lead {lead_names[i]} (Scalogram)')
        
        # 为了给GAN使用，关闭所有标签和坐标轴
        ax.set_ylabel('Frequency [Hz]')
        ax.set_xlabel('Time [sec]')
        # ax.axis('off') # <-- 如果用于GAN，取消注释此行
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150) # dpi可调
    plt.close(fig)
    print(f"Scalogram image saved to: {output_path}")


# --- 3. 主执行函数 ---

if __name__ == "__main__":
    
    # --- 配置 ---
    # !! 修改为你本地的 MIMIC-IV-ECG 数据路径
    # (注意：路径不包含 .hea 或 .dat 扩展名)
    rec_path = '/eresearch/ecg-echo-xai/khan271/MIMIC-IV-ECG-1.0/physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238'
    
    # !! 修改为你希望保存图像的目录
    output_dir = '/eresearch/ecg-echo-xai/khan271/picoutput'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # --- 步骤 1: 加载数据 ---
    try:
        record = wfdb.rdrecord(rec_path)
        signals = record.p_signal # 物理信号 (mV)
        fs = record.fs             # 采样率 (500 Hz)
        lead_names = record.sig_name # 导联名称
        
        print(f"成功加载 record: {rec_path}")
        print(f"  信号 shape: {signals.shape}")
        print(f"  采样率: {fs} Hz")
        
    except Exception as e:
        print(f"加载 record {rec_path} 失败: {e}")
        exit()

    # --- 步骤 2: 预处理信号 ---
    print("开始预处理信号 (滤波)...")
    filtered_signals = preprocess_signals(signals, fs)
    print("信号预处理完成。")
    
    # --- 步骤 3: 方法B (选项 1) - 生成谱图 ---
    spectrogram_out_path = os.path.join(output_dir, f'{os.path.basename(rec_path)}_spectrogram_name.png')
    create_spectrogram_image(filtered_signals, fs, lead_names, spectrogram_out_path)
    
    # --- 步骤 4: 方法B (选项 2) - 生成尺度图 (推荐) ---
    scalogram_out_path = os.path.join(output_dir, f'{os.path.basename(rec_path)}_scalogram_name.png')
    create_scalogram_image(filtered_signals, fs, lead_names, scalogram_out_path)
    
    print("\n所有图像生成完毕。")