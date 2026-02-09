import wfdb
from wfdb import processing
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, stft
from skimage.transform import resize
import os

# ===============================
# 1️⃣ 信号预处理
# ===============================

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, notch_freq=50, fs=500, Q=30):
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, Q)
    return filtfilt(b, a, data, axis=0)

def preprocess_signals(signals, fs=500):
    signals = np.nan_to_num(signals)
    signals_notched = notch_filter(signals, notch_freq=60, fs=fs)
    signals_filtered = bandpass_filter(signals_notched, lowcut=0.5, highcut=40, fs=fs)
    return signals_filtered

# ===============================
# 2️⃣ R 波检测 & 单心搏截取
# ===============================

def detect_r_peaks(signal, fs, lead_idx=1):
    lead = signal[:, lead_idx]
    r_peaks = processing.gqrs_detect(sig=lead, fs=fs)
    return r_peaks

def extract_heartbeat(signal, r_peaks, fs, pre=0.2):
    """提取第一个心搏示例"""
    start_idx = int(max(r_peaks[0] - pre*fs, 0))
    end_idx = r_peaks[1]
    return signal[start_idx:end_idx, :]

# ===============================
# 3️⃣ 生成竖条拼接傅里叶图
# ===============================

def create_heartbeat_strip(heartbeat, fs, lead_names, output_path, gap_px=2, tile_height=256):
    """
    - 每个导联竖条排列
    - 间隔黑色最小
    - 保持竖条清晰
    """
    n_leads = heartbeat.shape[1]
    nperseg = 128
    strips = []

    for i in range(n_leads):
        f, t, Zxx = stft(heartbeat[:, i], fs=fs, nperseg=nperseg)
        magnitude = 20 * np.log10(np.abs(Zxx) + 1e-12)
        magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-8)
        # 缩放高度一致，保持竖条清晰
        img_resized = resize(magnitude, (tile_height, magnitude.shape[1]), anti_aliasing=True)
        strips.append(img_resized)

    # 拼接竖条
    gap = np.zeros((tile_height, gap_px))
    combined = strips[0]
    for s in strips[1:]:
        combined = np.concatenate((combined, gap, s), axis=1)

    plt.figure(figsize=(12,6))
    plt.imshow(combined, aspect='auto', origin='lower', cmap='viridis')
    plt.title("12-Lead Heartbeat Spectrogram (竖条)", fontsize=14)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0, dpi=200)
    plt.close()
    print(f"🖼️ Saved heartbeat strip spectrogram: {output_path}")

# ===============================
# 4️⃣ 主程序
# ===============================

if __name__ == "__main__":
    rec_path = '/eresearch/ecg-echo-xai/khan271/MIMIC-IV-ECG-1.0/physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238'
    output_dir = '/eresearch/ecg-echo-xai/khan271/picoutput_heartbeat_strip'
    os.makedirs(output_dir, exist_ok=True)

    # 加载 ECG
    try:
        record = wfdb.rdrecord(rec_path)
        signals = record.p_signal
        fs = record.fs
        lead_names = record.sig_name
        print(f"✅ Loaded record: {rec_path}, shape: {signals.shape}, fs: {fs} Hz")
    except FileNotFoundError:
        print(f"❌ Record not found. 请确认路径下有 .dat 和 .hea 文件: {rec_path}")
        exit()

    # 预处理
    filtered_signals = preprocess_signals(signals, fs)
    print("⚙️ Preprocessing done.")

    # 检测 R 波
    r_peaks = detect_r_peaks(filtered_signals, fs, lead_idx=1)
    print(f"⚡ Detected {len(r_peaks)} R peaks")

    # 截取单心搏（第一个心搏）
    heartbeat = extract_heartbeat(filtered_signals, r_peaks, fs, pre=0.2)
    print(f"💓 Extracted single heartbeat: shape {heartbeat.shape}")

    # 生成竖条傅里叶图
    out_path = os.path.join(output_dir, 'heartbeat_strip.png')
    create_heartbeat_strip(heartbeat, fs, lead_names, out_path)

    print("🎯 Heartbeat strip spectrogram generated successfully.")
