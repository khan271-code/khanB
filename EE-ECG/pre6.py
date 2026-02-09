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
# 2️⃣ R 波检测 & 多心搏截取
# ===============================

def detect_r_peaks(signal, fs, lead_idx=1):
    """检测R波并去除过密误检"""
    lead = signal[:, lead_idx]
    r_peaks = processing.gqrs_detect(sig=lead, fs=fs)
    # 去除过近R波（<0.3s）
    valid_r = [r_peaks[0]]
    for r in r_peaks[1:]:
        if (r - valid_r[-1]) > int(0.3 * fs):
            valid_r.append(r)
    return np.array(valid_r)

def extract_heartbeats(signal, r_peaks, fs, pre=0.2):
    """从R波序列中截取多个完整心搏周期"""
    heartbeats = []
    for i in range(len(r_peaks) - 1):
        start_idx = int(max(r_peaks[i] - pre * fs, 0))
        end_idx = r_peaks[i + 1]
        heartbeats.append(signal[start_idx:end_idx, :])
    return heartbeats

def select_representative_heartbeat(heartbeats, r_peaks, fs):
    """选出最代表性的心搏：RR间期接近中位值"""
    rr_intervals = np.diff(r_peaks) / fs
    median_rr = np.median(rr_intervals)
    diffs = np.abs(rr_intervals - median_rr)
    idx = np.argmin(diffs)
    return heartbeats[idx], idx

# ===============================
# 3️⃣ 生成竖条拼接傅里叶图
# ===============================

def create_heartbeat_strip(heartbeat, fs, lead_names, output_path, gap_px=2, tile_height=256):
    n_leads = heartbeat.shape[1]
    nperseg = 128
    strips = []

    for i in range(n_leads):
        f, t, Zxx = stft(heartbeat[:, i], fs=fs, nperseg=nperseg)
        magnitude = 20 * np.log10(np.abs(Zxx) + 1e-12)
        magnitude = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + 1e-8)
        img_resized = resize(magnitude, (tile_height, magnitude.shape[1]), anti_aliasing=True)
        strips.append(img_resized)

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

    try:
        record = wfdb.rdrecord(rec_path)
        signals = record.p_signal
        fs = record.fs
        lead_names = record.sig_name
        print(f"✅ Loaded record: {rec_path}, shape: {signals.shape}, fs: {fs} Hz")
    except FileNotFoundError:
        print(f"❌ Record not found. 请确认路径下有 .dat 和 .hea 文件: {rec_path}")
        exit()

    # ================= 预处理 =================
    filtered_signals = preprocess_signals(signals, fs)
    print("⚙️ Preprocessing done.")

    # ================= 检测 R 波 =================
    r_peaks = detect_r_peaks(filtered_signals, fs, lead_idx=1)
    print(f"⚡ Detected {len(r_peaks)} R peaks")

    # ================= 限制到前 10 秒 =================
    max_samples = int(10 * fs)
    r_peaks_10s = r_peaks[r_peaks < max_samples]
    print(f"📏 Using {len(r_peaks_10s)} R peaks within first 10 seconds")

    # ================= 提取多个心搏 =================
    heartbeats = extract_heartbeats(filtered_signals, r_peaks_10s, fs)
    print(f"💓 Extracted {len(heartbeats)} heartbeats")

    if len(heartbeats) == 0:
        print("⚠️ No valid heartbeats detected within 10 seconds.")
        exit()

    # ================= 选择最具代表性心搏 =================
    rep_heartbeat, idx = select_representative_heartbeat(heartbeats, r_peaks_10s, fs)
    print(f"⭐ Selected representative heartbeat index: {idx}")

    # ================= 生成竖条傅里叶图 =================
    out_path = os.path.join(output_dir, f'heartbeat_strip_rep.png')
    create_heartbeat_strip(rep_heartbeat, fs, lead_names, out_path)

    print("🎯 Heartbeat strip spectrogram generated successfully.")
