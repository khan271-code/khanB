import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, stft
import pywt
import os

# ==============================================================
# 1. 信号预处理
# ==============================================================

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=5):
    """带通滤波"""
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def notch_filter(data, notch_freq=50, fs=500, Q=30):
    """陷波滤波，去除工频干扰"""
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, Q)
    return filtfilt(b, a, data, axis=0)

def preprocess_signals(signals, fs=500):
    """综合预处理：NaN填充 → 陷波 → 带通"""
    signals = np.nan_to_num(signals)
    signals_notched = notch_filter(signals, notch_freq=60, fs=fs)
    signals_filtered = bandpass_filter(signals_notched, lowcut=0.5, highcut=40, fs=fs)
    return signals_filtered


# ==============================================================
# 2. 生成完整谱图 / 尺度图
# ==============================================================

def create_spectrogram_image(signals, fs, lead_names, output_path):
    """生成完整12导联频谱图"""
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()

    for i in range(12):
        lead_signal = signals[:, i]
        f, t, Zxx = stft(lead_signal, fs=fs, nperseg=128)
        ax = axes[i]
        ax.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-12),
                      shading='gouraud', cmap='viridis')
        ax.set_title(f'Lead {lead_names[i]} (Spectrogram)')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Spectrogram image saved to: {output_path}")


def create_scalogram_image(signals, fs, lead_names, output_path):
    """生成完整12导联尺度图（小波变换）"""
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()
    time_array = np.arange(signals.shape[0]) / fs
    scales = np.arange(1, 128)
    wavelet = 'morl'

    for i in range(12):
        lead_signal = signals[:, i]
        coefficients, frequencies = pywt.cwt(lead_signal, scales, wavelet, sampling_period=1/fs)
        ax = axes[i]
        ax.pcolormesh(time_array, frequencies, np.abs(coefficients), cmap='viridis')
        ax.set_title(f'Lead {lead_names[i]} (Scalogram)')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"✅ Scalogram image saved to: {output_path}")


# ==============================================================
# 3. 生成每秒竖条拼接图
# ==============================================================

def create_spectrogram_strip_each_second(signals, fs, lead_names, output_dir, total_seconds=10, gap_px=5):
    """
    为每一秒生成一张12导联竖条拼接图（频谱特征竖向排列）
    """
    n_leads = signals.shape[1]
    nperseg = 128

    for sec in range(total_seconds):
        strips = []
        t_start, t_end = sec, sec + 1.0

        for i in range(n_leads):
            f, t, Zxx = stft(signals[:, i], fs=fs, nperseg=nperseg)
            magnitude = 20 * np.log10(np.abs(Zxx) + 1e-12)
            idx = np.where((t >= t_start) & (t <= t_end))[0]
            if len(idx) == 0:
                continue

            cropped = magnitude[:, idx]
            cropped = (cropped - np.min(cropped)) / (np.max(cropped) - np.min(cropped) + 1e-8)
            strips.append(cropped)

        if len(strips) == 0:
            continue

        # 统一高度并插入间隔
        min_height = min(strip.shape[0] for strip in strips)
        strips = [strip[:min_height, :] for strip in strips]
        gap = np.zeros((min_height, gap_px))
        combined = strips[0]
        for s in strips[1:]:
            combined = np.concatenate((combined, gap, s), axis=1)

        plt.figure(figsize=(16, 4))
        plt.imshow(combined, aspect='auto', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.title(f"12-Lead Spectrogram {t_start:.0f}-{t_end:.0f}s", fontsize=14)

        out_path = os.path.join(output_dir, f'spectrogram_strip_{sec}_{sec+1}s.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print(f"🖼️ Saved: {out_path}")


# ==============================================================
# 4. 主程序入口
# ==============================================================

if __name__ == "__main__":
    rec_path = '/eresearch/ecg-echo-xai/khan271/MIMIC-IV-ECG-1.0/physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238'
    output_dir = '/eresearch/ecg-echo-xai/khan271/picoutput'
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: 加载 ECG
    try:
        record = wfdb.rdrecord(rec_path)
        signals = record.p_signal
        fs = record.fs
        lead_names = record.sig_name
        print(f"✅ Loaded record: {rec_path}")
        print(f"   shape: {signals.shape}, fs: {fs} Hz")
    except Exception as e:
        print(f"❌ Failed to load record: {e}")
        exit()

    # Step 2: 预处理
    print("⚙️ Preprocessing signals...")
    filtered_signals = preprocess_signals(signals, fs)
    print("✅ Preprocessing done.")

    # Step 3: 生成完整谱图
    spectrogram_out = os.path.join(output_dir, f'{os.path.basename(rec_path)}_spectrogram.png')
    create_spectrogram_image(filtered_signals, fs, lead_names, spectrogram_out)

    # Step 4: 生成尺度图
    scalogram_out = os.path.join(output_dir, f'{os.path.basename(rec_path)}_scalogram.png')
    create_scalogram_image(filtered_signals, fs, lead_names, scalogram_out)

    # Step 5: 生成每秒竖条拼接图
    create_spectrogram_strip_each_second(filtered_signals, fs, lead_names, output_dir,
                                         total_seconds=10, gap_px=5)

    print("🎯 All images generated successfully.")
