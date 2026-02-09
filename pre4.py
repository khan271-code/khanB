import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, iirnotch, stft
import pywt
import os
from skimage.transform import resize

# ------------------------------
# 1. 信号预处理
# ------------------------------

def bandpass_filter(data, lowcut=0.5, highcut=40, fs=500, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, axis=0)
    return y

def notch_filter(data, notch_freq=50, fs=500, Q=30):
    nyquist = 0.5 * fs
    freq = notch_freq / nyquist
    b, a = iirnotch(freq, Q)
    y = filtfilt(b, a, data, axis=0)
    return y

def preprocess_signals(signals, fs=500):
    signals = np.nan_to_num(signals)
    signals_notched = notch_filter(signals, notch_freq=60, fs=fs)
    signals_filtered = bandpass_filter(signals_notched, lowcut=0.5, highcut=40, fs=fs)
    return signals_filtered

# ------------------------------
# 2. 生成谱图 / 尺度图
# ------------------------------

def create_spectrogram_image(signals, fs, lead_names, output_path):
    fig, axes = plt.subplots(4, 3, figsize=(20, 15))
    axes = axes.flatten()
    for i in range(12):
        lead_signal = signals[:, i]
        f, t, Zxx = stft(lead_signal, fs=fs, nperseg=128)
        ax = axes[i]
        ax.pcolormesh(t, f, 20 * np.log10(np.abs(Zxx) + 1e-12), shading='gouraud', cmap='viridis')
        ax.set_title(f'Lead {lead_names[i]} (Spectrogram)')
        ax.set_xlabel('Time [sec]')
        ax.set_ylabel('Frequency [Hz]')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Spectrogram image saved to: {output_path}")

def create_scalogram_image(signals, fs, lead_names, output_path):
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
    print(f"Scalogram image saved to: {output_path}")

# ------------------------------
# 3. 改进版：均匀分布正方形拼接图
# ------------------------------

def create_spectrogram_strip_each_second_square_uniform(signals, fs, lead_names, output_dir, total_seconds=10, tile_size=128, gap_px=5):
    """
    每秒生成一张12导联的正方形拼接图（6列×2行，竖条均匀分布）
    """
    n_leads = signals.shape[1]
    nperseg = 128

    for sec in range(total_seconds):
        lead_images = []
        t_start, t_end = sec, sec + 1.0

        # 计算每个导联的 STFT，并统一尺寸
        for i in range(n_leads):
            f, t, Zxx = stft(signals[:, i], fs=fs, nperseg=nperseg)
            magnitude = 20 * np.log10(np.abs(Zxx) + 1e-12)
            idx = np.where((t >= t_start) & (t <= t_end))[0]
            if len(idx) == 0:
                continue
            cropped = magnitude[:, idx]

            # 归一化并缩放为 tile_size×tile_size
            cropped = (cropped - np.min(cropped)) / (np.max(cropped) - np.min(cropped) + 1e-8)
            img_resized = resize(cropped, (tile_size, tile_size), anti_aliasing=True)
            lead_images.append(img_resized)

        if len(lead_images) < 12:
            print(f"Warning: Only {len(lead_images)} leads available at second {sec}")
            continue

        # 拼成 2×6 网格（均匀间隔）
        rows = 2
        cols = 6
        gap_color = 0.0  # 黑色间隔
        h_gap = np.ones((tile_size, gap_px)) * gap_color
        v_gap = np.ones((gap_px, (tile_size + gap_px) * cols - gap_px)) * gap_color

        # 上6导联
        top_row = lead_images[0]
        for i in range(1, 6):
            top_row = np.concatenate((top_row, h_gap, lead_images[i]), axis=1)

        # 下6导联
        bottom_row = lead_images[6]
        for i in range(7, 12):
            bottom_row = np.concatenate((bottom_row, h_gap, lead_images[i]), axis=1)

        # 上下拼接
        combined = np.concatenate((top_row, v_gap, bottom_row), axis=0)

        # 调整为正方形比例
        size = max(combined.shape)
        square = np.zeros((size, size))
        square[:combined.shape[0], :combined.shape[1]] = combined

        # 保存结果
        plt.figure(figsize=(10, 10))
        plt.imshow(square, aspect='equal', origin='lower', cmap='viridis')
        plt.axis('off')
        plt.title(f"12-Lead Spectrogram {t_start:.0f}-{t_end:.0f}s", fontsize=14)

        out_path = os.path.join(output_dir, f'spectrogram_square_{sec}_{sec+1}s.png')
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
        plt.close()
        print(f"Saved (uniform square): {out_path}")

# ------------------------------
# 4. 主程序
# ------------------------------

if __name__ == "__main__":
    rec_path = '/eresearch/ecg-echo-xai/khan271/MIMIC-IV-ECG-1.0/physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238'
    output_dir = '/eresearch/ecg-echo-xai/khan271/picoutput'
    os.makedirs(output_dir, exist_ok=True)

    # 1. 加载 ECG
    try:
        record = wfdb.rdrecord(rec_path)
        signals = record.p_signal
        fs = record.fs
        lead_names = record.sig_name
        print(f"Loaded record: {rec_path}, shape: {signals.shape}, fs: {fs} Hz")
    except Exception as e:
        print(f"Failed to load record: {e}")
        exit()

    # 2. 预处理
    print("Preprocessing signals...")
    filtered_signals = preprocess_signals(signals, fs)
    print("Preprocessing done.")

    # 3. 生成完整谱图
    spectrogram_out_path = os.path.join(output_dir, f'{os.path.basename(rec_path)}_spectrogram.png')
    create_spectrogram_image(filtered_signals, fs, lead_names, spectrogram_out_path)

    # 4. 生成尺度图
    scalogram_out_path = os.path.join(output_dir, f'{os.path.basename(rec_path)}_scalogram.png')
    create_scalogram_image(filtered_signals, fs, lead_names, scalogram_out_path)

    # 5. 生成正方形拼接图（均匀分布）
    create_spectrogram_strip_each_second_square_uniform(
        filtered_signals, fs, lead_names, output_dir,
        total_seconds=10, tile_size=128, gap_px=5
    )

    print("All images generated successfully.")
