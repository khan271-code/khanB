import wfdb

record_path = '/eresearch/ecg-echo-xai/khan271/MIMIC-IV-ECG-1.0/physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238'  # 不要加 .dat 或 .hea 后缀
record = wfdb.rdrecord(record_path)

# 采样频率（Hz）
fs = record.fs

# 信号总样本点数
num_samples = record.sig_len

# 记录总时长（秒）
duration_sec = num_samples / fs

print(f"采样率: {fs} Hz")
print(f"总样本数: {num_samples}")
print(f"记录长度: {duration_sec:.2f} 秒 ({duration_sec/60:.2f} 分钟)")
