import wfdb
import matplotlib.pyplot as plt

rec_path = '/eresearch/ecg-echo-xai/khan271/MIMIC-IV-ECG-1.0/physionet.org/files/mimic-iv-ecg/1.0/files/p1000/p10000032/s40689238/40689238'
record = wfdb.rdrecord(rec_path)

# 获取 ECG 数据
data = record.p_signal
leads = record.sig_name
fs = record.fs
time = [i/fs for i in range(data.shape[0])]

# 绘制 12-lead ECG
plt.figure(figsize=(20,12))
for i in range(12):
    plt.subplot(12,1,i+1)
    plt.plot(time, data[:,i])
    plt.title(leads[i])
    plt.xticks([])
plt.xlabel('Time (s)')
plt.tight_layout()

# 保存图片
plt.savefig('/eresearch/ecg-echo-xai/khan271/picoutput/40689238.png')
plt.close()
