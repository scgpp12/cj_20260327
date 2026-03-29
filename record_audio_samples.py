"""
record_audio_samples.py - 录制攻击音效样本，自动切分，保存指纹模板
使用方法：
  1. 运行脚本
  2. 在游戏中连续攻击 5-10 次
  3. 脚本自动检测音量爆发，切分出每次攻击的音效
  4. 保存为指纹模板文件 assets/attack_fingerprint.npz
"""

import os
import numpy as np
import time
import pyaudiowpatch as pyaudio
from numpy.fft import rfft, rfftfreq

ASSETS_DIR = "assets"
FINGERPRINT_PATH = os.path.join(ASSETS_DIR, "attack_fingerprint.npz")


def find_loopback():
    """找到 WASAPI Loopback 设备"""
    p = pyaudio.PyAudio()
    loopback = None
    for i in range(p.get_device_count()):
        dev = p.get_device_info_by_index(i)
        if dev.get("isLoopbackDevice", False):
            loopback = dev
            break
    return p, loopback


def compute_spectrogram(audio, sr, n_fft=1024, hop=512):
    """计算短时傅里叶变换频谱图"""
    frames = []
    for start in range(0, len(audio) - n_fft, hop):
        window = audio[start:start + n_fft] * np.hanning(n_fft)
        spectrum = np.abs(rfft(window))
        frames.append(spectrum)
    return np.array(frames)


def compute_mfcc_like(audio, sr, n_bands=20):
    """
    计算简易频谱特征（类 MFCC，不依赖 librosa）
    将频谱分成 n_bands 个频段，取对数能量
    """
    spec = compute_spectrogram(audio, sr)
    if len(spec) == 0:
        return np.zeros((1, n_bands))

    n_freq = spec.shape[1]
    band_size = n_freq // n_bands

    features = []
    for frame in spec:
        bands = []
        for b in range(n_bands):
            start = b * band_size
            end = start + band_size if b < n_bands - 1 else n_freq
            energy = np.sum(frame[start:end] ** 2)
            bands.append(np.log(energy + 1e-10))
        features.append(bands)

    return np.array(features)


def extract_attacks(audio, sr, threshold=0.015, min_gap=0.3, attack_len=0.4):
    """
    从连续录音中自动切分出每次攻击的音效片段

    Args:
        audio: 单声道音频
        sr: 采样率
        threshold: 音量阈值，超过此值视为攻击开始
        min_gap: 两次攻击最小间隔（秒）
        attack_len: 每段攻击截取长度（秒）

    Returns:
        list of np.array: 每段攻击音效
    """
    # 计算每帧 RMS（10ms 窗口）
    frame_size = int(sr * 0.01)
    rms_list = []
    for i in range(0, len(audio) - frame_size, frame_size):
        rms = np.sqrt(np.mean(audio[i:i + frame_size] ** 2))
        rms_list.append(rms)

    # 找到超过阈值的位置
    attacks = []
    last_attack_frame = -int(min_gap / 0.01)

    for i, rms in enumerate(rms_list):
        if rms > threshold and (i - last_attack_frame) > int(min_gap / 0.01):
            # 攻击起始点（往前偏移一点捕获完整起音）
            start_sample = max(0, int((i - 2) * frame_size))
            end_sample = min(len(audio), start_sample + int(sr * attack_len))
            segment = audio[start_sample:end_sample]

            if len(segment) >= int(sr * 0.1):  # 至少 100ms
                attacks.append(segment)
                last_attack_frame = i

    return attacks


def build_fingerprint(attacks, sr):
    """
    从多个攻击样本中构建指纹模板

    Returns:
        dict: 指纹数据
    """
    all_features = []
    all_spectrograms = []

    # 统一长度（取最短的）
    min_len = min(len(a) for a in attacks)
    attacks_uniform = [a[:min_len] for a in attacks]

    for atk in attacks_uniform:
        feat = compute_mfcc_like(atk, sr, n_bands=20)
        spec = compute_spectrogram(atk, sr)
        all_features.append(feat)
        all_spectrograms.append(spec)

    # 统一特征帧数
    min_frames = min(f.shape[0] for f in all_features)
    all_features = [f[:min_frames] for f in all_features]
    all_spectrograms = [s[:min_frames] for s in all_spectrograms]

    # 平均模板
    avg_feature = np.mean(all_features, axis=0)
    avg_spectrogram = np.mean(all_spectrograms, axis=0)

    # 计算每个样本与平均的相似度（验证一致性）
    similarities = []
    for feat in all_features:
        sim = np.corrcoef(avg_feature.flatten(), feat.flatten())[0, 1]
        similarities.append(sim)

    return {
        "avg_feature": avg_feature,
        "avg_spectrogram": avg_spectrogram,
        "sample_count": len(attacks),
        "sample_rate": sr,
        "attack_duration": min_len / sr,
        "similarities": similarities,
        "feature_frames": min_frames,
    }


def main():
    print("=" * 60)
    print("  攻击音效录制 & 指纹提取工具")
    print("=" * 60)

    os.makedirs(ASSETS_DIR, exist_ok=True)

    p, loopback = find_loopback()
    if loopback is None:
        print("[ERROR] 没有找到 WASAPI Loopback 设备")
        return

    sr = int(loopback["defaultSampleRate"])
    ch = loopback["maxInputChannels"]
    print(f"  设备: {loopback['name']}")
    print(f"  采样率: {sr} Hz")
    print()
    print("  请在游戏中连续攻击 5-10 次")
    print("  录制 8 秒后自动停止...")
    print()

    input("  按回车开始录制 >> ")

    # 录制
    all_data = []
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=ch,
        rate=sr,
        input=True,
        input_device_index=loopback["index"],
        frames_per_buffer=int(sr * 0.05),
    )

    duration = 8
    print(f"  录制中... ({duration}秒)")
    start = time.time()
    while time.time() - start < duration:
        data = stream.read(int(sr * 0.05), exception_on_overflow=False)
        arr = np.frombuffer(data, dtype=np.float32).reshape(-1, ch)
        all_data.append(arr)
        # 进度
        elapsed = time.time() - start
        bar_len = int(elapsed / duration * 30)
        print(f"\r  [{'█' * bar_len}{'░' * (30 - bar_len)}] {elapsed:.1f}s/{duration}s", end="")

    stream.stop_stream()
    stream.close()
    p.terminate()
    print("\n  录制完成!")

    # 合并为单声道
    audio = np.concatenate(all_data).mean(axis=1)
    print(f"  总样本数: {len(audio)} ({len(audio)/sr:.1f}s)")

    # 自动切分攻击片段
    attacks = extract_attacks(audio, sr)
    print(f"  检测到 {len(attacks)} 次攻击")

    if len(attacks) < 3:
        print("[WARNING] 攻击次数太少（至少需要3次），请增大攻击次数或降低阈值")
        print("  尝试降低阈值重新检测...")
        attacks = extract_attacks(audio, sr, threshold=0.008)
        print(f"  低阈值检测到 {len(attacks)} 次攻击")

    if len(attacks) < 2:
        print("[ERROR] 无法提取足够的攻击样本，请确保游戏有声音输出")
        return

    # 显示每段信息
    for i, atk in enumerate(attacks):
        rms = np.sqrt(np.mean(atk ** 2))
        peak = np.max(np.abs(atk))
        print(f"  #{i+1}: 长度={len(atk)/sr:.3f}s  RMS={rms:.4f}  Peak={peak:.4f}")

    # 构建指纹
    print("\n  构建指纹模板...")
    fp = build_fingerprint(attacks, sr)

    print(f"  样本数: {fp['sample_count']}")
    print(f"  攻击时长: {fp['attack_duration']:.3f}s")
    print(f"  特征帧数: {fp['feature_frames']}")
    print(f"  样本一致性:")
    for i, sim in enumerate(fp["similarities"]):
        bar = "█" * int(sim * 20)
        print(f"    #{i+1}: {sim:.3f} {bar}")

    avg_sim = np.mean(fp["similarities"])
    if avg_sim < 0.7:
        print(f"\n  [WARNING] 平均一致性 {avg_sim:.3f} 偏低，可能混入了其他声音")
    else:
        print(f"\n  平均一致性: {avg_sim:.3f} — {'良好' if avg_sim > 0.85 else '尚可'}")

    # 保存
    np.savez(
        FINGERPRINT_PATH,
        avg_feature=fp["avg_feature"],
        avg_spectrogram=fp["avg_spectrogram"],
        sample_count=fp["sample_count"],
        sample_rate=fp["sample_rate"],
        attack_duration=fp["attack_duration"],
        feature_frames=fp["feature_frames"],
    )
    print(f"\n  指纹已保存: {FINGERPRINT_PATH}")
    print("  可以在 main.py 中使用音频攻击检测了")


if __name__ == "__main__":
    main()
