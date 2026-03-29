"""
audio_detector.py - 音频指纹匹配检测攻击命中
使用预录制的攻击音效指纹，实时匹配游戏音频流
"""

import os
import threading
import time
import numpy as np
from numpy.fft import rfft, rfftfreq

FINGERPRINT_PATH = os.path.join("assets", "attack_fingerprint.npz")


class AudioDetector:
    """
    音频指纹匹配检测器

    工作方式：
    1. 加载预录制的攻击音效指纹
    2. 后台线程持续录制游戏音频
    3. 滑动窗口与指纹做相关性匹配
    4. 超过阈值 → 检测到攻击命中
    """

    def __init__(self, match_threshold=0.65, cooldown=0.5):
        """
        Args:
            match_threshold: 匹配分数阈值（0-1），越高越严格
            cooldown: 两次检测之间最小间隔（秒），避免一次攻击重复触发
        """
        self.match_threshold = match_threshold
        self.cooldown = cooldown

        # 指纹数据
        self.fingerprint = None
        self.fp_feature = None
        self.fp_sr = 48000
        self.fp_duration = 0.4
        self.fp_frames = 0

        # 状态
        self.is_running = False
        self.attack_detected = False
        self.last_attack_time = 0
        self.match_score = 0.0
        self.current_rms = 0.0

        # 音频缓冲（环形缓冲区）
        self.buffer_lock = threading.Lock()
        self.audio_buffer = np.zeros(0)
        self.buffer_max_len = 0  # 在 start() 中设置

        # 后台线程
        self._thread = None
        self._stop_event = threading.Event()

        # 加载指纹
        self._load_fingerprint()

    def _load_fingerprint(self):
        """加载攻击音效指纹文件"""
        if not os.path.exists(FINGERPRINT_PATH):
            print(f"[AUDIO] 指纹文件不存在: {FINGERPRINT_PATH}")
            print(f"[AUDIO] 请先运行 python record_audio_samples.py 录制攻击音效")
            return False

        data = np.load(FINGERPRINT_PATH)

        # 兼容新旧格式
        if "avg_feature" in data:
            self.fp_feature = data["avg_feature"]
            self.fp_duration = float(data["attack_duration"])
            self.fp_frames = int(data["feature_frames"])
        elif "fingerprint" in data:
            fp = data["fingerprint"]
            self.fp_feature = np.abs(np.fft.rfft(fp))  # 转为频谱特征
            self.fp_duration = float(data["duration"])
            self.fp_frames = len(fp) // 2205 if len(fp) > 0 else 1
        else:
            print(f"[AUDIO] 指纹格式不识别: {list(data.keys())}")
            return False

        self.fp_sr = int(data["sample_rate"])
        self.fingerprint = data

        print(f"[AUDIO] 已加载攻击指纹: {self.fp_frames} 帧, "
              f"{self.fp_duration:.3f}s, sr={self.fp_sr}")
        return True

    def start(self):
        """启动后台音频录制线程"""
        if self.fp_feature is None:
            print("[AUDIO] 没有指纹，跳过音频检测")
            return False

        try:
            import pyaudiowpatch as pyaudio
        except ImportError:
            print("[AUDIO] pyaudiowpatch 未安装，跳过音频检测")
            return False

        # 找 loopback 设备
        p = pyaudio.PyAudio()
        loopback = None
        for i in range(p.get_device_count()):
            dev = p.get_device_info_by_index(i)
            if dev.get("isLoopbackDevice", False):
                loopback = dev
                break

        if loopback is None:
            print("[AUDIO] 没有找到 Loopback 设备")
            p.terminate()
            return False

        sr = int(loopback["defaultSampleRate"])
        ch = loopback["maxInputChannels"]

        # 缓冲区大小：保留 1 秒音频
        self.buffer_max_len = sr
        self.audio_buffer = np.zeros(self.buffer_max_len)

        self._stop_event.clear()
        self.is_running = True

        def _record_thread():
            try:
                stream = p.open(
                    format=pyaudio.paFloat32,
                    channels=ch,
                    rate=sr,
                    input=True,
                    input_device_index=loopback["index"],
                    frames_per_buffer=int(sr * 0.05),
                )

                while not self._stop_event.is_set():
                    try:
                        data = stream.read(int(sr * 0.05), exception_on_overflow=False)
                        arr = np.frombuffer(data, dtype=np.float32).reshape(-1, ch)
                        mono = arr.mean(axis=1)

                        with self.buffer_lock:
                            # 追加到环形缓冲区
                            self.audio_buffer = np.concatenate([
                                self.audio_buffer[len(mono):],
                                mono
                            ])
                            self.current_rms = float(np.sqrt(np.mean(mono ** 2)))

                        # 音量超过阈值时才做指纹匹配（节省 CPU）
                        if self.current_rms > 0.008:
                            self._try_match(sr)

                    except Exception:
                        time.sleep(0.01)

                stream.stop_stream()
                stream.close()
            except Exception as e:
                print(f"[AUDIO] 录音线程异常: {e}")
            finally:
                p.terminate()
                self.is_running = False

        self._thread = threading.Thread(target=_record_thread, daemon=True)
        self._thread.start()
        print(f"[AUDIO] 后台录音已启动 (设备: {loopback['name'][:30]})")
        return True

    def stop(self):
        """停止后台录音"""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)
        self.is_running = False

    def _try_match(self, sr):
        """尝试将当前音频与指纹匹配"""
        now = time.time()
        if now - self.last_attack_time < self.cooldown:
            return

        with self.buffer_lock:
            # 取最近一段音频（和指纹等长）
            attack_samples = int(self.fp_duration * sr)
            if len(self.audio_buffer) < attack_samples:
                return
            segment = self.audio_buffer[-attack_samples:].copy()

        # 提取特征
        feat = self._compute_feature(segment, sr)
        if feat is None:
            return

        # 与指纹做相关性匹配
        score = self._compute_similarity(feat, self.fp_feature)
        self.match_score = score

        if score >= self.match_threshold:
            self.attack_detected = True
            self.last_attack_time = now

    def _compute_feature(self, audio, sr, n_bands=20, n_fft=1024, hop=512):
        """计算音频的频段特征（和录制时一致）"""
        frames = []
        for start in range(0, len(audio) - n_fft, hop):
            window = audio[start:start + n_fft] * np.hanning(n_fft)
            spectrum = np.abs(rfft(window))
            frames.append(spectrum)

        if len(frames) == 0:
            return None

        spec = np.array(frames)
        n_freq = spec.shape[1]
        band_size = n_freq // n_bands

        features = []
        for frame in spec:
            bands = []
            for b in range(n_bands):
                s = b * band_size
                e = s + band_size if b < n_bands - 1 else n_freq
                energy = np.sum(frame[s:e] ** 2)
                bands.append(np.log(energy + 1e-10))
            features.append(bands)

        result = np.array(features)

        # 对齐帧数
        if result.shape[0] > self.fp_frames:
            result = result[:self.fp_frames]
        elif result.shape[0] < self.fp_frames:
            pad = np.zeros((self.fp_frames - result.shape[0], n_bands))
            result = np.vstack([result, pad])

        return result

    def _compute_similarity(self, feat_a, feat_b):
        """计算两个特征的相关性（皮尔逊相关系数）"""
        a = feat_a.flatten()
        b = feat_b.flatten()

        if len(a) != len(b):
            min_len = min(len(a), len(b))
            a = a[:min_len]
            b = b[:min_len]

        a_mean = a - np.mean(a)
        b_mean = b - np.mean(b)

        num = np.dot(a_mean, b_mean)
        denom = np.sqrt(np.dot(a_mean, a_mean) * np.dot(b_mean, b_mean))

        if denom < 1e-10:
            return 0.0

        return float(num / denom)

    def get_state(self):
        """获取当前状态（给主循环和可视化用）"""
        detected = self.attack_detected
        self.attack_detected = False  # 读取后重置

        return {
            "is_running": self.is_running,
            "attack_hit": detected,
            "match_score": self.match_score,
            "rms": self.current_rms,
            "has_fingerprint": self.fp_feature is not None,
        }
