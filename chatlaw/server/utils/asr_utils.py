"""服务端语音转文字（ASR）工具。

从原客户端 ``common_utils.py`` 迁移而来。重构后语音转文字在服务端完成，
客户端只负责采集并上传原始音频。

本模块不依赖任何推理引擎（vllm / transformers），可独立导入测试。
"""
import os
import time
import uuid

import numpy as np
import soundfile as sf
import librosa


def speech_to_text(audio, target_sr, audio_cache_dir, audio_model):
    """
    audio: (sample_rate, numpy_array)
    return: 中文文本
    """
    if audio is None:
        return "未检测到语音输入"

    sr, data = audio

    # ---------- 1. 转单声道 ----------
    if data.ndim > 1:
        data = data.mean(axis=1)

    data = data.astype(np.float32)

    # ---------- 2. 重采样到 16k ----------
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)

    # ---------- 3. 幅值归一化（防止过小） ----------
    max_val = np.max(np.abs(data))
    if max_val > 0:
        data = data / max_val * 0.9

    # ---------- 4. 时长校验 ----------
    duration = len(data) / target_sr
    if duration < 0.8:
        return "语音过短，请说完整一句话"

    # ---------- 5. 写入固定目录 wav（避免 tempfile 坑） ----------
    wav_name = f"{uuid.uuid4().hex}.wav"
    wav_path = os.path.join(audio_cache_dir, wav_name)

    sf.write(wav_path, data, target_sr, subtype="PCM_16")

    # 确保文件完全落盘（Windows 必须）
    time.sleep(0.05)

    # ---------- 6. 调用 Paraformer ONNX ----------
    try:
        result = audio_model([wav_path])
    except Exception as e:
        return f"ASR 推理异常: {e}"
    finally:
        try:
            os.remove(wav_path)
        except Exception:
            pass

    # ---------- 7. 正确解析 funasr-onnx 返回 ----------
    # 返回格式示例：
    # [{'preds': ('如何理解等差数列', ['如','何','理','解','等','差','数','列'])}]
    if not result or not isinstance(result, list):
        return "未识别到有效内容"

    preds = result[0].get("preds", None)
    if not preds or not isinstance(preds, tuple):
        return "未识别到有效内容"

    text = preds[0].strip()
    if not text:
        raise ValueError("语音未识别到有效内容！")

    return text


__all__ = ["speech_to_text"]
