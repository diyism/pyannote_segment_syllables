import onnxruntime as ort
import librosa
import numpy as np
from typing import List, Tuple
import os
import subprocess
from pathlib import Path

def init_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 1
    opts.log_severity_level = 3
    sess = ort.InferenceSession(model_path, sess_options=opts)
    return sess

def read_wav(path: str) -> Tuple[np.ndarray, int]:
    return librosa.load(path, sr=16000)

def compute_syllable_features(window: np.ndarray, sr: int, frame_length: int = 160) -> Tuple[np.ndarray, np.ndarray]:
    """
    计算用于音节分割的特征
    frame_length=160 对应 10ms
    """
    # 确保输入不为空
    if len(window) < frame_length:
        return np.array([]), np.array([])
        
    # 计算短时能量
    energy = librosa.feature.rms(
        y=window, 
        frame_length=frame_length,
        hop_length=frame_length//2
    )[0]
    
    # 计算过零率
    zcr = librosa.feature.zero_crossing_rate(
        window,
        frame_length=frame_length,
        hop_length=frame_length//2,
        center=False  # 避免填充问题
    )[0]
    
    return energy, zcr

def find_syllable_boundaries(
    energy: np.ndarray,
    zcr: np.ndarray,
    sr: int,
    min_duration: float = 0.18,  # 最小音节长度 0.2s
    max_duration: float = 0.35   # 最大音节长度 0.3s
) -> List[float]:
    """
    基于能量和过零率查找音节边界
    """
    if len(energy) == 0:
        return []
        
    # 使用动态阈值
    energy_mean = np.mean(energy)
    energy_threshold = energy_mean * 0.4  # 降低阈值
    
    # 寻找能量的局部最小值点
    boundaries = []
    win_size = int(min_duration * sr / (160/2))  # 将时间转换为帧数
    
    # 确保有足够的数据进行处理
    if len(energy) <= 2 * win_size:
        return []
        
    for i in range(win_size, len(energy) - win_size):
        # 使用更灵活的局部最小值检测
        if energy[i] < energy_threshold:
            local_min = True
            for j in range(1, min(win_size, 5)):  # 缩小检查范围
                if i-j < 0 or i+j >= len(energy):
                    local_min = False
                    break
                if energy[i] > energy[i-j] or energy[i] > energy[i+j]:
                    local_min = False
                    break
            if local_min:
                time = i * (160/2) / sr
                boundaries.append(time)
    
    return boundaries

def merge_close_boundaries(
    boundaries: List[float],
    min_duration: float = 0.18,
    max_duration: float = 0.35
) -> List[float]:
    """
    合并过近的边界点，分割过长的片段
    """
    if not boundaries:
        return []
    
    result = [0.0]  # 添加起始点
    
    for i in range(len(boundaries)):
        curr_time = boundaries[i]
        last_time = result[-1]
        duration = curr_time - last_time
        
        if duration < min_duration * 0.8:  # 稍微放宽最小间隔限制
            continue
            
        elif duration > max_duration:
            # 更平滑的分割
            n_points = max(1, int(duration / max_duration))
            interval = duration / (n_points + 1)
            for j in range(n_points):
                new_point = last_time + interval * (j + 1)
                if new_point - result[-1] >= min_duration * 0.8:
                    result.append(new_point)
        else:
            result.append(curr_time)
    
    return result

def segment_syllables(samples: np.ndarray, sr: int) -> List[Tuple[float, float]]:
    """
    对整个音频进行音节分割
    """
    # 检查输入是否为空
    if len(samples) == 0:
        return []
        
    # 计算特征
    energy, zcr = compute_syllable_features(samples, sr)
    
    # 如果特征为空，返回空列表
    if len(energy) == 0:
        return []
    
    # 查找边界点
    boundaries = find_syllable_boundaries(energy, zcr, sr)
    
    # 处理边界点
    boundaries = merge_close_boundaries(boundaries)
    
    # 确保添加终点
    if boundaries and len(samples) / sr - boundaries[-1] > 0.1:  # 如果末尾还有足够长的音频
        boundaries.append(len(samples) / sr)
    elif not boundaries and len(samples) > 0:  # 如果没有检测到边界但有音频
        boundaries = [0.0, len(samples) / sr]
    
    # 转换为时间段
    segments = []
    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        if end - start >= 0.1:  # 只添加足够长的段
            segments.append((start, end))
        
    return segments

def detect_silence(audio: np.ndarray, sr: int, frame_length: int = 160, 
                  silence_threshold: float = 0.1, min_silence_dur: float = 0.05) -> List[Tuple[float, float]]:
    """
    检测音频中的静音段
    """
    # 计算短时能量
    energy = librosa.feature.rms(
        y=audio,
        frame_length=frame_length,
        hop_length=frame_length//2
    )[0]
    
    # 动态确定静音阈值
    energy_threshold = np.max(energy) * silence_threshold
    
    # 找到低于阈值的帧
    is_silence = energy < energy_threshold
    
    # 将连续的静音帧组合成区间
    silence_regions = []
    start_frame = None
    min_frames = int(min_silence_dur * sr / (frame_length//2))
    
    for i, silent in enumerate(is_silence):
        if silent and start_frame is None:
            start_frame = i
        elif not silent and start_frame is not None:
            if i - start_frame >= min_frames:
                start_time = start_frame * (frame_length/2) / sr
                end_time = i * (frame_length/2) / sr
                silence_regions.append((start_time, end_time))
            start_frame = None
            
    # 处理最后一个静音区间
    if start_frame is not None and len(is_silence) - start_frame >= min_frames:
        start_time = start_frame * (frame_length/2) / sr
        end_time = len(is_silence) * (frame_length/2) / sr
        silence_regions.append((start_time, end_time))
    
    return silence_regions

def refine_syllable_segments(syllables: List[Tuple[float, float]], 
                           silence_regions: List[Tuple[float, float]], 
                           min_duration: float = 0.1) -> List[Tuple[float, float]]:
    """
    基于静音检测结果优化音节分割
    """
    refined_segments = []
    
    for start, end in syllables:
        # 检查当前音节是否与任何静音区域重叠
        segment_splits = [(start, end)]
        
        for silence_start, silence_end in silence_regions:
            new_splits = []
            for seg_start, seg_end in segment_splits:
                # 如果有重叠
                if silence_start < seg_end and silence_end > seg_start:
                    # 分割前半部分（如果足够长）
                    if silence_start - seg_start >= min_duration:
                        new_splits.append((seg_start, silence_start))
                    # 分割后半部分（如果足够长）
                    if seg_end - silence_end >= min_duration:
                        new_splits.append((silence_end, seg_end))
                else:
                    new_splits.append((seg_start, seg_end))
            segment_splits = new_splits
        
        # 添加所有有效的分割结果
        refined_segments.extend(segment_splits)
    
    # 按时间排序
    refined_segments.sort(key=lambda x: x[0])
    
    return refined_segments

def process_audio(audio_path: str, model_path: str) -> List[Tuple[float, float]]:
    """
    处理完整的音频文件
    """
    # 初始化VAD模型
    session = init_session(model_path)
    
    # 读取音频
    samples, sample_rate = read_wav(audio_path)
    
    # VAD处理
    frame_size = 160
    frame_start = 400
    window_size = sample_rate
    
    is_speech = False
    offset = frame_start
    speech_segments = []
    
    samples = np.pad(samples, (0, window_size), 'constant')
    
    step_size = window_size // 2
    for start in range(0, len(samples) - window_size, step_size):
        window = samples[start:start + window_size]
        ort_outs = session.run(None, {'input': window[None, None, :]})[0][0]
        
        for probs in ort_outs:
            predicted_id = np.argmax(probs)
            if predicted_id != 0:
                if not is_speech:
                    speech_start = start + offset
                    is_speech = True
            elif is_speech:
                speech_end = start + offset
                if speech_end - speech_start >= 0.1 * sample_rate:
                    speech_segments.append((
                        speech_start / sample_rate,
                        speech_end / sample_rate
                    ))
                is_speech = False
            offset += frame_size
            
    if is_speech:
        speech_end = len(samples)
        if speech_end - speech_start >= 0.1 * sample_rate:
            speech_segments.append((
                speech_start / sample_rate,
                speech_end / sample_rate
            ))
    
    # 对每个语音段进行处理
    all_syllables = []
    for start, end in speech_segments:
        seg_start = int(start * sample_rate)
        seg_end = int(end * sample_rate)
        speech_segment = samples[seg_start:seg_end]
        
        if len(speech_segment) > 0:
            # 检测静音段
            silence_regions = detect_silence(speech_segment, sample_rate)
            
            # 调整静音段的时间戳以匹配原始音频
            adjusted_silence = [(s + start, e + start) for s, e in silence_regions]
            
            # 获取初始音节分割
            syllables = segment_syllables(speech_segment, sample_rate)
            adjusted_syllables = [(s + start, e + start) for s, e in syllables]
            
            # 使用静音信息优化分割结果
            refined_syllables = refine_syllable_segments(adjusted_syllables, adjusted_silence)
            all_syllables.extend(refined_syllables)
    
    return all_syllables

def save_syllables_to_wav(
    input_wav: str,
    syllables: List[Tuple[float, float]],
    output_dir: str = "syllables"
) -> None:
    """
    Save each syllable as a separate WAV file using sox command.
    """
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Process each syllable
    for i, (start, end) in enumerate(syllables, 1):
        duration = end - start
        output_file = os.path.join(output_dir, f"{i:03d}.wav")
        
        # Construct sox command
        cmd = [
            "sox",
            input_wav,
            output_file,
            "trim",
            f"{start:.3f}",
            f"{duration:.3f}"
        ]
        
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Saved syllable {i:03d}: {start:.3f}s - {end:.3f}s (duration: {duration:.3f}s)")
        except subprocess.CalledProcessError as e:
            print(f"Error processing syllable {i}: {e}")
            print(f"Sox stderr: {e.stderr.decode()}")

if __name__ == '__main__':
    import sys
    if len(sys.argv) != 2:
        print("Usage: python script.py <audio_file>")
        sys.exit(1)
        
    # Process audio file
    audio_file = sys.argv[1]
    syllables = process_audio(
        audio_file,
        'segmentation-3.0.onnx'
    )
    
    print(f"Found {len(syllables)} syllables:")
    for start, end in syllables:
        print(f"{start:.3f}s - {end:.3f}s")
        
    # Save syllables to WAV files
    save_syllables_to_wav(audio_file, syllables)
