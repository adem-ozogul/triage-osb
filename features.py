import random

def extract_video_features(video_path, name_call_sec=None, frame_stride=3):
    """
    Dummy feature extractor for simulation.
    In the real version, this would analyze the video with MediaPipe.
    """
    # Örnek: 10 adet rastgele özellik üretelim
    feats = {f"feat_{i}": round(random.uniform(0, 1), 3) for i in range(10)}
    meta = {"frames_analyzed": 100, "duration": 30, "name_call_sec": name_call_sec}
    return feats, meta
