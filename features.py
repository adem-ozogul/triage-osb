import random

def extract_video_features(video_path, frame_stride=3):
    """Simülasyon amaçlı sahte özellik çıkarımı"""
    feats = {f"feat_{i}": round(random.uniform(0,1),3) for i in range(5)}
    meta = {"frames": 120, "duration": 30}
    return feats, meta
