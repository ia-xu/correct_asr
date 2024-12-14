from pathlib import Path

def work_dir():
    return Path(__file__).resolve().parents[2]