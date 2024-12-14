# 第二步,从你的数据当中找到
# 1. ASR 和单帧对应的
# 2. ASR 和多帧对应的
# 3. 连续正确的 ASR 识别结果


from pathlib import Path
from correct_asr.utils.path import work_dir
from correct_asr.data import VideoLoader
from correct_asr.data.task import *
from correct_asr.data.gt_maker import *
from correct_asr.utils.mllm import mllm_query

import random
import json
import numpy as np
from tqdm import tqdm
import re

# 从当前视频的 asr-ocr 结果当中，找到结果不匹配的数据
# 对这些数据,利用本项目提供的训练好的模型进行预标注，将结果保存
# 记录这些数据，并在后期利用标注工具进行标注


def gather():
    mismatch_all = []
    for video_dir in (work_dir()  / 'demo' / 'video' ).glob('*'):
        mp4_file = list(video_dir.glob('*.mp4'))[0]
        loader = VideoLoader(mp4_file)
        mismatch = loader.gather_mismatch()
        mismatch_all.extend(mismatch)
    return mismatch_all



if __name__ == '__main__':
    exported_dir = work_dir() / 'demo' / 'exported'
    exported_dir.mkdir(exist_ok=True)

    raw_dir = work_dir() / 'demo' / 'raw'
    raw_dir.mkdir(exist_ok=True)

    mismatch_all = gather()
    random.shuffle(mismatch_all)

    # 抽样个10个展示一下效果即可
    task3_3(mismatch_all, exported_dir)