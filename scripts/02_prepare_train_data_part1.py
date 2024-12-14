# 第二步,从你的数据当中找到
# 1. ASR 和单帧对应的
# 2. ASR 和多帧对应的
# 3. 连续正确的 ASR 识别结果


from pathlib import Path
from correct_asr.utils.path import work_dir
from correct_asr.data import VideoLoader
from correct_asr.data.task import *
from correct_asr.data.gt_maker import *

import random
import json
import numpy as np
from tqdm import tqdm
import re


def gather():
    one_list_all = []
    interval_list_all = []
    for video_dir in (work_dir()  / 'demo' / 'video' ).glob('*'):
        mp4_file = list(video_dir.glob('*.mp4'))[0]
        loader = VideoLoader(mp4_file)
        one_list , interval_list = loader.gather_gt_for_task1_and_task2()

        one_list_all.extend(one_list)
        interval_list_all.extend(interval_list)
    return one_list_all , interval_list_all



if __name__ == '__main__':
    exported_dir = work_dir() / 'demo' / 'exported'
    exported_dir.mkdir(exist_ok=True)

    raw_dir = work_dir() / 'demo' / 'raw'
    raw_dir.mkdir(exist_ok=True)

    # 获取制作训练数据所需要的前置数据
    one_list_all, interval_list_all = gather()

    random.shuffle(one_list_all)
    random.shuffle(interval_list_all)


    # 设计针对单图的任务
    # 制作任务1数据
    # 给定一张图片,希望返回这张图片的字幕
    task1(one_list_all[: int(len(one_list_all) * 2//3) ] , exported_dir)
    make_gt1(exported_dir / 'single' , raw_dir)

    # 制作任务3的不修改数据
    # 给定一些连续的图片,但是这些图片对应的字幕内容都一致,希望模型能够去重并返回结果
    task3_1(one_list_all[int(len(one_list_all) * 2//3):] , exported_dir)
    make_gt3_1( exported_dir / 'multi' , raw_dir)


    # 设计针对多图的任务
    # 制作任务2的数据
    # 识别连续frame 当中的字幕,但是这些字幕内容可能存在一致的部分，也存在不一致的部分
    task2(interval_list_all[: int(len(interval_list_all)  //2) ] , exported_dir)
    make_gt2( exported_dir / 'multi-read' , raw_dir)


    # 利用大语言模型制作任务3的部分数据
    # 利用大语言模型生成一些错误的 asr 结果 （这部分建议使用 72B 的模型）
    task3_2(interval_list_all[int(len(interval_list_all) //2) :] , exported_dir)
    make_gt3_2(exported_dir / 'multi-correct' , raw_dir)



