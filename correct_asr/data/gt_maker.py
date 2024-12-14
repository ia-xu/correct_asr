# 对若干任务,制作训练所需要的GT,构成若干个 json 文件
import os
from pathlib import Path
import json
import random
from loguru import logger
from tqdm import tqdm
import pandas as pd
import re
task1_prompts = [
    '<image>请识别图片当中的字幕内容。',
    '<image>请找出图片中的字幕。',
    '<image>这张图片中的字幕是什么？',
    '<image>请描述图片中的字幕。',
    '<image>图片中的字幕内容是什么？',
    '<image>请找出图片中的字幕内容。',
    '<image>图片中的字幕是什么？',
    '<image>请描述图片中的字幕内容。',
    '<image>图片中的字幕内容有哪些？',
]

task2_prompts = [
    '<image>请识别如下多张视频截图当中的字幕内容。如果有重复的字幕，请去除重复项',
    '<image>请找出如下多张视频截图中的字幕。如果有重复的字幕，请去除重复项',
    '<image>这些视频截图中的字幕是什么？如果有重复的字幕，请去除重复项',
    '<image>请描述这些视频截图中的字幕。如果有重复的字幕，请去除重复项',
    '<image>这些视频截图中的字幕内容是什么？如果有重复的字幕，请去除重复项',
    '<image>请找出这些视频截图中的字幕内容。如果有重复的字幕，请去除重复项',
    '<image>这些视频截图中的字幕是什么？如果有重复的字幕，请去除重复项',
    '<image>请描述这些视频截图中的字幕内容。如果有重复的字幕，请去除重复项',
    '<image>这些视频截图中的字幕内容有哪些？如果有重复的字幕，请去除重复项',
]

task3_prompts = """
我将向你提供一段视频内容对应的音频的ASR语音识别的结果,该结果可能正确,也可能包含识别错误的内容。
同时,我会向你提供这段视频内容的一些截图,截图当中含有这段音频对应的视频片段的字幕内容。
请根据这些截图中的字幕内容,对ASR语音识别的结果进行修正。
1. 截图内容和ASR识别结果没有完全对应,你需要从截图当中的字幕中找到ASR语音识别结果对应的片段
2. 如果截图内容和ASR语音识别结果不匹配,你不需要修改ASR的结果
3. 视频当中可能存在错别字,因此你需要仔细判断是否需要对ASR结果进行修正

ASR语音识别结果:
<asr>
{asr_result}
</asr>

截图:
<image>

请根据截图中的字幕内容,对ASR语音识别的结果进行修正。
"""

def remove(train_dir, test_dir, task_file='task1.json'):
    """Remove existing JSON files from the train and test directories."""
    json_files = [train_dir / task_file, test_dir / task_file]
    for json_file in json_files:
        if json_file.exists():
            os.remove(json_file)
            logger.info(f"Removed file: {json_file}")
def save(data, train_dir, test_dir, task_file='task1.json' , r = 0.9):
    """Split the data into training and testing sets and save them as JSON files."""
    random.shuffle(data)
    train_size = int(len(data) * r)
    train_data = data[:train_size]
    test_data = data[train_size:]

    train_json_file = train_dir / task_file
    test_json_file = test_dir / task_file

    with open(train_json_file, 'w' , encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=4)

    with open(test_json_file, 'w' , encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=4)

    logger.info(f"Saved {len(train_data)} training samples to {train_json_file}")
    logger.info(f"Saved {len(test_data)} testing samples to {test_json_file}")



def make_gt1(single_dir,raw_dir):

    train_dir = raw_dir / 'train'
    test_dir = raw_dir / 'test'

    remove(train_dir, test_dir, 'task1.json')

    tasks = list(single_dir.glob('*.json'))
    data = []
    for task_file in tasks:
        with open(task_file) as f:
            ann = json.load(f)
        img_file = task_file.with_suffix('.jpg')
        assert img_file.exists()
        formatted_data = {
            'query': random.choice(task1_prompts),
            'response': ann['caption'],
            'images': [str(img_file)]
        }
        data.append(formatted_data)

    save(data, train_dir, test_dir, 'task1.json')


def make_gt3_1( multi_dir , raw_dir):
    train_dir = raw_dir / 'train'
    test_dir = raw_dir / 'test'
    remove(train_dir, test_dir, 'task3_1.json')
    data = []
    for task_dir in multi_dir.glob('*'):
        try:
            with open(task_dir / 'index.json', 'r') as f:
                index_data = json.load(f)
        except:
            continue

        # 读取图片
        image_files = sorted(task_dir.glob('*.jpg'), key=lambda x: x.stem)
        images = [
            str(_) for _ in image_files
        ]
        task3_prompts_ = task3_prompts.format(
            asr_result=index_data['text'],
        )
        data.append(
            {
                'query': task3_prompts_.replace('<image>', '<image>' * len(images)),
                'response': index_data['text'],
                'images': images
            }
        )

    save(data, train_dir, test_dir, 'task3_1.json')


def make_gt2(multi_read , raw_dir):
    train_dir = raw_dir / 'train'
    test_dir = raw_dir / 'test'

    remove(train_dir , test_dir , 'task2.json')

    data = []
    for task_dir in tqdm(multi_read.glob('*')):
        index_json_path = task_dir / 'index.json'

        assert index_json_path.exists()
        with open(index_json_path, 'r') as f:
            index_data = json.load(f)
            # Process index_data as needed
            # print(index_data)

        caption = index_data['jtext']
        # 遍历这个文件夹下面的所有图片，并按照文件名排序

        image_files = sorted(task_dir.glob('*.jpg'), key=lambda x: x.stem)
        images = [
            str(_) for _ in image_files
        ]
        data.append(
            {
                'query': random.choice(task2_prompts).replace('<image>', '<image>' * len(images)),
                'response': caption,
                'images': images
            }
        )
    save(data, train_dir, test_dir, 'task2.json')

def make_gt3_2(multi_correct , raw_dir):

    train_dir = raw_dir/ 'train'
    test_dir = raw_dir / 'test'

    remove(train_dir, test_dir, 'task3_2.json')

    folders = list(multi_correct.glob('*'))
    data = []
    for folder in folders:
        # read index.json
        try:
            with open(folder / 'index.json', 'r') as f:
                index_data = json.load(f)
        except:
            continue
            # Process index_data as needed
        # gather images
        image_files = sorted(folder.glob('*.jpg'), key=lambda x: x.stem)
        images = [
            str(_) for _ in image_files
        ]
        task3_prompts_ = task3_prompts.format(
            asr_result=index_data['asr'],
        )
        data.append(
            {
                'query': task3_prompts_.replace('<image>', '<image>' * len(images)),
                'response': index_data['jtext'],
                'images': images
            }
        )
    save(data, train_dir, test_dir, 'task3_2.json')


def make_gt3_3(ann_file , raw_dir):
    train_dir = raw_dir / 'train'
    test_dir = raw_dir / 'test'


    remove(train_dir, test_dir, 'task3_3.json')

    data_frame = pd.read_csv(ann_file)

    data = []
    for _, raw in data_frame[data_frame['status'] == '已确认'].iterrows():
        source = Path(raw['source'])
        with open(source, encoding='utf-8') as f:
            index_data = json.load(f)

        image_files = sorted(source.parent.glob('*.jpg'), key=lambda x: x.stem)
        images = [
            str(_) for _ in image_files
        ]
        task3_prompts_ = task3_prompts.format(
            asr_result=index_data['text'],
        )
        data.append(
            {
                'query': task3_prompts_.replace('<image>', '<image>' * len(images)),
                'response': raw['rectified'],
                'images': images
            }
        )

    save(data, train_dir, test_dir, 'task3_3.json', 0.9)
