from .video import VideoLoader
import json
from tqdm import tqdm
import numpy as np

def task1(single_list, exported_dir):
    exported_dir_task1 = exported_dir / 'single'
    exported_dir_task1.mkdir(exist_ok=True)
    for frame in single_list:
        try:
            loader = VideoLoader(frame['source'])
            frame_im = loader.get_frame(frame['frame'])
            new_name = frame['source'].with_suffix(f"._{frame['frame']}.jpg").name
            new_raw = frame['source'].with_suffix(f"._{frame['frame']}.json").name
            frame_im.save(exported_dir_task1 / new_name)
            frame['source'] = str(frame['source'])
            with open(exported_dir_task1 / new_raw , 'w') as f:
                json.dump(
                    {
                        'caption' : frame['text'],
                        **frame
                    } , f ,
                    ensure_ascii=False
                )
        except:
            pass


def task3_1(single_list, exported_dir):
    exported_dir_task2 = exported_dir / 'multi'
    exported_dir_task2.mkdir(exist_ok=True)
    for sample in single_list:
        try:
            loader = VideoLoader(sample['source'])
            start = sample['fstart'] - 0.5
            fend = sample['fend'] + 0.5
            frames = np.linspace(start , fend , 5 )
            new_name = sample['source'].with_suffix(f"._{sample['frame']}.jpg").stem
            folder_name = exported_dir_task2 / new_name
            folder_name.mkdir(exist_ok=True)
            for frame in frames:
                frame_im = loader.get_frame(frame)
                frame_im.save(folder_name / f'{frame}.jpg')

            sample['source'] = str(sample['source'])
            with open(folder_name / 'index.json','w') as f:
                json.dump(
                    sample ,  f , ensure_ascii=False
                )
        except:
            pass

def task2(multi , exported_dir):
    exported_dir_task3 = exported_dir / 'multi-read'
    exported_dir_task3.mkdir(exist_ok=True)
    for sample in tqdm(multi):
        try:
            # 直接读取5 frame 的内容
            loader = VideoLoader(sample['source'])
            start = sample['bbf_start']
            end = sample['bbf_end']
            frame_ts = np.linspace(start, end, 5)

            new_name = sample['source'].with_suffix(f"._{start}_{end}.jpg").stem
            folder_name = exported_dir_task3 / new_name
            folder_name.mkdir(exist_ok=True)
            for frame_t in frame_ts:
                frame_im = loader.get_frame(frame_t)
                frame_im.save(folder_name / f'{frame_t}.jpg')

            sample['source'] = str(sample['source'])
            with open(folder_name / 'index.json', 'w') as f:
                json.dump(
                    sample, f, ensure_ascii=False
                )
        except:
            pass

def task3_2(multi , exported_dir):
    exported_dir_task4 = exported_dir / 'multi-correct'
    exported_dir_task4.mkdir(exist_ok=True)
    try:
        from langchain_community.chat_models import ChatOpenAI
        # 更换为你自己的 openai 模型
        llm = ChatOpenAI(
            model_name="qwen2_5-14b-instruct",
            openai_api_base="http://127.0.0.1:11004/v1",
            openai_api_key="none"
        )

        import pylcs
        for frames in multi:
            original_text = frames['jtext']
            out = llm.invoke(
                f"""
帮我往当前asr结果当中随机执行如下操作
替换内容为一些错别字,让它听上去更像是真实场景下的asr结果,并且不要改变原意
你应该尽量使用同音替换,多字或者漏字等方式修改原始结果
asr结果:{original_text}
如果不好添加错别字,直接返回原始结果。返回的结果不能有繁体字
你只需要输出添加了错别字的结果,返回方式如下
<wrong>
...
</wrong>
"""
            ).content
            # extract
            out = re.search(r'<wrong>(.*?)</wrong>', out, re.DOTALL).group(1)
            out = out.strip()
            c = pylcs.lcs(out, original_text)
            if c / len(original_text) < 0.8:
                continue
            # 认为大体相似
            print(out, original_text)
            # 保存5帧内容
            loader = VideoLoader(frames['source'])
            start = frames['start'] - 0.5
            fend = frames['end'] + 0.5
            frame_ts = np.linspace(start, fend, 5)

            new_name = frames['source'].with_suffix(f"._{frames['start']}_{frames['end']}.jpg").stem
            folder_name = exported_dir_task4 / new_name
            folder_name.mkdir(exist_ok=True)
            for frame_t in frame_ts:
                frame_im = loader.get_frame(frame_t)
                frame_im.save(folder_name / f'{frame_t}.jpg')

            frames['source'] = str(frames['source'])
            frames['asr'] = out
            with open(folder_name / 'index.json', 'w') as f:
                json.dump(
                    frames, f, ensure_ascii=False
                )

    except:
        pass


def task3_3(mismatch_all , exported_dir):


    from correct_asr.config import CAPTION_URL_LOCAL
    from correct_asr.utils.mllm import mllm_query
    from .gt_maker import task3_prompts
    from correct_asr.utils import video_utils
    from correct_asr.infer.process import rectify

    task3_3_dir = exported_dir / 'multi-mismatch'
    task3_3_dir.mkdir(exist_ok = True)

    task3_3_dir = task3_3_dir / '0'
    task3_3_dir.mkdir(exist_ok=True)

    # 生成模型的矫正结果
    for sentence in mismatch_all:
        loader = VideoLoader(sentence['source'])
        s = sentence['start'] - 0.5
        e = sentence['end'] + 0.5

        response = rectify(loader.file_path , sentence['text'] , s , e , 4)
        sentence['correct-ocr'] = response
        print(sentence['text'] ,  '->' ,  sentence['correct-ocr'])


        # 制作数据
        s_, e_ = sentence['start'] , sentence['end']

        # 正式识别的时候也会做这个 offset
        s = s_ - 0.5
        e = e_ + 0.5
        new_name = sentence['source'].with_suffix(f"._{s_}_{e_}.jpg").stem
        save_dir = task3_3_dir / new_name
        save_dir.mkdir(exist_ok=True)
        try:
            for frame_t in np.linspace(s, e, 5):
                frame_im = loader.get_frame(frame_t)
                frame_im.save(save_dir / f'{frame_t}.jpg')
        except:
            continue

        sentence['source'] = str(sentence['source'])


        with open(save_dir / 'index.json', 'w') as f:
            json.dump(
                sentence, f, ensure_ascii=False
            )







