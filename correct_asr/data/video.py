
from pathlib import Path
from correct_asr.utils import dl_models , video_utils
from correct_asr.config import ASR_URL_LOCAL ,OCR_URL_LOCAL
from loguru import logger
import json
import numpy as np

from tqdm import tqdm
from correct_asr.utils.mllm import mllm_query
import os
from PIL import  Image

class VideoLoader():

    def __init__(self, video_path):
        self.file_path = video_path
        self.file_dir = Path(video_path).parent
    @property
    def audio_file_path(self):
        # audio_file_path = self.file_path.parent / f'_{self.file_path.stem}.wav'
        audio_file_path = self.file_dir / f'{self.file_path.stem}.mp3'
        return audio_file_path

    @property
    def asr_file_path(self):
        asr_file_path = self.file_dir / f'asr_{self.file_path.stem}.json'
        return asr_file_path

    @property
    def ocr_file_path(self):
        ocr_file_path = self.file_dir / f'ocr_{self.file_path.stem}.json'
        return ocr_file_path

    def asr(self):
        if self.asr_file_path.exists():
            return
        logger.info('cvt2mp3 start...')
        video_utils.parser_audio(self.file_path, self.audio_file_path)
        logger.info('cvt2mp3 done...')
        asr_result = dl_models.asr(
            self.audio_file_path,
            'Whisper-Large',
            ASR_URL_LOCAL
        )
        with open(self.asr_file_path , 'w') as f:
            json.dump( asr_result , f , indent=2 , ensure_ascii=False )



    def ocr(self):
        if self.ocr_file_path.exists():
            return
        # 相对比较复杂，对每一个asr 的句子，尽可能找到能够进行匹配的 ocr 结果
        with open(self.asr_file_path) as f:
            asr_data = json.load(f)
        # 遍历每一个句子，进行 ocr 解析
        for sentence in tqdm(asr_data):
            s , e = sentence['offset_timestamp']
            # 先识别 mid 位置
            mid = ( s + e ) / 2
            ocrs = self.ocr_one_frame(mid )

            if any([sentence['text'] in _['text'] for _ in ocrs]):
                # 说明仅仅使用中间帧就可以搞定
                sentence['ocr'] = ocrs
            else:

                # 这种情况有可能是识别出错了
                # 尝试一下做更多的识别
                ocrs = []
                for frame_t in np.linspace(s, e, 4):
                    ocrs += self.ocr_one_frame(frame_t)

                # 去重
                prev = []
                filter_results = []
                for ocr in ocrs:
                    if ocr['text'] not in prev:
                        prev.append(ocr['text'])
                        filter_results.append(ocr)

                sentence['ocr'] = filter_results


                # in_text_ocr = [ ocr['rec'] for ocr in filter_results if ocr['rec'] in sentence['text']]
                # in_text_ocr = ''.join(in_text_ocr)
        with open(self.ocr_file_path , 'w') as f:
            json.dump(asr_data , f ,ensure_ascii=False , indent=2)

    def get_frame(self , timestamp):
        return video_utils.get_frame(
                self.file_path, timestamp
        )

    def ocr_one_frame(self ,timestamp):
        try:
            frame = video_utils.get_frame(
                self.file_path, timestamp
            )
            ocrs = dl_models.ocr(
                frame,
                'torchocr'
                # 'gotocr/ocr'
                , OCR_URL_LOCAL
            )
            for ocr in ocrs:
                ocr['frame'] = timestamp
                ocr['text'] = ocr.pop('rec')
            return ocrs
        except:
            return []



    def gather_gt_for_task1_and_task2(self):
        ocr_file = self.ocr_file_path
        with open(ocr_file) as f:
            ann = json.load(f)

        # 找到所有能够匹配上单帧结果的情况
        # 这一步是希望搜集这种数据
        # 1 输入单帧图片，返回字幕结果
        # 2 输入从asr start - asr end 的多帧图片。模型自动去重并返回去重后的和 asr 一致的结果
        matched = []
        for frame in ann:
            text = frame['text']
            match = False
            if any([ text == tb['text'] for tb in frame['ocr']]):
                match = True
            matched.append(match)

        # 找到上述结果当中的所有连续的匹配
        # 这一步是希望搜集这种数据：输入连续帧图片，让模型自动去重并返回多个字幕结果
        consecutive_matches = []
        start = 0
        while start < len(matched):
            if matched[start]:
                end = start
                while end < len(matched) and matched[end]:
                    end += 1
                consecutive_matches.append((start, end))
                start = end
            else:
                start += 1


        consecutive_matches = np.array(consecutive_matches)
        # 不能够支持做连续数据的，记为 one
        one_  = consecutive_matches[(consecutive_matches[:,1] - consecutive_matches[:,0]) == 1]
        # 能够支持做连续数据的，记为 other
        other = consecutive_matches[(consecutive_matches[:,1] - consecutive_matches[:,0]) >= 2]

        one_list = []
        interval_list = []

        # 从 one 当中随机抽样
        for idx in np.random.choice(one_[:,0],min(len(one_),10),replace = False)      :
            text = ann[idx]['text']
            fstart = ann[idx]['offset_timestamp'][0]
            fend = ann[idx]['offset_timestamp'][1]
            for bb in ann[idx]['ocr']:
                if bb['text'] == text:
                    one_list.append(
                        {
                            'text': text,
                            # 记录当前 asr 结果所匹配的 帧
                            'frame': bb['frame'],
                            'source': self.file_path,
                            'fstart': fstart,
                            'fend': fend
                        }
                    )
        # 从 consecutive 抽样
        interval_ids = np.random.choice(len(other), min(len(other), 10), replace=False)
        for interval_id in interval_ids:
            interval = other[interval_id]
            s = interval[0]
            e = interval[1]
            if e - s > 3:
                # 抽取一个区间
                s = np.random.randint(s,e-3)
                e = s + 3

            sub_ann = ann[s:e]
            texts = []
            bbs = []
            for sub in sub_ann:
                texts.append(sub['text'])
                for ocr in sub['ocr']:
                    if ocr['text'] in [sub['text'],sub.get('funasr_result','~~~')]:
                        bbs.append(ocr)


            jtext = ' '.join(texts)
            fstart = sub_ann[0]['offset_timestamp'][0]
            estart = sub_ann[-1]['offset_timestamp'][0]

            bbf_start = min([ bb['frame'] for bb in bbs])
            bbf_end = max([ bb['frame'] for bb in bbs])

            interval_list.append(
                {
                    'jtext':jtext,
                    'texts': texts ,
                    # asr 的开始和结束
                    'start': fstart,
                    'end': estart,
                    # 找到 匹配的 bb 的开始和结束的位置，精确对应到 frame 上
                    'bbf_start': bbf_start,
                    'bbf_end': bbf_end,
                    'source' : self.file_path
                }
            )

        return one_list , interval_list


    def gather_mismatch(self):
        ocr_file = self.ocr_file_path
        with open(ocr_file) as f:
            ann = json.load(f)

        mismatch = []
        for sentence in ann:
            text = sentence['text']
            if self.match_ocr(sentence):
                continue
            # 剩下的就是 ocr 和 asr 无法匹配的结果
            mismatch.append({
                'text' : sentence['text'],
                'start' : sentence['offset_timestamp'][0],
                'end' : sentence['offset_timestamp'][1],
                'source' : self.file_path
            })
        return mismatch

    def _clean(self, text):
        for spec in ['，', ',', '.', '。', '、', '：', ':', '；', '；', '?', '？','！','!']:
            text = text.replace(spec, '')
        text = text.replace(' ', '')
        # 移除语气词
        for spec in ['啊', '呢', '吧', '的', '那', '得', '了', '地', '么',
                     '嘛', '呀', '哦', '嗯', '呵', '嘿', '哈', '哟', '呦', '呐', '咦', '唉', '哎', '喔', '哇', '啦',
                     ]:
            text = text.replace(spec, '')
        from collections import OrderedDict
        chinese_to_arabic = OrderedDict({
            '零': '0', '十': '10', '百': '00', '一': '1', '二': '2', '三': '3', '四': '4',
            '五': '5', '六': '6', '七': '7', '八': '8', '九': '9'
        })

        for pattern, tgt in chinese_to_arabic.items():
            text = text.replace(pattern, tgt)
        return text


    def match_ocr(self , sentence):
        text = sentence['text']
        ocr = sentence['ocr']
        match = False
        if any([self._clean(text) == self._clean(tb['text']) for tb in sentence['ocr']]):
            match = True
        else:
            # 尝试用多个结果去匹配
            in_text_ocr = [ocr['text'] for ocr in sentence['ocr'] if ocr['text'] in sentence['text']]
            in_text_ocr = ''.join(in_text_ocr)
            if self._clean(in_text_ocr) == self._clean(sentence['text']):
                match = True

            flag = np.array([False]* len(text))
            for ocr in sentence['ocr']:
                if ocr['text'] in sentence['text']:
                    start = sentence['text'].index(ocr['text'])
                    flag[start: start + len(ocr['text'])] = True
            if all(flag):
                match = True
        return match

    def small_frame(self, frame, pix=480):
        if min(frame.size) > pix:
            frame_small = frame.resize((int(frame.width * pix / min(frame.width, frame.height)), pix), Image.ANTIALIAS)
        else:
            frame_small = frame
        return frame_small

    def rectify(self,  model , url):

        with open(self.ocr_file_path) as f:
            asr = json.load(f)

        for sentence in asr:
            text = sentence['text']
            s, e = sentence['offset_timestamp']
            mid = (s + e) / 2
            frame = self.get_frame(mid)

            if self.match_ocr(sentence):
                continue

            out_mid = mllm_query(
                '<image>请识别图片当中的字幕内容。',
                [self.small_frame(frame)],
                model=model,
                url=url,
            )
            if out_mid == sentence['text']:
                continue

            images = video_utils.interpolate(
                self.file_path, s, e, 2
            )

            out_ht = mllm_query(
                '<image><image>请识别如下多张视频截图当中的字幕内容。如果有重复的字幕，请去除重复项',
                [self.small_frame(im) for im in images],
                url,
                model,
            )
            if self._clean(sentence['text']) in self._clean(out_ht):
                # 说明只是字幕位置不对
                continue

            if len(out_ht.split(' ')) == 2:
                out_hmt = out_ht.split(' ')
                out_hmt = out_hmt[0] + out_mid + out_hmt[1]
                if self._clean(sentence['text']) in self._clean(out_hmt):
                    continue

            rectified_result = self._rectify(
                self.file_path,
                sentence['text'],
                s,
                e,
                model = model,
                url=url
            )
            if rectified_result == sentence['text']:
                continue
            print(sentence['text'], rectified_result)

    def _rectify(self ,mp4_file , text , start , end , model , url):
        prompt = """
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
        start = start - 0.5
        end = end + 0.5
        images = video_utils.interpolate(
            mp4_file, start, end, 5
        )
        images = [self.small_frame(im, 720) for im in images]
        query = prompt.format(
            asr_result=text
        ).replace('<image>', '<image>' * len(images))

        response = mllm_query(
            query,
            images,
            url=url,
            model=model

        )
        return response
