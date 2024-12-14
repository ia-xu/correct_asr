
from correct_asr.data import VideoLoader
from correct_asr.utils.path import work_dir


url = 'http://127.0.0.1:11005/v1'
model = 'Qwen2-VL-2B-Instruct'

for video_dir in (work_dir()  / 'demo' / 'video' ).glob('*'):
    mp4_file = list(video_dir.glob('*.mp4'))[0]
    loader = VideoLoader(mp4_file)
    # 首先做 asr
    loader.asr()
    # 其次做 ocr
    loader.ocr()
    # 随后，使用当前模型结果对 ocr 结果进行矫正
    loader.rectify(model = model , url = url)
