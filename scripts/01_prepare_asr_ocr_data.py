

# 第一步,你需要对自己关注的视频数据进行 asr 和 ocr 解析
from pathlib import Path
from correct_asr.utils.path import work_dir
from correct_asr.data import VideoLoader
# 获取 ocr 和 asr 结果
## 参考项目 https://github.com/ia-xu/aio_exporter/
## 启动该项目下的 asr / ocr 服务,以获取 asr / ocr 结果


for video_dir in (work_dir()  / 'demo' / 'video' ).glob('*'):
    mp4_file = list(video_dir.glob('*.mp4'))[0]
    loader = VideoLoader(mp4_file)
    # 首先做 asr
    loader.asr()
    # 其次做 ocr
    loader.ocr()

