# Copyright (c) Alibaba, Inc. and its affiliates.
from swift.llm import deploy_main
from correct_asr.utils.auto_resize_qwen import _process_image_qwen_with_autoresize,replace_tag
from swift.llm.template.template import qwen
from swift.llm.template.template.qwen import Qwen2VLTemplate

Qwen2VLTemplate.replace_tag = replace_tag
# 替换原先 qwen 的 resize 方法
# qwen._process_image_qwen = _process_image_qwen_with_autoresize

if __name__ == '__main__':
    # CUDA_VISIBLE_DEVICES=4 python3 correct_asr/infer/deploy.py configs/qwen2vl-correct-asr.json
    deploy_main()


