
# 替换原来的 qwen 处理函数，增加一个对多图自动进行 resize 的逻辑
# from swift.llm.utils.template import get_env_args
from swift.llm.template.template.qwen import get_env_args
import numpy as np
from typing import Any, Dict, List, Literal, Optional
from swift.llm.template.template_inputs import StdTemplateInputs
from swift.llm.template.utils import Context
from swift.llm.template.vision_utils import load_video_qwen2

def _process_image_qwen_with_autoresize(image , size_list , index):
    from qwen_vl_utils.vision_process import IMAGE_FACTOR, MIN_PIXELS, MAX_PIXELS, smart_resize
    size_factor = get_env_args('size_factor', int, IMAGE_FACTOR)
    # resize
    resized_height = get_env_args('resized_height', int, None)
    resized_width = get_env_args('resized_width', int, None)
    if resized_height and resized_width:
        resized_height, resized_width = smart_resize(
            resized_height,
            resized_width,
            factor=size_factor,
        )
    else:
        # # 如果设置了这个参数，就把上面 1 / 3 的部分切掉
        # cut_up = get_env_args('cut_up', bool, False)
        # if cut_up:
        #     # print('Removing top 1/3 of the image')
        #     height = image.size[1]
        #     new_height = height // 3
        #     image = image.crop((0, new_height, image.size[0], height))

        width, height = image.size
        min_pixels = get_env_args('min_pixels', int, MIN_PIXELS)
        max_pixels = get_env_args('max_pixels', int, MAX_PIXELS)

        if len(size_list) > 1:
            original_pixels = size_list[index][0] * size_list[index][1]
            total_original_pixels = sum(height * width for width, height in size_list)
            max_pixels = int(original_pixels / total_original_pixels * max_pixels)
        else:
            # 单图的话,不需要用太大的推理成本
            max_pixels = 960 * 28 * 28

        resized_height, resized_width = smart_resize(
            height,
            width,
            factor=size_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )
    image = image.resize((resized_width, resized_height))
    return image


def replace_tag(self, media_type: Literal['image', 'video', 'audio'], index: int,
                inputs: StdTemplateInputs) -> List[Context]:
    assert media_type in {'image', 'video'}
    if media_type == 'image':
        size_list = [image.size for image in inputs.images]
        inputs.images[index] = _process_image_qwen_with_autoresize(inputs.images[index] ,size_list , index)
        return ['<|vision_start|><|image_pad|><|vision_end|>']
    else:
        inputs.videos[index] = load_video_qwen2(inputs.videos[index])
        return ['<|vision_start|><|video_pad|><|vision_end|>']