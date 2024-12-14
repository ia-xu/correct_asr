
import random
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from correct_asr.utils.auto_resize_qwen import _process_image_qwen_with_autoresize,replace_tag

from pathlib import Path
from swift.llm.template.template import qwen
from swift.llm.template.template.qwen import Qwen2VLTemplate
from swift.cli.sft import sft_main


Qwen2VLTemplate.replace_tag = replace_tag

from swift.llm.dataset.register import DatasetMeta , register_dataset
from swift.llm.train.sft import SwiftSft
from swift.llm import get_model_tokenizer



train_files = list(Path('/root/training/train').glob('*.json'))
train_files = [str(_) for _ in train_files]

test_files = ['/root/training/test/task5.json' , '/root/training/test/task6.json']


for idx , file in enumerate(train_files):
    register_dataset(
        DatasetMeta(
            f'correct_asr_{idx}',
            dataset_path=file,
        )
    )

for idx , file in enumerate(test_files):
    register_dataset(
        DatasetMeta(
            f'val_correct_asr_{idx}',
            dataset_path=file,
        )
    )


# PYTHONPATH=`pwd` CUDA_VISIBLE_DEVICES=2,3,4,5  MAX_PIXELS=1254400 torchrun --nproc_per_node=4 --master_port 29501 correct_asr/training/sft.py correct_asr/training/sft_config.json
if __name__ == '__main__':
    sft_main()


