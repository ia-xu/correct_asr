
import random
from tqdm import tqdm
from dataclasses import dataclass
from pathlib import Path
from tqdm import tqdm
from correct_asr.utils.auto_resize_qwen import _process_image_qwen_with_autoresize,replace_tag

from pathlib import Path
from swift.llm.template.template import qwen
from swift.llm.template.template.qwen import Qwen2VLTemplate,Template

# qwen._process_image_qwen = _process_image_qwen_with_autoresize
Qwen2VLTemplate.replace_tag = replace_tag

from swift.llm.dataset.register import DatasetMeta , register_dataset
from swift.llm.train.sft import SwiftSft
from swift.llm.base import SwiftPipeline
from swift.llm import get_model_tokenizer

class nomodel(SwiftSft):
    def __init__(self,args = None):
        SwiftPipeline.__init__(self, args)
        self.args.save_args()
        self.train_msg = {}
        self._prepare_model_tokenizer()
        self._prepare_template(True)

    def _prepare_model_tokenizer(self):
        args = self.args
        self.model, self.processor = self._get_model_tokenizer(args.model, args.model_type, args.model_revision)
    def _get_model_tokenizer(self, model, model_type, model_revision):
        args = self.args
        kwargs = args.get_model_kwargs()
        # compat rlhf
        kwargs['model_id_or_path'] = model
        kwargs['model_type'] = model_type
        kwargs['model_revision'] = model_revision
        model_kwargs = {}
        if args.num_labels is not None:
            from transformers import AutoModelForSequenceClassification
            kwargs['automodel_class'] = AutoModelForSequenceClassification
            model_kwargs = {'num_labels': args.num_labels}
        if args.tuner_backend == 'unsloth':
            kwargs['unsloth_kwargs'] = {'load_in_4bit': args.quant_bits == 4}

        # 只是为了加载数据，不需要加载模型
        model, processor = get_model_tokenizer(
            **kwargs, model_kwargs=model_kwargs, use_unsloth=(args.tuner_backend == 'unsloth'),
            load_model = False
        )
        return model, processor


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



def load_data():
    sft = nomodel()
    train_dataset, val_dataset = sft._get_dataset()
    train_dataset, val_dataset = sft._encode_dataset(train_dataset, val_dataset)


load_data()

