

# 尝试基于视频字幕对 ASR 结果进行矫正

## data preparation

    - scripts/01_prepare_asr_ocr_data.py
        - 为每一个视频准备基础的 asr 数据和 ocr 数据
    - scripts/02_prepare_train_data_part1.py
        - 尝试为每一个视频准备一些不需要人工标注就能够生成的训练数据
    - scripts/03_prepare_train_data_part2.py        \
        - 从视频当中搜集一些 asr 结果可能错误的数据
    - streamlit run scripts/04_annotate_mismatch.py
        - 对 03 脚本生成的结果，利用 streamlit 启动的标注页面进行标注
    - scripts/05_prepare_train_data_part3.py
        - 对标注结果处理成数据

## training
    - 首先执行 correct_asr/traning/load_data.py 
        - 函数的目的是在避免加载模型参数的情况下加载模型训练相关的数据/调试数据相关的 processor (如果有的话)
        - 修改函数加载 train_files / test_files 的部分
        - 进行数据加载部分的调试
        - 你需要同时对 sft_debug.json 文件进行修改，调整成适合自己训练的参数
        - python3 correct_asr/training/load_data.py correct_asr/training/sft_debug.json 

    - 完成代码调试后，将  sft_debug.json 相关改动的代码调整到 correct_asr/training/sft_config.json 当中
        - 并修改 deepseed 指向 training/zero2.json 
    - 开始训练
        - 修改 training/sft.py 当中模型训练相关的数据
        - PYTHONPATH=`pwd` CUDA_VISIBLE_DEVICES=2,3,4,5  MAX_PIXELS=1254400 torchrun --nproc_per_node=4 --master_port 29501 correct_asr/training/sft.py correct_asr/training/sft_config.json


## infer server 启动

    - 修改 configs/qwen2vl-correct-asr.json   
    - 执行
        - CUDA_VISIBLE_DEVICES=4 python3 correct_asr/infer/deploy.py configs/qwen2vl-correct-asr.json

## 调用

    - 跑一下 correct_asr/infer/infer.py 进行矫正
    