from pathlib import Path
from correct_asr.utils.path import work_dir
from correct_asr.data.gt_maker import make_gt3_3

if __name__ == '__main__':

    raw_dir = work_dir() / 'demo' / 'raw'
    raw_dir.mkdir(exist_ok=True)

    train_dir = raw_dir / 'train'
    test_dir = raw_dir / 'test'
    train_dir.mkdir(exist_ok=True)
    test_dir.mkdir(exist_ok=True)

    exported_dir = work_dir() / 'demo' / 'exported'
    for subdir in (exported_dir / 'multi-mismatch').glob('*'):
        ann_file = subdir.parent / f'{subdir.stem}.csv'
        make_gt3_3(ann_file , raw_dir)
