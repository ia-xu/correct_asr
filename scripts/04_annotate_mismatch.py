import time

# 对第7步制作的标签利用 streamlit 进行修正
import streamlit as st
from pathlib import Path
import json
import pandas as pd
from PIL import Image
import numpy as np
import random
from mmcv import imrescale


from correct_asr.utils.path import work_dir

st.set_page_config(page_title='本地标注', page_icon="🧊", layout="wide")


def create_gt(current_dir, csv_filename):
    labels = []
    for jsonfile in current_dir.rglob('*.json'):
        with open(jsonfile,encoding='utf-8') as f:
            data = json.load(f)
        asr = data['text']
        correct = data['correct-ocr']
        labels.append(
            {
                'org':  asr,
                'rectified' : correct,
                'equal' :  '相等' if asr == correct else '不相等',
                'status' : '未确认',
                'source' : str(jsonfile)
            }
        )

    # Convert labels to DataFrame
    labels = pd.DataFrame(labels)

    # Random shuffle the DataFrame
    labels = labels.sample(frac=1).reset_index(drop=True)

    # Add an 'order' column with sequential numbers
    labels['order'] = range(1, len(labels) + 1)

    # Save the shuffled DataFrame to CSV
    labels.to_csv(csv_filename, index=False)
    # pd.DataFrame(labels).to_csv(csv_filename , index= False)
    return


def clean(x):
    return x.replace(' ','')

def cat_img(images):
    # 判断是否是竖屏视频
    v = None
    if images[0].size[1] > images[0].size[0]:
        v = True
        # 如果是一个竖屏视频，把这些图片横向拼接起来
        total_width = sum(img.size[0] for img in images)
        max_height = max(img.size[1] for img in images)
        combined_image = Image.new('RGB', (total_width + images[0].size[1], max_height))
        x_offset = 0
        for img in images:
            combined_image.paste(img, (x_offset, 0))
            x_offset += img.size[0]

    if images[0].size[1] <= images[0].size[0]:
        v = False
        # 如果是横屏视频，按照两行三列的方式进行拼接
        rows = 2
        cols = 3
        total_width = max(img.size[0] for img in images) * cols
        total_height = max(img.size[1] for img in images) * rows
        combined_image = Image.new('RGB', (total_width, total_height), color='white')
        x_offset = 0
        y_offset = 0
        for i, img in enumerate(images):
            combined_image.paste(img, (x_offset, y_offset))
            x_offset += img.size[0]
            if (i + 1) % cols == 0:
                x_offset = 0
                y_offset += img.size[1]

    image = np.array(combined_image)
    image = imrescale(image, (min(image.shape[:2]), 4096))
    return image , v

def load_image(row):

    images = sorted(list(Path(row['source']).parent.glob('*.jpg')))
    images = [Image.open(_) for _ in images]
    image, _ = cat_img(images)
    return Image.fromarray(image)


def vis2x3(image):
    # 将图片切分成 2 x 3 的小图片
    w, h = image.size
    small_w = w // 3
    small_h = h // 2
    cut_images = []
    for i in range(2):
        for j in range(3):
            # Calculate the height range for the small image

            start_h = small_h * i + int(small_h * 2 / 3)
            end_h = small_h * i + small_h
            box = (j * small_w, start_h, (j + 1) * small_w, end_h)
            small_image = image.crop(box)
            cut_images.append(small_image)

    new_im = np.zeros((2 * small_h // 3, small_w * 3, 3), dtype=np.uint8)
    # Display the cut images in a 2x3 grid and stitch them back into new_im
    for i in range(2):
        for j in range(3):
            cut_image = cut_images[i * 3 + j]
            # Stitch the cut image back into new_im
            start_h = int(small_h / 3 * i)
            end_h = int(small_h / 3 * (i + 1))
            new_im[start_h:end_h, j * small_w:(j + 1) * small_w] = np.array(cut_image)
    new_im = Image.fromarray(new_im)
    if new_im.size[1] > 250:
        # Resize the height to 250 while maintaining the aspect ratio
        new_im = new_im.resize((int(new_im.size[0] * (250 / new_im.size[1])), 250))
    st.image(new_im)


def vis1x6(image):
    w , h = image.size
    real = w / 6.4 * 5
    sub_im = image.crop(( 0, h // 2 , real , h ))
    if sub_im.size[1] > 250:
        # Resize the height to 250 while maintaining the aspect ratio
        sub_im = sub_im.resize((int(sub_im.size[0] * (250 / sub_im.size[1])), 250))
    st.image(sub_im)

def main():
    root_dir = work_dir() / 'demo'  / 'exported' / 'multi-mismatch'

    subdir = list(root_dir.glob('*'))
    subdir = [_ for _ in subdir if _.is_dir()]
    current_dir = st.sidebar.selectbox(
        '选择任务',
        subdir , 0
    )
    # 搜集所有的标注的结果,保存到一个 csv 当中
    csv_filename = root_dir / (current_dir.name + '.csv')
    if not csv_filename.exists():
        create_gt(current_dir , csv_filename)

    data = pd.read_csv(csv_filename)



    # data = st.data_editor(data)
    allowed_status = ['未确认', '已确认' , '待检查']

    data['equal'] = data.apply(lambda x: '相等' if clean(x['org']) == clean(x['rectified']) else '不相等', axis=1)

    # random shuffle
    # Define the custom order for sorting
    custom_order = ['已确认', '待检查', '未确认']
    # Sort the data based on the custom order
    data['status_order'] = data['status'].apply(lambda x: custom_order.index(x) if x in custom_order else 3)
    data = data.sort_values(by=['status_order','order']).drop(columns=['status_order']).reset_index(drop=True)


    st.write('- 默认只加载图片的局部内容,当局部内容无法判断,再加载高清的完整图片')
    st.write('- 修正当前图片后可再次取消勾选,加载低清图片')
    high_res = st.checkbox('加载完整图片',value = False , )


    data = st.data_editor(
        data,
        column_config={
            'status': st.column_config.SelectboxColumn(
                options=allowed_status
            )
        },
    )

    # 统计当前已确认内容的数量
    st.sidebar.write(f'- 当前已确认内容的数量:\n  - {data[data["status"] == "已确认"].shape[0]}')
    # 统计当前仍然未标注内容的数量
    st.sidebar.write(f'- 当前仍然未标注内容的数量:\n  - {data[data["status"] == "未确认"].shape[0]}')


    # 从上述数据当中遍历所有的待确认
    need_check_files = data[data['status'] == '待检查']
    need_check_files2 = data[data['status'] == '未确认'].iloc[:1,:]
    need_check_files = pd.concat([need_check_files , need_check_files2])


    for _ , row in need_check_files.iterrows():
        with open(row['source'],encoding='utf-8') as f:
            source = json.load(f)
        # 获取到 labelme 当中的图像路径

        try:
            image = load_image(row)
            if not high_res:
                w , h = image.size
                if h / 2 in [720, 480]:
                    vis2x3(image)
                elif w / 6.4 == 640 and h > 640:
                    vis1x6(image)
                elif (w , h) == (3252 , 852):
                    vis1x6(image)
                else:
                    print('debug' , w ,h )
                    st.image(image.resize( (w // 2 , h // 2 ))              )
            else:

                st.image(image)
        except:
            import traceback
            traceback.print_exc()
            st.warning('当前图片加载失败，请将当前图片的label设置为空并及时保存!')





    confirm = st.button('保存更新结果')
    if confirm:
        data['equal'] = data.apply(lambda x: '相等' if x['org'] == x['rectified'] else '不相等', axis=1)
        data.to_csv(csv_filename, index= False)
        st.success('更新完成')
        time.sleep(0.5)
        st.rerun()
        return

    st.warning('请随时保存结果')
    return








main()
