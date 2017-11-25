import os
from PIL import Image
import glob
import numpy as np

def resize_image(im, size, mode='RGB'):
    w, h = im.size
    im.thumbnail((size, size), Image.ANTIALIAS)
    background = Image.new(mode, (size, size))
    background.paste(
        im, (int((size - im.size[0]) / 2), int((size - im.size[1]) / 2))
    )
    return background

def resize(in_path, out_path, size, mode='RGB'):
    im = Image.open(in_path)
    w, h = im.size
    im.thumbnail((size, size), Image.ANTIALIAS)
    background = Image.new(mode, (size, size))
    background.paste(
        im, (int((size - im.size[0]) / 2), int((size - im.size[1]) / 2))
    )
    if out_path:
        background.save(out_path)

    return background


def reshape(in_dir, glob_pattern, out_dir, mode, size, allowed=None):
    in_dir = os.path.realpath(in_dir)
    out_dir = os.path.realpath(out_dir)

    for f in glob.glob(os.path.join(in_dir, glob_pattern), recursive=True):
        out_f = f.replace(in_dir, out_dir)
        print(f)
        os.makedirs(os.path.dirname(out_f), exist_ok=True)
        resize(f, out_f, size, mode)


def convert_test(img_dir, masks_dir, class_count, out_dir, size=640, allowed_formats=None):
    if not allowed_formats:
        allowed_formats = ['.bmp']
    for img in os.listdir(img_dir):
        print(img)
        name, ext = os.path.splitext(img)
        img_path = os.path.join(img_dir, img)
        out_img_path = os.path.join(out_dir, img)
        if ext not in allowed_formats:
            continue

        resize(img_path, out_img_path, mode='RGB', size=size)
        os.makedirs(os.path.dirname(out_img_path), exist_ok=True)

        os.makedirs(os.path.join(out_dir, name), exist_ok=True)
        for label_id in range(1, class_count + 1):
            mask_path = os.path.join(masks_dir, '{}_{}{}'.format(name, label_id, ext))
            out_mask_path = os.path.join(out_dir, name, '{}_{}{}'.format(name, label_id, ext))

            im = Image.open(mask_path)
            im = im.convert("RGB")
            arr = np.array(im, dtype=np.uint8)
            # im = Image.fromarray(arr, 'L')

            im_arr = (((arr != 255).any(axis=-1)).astype(np.uint8)) * 255
            im = Image.fromarray(im_arr, 'P')
            im = resize_image(im, size, 'P')
            im.save(out_mask_path)



# reshape('sample_0', '*.bmp', 'sample', 'RGB', 640)
# reshape('sample_0', '*/*.bmp', 'sample', 'P', 640)

# convert_test('dataset0/test', 'dataset0/eval', class_count=7, out_dir='files/test')

# reshape('dataset0/train', '*.bmp', 'files/train', 'RGB', 640)
# reshape('dataset0/train', '*/*.bmp', 'files/train', 'P', 640)