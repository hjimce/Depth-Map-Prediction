#coding=utf-8
import os
import sys
import numpy as np

from PIL import Image

import net

def main():
    # location of depth module, config and parameters
    module_fn = 'models/depth.py'
    config_fn = 'models/depth.conf'#网络结构
    params_dir = 'weights/depth'#网络相关参数

    # load depth network
    machine = net.create_machine(module_fn, config_fn, params_dir)

    # demo image
    rgb = Image.open('demo_nyud_rgb.jpg')
    rgb = rgb.resize((320, 240), Image.BICUBIC)

    # build depth inference function and run
    rgb_imgs = np.asarray(rgb).reshape((1, 240, 320, 3))
    pred_depths = machine.infer_depth(rgb_imgs)

    # save prediction
    (m, M) = (pred_depths.min(), pred_depths.max())
    depth_img_np = (pred_depths[0] - m) / (M - m)
    depth_img = Image.fromarray((255*depth_img_np).astype(np.uint8))
    depth_img.save('demo_nyud_depth_prediction.png')


if __name__ == '__main__':
    main()