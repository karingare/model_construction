#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 15 09:18:09 2023

@author: forskningskarin
"""

import cv2
from torchvision.transforms.functional import pad, affine
from torchvision import transforms
import numpy as np
import numbers
from PIL import Image

def mode_pixel_value(img):
    """Returns the most common pixel value for image i.e. mode value."""

    np_image = np.asarray(img)  # calcHist doesnt work with PIL images
    hist = cv2.calcHist([np_image], [0], None, [256], [0, 256])
    mode_np = int(np.argmax(hist))
    mode_pil = np.array([mode_np] * 3, dtype=np.uint8)

    # Create a PIL image from the mode_pil numpy array
    mode_pil_img = Image.fromarray(mode_pil)

    return  mode_pil_img


def get_padding(image):
    w, h = image.size
    max_wh = 180
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding + 0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding + 0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding - 0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding - 0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPadBase:
    def __init__(self, padding_mode='constant', max_translate=10):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']
        self.padding_mode = padding_mode
        self.max_translate = max_translate

    def pad_image(self, img):
        max_size = 180

        # Resize image to be 180 wide
        aspect_ratio = img.size[1] / img.size[0]  # height/width
        img = img.resize((max_size, int(max_size * aspect_ratio)))

        fill = tuple(mode_pixel_value(img).getdata())

        return pad(img, get_padding(img), fill, self.padding_mode)


class NewPad(NewPadBase):
    def __call__(self, img):
        img = self.pad_image(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={0}, max_translate={1})'.format(self.padding_mode, self.max_translate)


class NewPadAndTransform(NewPadBase):
    def __call__(self, img):
        img = self.pad_image(img)

        fill = tuple(mode_pixel_value(img).getdata())
        translate = (0, np.random.randint(-self.max_translate, self.max_translate))  # set the scale of the translation
        scale = np.random.uniform(0.6, 1.4)
        img = affine(img, angle=0, translate=translate, scale=scale, shear=0, fill=fill)  # translation
        rotater = transforms.RandomRotation(degrees=(-10, 10), fill=fill)
        img = rotater(img)
        return img

    def __repr__(self):
        return self.__class__.__name__ + '(padding_mode={0}, max_translate={1})'.format(self.padding_mode, self.max_translate)