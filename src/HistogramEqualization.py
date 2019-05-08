#!/usr/bin/python3.6.8
# -*- coding: utf-8 -*-
#   Author: HowkeWayne
#     Date: 2019/5/7 - 14:50
"""
Life is Short I Use Python!!!
=== If this runs wrong,don't ask me,I don't know why;
=== If this runs right,thank god,and I don't know why. 
=== Maybe the answer,my friend,is blowing in the wind.
==========================================================
File Description...
直方图均衡化 opencv
"""
import os

import cv2

from b_comm import get_conf


def histogram_equalization_Color(img):
    ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCR_CB)
    channels = cv2.split(ycrcb)
    print(len(channels))
    cv2.equalizeHist(channels[0], channels[0])
    cv2.merge(channels, ycrcb)
    cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2RGB, img)
    return img


if __name__ == '__main__':
    img = cv2.imread(os.path.join(get_conf('client', 'Info_path'), '0000073.png'))
    cv2.imshow('original',img)
    cv2.imshow('Ahis_equal',histogram_equalization_Color(img))
    cv2.waitKey(0)