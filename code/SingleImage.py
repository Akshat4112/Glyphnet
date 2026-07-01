# Import Libraries
 
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import pandas as pd
import time, multiprocessing
from multiprocessing import Process
import os
import argparse

font_path = os.path.join('../data/', "ARIAL.TTF")

def img(text, path):
    os.makedirs(path, exist_ok=True)
    img = Image.new('L', (256, 256))
    fnt = ImageFont.truetype(font_path, 28)
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill = (255))
    #enhancer = ImageEnhance.Contrast(img)
    #im_output = enhancer.enhance(1.5)
    #transposed  = img.transpose(Image.ROTATE_90)
    p = os.path.join(path, text + 'glp.png')
    img.save(p, 'PNG')

img('facebοοk.com', '../data/singleImages/')