# Import Libraries
 
from PIL import Image, ImageDraw, ImageFont, ImageEnhance

import pandas as pd
import time, multiprocessing
from multiprocessing import Process
import os
import argparse

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, help="Define path for the data")
path_arg  = parser.parse_args().path_data


# font_path = '../data/ARIAL.TTF'
# file_path = '../data/dataset_final.csv'

font_path = os.path.join(path_arg, "ARIAL.TTF")
file_path = os.path.join(path_arg, "dataset_final.csv")


try:
    os.makedirs(os.path.join(path_arg, "real"))
except OSError as e:
    print('Real Directory Exists!')

try:
    os.makedirs(os.path.join(path_arg, "fake"))
except OSError as e:
    print('Fake Directory Exists!')


data = pd.read_csv(file_path, nrows=2000000)

print(len(data))

def img(text, path):     
    img = Image.new('L', (256, 256))
    fnt = ImageFont.truetype(font_path, 28)
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill = (255))
    #enhancer = ImageEnhance.Contrast(img)
    #im_output = enhancer.enhance(1.5)
    #transposed  = img.transpose(Image.ROTATE_90)
    p = os.path.join(path, text + '.png')
    img.save(p, 'PNG')


def generate_real(data):
  data.apply(img, path = os.path.join(path_arg, "real"))

def generate_fake(data):
  data.apply(img, path = os.path.join(path_arg, "fake"))


# Split points used to shard the data across worker processes.
CHUNKS = [(0, 500000), (500000, 1000000), (1000000, 1500000), (1500000, None)]


def run_in_parallel(target, column):
    """Render one column of `data` in parallel across CHUNKS worker processes.

    column 0 = real domains, column 1 = homoglyph (fake) domains.
    """
    processes = []
    for start, stop in CHUNKS:
        p = multiprocessing.Process(target=target, args=(data.iloc[start:stop, column],))
        processes.append(p)
        p.start()
    for process in processes:
        process.join()


if __name__ == '__main__':
    starttime = time.time()

    run_in_parallel(generate_real, column=0)
    run_in_parallel(generate_fake, column=1)

    print('Time taken = {} seconds'.format(time.time() - starttime))