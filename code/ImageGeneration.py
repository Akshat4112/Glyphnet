# Import Libraries

import argparse
import multiprocessing
import os
import time

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, help="Define path for the data")
path_arg = parser.parse_args().path_data

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

print("Length of data", len(data))


def img(text, path):
    img = Image.new('L', (256, 256))
    fnt = ImageFont.truetype(font_path, 28)
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill=(255))
    # enhancer = ImageEnhance.Contrast(img)
    # im_output = enhancer.enhance(1.5)
    # transposed  = img.transpose(Image.ROTATE_90)
    p = os.path.join(path, text + '.png')
    img.save(p, 'PNG')


def generate_real(data):
    data.apply(img, path=os.path.join(path_arg, "real"))


def generate_fake(data):
    data.apply(img, path=os.path.join(path_arg, "fake"))


def multiprocessing_func_1(data):
    generate_real(data)


def multiprocessing_func_2(data):
    generate_fake(data)


def multiprocessing_func_3(data):
    generate_real(data)


def multiprocessing_func_4(data):
    generate_fake(data)


def multiprocessing_func_5(data):
    generate_real(data)


def multiprocessing_func_6(data):
    generate_fake(data)


def multiprocessing_func_7(data):
    generate_real(data)


def multiprocessing_func_8(data):
    generate_fake(data)


'''
def multiprocessing_func_9(data):
  generate_real(data)

def multiprocessing_func_10(data):
  generate_fake(data)
'''

if __name__ == '__main__':
    starttime = time.time()

    processes_1 = []
    processes_2 = []

    ## Multiprovessing for the real data
    p1 = multiprocessing.Process(target=multiprocessing_func_1, args=(data.iloc[:500000, 0],))
    processes_1.append(p1)
    p2 = multiprocessing.Process(target=multiprocessing_func_3, args=(data.iloc[500000:1000000, 0],))
    processes_1.append(p2)
    p3 = multiprocessing.Process(target=multiprocessing_func_5, args=(data.iloc[1000000:1500000, 0],))
    processes_1.append(p3)
    p4 = multiprocessing.Process(target=multiprocessing_func_7, args=(data.iloc[1500000:, 0],))
    processes_1.append(p4)
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    for process in processes_1:
        process.join()

    ## Multiprocessinf for the phish data
    p1 = multiprocessing.Process(target=multiprocessing_func_2, args=(data.iloc[:500000, 1],))
    processes_2.append(p1)
    p2 = multiprocessing.Process(target=multiprocessing_func_4, args=(data.iloc[500000:1000000, 1],))
    processes_2.append(p2)
    p3 = multiprocessing.Process(target=multiprocessing_func_6, args=(data.iloc[1000000:1500000, 1],))
    processes_2.append(p3)
    p4 = multiprocessing.Process(target=multiprocessing_func_8, args=(data.iloc[1500000:, 1],))
    processes_2.append(p4)
    p1.start()
    p2.start()
    p3.start()
    p4.start()

    for process in processes_2:
        process.join()

    print('Time taken = {} seconds'.format(time.time() - starttime))
