import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import urllib.request
import pickle
import os
import argparse

data = pickle.load(urllib.request.urlopen('https://github.com/endgameinc/homoglyph/blob/master/data/domains_spoof.pkl?raw=true'))
font = '../data/ARIAL.TTF'
# BASE_DIR = ''

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, help="Define path for the data")
path_arg  = parser.parse_args().path_data

print(path_arg)

def img(text, path):     
    img = Image.new('L', (256, 256))
    fnt = ImageFont.truetype(font, 28)
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill = (255))
    #enhancer = ImageEnhance.Contrast(img)
    #im_output = enhancer.enhance(1.5)
    #transposed  = img.transpose(Image.ROTATE_90)

    path = os.path.join(path, text + ".png")
    print("Path of img", path)
    img.save(path, 'PNG')
    
train = [dom[0] for dom in data['train']]
test = [dom[0] for dom in data['test']]
valid = [dom[0] for dom in data['validate']]

uniq_train = list(set(train))
uniq_test = list(set(test))
uniq_valid = list(set(valid))

print(len(uniq_train), len(uniq_test), len(uniq_valid))

##------------------- Dumping the data to .txts from the URL -------------------##
train_path = os.path.join(path_arg, "domains_train.txt")
test_path = os.path.join(path_arg, "domains_test.txt")
val_path = os.path.join(path_arg, "domains_val.txt")
print(train_path)


with open(train_path, "wb") as fp:
    pickle.dump(uniq_train, fp)

with open(test_path, "wb") as fp:
    pickle.dump(uniq_test, fp)

with open(val_path, "wb") as fp:
    pickle.dump(uniq_valid, fp)


##--------------------- Loading the dump txts --------------------------##
with open(train_path, "rb") as fp:
    domains_train = pickle.load(fp)

with open(test_path, "rb") as fp:
    domains_test = pickle.load(fp)

with open(val_path, "rb") as fp:
    domains_valid = pickle.load(fp)

print(len(domains_train), len(domains_valid), len(domains_test))


##------------------ Creating the images for stored txts ---------------------##
print("Creating Directories for data schema..")

domain_pics_folder_path = os.path.join(path_arg, "domain_pics")
domain_pics_folder_train_path = os.path.join(path_arg, "domain_pics", "train")
domain_pics_folder_test_path = os.path.join(path_arg, "domain_pics", "test")
domain_pics_folder_val_path = os.path.join(path_arg, "domain_pics", "valid")

print(domain_pics_folder_train_path)

if not os.path.exists(domain_pics_folder_path):
    os.mkdir(domain_pics_folder_path)
if not os.path.exists(domain_pics_folder_train_path):
    os.mkdir(domain_pics_folder_train_path)
if not os.path.exists(domain_pics_folder_test_path):
    os.mkdir(domain_pics_folder_test_path)
if not os.path.exists(domain_pics_folder_val_path):
    os.mkdir(domain_pics_folder_val_path)

# os.mkdir('../data/domain_pics/train')
# os.mkdir('../data/domain_pics/test')
# os.mkdir('../data/domain_pics/valid')

print("Directories for data schema created..")

print("Creating images for real train domains...")
for domain in domains_train:
    img(domain, domain_pics_folder_train_path)

print("Creating images for real test domains...")
for domain in domains_test:
    img(domain, domain_pics_folder_test_path)

print("Creating images for real valid domains...")
for domain in domains_valid:
    img(domain, domain_pics_folder_val_path)


##------------------- Dumping the fake data to .txts from the URL -------------------##
train = [dom[1] for dom in data['train']][:69723]
test = [dom[1] for dom in data['test']][:18349]
valid = [dom[1] for dom in data['validate']][:3670]

train_fake_path = os.path.join(path_arg, "fake_train.txt")
test_fake_path = os.path.join(path_arg, "fake_test.txt")
val_fake_path = os.path.join(path_arg, "fake_valid.txt")

with open(train_fake_path, "wb") as fp:
    pickle.dump(train, fp)

with open(test_fake_path, "wb") as fp:
    pickle.dump(test, fp)

with open(val_fake_path, "wb") as fp:
    pickle.dump(valid, fp)

##--------------------- Loading the fake txts ------------------##
with open(train_fake_path, "rb") as fp:
    fake_train = pickle.load(fp)

with open(test_fake_path, "rb") as fp:
    fake_test = pickle.load(fp)

with open(val_fake_path, "rb") as fp:
    fake_valid = pickle.load(fp)


##------------------ Creating the images for stored fake txts ---------------------##
print("Creating directories for fake domains")

domain_fakepics_folder_path = os.path.join(path_arg, "fake_pics")
domain_fakepics_folder_train_path = os.path.join(path_arg, "fake_pics", "train")
domain_fakepics_folder_test_path = os.path.join(path_arg, "fake_pics", "test")
domain_fakepics_folder_val_path = os.path.join(path_arg, "fake_pics", "valid")

if not os.path.exists(domain_fakepics_folder_path):
    os.mkdir(domain_fakepics_folder_path)
if not os.path.exists(domain_fakepics_folder_train_path):
    os.mkdir(domain_fakepics_folder_train_path)
if not os.path.exists(domain_fakepics_folder_test_path):
    os.mkdir(domain_fakepics_folder_test_path)
if not os.path.exists(domain_fakepics_folder_val_path):
    os.mkdir(domain_fakepics_folder_val_path)


# os.mkdir('../data/fake_pics/train')
# os.mkdir('../data/fake_pics/test')
# os.mkdir('../data/fake_pics/valid')

print("Creating images for fake train domains...")
for fake in fake_train:
    img(fake, domain_fakepics_folder_train_path)

print("Creating images for fake test domains...")
for fake in fake_test:
    img(fake, domain_fakepics_folder_test_path)

print("Creating images for fake valid domains...")
for fake in fake_valid:
    img(fake, domain_fakepics_folder_val_path)


## Creating the directories for training the model

final_train_path = os.path.join(path_arg, "final_train")
final_train_real_path = os.path.join(path_arg, "final_train", "real")
final_train_phish_path = os.path.join(path_arg, "final_train", "phish")

final_test_path = os.path.join(path_arg, "final_test")
final_test_real_path = os.path.join(path_arg, "final_test", "real")
final_test_phish_path = os.path.join(path_arg, "final_test", "phish")

final_valid_path = os.path.join(path_arg, "final_valid")
final_valid_real_path = os.path.join(path_arg, "final_valid", "real")
final_valid_phish_path = os.path.join(path_arg, "final_valid", "phish")

for folder in [final_train_path, final_train_real_path, final_train_phish_path, final_test_path, final_test_real_path, final_test_phish_path, final_valid_path, final_valid_real_path, final_valid_phish_path]:
    if not os.path.exists(folder):
        os.mkdir(folder)


os.mkdir('../data/final_train')
os.mkdir('../data/final_train/real')
os.mkdir('../data/final_train/phish')

os.mkdir('../data/final_valid')
os.mkdir('../data/final_valid/real')
os.mkdir('../data/final_valid/phish')

os.mkdir('../data/final_test')
os.mkdir('../data/final_test/real')
os.mkdir('../data/final_test/phish')

print("Get data completed..")
                        