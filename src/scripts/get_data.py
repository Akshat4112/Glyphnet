import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import urllib.request
import pickle
import os

data = pickle.load(urllib.request.urlopen('https://github.com/endgameinc/homoglyph/blob/master/data/domains_spoof.pkl?raw=true'))
font = 'ARIAL.TTF'
# BASE_DIR = ''

def img(text, path):     
    img = Image.new('L', (256, 256))
    fnt = ImageFont.truetype(font, 28)
    d = ImageDraw.Draw(img)
    d.text((0, 128), text, font=fnt, fill = (255))
    #enhancer = ImageEnhance.Contrast(img)
    #im_output = enhancer.enhance(1.5)
    #transposed  = img.transpose(Image.ROTATE_90)
    img.save(path + text + '.png', 'PNG')
    
train = [dom[0] for dom in data['train']]
test = [dom[0] for dom in data['test']]
valid = [dom[0] for dom in data['validate']]

uniq_train = list(set(train))
uniq_test = list(set(test))
uniq_valid = list(set(valid))

print(len(uniq_train), len(uniq_test), len(uniq_valid))

with open("../../data/domains_train.txt", "wb") as fp:
    pickle.dump(uniq_train, fp)

with open("../../data/domains_test.txt", "wb") as fp:
    pickle.dump(uniq_test, fp)

with open("../../data/domains_valid.txt", "wb") as fp:
    pickle.dump(uniq_valid, fp)
    
with open("../../data/domains_train.txt", "rb") as fp:
    domains_train = pickle.load(fp)

with open("../../data/domains_test.txt", "rb") as fp:
    domains_test = pickle.load(fp)
    
with open("../../data/domains_valid.txt", "rb") as fp:
    domains_valid = pickle.load(fp)
    
print(len(domains_train), len(domains_valid), len(domains_test))

print("Creating Directories for data schema..")

os.mkdir('../../data/domain_pics')
os.mkdir('../../data/domain_pics/train')
os.mkdir('../../data/domain_pics/test')
os.mkdir('../../data/domain_pics/valid')

print("Directories for data schema created..")

print("Creating images for real train domains...")
for domain in domains_train:
    img(domain, '../../data/domain_pics/train/')

print("Creating images for real test domains...")
for domain in domains_test:
    img(domain, '../../data/domain_pics/test/')

print("Creating images for real valid domains...")    
for domain in domains_valid:
    img(domain, '../../data/domain_pics/valid/')
    
train = [dom[1] for dom in data['train']][:69723]
test = [dom[1] for dom in data['test']][:18349]
valid = [dom[1] for dom in data['validate']][:3670]

with open("../../data/fake_train.txt", "wb") as fp:
    pickle.dump(train, fp)
    
with open("../../data/fake_test.txt", "wb") as fp:
    pickle.dump(test, fp)
    
with open("../../data/fake_valid.txt", "wb") as fp:
    pickle.dump(valid, fp)
    
with open("../../data/fake_train.txt", "rb") as fp:
    fake_train = pickle.load(fp)
    
with open("../../data/fake_test.txt", "rb") as fp:
    fake_test = pickle.load(fp)
    
with open("../../data/fake_valid.txt", "rb") as fp:
    fake_valid = pickle.load(fp)

print("Creating directories for fake domains")    
os.mkdir('../../data/fake_pics')
os.mkdir('../../data/fake_pics/train')
os.mkdir('../../data/fake_pics/test')
os.mkdir('../../data/fake_pics/valid')

print("Creating images for fake train domains...")

for fake in fake_train:
    img(fake,'../../data/fake_pics/train/')

print("Creating images for fake test domains...")    
for fake in fake_test:
    img(fake,'../../data/fake_pics/test/')

print("Creating images for fake valid domains...")    
for fake in fake_valid:
    img(fake,'../../data/fake_pics/valid/')
    
os.mkdir('../../data/final_train')
os.mkdir('../../data/final_train/real')
os.mkdir('../../data/final_train/phish')

os.mkdir('../../data/final_valid')
os.mkdir('../../data/final_valid/real')
os.mkdir('../../data/final_valid/phish')

os.mkdir('../../data/final_test')
os.mkdir('../../data/final_test/real')
os.mkdir('../../data/final_test/phish')

print("Get data completed..")
                        