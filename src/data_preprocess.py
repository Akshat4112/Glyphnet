import os
from os import path
import shutil

src = "../data/domain_pics/train/"
dst = "../data/final_train/real/"
print("Data moved from real train to final_train real..")

cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
files = [i for i in os.listdir(src)]
for f in files:
        shutil.copy(path.join(src, f), dst)

src = "../data/domain_pics/test/"
dst = "../data/final_test/real/"
print("Data moved from real test to final_test real..")    
    
cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
files = [i for i in os.listdir(src)]
for f in files:
        shutil.copy(path.join(src, f), dst)
        
src = "../data/fake_pics/train/"
dst = "../data/final_train/phish/"
print("Data moved from fake_train to final_train fake..")        

cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
files = [i for i in os.listdir(src)]
for f in files:
        shutil.copy(path.join(src, f), dst)
        
src = "../data/fake_pics/test/"
dst = "../data/final_test/phish/"
print("Data moved from fake test to final test fake..")        

cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
files = [i for i in os.listdir(src)]
for f in files:
        shutil.copy(path.join(src, f), dst)
        
src = "../data/domain_pics/valid/"
dst = "../data/final_valid/real/"
print("Data moved from real valid to final valid real..")        
cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
files = [i for i in os.listdir(src)]
for f in files:
        shutil.copy(path.join(src, f), dst)
        
src = "../data/fake_pics/valid/"
dst = "../data/final_valid/phish/"
print("Data moved from fake valid to final_valid fake..")        

cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
files = [i for i in os.listdir(src)]
for f in files:
        shutil.copy(path.join(src, f), dst)
        
                                            