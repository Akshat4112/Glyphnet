import os
from os import path
import shutil
import argparse

parser = argparse.ArgumentParser(description="Parameters while pasing the argument..")
parser.add_argument("--path_data", type=str, help="Define path for the data")
path_arg  = parser.parse_args().path_data

src_train = os.path.join(path_arg, "domain_pics", "train")
dst_train = os.path.join(path_arg, "final_train", "real")
src_test = os.path.join(path_arg, "domain_pics", "test")
dst_test = os.path.join(path_arg, "final_test", "real")
src_train_phish = os.path.join(path_arg, "fake_pics", "train")
dst_train_phish = os.path.join(path_arg, "final_train", "phish")
src_test_phish = os.path.join(path_arg, "fake_pics", "test")
dst_test_phish = os.path.join(path_arg, "final_test", "phish")
src_val = os.path.join(path_arg, "domain_pics", "valid")
dst_val = os.path.join(path_arg, "final_valid", "real")
src_val_phish = os.path.join(path_arg, "fake_pics", "valid")
dst_val_phish = os.path.join(path_arg, "final_valid", "phish")



# src = "../data/domain_pics/train/"
# dst = "../data/final_train/real/"
print("Data moved from real train to final_train real..")
cnt = 0
#files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]

for src, dst in [(src_train, dst_train), (src_test, dst_test), (src_train_phish, dst_train_phish), (src_test_phish, dst_test_phish), (src_val, dst_val), (src_val_phish, dst_val_phish)]:
        files = [i for i in os.listdir(src)]
        print(src, dst)
        for f in files:
                shutil.copy(path.join(src, f), dst)

# src = "../data/domain_pics/test/"
# dst = "../data/final_test/real/"
# print("Data moved from real test to final_test real..")
#
# cnt = 0
# #files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
# files = [i for i in os.listdir(src)]
# for f in files:
#         shutil.copy(path.join(src, f), dst)
        
# src = "../data/fake_pics/train/"
# dst = "../data/final_train/phish/"
# print("Data moved from fake_train to final_train fake..")
#
# cnt = 0
# #files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
# files = [i for i in os.listdir(src)]
# for f in files:
#         shutil.copy(path.join(src, f), dst)
        
# src = "../data/fake_pics/test/"
# dst = "../data/final_test/phish/"
# print("Data moved from fake test to final test fake..")

# cnt = 0
# #files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
# files = [i for i in os.listdir(src)]
# for f in files:
#         shutil.copy(path.join(src, f), dst)
        
# src = "../data/domain_pics/valid/"
# dst = "../data/final_valid/real/"
# print("Data moved from real valid to final valid real..")
# cnt = 0
# #files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
# files = [i for i in os.listdir(src)]
# for f in files:
#         shutil.copy(path.join(src, f), dst)
        
# src = "../data/fake_pics/valid/"
# dst = "../data/final_valid/phish/"
# print("Data moved from fake valid to final_valid fake..")
#
# cnt = 0
# #files = [i for i in os.listdir(src) if i.startswith("CTASK") and path.isfile(path.join(src, i))]
# files = [i for i in os.listdir(src)]
# for f in files:
#         shutil.copy(path.join(src, f), dst)
        
                                            