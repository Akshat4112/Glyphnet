
import os

print('Total Train Real images: ', len(os.listdir('../../data/final_train/real/')))
print('Total Test Real images: ', len(os.listdir('../../data/final_test/real/')))
print('Total Train Fake images: ', len(os.listdir('../../data/final_train/phish/')))
print('Total Test Fake images: ', len(os.listdir('../../data/final_test/phish/')))
print('Total Validation Real images: ', len(os.listdir('../../data/final_valid/real/')))
print('Total Validation Fake images: ', len(os.listdir('../../data/final_valid/phish/')))