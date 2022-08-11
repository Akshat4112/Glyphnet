import shutil
import os 

BASE_PATH = '../data/'
len_real = len(os.listdir(BASE_PATH +  'real'))
len_fake = len(os.listdir(BASE_PATH + 'fake'))

print(len_real)

len_real_train = int(0.7*len_real)
len_fake_train = int(0.7*len_fake)

len_real_valid = int(0.2*len_real)
len_fake_valid = int(0.2*len_fake)

len_real_test = int(0.1*len_real)
len_fake_test = int(0.1*len_fake)


os.mkdir(BASE_PATH + 'train')
os.mkdir(BASE_PATH + 'test')
os.mkdir(BASE_PATH + 'valid')

os.mkdir(BASE_PATH + 'train/real')
os.mkdir(BASE_PATH + 'train/fake')

os.mkdir(BASE_PATH + 'test/real')
os.mkdir(BASE_PATH + 'test/fake')

os.mkdir(BASE_PATH + 'valid/real')
os.mkdir(BASE_PATH + 'valid/fake')



print(len_real_train)

c = 0
for item in os.listdir(BASE_PATH + 'real/'):
    if c==len_real_train:
        break
    else:
        shutil.move(BASE_PATH + 'real/'+item, BASE_PATH + 'train/real/')
        c+=1

c = 0        
for item in os.listdir(BASE_PATH + 'real/'):
    if c==len_real_test:
        break
    else:
        shutil.move(BASE_PATH + 'real/'+item, BASE_PATH + 'test/real/')
        c+=1

c = 0
for item in os.listdir(BASE_PATH + 'real/'):
    if c==len_real_valid:
        break
    else:
        shutil.move(BASE_PATH + 'real/'+item, BASE_PATH +'valid/real/')        
        c+=1


c = 0
for item in os.listdir(BASE_PATH + 'fake/'):
    if c==len_fake_train:
        break
    else:
        shutil.move(BASE_PATH + 'fake/'+item,  BASE_PATH + 'train/fake/')
        c+=1

c = 0        
for item in os.listdir(BASE_PATH + 'fake/'):
    if c==len_fake_test:
        break
    else:
        shutil.move(BASE_PATH + 'fake/'+item, BASE_PATH + 'test/fake/')
        c+=1

c = 0
for item in os.listdir(BASE_PATH + 'fake/'):
    if c==len_fake_valid:
        break
    else:
        shutil.move(BASE_PATH + 'fake/'+item, BASE_PATH + 'valid/fake/')        
        c+=1

os.rmdir(BASE_PATH + 'fake')
os.rmdir(BASE_PATH + 'real')