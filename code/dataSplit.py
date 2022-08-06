import shutil
import os 

len_real = len(os.listdir('../data/real'))
len_fake = len(os.listdir('../data/fake'))

print(len_real)

len_real_train = int(0.7*len_real)
len_fake_train = int(0.7*len_fake)

len_real_valid = int(0.2*len_real)
len_fake_valid = int(0.2*len_fake)

len_real_test = int(0.1*len_real)
len_fake_test = int(0.1*len_fake)


os.mkdir('../data/train')
os.mkdir('../data/test')
os.mkdir('../data/valid')

os.mkdir('../data/train/real')
os.mkdir('../data/train/fake')

os.mkdir('../data/test/real')
os.mkdir('../data/test/fake')

os.mkdir('../data/valid/real')
os.mkdir('../data/valid/fake')



print(len_real_train)

c = 0
for item in os.listdir('../data/real/'):
    if c==len_real_train:
        break
    else:
        shutil.move('../data/real/'+item, '../data/train/real/')
        c+=1

c = 0        
for item in os.listdir('../data/real/'):
    if c==len_real_test:
        break
    else:
        shutil.move('../data/real/'+item, '../data/test/real/')
        c+=1

c = 0
for item in os.listdir('../data/real/'):
    if c==len_real_valid:
        break
    else:
        shutil.move('../data/real/'+item, '../data/valid/real/')        
        c+=1


c = 0
for item in os.listdir('../data/fake/'):
    if c==len_fake_train:
        break
    else:
        shutil.move('../data/fake/'+item, '../data/train/fake/')
        c+=1

c = 0        
for item in os.listdir('../data/fake/'):
    if c==len_fake_test:
        break
    else:
        shutil.move('../data/fake/'+item, '../data/test/fake/')
        c+=1

c = 0
for item in os.listdir('../data/fake/'):
    if c==len_fake_valid:
        break
    else:
        shutil.move('../data/fake/'+item, '../data/valid/fake/')        
        c+=1

os.rmdir('../data/fake')
os.rmdir('../data/real')