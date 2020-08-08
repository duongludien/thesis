from sklearn.model_selection import train_test_split
import os


f = open('data.txt', 'rt')
data = f.read().split('\n')
train, test = train_test_split(data, test_size=0.2)

for item in test:
    basename = os.path.basename(item)
    labelname = basename.replace('png', 'txt')

    dst1 = os.path.join('/home/diendl/Desktop/test', basename)
    dst2 = os.path.join('/home/diendl/Desktop/test', labelname)

    print('Moving {} => {}'.format(item, dst1))
    os.rename(item, dst1)
    print('Moving {} => {}'.format(item.replace('png', 'txt'), dst2))
    os.rename(item.replace('png', 'txt'), dst2)
