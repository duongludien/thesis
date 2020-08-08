import os
import cv2

"""
Convert images to PNG and rename them to a standard format
"""

DATA_DIR = '/home/diendl/Desktop/tonghop'
OUT_DIR = '/home/diendl/Desktop/rename'

img_dirs = os.listdir(DATA_DIR)
print(img_dirs)

for img_dir in img_dirs:
    imgs_list = os.listdir(os.path.join(DATA_DIR, img_dir))
    print('Processing', img_dir)

    os.mkdir(os.path.join(OUT_DIR, img_dir))
    count = 1
    for img_file in imgs_list:
        print('Converting', img_file)
        img = cv2.imread(os.path.join(DATA_DIR, img_dir, img_file))
        cv2.imwrite(os.path.join(OUT_DIR, img_dir, '%s_%04d.png' % (img_dir, count)), img)
        count = count + 1
