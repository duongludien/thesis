import cv2
import os


VIDEOS_DIR = '/home/dldien/data/cutted'
SAVE_DIR = '/home/dldien/data/images'

videos_list = os.listdir(VIDEOS_DIR)

for video in videos_list:
    video_path = os.path.join(VIDEOS_DIR, video)
    print('Processing', video_path)

    video_capture = cv2.VideoCapture(video_path)
    success, image = video_capture.read()
    count = 0
    while success:
        if count % 6 == 0:
            img_path = os.path.join(SAVE_DIR, "{}_{}.jpg".format(video, count))
            cv2.imwrite(img_path, image)
        success, image = video_capture.read()
        count += 1
