from model import YOLOv3
import utils
import tensorflow as tf
import cv2
import numpy as np
import time
import torch
import os
import glob
import imutils


DARKNET_WEIGHTS_PATH = '/home/diendl/tiny_model/yolov3_tiny_traffic_train_4000.weights'
IMAGE_DIR = '/home/diendl/Desktop/new_dataset/data/test'
LABELMAP = 'cfg/traffic_signs.names'
CFG_PATH = 'cfg/yolov3_tiny_traffic_inference.cfg'


model = YOLOv3(CFG_PATH)

graph = tf.Graph()

with graph.as_default():
    inputs = tf.placeholder(dtype=tf.float32,
                            name='input_images',
                            shape=[model.BATCH, model.WIDTH, model.HEIGHT, model.CHANNELS])
    weights_list, predictions = model.forward(inputs)
    load_weights_ops = utils.load_darknet_weights(DARKNET_WEIGHTS_PATH, weights_list)

with tf.Session(graph=graph) as sess:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Loading weights from Darknet weights file...')
    sess.run(load_weights_ops)
    classes = utils.load_classes(LABELMAP)

    image_list = glob.glob(os.path.join(IMAGE_DIR, '*.png'))
    image_list += glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    # image_list.sort()
    index = 0
    while index < len(image_list):
        image = cv2.imread(image_list[index])
        if image.shape[0] > image.shape[1]:
            image = imutils.resize(image, width=450)
        else:
            image = imutils.resize(image, width=900)
        original_height = image.shape[0]
        original_width = image.shape[1]
        resized_image = utils.resize_image(image, model.WIDTH)
        resized_image = np.expand_dims(resized_image, 0)

        start = time.time()
        tf.logging.info('Predicting {}...'.format(os.path.basename(image_list[index])))
        detections = sess.run(predictions, feed_dict={inputs: resized_image})
        tf.logging.info('Performing non-maximum suppression...')
        result = utils.torch_non_max_suppression(detections=torch.from_numpy(detections),
                                                 confidence_threshold=0.25,
                                                 num_classes=model.NUM_CLASSES,
                                                 nms_conf=0.4)
        end = time.time()
        tf.logging.info('Total time: {}'.format(end - start))

        if result is not None:
            for box in result:
                p1 = (int(box[1]), int(box[2]))
                p1 = utils.convert_to_original_size(p1, original_height, original_width, model.HEIGHT, model.WIDTH)
                p2 = (int(box[3]), int(box[4]))
                p2 = utils.convert_to_original_size(p2, original_height, original_width, model.HEIGHT, model.WIDTH)
                class_name = int(box[-1])
                class_name = classes[class_name]
                confidence = box[5] * 100.
                print('%s: %.2f%%' % (class_name, confidence))
                image = utils.draw_bounding_box_with_text(image, p1, p2, class_name,
                                                          box_color=(0, 0, 255),
                                                          text_color=(255, 255, 255))

        cv2.imshow(os.path.basename(image_list[index]), image)
        key = cv2.waitKey(0)
        if key == 81:
            index = index - 1
        else:
            index = index + 1
        cv2.destroyAllWindows()
