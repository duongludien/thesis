import utils
import argparse
import tensorflow as tf
import cv2
import numpy as np
import time
import torch
import os
import glob
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('IMAGES_DIR', help='Directory which contains images to predict')
args = parser.parse_args()
images_dir = filename = args.IMAGES_DIR

LABELMAP = 'cfg/traffic_signs.names'
FROZEN_MODEL = 'frozen_tiny_model_4000.pb'


with tf.gfile.GFile(FROZEN_MODEL, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

graph = tf.Graph()
with graph.as_default():
    tf.import_graph_def(graph_def, name="")
    inputs = graph.get_tensor_by_name('input_images:0')
    predictions = graph.get_tensor_by_name('outputs:0')

with tf.Session(graph=graph) as sess:
    tf.logging.set_verbosity(tf.logging.INFO)
    classes = utils.load_classes(LABELMAP)

    image_list = glob.glob(os.path.join(images_dir, '*.png'))
    image_list += glob.glob(os.path.join(images_dir, '*.jpg'))
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
        resized_image = utils.resize_image(image, 416)
        resized_image = np.expand_dims(resized_image, 0)

        start = time.time()
        tf.logging.info('Predicting {}...'.format(os.path.basename(image_list[index])))
        detections = sess.run(predictions, feed_dict={inputs: resized_image})
        tf.logging.info('Performing non-maximum suppression...')
        result = utils.torch_non_max_suppression(detections=torch.from_numpy(detections),
                                                 confidence_threshold=0.25,
                                                 num_classes=22,
                                                 nms_conf=0.4)
        end = time.time()
        tf.logging.info('Total time: {}'.format(end - start))

        if result is not None:
            for box in result:
                p1 = (int(box[1]), int(box[2]))
                p1 = utils.convert_to_original_size(p1, original_height, original_width, 416, 416)
                p2 = (int(box[3]), int(box[4]))
                p2 = utils.convert_to_original_size(p2, original_height, original_width, 416, 416)
                class_name = int(box[-1])
                class_name = classes[class_name]
                confidence = box[5] * 100.
                print('%s: %.2f%%' % (class_name, confidence))
                image = utils.draw_bounding_box_with_text(image, p1, p2, class_name,
                                                          box_color=(0, 0, 255),
                                                          text_color=(255, 255, 255))

        cv2.imshow(os.path.basename(image_list[index]), image)
        cv2.imwrite(os.path.join('/home/diendl/result', os.path.basename(image_list[index])), image)
        key = cv2.waitKey(0)
        if key == 81:
            index = index - 1
        elif key == 113:
            break
        else:
            index = index + 1
        cv2.destroyAllWindows()
