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
    classes = utils.load_classes('cfg/traffic_signs.names')

    image_list = glob.glob(os.path.join(IMAGE_DIR, '*.png'))
    image_list.sort()

    for image_path in image_list:
        image = cv2.imread(os.path.join(image_path))
        if image.shape[0] > image.shape[1]:
            image = imutils.resize(image, width=450)
        else:
            image = imutils.resize(image, width=900)
        original_height = image.shape[0]
        original_width = image.shape[1]
        resized_image = utils.resize_image(image, model.WIDTH)
        resized_image = np.expand_dims(resized_image, 0)

        # Load label
        base_name = os.path.basename(image_path)
        label_name = '.'.join(base_name.split('.')[:-1]) + '.txt'
        label_path = os.path.join(IMAGE_DIR, label_name)
        true_boxes = utils.read_label(label_path)

        start = time.time()
        tf.logging.info('Predicting {}...'.format(os.path.basename(image_path)))
        detections = sess.run(predictions, feed_dict={inputs: resized_image})
        tf.logging.info('Performing non-maximum suppression...')
        result = utils.torch_non_max_suppression(detections=torch.from_numpy(detections),
                                                 confidence_threshold=0.25,
                                                 num_classes=model.NUM_CLASSES,
                                                 nms_conf=0.4)
        end = time.time()
        tf.logging.info('Total time: {}'.format(end - start))

        for true_box in true_boxes:
            true_class_index = int(true_box[0])
            true_class_name = classes[true_class_index]
            x1, y1, x2, y2 = utils.yolo_ratios_to_real_voc(true_box[1], true_box[2], true_box[3], true_box[4],
                                                           original_height, original_width)
            image = utils.draw_bounding_box_with_text(image=image,
                                                      top_left=(int(x1), int(y1)),
                                                      right_bottom=(int(x2), int(y2)),
                                                      text=true_class_name,
                                                      box_color=(0, 255, 255),
                                                      text_color=(0, 0, 0))

        if result is not None:

            for box in result:

                # Draw bounding box for visualization
                p1 = (int(box[1]), int(box[2]))
                p1 = utils.convert_to_original_size(p1, original_height, original_width, model.HEIGHT, model.WIDTH)
                p2 = (int(box[3]), int(box[4]))
                p2 = utils.convert_to_original_size(p2, original_height, original_width, model.HEIGHT, model.WIDTH)
                class_index = int(box[-1])
                class_name = classes[class_index]
                image = utils.draw_bounding_box_with_text(image=image,
                                                          top_left=p1,
                                                          right_bottom=p2,
                                                          text=class_name,
                                                          box_color=(0, 0, 255),
                                                          text_color=(255, 255, 255))

        cv2.imshow(os.path.basename(image_path), image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
