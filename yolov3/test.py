import model
import utils
import tensorflow as tf
import cv2
import numpy as np
import time


# NOT WORKING - Try to perform NMS using NumPy

DARKNET_WEIGHTS_PATH = '/home/diendl/yolo_weights/yolov3-tiny.weights'
IMAGE_DIR = ''
LABELMAP = 'cfg/coco.names'

graph = tf.Graph()

with graph.as_default():
    inputs = tf.placeholder(dtype=tf.float32,
                            name='input_images',
                            shape=[model.BATCH, model.WIDTH, model.HEIGHT, model.CHANNELS])

    weights_list, predictions = model.forward(inputs)
    load_weights_ops = utils.load_darknet_weights(DARKNET_WEIGHTS_PATH, weights_list)
    thresholded = utils.threshold(detections=predictions, object_confidence_threshold=0.25)

with tf.Session(graph=graph) as sess:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Loading weights from Darknet weights file...')
    sess.run(load_weights_ops)
    classes = utils.load_classes(LABELMAP)

    image1 = cv2.imread('dog-cycle-car.png')
    image2 = cv2.imread('scream.jpg')

    resized_image1 = utils.resize_image(image1, model.WIDTH)
    resized_image2 = utils.resize_image(image2, model.WIDTH)
    resized_image1 = np.expand_dims(resized_image1, 0)
    resized_image2 = np.expand_dims(resized_image2, 0)
    input_images = np.concatenate((resized_image1, resized_image2))

    start = time.time()
    tf.logging.info('Predicting...')
    detections = sess.run(thresholded, feed_dict={inputs: input_images})
    tf.logging.info('Performing non-maximum suppression...')
    result = utils.numpy_non_max_suppression(thresholded_detections=detections,
                                             num_classes=model.NUM_CLASSES,
                                             iou_threshold=0.4)
    end = time.time()
    tf.logging.info('Total time: {}'.format(end - start))

    # if result is not None:
    #     for box in result:
    #         print(box)
    #         p1 = (int(box[1]), int(box[2]))
    #         p1 = utils.convert_to_original_size(p1, original_height, original_width, model.HEIGHT, model.WIDTH)
    #         p2 = (int(box[3]), int(box[4]))
    #         p2 = utils.convert_to_original_size(p2, original_height, original_width, model.HEIGHT, model.WIDTH)
    #         class_name = int(box[-1])
    #         class_name = classes[class_name]
    #         image = utils.draw_bounding_box_with_text(image, p1, p2, class_name,
    #                                                   box_color=(0, 0, 255),
    #                                                   text_color=(255, 255, 255))
    #
    # cv2.imshow(os.path.basename(image_path), image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
