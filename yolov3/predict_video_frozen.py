import glob, os
import argparse
import utils
import tensorflow as tf
import cv2
import numpy as np
import torch
import imutils

parser = argparse.ArgumentParser()
parser.add_argument('VIDEOS_DIR', help='Directory which contains videos to predict')
args = parser.parse_args()
video_dir = filename = args.VIDEOS_DIR

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

    videos_list = glob.glob(os.path.join(video_dir, '*.mp4'))
    videos_list.sort()

    for item in videos_list:
        cap = cv2.VideoCapture(item)
        frame_num = 0
        while cap.isOpened():
            _, image = cap.read()

            if image is None:
                break

            if frame_num % 4 == 0:
                if image.shape[0] > image.shape[1]:
                    image = imutils.resize(image, width=450)
                else:
                    image = imutils.resize(image, width=900)
                original_height = image.shape[0]
                original_width = image.shape[1]

                resized_image = utils.resize_image(image, 416)
                resized_image = np.expand_dims(resized_image, 0)

                detections = sess.run(predictions, feed_dict={inputs: resized_image})
                result = utils.torch_non_max_suppression(detections=torch.from_numpy(detections),
                                                         confidence_threshold=0.25,
                                                         num_classes=22,
                                                         nms_conf=0.4)
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
                cv2.imshow(os.path.basename(item), image)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            frame_num = frame_num + 1

        cap.release()
        cv2.destroyAllWindows()
