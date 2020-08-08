# TODO: Implement LATER


from model import YOLOv3
import utils
import tensorflow as tf

CFG_PATH = 'cfg/yolov3_tiny_traffic_inference.cfg'
DARKNET_WEIGHTS_PATH = '/home/diendl/tiny_model/yolov3_tiny_traffic_train_4000.weights'
LABELMAP = 'cfg/coco.names'
TRAIN_FILES = ['dataset_tools/train.tf']

model = YOLOv3(CFG_PATH)
graph = tf.Graph()

with graph.as_default():
    dataset = utils.create_dataset(TRAIN_FILES, model.BATCH, model.HEIGHT, model.WIDTH, model.CHANNELS)
    images_batch, dense_indices_batch, dense_bxs_batch, dense_bys_batch, dense_bws_batch, dense_bhs_batch \
        = utils.get_next_batch(dataset)

    inputs = tf.placeholder(dtype=tf.float32,
                            name='input_images',
                            shape=[model.BATCH, model.WIDTH, model.HEIGHT, model.CHANNELS])

    weights_list, predictions = model.forward(inputs)
    load_weights_ops = utils.load_darknet_weights(DARKNET_WEIGHTS_PATH, weights_list)

    num_detections = [(item ** 2) * 3 for item in model.GRID_SIZES]
    num_detections = sum(num_detections)
    outputs = tf.placeholder(shape=[model.BATCH, num_detections, model.NUM_CLASSES + 5],
                             dtype=tf.float32,
                             name='true_vector')

    # No object loss
    no_obj_mask = tf.equal(outputs[:, :, 4], 0)
    no_obj_mask = tf.cast(no_obj_mask, tf.float32)
    no_obj_mask = tf.expand_dims(no_obj_mask, -1)
    no_obj_predictions = tf.multiply(predictions, no_obj_mask)
    no_obj_scale = tf.constant(0.5, dtype=tf.float32)
    no_obj_loss = no_obj_scale * tf.reduce_sum((0 - no_obj_predictions[:, :, 4]) ** 2)

    # Object loss
    object_mask = tf.not_equal(outputs[:, :, 4], 0)
    object_mask = tf.cast(object_mask, tf.float32)
    object_mask = tf.expand_dims(object_mask, axis=-1)
    obj_predictions = tf.multiply(predictions, object_mask)
    obj_scale = tf.constant(1.0, dtype=tf.float32)
    obj_loss = obj_scale * tf.reduce_sum((0 - obj_predictions[:, :, 4]) ** 2)

    # Class loss
    class_scale = tf.constant(1.0, dtype=tf.float32)
    class_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=outputs[:, :, 5:],
                                                 logits=obj_predictions[:, :, 5:])

    # Bounding-boxes loss
    coord_scale = tf.constant(1.0, dtype=tf.float32)
    bx_loss = (outputs[:, :, 0] - obj_predictions[:, :, 0]) ** 2
    by_loss = (outputs[:, :, 1] - obj_predictions[:, :, 1]) ** 2
    bx_by_loss = coord_scale * tf.reduce_sum(bx_loss + by_loss)
    bw_loss = (tf.sqrt(outputs[:, :, 2]) - tf.sqrt(obj_predictions[:, :, 2])) ** 2
    bh_loss = (tf.sqrt(outputs[:, :, 3]) - tf.sqrt(obj_predictions[:, :, 3])) ** 2
    bw_bh_loss = coord_scale * tf.reduce_sum(bw_loss + bh_loss)

with tf.Session(graph=graph) as sess:
    tf.logging.set_verbosity(tf.logging.INFO)

    tf.logging.info('Loading weights from Darknet weights file...')
    sess.run(load_weights_ops)

    for step in range(100):
        images, indices_batch, bxs_batch, bys_batch, bws_batch, bhs_batch \
            = sess.run([images_batch, dense_indices_batch,
                        dense_bxs_batch, dense_bys_batch, dense_bws_batch, dense_bhs_batch])

        # import cv2
        # import numpy as np
        #
        # images = cv2.imread('/home/diendl/out.png')
        # images = utils.resize_image(images, 416)
        # images = np.expand_dims(images, 0)
        #
        # indices_batch = np.array([[16, 1, 7]])
        # bxs_batch = np.array([[0.289663, 0.451923, 0.743990]])
        # bys_batch = np.array([[0.621394, 0.486779, 0.282452]])
        # bws_batch = np.array([[0.233173, 0.576923, 0.276442]])
        # bhs_batch = np.array([[0.415865, 0.372596, 0.122596]])

        true_vector = utils.construct_batch_true_vector(indices_batch, bxs_batch, bys_batch, bws_batch, bhs_batch,
                                                        model.NUM_CLASSES, model.GRID_SIZES, model.ANCHORS,
                                                        model.WIDTH, model.HEIGHT)
        print(indices_batch)

        print(sess.run(fetches=[no_obj_loss, obj_loss, class_loss],
                       feed_dict={inputs: images, outputs: true_vector}))
