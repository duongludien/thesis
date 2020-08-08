from utils import *
import numpy as np
import cv2

NUM_CLASSES = 22
TRAIN_FILES = ['dataset_tools/train.tf']
CFG_FILE = 'cfg/yolov3_old.cfg'

config_blocks = load_config(CFG_FILE)

# Parsing network information
net_info = config_blocks[0]
BATCH = int(net_info['batch'])
SUBDIVISIONS = int(net_info['subdivisions'])
WIDTH = int(net_info['width'])
HEIGHT = int(net_info['height'])
CHANNELS = int(net_info['channels'])
MOMENTUM = float(net_info['momentum'])
DECAY = float(net_info['decay'])
ANGLE = int(net_info['angle'])
SATURATION = float(net_info['saturation'])
EXPOSURE = float(net_info['exposure'])
HUE = float(net_info['hue'])
LEARNING_RATE = float(net_info['learning_rate'])
BURN_IN = int(net_info['burn_in'])
MAX_BATCHES = int(net_info['max_batches'])
POLICY = net_info['policy']
STEPS = [int(x) for x in net_info['steps'].split(',')]
SCALES = [float(x) for x in net_info['scales'].split(',')]


def _parse_function(example_proto):
    keys_to_features = {
        'image': tf.VarLenFeature(tf.string),
        'class_indices': tf.VarLenFeature(tf.int64),
        'bx': tf.VarLenFeature(tf.float32),
        'by': tf.VarLenFeature(tf.float32),
        'bw': tf.VarLenFeature(tf.float32),
        'bh': tf.VarLenFeature(tf.float32)
    }
    parsed_features = tf.parse_single_example(example_proto, keys_to_features)

    image = tf.decode_raw(parsed_features['image'].values, tf.uint8)
    image = tf.reshape(image, [HEIGHT, WIDTH, CHANNELS])
    image = tf.cast(image, tf.float32)

    return image / 255., parsed_features['class_indices'], parsed_features['bx'], \
        parsed_features['by'], parsed_features['bw'], parsed_features['bh']





training_set = tf.data.TFRecordDataset(TRAIN_FILES)
training_set = training_set.map(_parse_function)
training_set = training_set.repeat().batch(BATCH)
training_iterator = training_set.make_one_shot_iterator()

images_batch, sparse_indices_batch, sparse_bxs_batch, sparse_bys_batch, sparse_bws_batch, sparse_bhs_batch \
    = training_iterator.get_next()

# Convert sparse tensor to dense
dense_indices_batch = tf.sparse_tensor_to_dense(sparse_indices_batch, default_value=-1)
dense_bxs_batch = tf.sparse_tensor_to_dense(sparse_bxs_batch, default_value=-1)
dense_bys_batch = tf.sparse_tensor_to_dense(sparse_bys_batch, default_value=-1)
dense_bws_batch = tf.sparse_tensor_to_dense(sparse_bws_batch, default_value=-1)
dense_bhs_batch = tf.sparse_tensor_to_dense(sparse_bhs_batch, default_value=-1)

with tf.Session() as sess:
    for step in range(1):

        images, indices_batch, bxs_batch, bys_batch, bws_batch, bhs_batch \
            = sess.run([images_batch, dense_indices_batch,
                        dense_bxs_batch, dense_bys_batch, dense_bws_batch, dense_bhs_batch])

        outputs_batch = None

        # Iterate over all images in this batch
        for index in range(BATCH):
            print('Image {}:'.format(index))

            img = images[index]
            indices = indices_batch[index]
            bxs = bxs_batch[index]
            bys = bys_batch[index]
            bws = bws_batch[index]
            bhs = bhs_batch[index]

            # Except sparse tensor default value (default: -1)
            indices = [indices[i] for i in range(len(indices)) if indices[i] != -1]
            bxs = [bxs[i] for i in range(len(bxs)) if bxs[i] != -1]
            bys = [bys[i] for i in range(len(bys)) if bys[i] != -1]
            bws = [bws[i] for i in range(len(bws)) if bws[i] != -1]
            bhs = [bhs[i] for i in range(len(bhs)) if bhs[i] != -1]

            num_boxes = len(indices)

            for j in range(num_boxes):
                print('Box {}:'.format(j))
                top = max((bys[j] * HEIGHT) - (bhs[j] * HEIGHT) / 2, 0)
                left = max((bxs[j] * WIDTH) - (bws[j] * WIDTH) / 2, 0)
                right = min((bxs[j] * WIDTH) + (bws[j] * WIDTH) / 2, WIDTH)
                bottom = min((bys[j] * HEIGHT) + (bhs[j] * HEIGHT) / 2, HEIGHT)
                print(left, top, right, bottom)
                cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 1)

            cv2.imshow('', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        #     print(bxs)
        #     print(bys)
        #     print(bws)
        #     print(bhs)
        #
        #     outputs = construct_true_vector(bxs, bys, bws, bhs, indices, 80,
        #                                     grid_sizes=[13, 26, 52],
        #                                     anchors=[
        #                                         [(116, 90), (156, 198), (373, 326)],
        #                                         [(30, 61), (62, 45), (59, 119)],
        #                                         [(10, 13), (16, 30), (33, 23)]
        #                                     ])
        #
        #     if outputs_batch is None:
        #         outputs_batch = outputs
        #     else:
        #         outputs_batch = np.concatenate((outputs_batch, outputs), axis=0)
        #
        # print('Batch input shape:', images.shape)
        # print('Batch output shape:', outputs_batch.shape)
