import tensorflow as tf
import cv2

NUM_CLASSES = 22
INPUT_SIZE = 416
INPUT_CHANNELS = 3

BATCH_SIZE = 3

TRAIN_FILES = ["./train.tf"]


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
    image = tf.reshape(image, [INPUT_SIZE, INPUT_SIZE, INPUT_CHANNELS])
    image = tf.cast(image, tf.float32)

    class_indices = tf.sparse_tensor_to_dense(parsed_features['class_indices'])
    class_indices = tf.one_hot(indices=class_indices, depth=NUM_CLASSES, axis=-1)

    bx = tf.sparse_tensor_to_dense(parsed_features['bx'])
    by = tf.sparse_tensor_to_dense(parsed_features['by'])
    bw = tf.sparse_tensor_to_dense(parsed_features['bw'])
    bh = tf.sparse_tensor_to_dense(parsed_features['bh'])

    return image / 255., class_indices, bx, by, bw, bh


training_set = tf.data.TFRecordDataset(TRAIN_FILES)
training_set = training_set.map(_parse_function)
training_set = training_set.repeat().batch(BATCH_SIZE)
training_iterator = training_set.make_one_shot_iterator()

with tf.Session() as sess:
    for k in range(9):
        imgs, indices, bxs, bys, bws, bhs = sess.run(training_iterator.get_next())

        for i in range(BATCH_SIZE):
            img = imgs[i]
            for j in range(len(bxs[i])):
                top = max((bys[i][j] * INPUT_SIZE) - (bhs[i][j] * INPUT_SIZE) / 2, 0)
                left = max((bxs[i][j] * INPUT_SIZE) - (bws[i][j] * INPUT_SIZE) / 2, 0)
                right = min((bxs[i][j] * INPUT_SIZE) + (bws[i][j] * INPUT_SIZE) / 2, INPUT_SIZE)
                bottom = min((bys[i][j] * INPUT_SIZE) + (bhs[i][j] * INPUT_SIZE) / 2, INPUT_SIZE)
                print(left, top, right, bottom)
                cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 1)

            cv2.imshow('', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            cv2.waitKey(0)
            cv2.destroyAllWindows()
