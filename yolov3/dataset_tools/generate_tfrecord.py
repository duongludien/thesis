import cv2
import numpy as np
import os, glob
import tensorflow as tf
import io


IMAGES_DIR = '/home/diendl/Desktop/new_dataset/data/train'
LABELS_DIR = '/home/diendl/Desktop/new_dataset/data/train'
IMAGES_FORMAT = '.png'
OUTPUT_SIZE = 416


def resize_image(img, new_size):

    height = img.shape[0]
    width = img.shape[1]
    portrait = height > width

    # Choose the large dimension to resize and calculate other dimension based on it
    if height > width:
        new_height = new_size
        stride = new_size / height
        new_width = int(width * stride)
    else:
        new_width = new_size
        stride = new_size / width
        new_height = int(height * stride)

    img = cv2.resize(img, (new_width, new_height))

    # Add canvas to keep aspect ratio
    canvas = np.full((new_size, new_size, 3), 128)

    canvas[(new_size - new_height) // 2: (new_size - new_height) // 2 + new_height,
           (new_size - new_width) // 2: (new_size - new_width) // 2 + new_width,
           :] = img

    # cv2.cvtColor requires uint8
    img = canvas.astype(np.uint8)

    # By default, image which is loaded from cv2 is BGR
    # we have to convert it into RGB format
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Next, we'll normalize our image
    # img = img / 255.

    return img, portrait, new_height, new_width


def transform_input(image_path, label_path):
    image = cv2.imread(image_path)
    image, portrait, new_height, new_width = resize_image(image, OUTPUT_SIZE)
    f = open(label_path, 'rt')
    label = f.read().split('\n')
    lines = [line for line in label if len(line) != 0]
    lines = [line.split(' ') for line in lines]
    lines = [[float(item) for item in line] for line in lines]

    for index in range(0, len(lines)):
        line = lines[index]
        line[0] = int(line[0])  # Class index must be integer
        if portrait:
            line[1] = (OUTPUT_SIZE - new_width + 2 * line[1] * new_width) / (2.0 * OUTPUT_SIZE)
            line[-2] = (new_width / OUTPUT_SIZE) * line[-2]
        else:
            line[2] = (OUTPUT_SIZE - new_height + 2 * line[2] * new_height) / (2.0 * OUTPUT_SIZE)
            line[-1] = (new_height / OUTPUT_SIZE) * line[-1]

    return image, lines


def write_transformed_labels(labels, dst_path):
    f = open(dst_path, 'wt')
    for line in labels:
        f.write(' '.join([str(item) for item in line]))
        f.write('\n')
    f.close()


def convert_image_to_bytes(image):
    image_bytes = io.BytesIO(image)
    return image_bytes.getvalue()


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


writer = tf.python_io.TFRecordWriter('./train.tf')

images_list = glob.glob(os.path.join(IMAGES_DIR, '*' + IMAGES_FORMAT))
for image_name in images_list:
    print('Processing {}'.format(image_name))

    image_path = os.path.join(IMAGES_DIR, image_name)
    label_path = os.path.join(LABELS_DIR, image_name.replace(IMAGES_FORMAT, '.txt'))
    image, labels = transform_input(image_path, label_path)
    bytes_image = convert_image_to_bytes(image)
    class_indices = []
    bx = []
    by = []
    bw = []
    bh = []
    for item in labels:
        class_indices.append(item[0])
        bx.append(item[1])
        by.append(item[2])
        bw.append(item[3])
        bh.append(item[4])

    feature_dict = {
        'image': bytes_list_feature([bytes_image]),
        'class_indices': int64_list_feature(class_indices),
        'bx': float_list_feature(bx),
        'by': float_list_feature(by),
        'bw': float_list_feature(bw),
        'bh': float_list_feature(bh)
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))

    writer.write(example.SerializeToString())
    print('Successfully: {}'.format(image_name))

writer.close()


# img, labels = transform_input('/home/diendl/Desktop/test/302_0003.png',
#                               '/home/diendl/Desktop/test/302_0003.txt')
# write_transformed_labels(labels, '/home/diendl/Desktop/test/out.txt')


# for index in range(0, len(labels)):
#     top = max((labels[index][2] * OUTPUT_SIZE) - (labels[index][-1] * OUTPUT_SIZE) / 2, 0)
#     left = max((labels[index][1] * OUTPUT_SIZE) - (labels[index][-2] * OUTPUT_SIZE) / 2, 0)
#     right = min((labels[index][1] * OUTPUT_SIZE) + (labels[index][-2] * OUTPUT_SIZE) / 2, OUTPUT_SIZE)
#     bottom = min((labels[index][2] * OUTPUT_SIZE) + (labels[index][-1] * OUTPUT_SIZE) / 2, OUTPUT_SIZE)
#
#     labels[index][1:5] = [left, top, right, bottom]
#
# print(labels)
#
# labels = [[int(item) for item in element] for element in labels]
#
# for label in labels:
#     cv2.rectangle(img, (label[1], label[2]), (label[3], label[4]), (255, 0, 0), 1)
#
# cv2.imshow('', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
