import tensorflow as tf
import cv2
import numpy as np
import torch
import random


def load_config(cfg_path):
    """
    Load network architecture from cfg file
    :param cfg_path: Path to cfg file
    :return: a list of dictionaries. Each dictionary is a layer in the neural network.
    Note that the first dictionary is network information, not a layer.
    """

    file = open(cfg_path, 'rt')
    lines = file.read().split('\n')
    file.close()

    # get rid of empty lines
    lines = [line for line in lines if len(line) != 0]

    # get rid of comment lines
    lines = [line for line in lines if line[0] != '#']

    # clean spaces
    lines = [line.lstrip().rstrip() for line in lines]

    # config blocks
    config_blocks = []

    # block is a layer
    block = {}

    for line in lines:
        # beginning of a layer
        if line[0] == '[':
            # append previous block to self.config_blocks, re-initialize block
            if block != {}:
                config_blocks.append(block)
                block = {}
            block['name'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            block[key.rstrip()] = value.lstrip()
    config_blocks.append(block)

    return config_blocks


def conv_layer(layer_index, layer_input, input_filters, output_filters,
               size, stride, pad, activation, batch_normalize):
    """
    Convolution, batch normalize, add bias, activate
    :param layer_index: layer index
    :param layer_input: 4D vector (BHWC)
    :param input_filters: layer_input depth
    :param output_filters: output depth
    :param size: kernel size
    :param stride: convolution stride
    :param pad: convolution padding size
    :param activation: activation function name
    :param batch_normalize: batch normalize or not?
    :return: Result and a list of weights, biases
    """

    weights_list = []

    with tf.variable_scope('{}_conv'.format(layer_index)):
        weights = tf.get_variable(
            initializer=tf.truncated_normal(shape=[size, size, input_filters, output_filters],
                                            stddev=1e-1,
                                            dtype=tf.float32),
            trainable=True,
            name='weights')

        output = tf.nn.conv2d(input=layer_input,
                              filter=weights,
                              strides=[1, stride, stride, 1],
                              padding=pad,
                              name='conv')

        if batch_normalize:
            beta_offset = tf.get_variable(initializer=tf.zeros_initializer(),
                                          shape=[output_filters],
                                          trainable=True,
                                          name='beta_offset')
            weights_list.append(beta_offset)

            gamma_scale = tf.get_variable(initializer=tf.zeros_initializer(),
                                          shape=[output_filters],
                                          trainable=True,
                                          name='gamma_scale')
            weights_list.append(gamma_scale)

            mean = tf.get_variable(initializer=tf.zeros_initializer(),
                                   shape=[output_filters],
                                   name='mean')
            weights_list.append(mean)

            variance = tf.get_variable(initializer=tf.zeros_initializer(),
                                       shape=[output_filters],
                                       name='variance')
            weights_list.append(variance)

            output = tf.nn.batch_normalization(x=output,
                                               mean=mean,
                                               variance=variance,
                                               offset=beta_offset,
                                               scale=gamma_scale,
                                               variance_epsilon=1e-05,
                                               name='batch_normalize')

        else:

            bias = tf.get_variable(initializer=tf.truncated_normal(shape=[output_filters],
                                                                   stddev=1e-1,
                                                                   dtype=tf.float32),
                                   name='bias')
            weights_list.append(bias)

            output = tf.nn.bias_add(value=output,
                                    bias=bias,
                                    name='add_bias')

        if activation == 'leaky':
            output = tf.nn.leaky_relu(features=output,
                                      alpha=0.1,
                                      name='leaky_relu')

        weights_list.append(weights)

    return output, weights_list


def transform_features_map(input_size, layer_index, features_map, anchors, num_classes):
    """
    Reshape, apply activation, add offset and convert feature map to detection result
    :param input_size: Input image size
    :param layer_index: Layer index
    :param features_map: feature map to transform
    :param anchors: a list of anchors which were loaded from cfg file
    :param num_classes: Number of classes
    :return: Detection result
    """

    features_map_shape = features_map.get_shape().as_list()

    batch = features_map_shape[0]
    grid_size = features_map_shape[1]
    num_anchors = len(anchors)
    stride = input_size // grid_size

    with tf.variable_scope('{}_yolo'.format(layer_index)):
        features_map = tf.reshape(features_map, [batch, grid_size, grid_size, num_anchors, 5 + num_classes])
        features_map = tf.reshape(features_map, [batch, grid_size * grid_size * num_anchors, 5 + num_classes])

        # Transform bx, by
        bx_by = features_map[:, :, 0:2]
        bx_by = tf.sigmoid(bx_by)

        grid = tf.range(start=0, limit=grid_size, delta=1, dtype=tf.float32)
        x_offset, y_offset = tf.meshgrid(grid, grid)
        x_offset = tf.reshape(x_offset, [-1, 1])
        y_offset = tf.reshape(y_offset, [-1, 1])
        offset = tf.concat([x_offset, y_offset], axis=1)
        offset = tf.tile(offset, [batch, num_anchors])
        offset = tf.reshape(offset, [batch, -1, 2])

        transformed_bx_by = tf.add(bx_by, offset)
        transformed_bx_by = tf.multiply(transformed_bx_by, stride, name='bx_by')

        # Transform bw, bh
        bw_bh = features_map[:, :, 2:4]
        anchors = [(a[0] / stride, a[1] / stride) for a in anchors]
        anchors = tf.tile(anchors, [grid_size * grid_size, 1])
        anchors = tf.expand_dims(anchors, axis=0)

        transformed_bw_bh = tf.multiply(tf.exp(bw_bh), anchors)
        transformed_bw_bh = tf.multiply(transformed_bw_bh, stride, name='bw_bh')

        # Transform object confidence
        p = features_map[:, :, 4]
        transformed_p = tf.sigmoid(p)
        transformed_p = tf.reshape(transformed_p, [batch, -1, 1], name='p')
        # print(transformed_p)

        # Transform class scores
        class_scores = features_map[:, :, 5:]
        transformed_class_scores = tf.sigmoid(class_scores, name='class_scores')
        # print(transformed_class_scores)

        transformed_features_map = tf.concat([transformed_bx_by,
                                              transformed_bw_bh,
                                              transformed_p,
                                              transformed_class_scores],
                                             name='transformed_feature_map',
                                             axis=-1)

    return transformed_features_map


def intersection_over_union(box1, box2):
    """
    Calculate IoU between 2 boxes
    :param box1: top-left, right-bottom (a list of 4 values)
    :param box2: same as box1
    :return: IoU
    """

    intersection_x1 = max(box1[0], box2[0])
    intersection_y1 = max(box1[1], box2[1])
    intersection_x2 = min(box1[2], box2[2])
    intersection_y2 = min(box1[3], box2[3])

    intersection_area = max(0, intersection_x2 - intersection_x1) * max(0, intersection_y2 - intersection_y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    if intersection_area > 0:
        union = box1_area + box2_area - intersection_area
    else:
        union = box1_area + box2_area

    iou = intersection_area / union

    return iou


def yolo_ratios_to_real_voc(bx, by, bw, bh, height, width):
    """
    Convert bounding-box info in cfg file to top-left right-bottom coordinates
    :param bx: The first value in cfg
    :param by: The second value in cfg
    :param bw: The third one
    :param bh: The fourth one
    :param height: image height in cfg
    :param width: image width in cfg
    :return: top-left, right-bottom coordinates
    """

    x1 = max((bx * width) - (bw * width) / 2, 0)
    y1 = max((by * height) - (bh * height) / 2, 0)
    x2 = min((bx * width) + (bw * width) / 2, width)
    y2 = min((by * height) + (bh * height) / 2, height)

    return x1, y1, x2, y2


def resize_image(img, new_size):
    """
    Resize image to standard size in cfg file but keep ratios and normalize it
    :param img: Input image which was loaded by OpenCV (BGR)
    :param new_size: destination size
    :return: normalized image (RGB)
    """

    height = img.shape[0]
    width = img.shape[1]

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
    img = img / 255.0

    return img


def load_darknet_weights(weights_path, weights_list):
    """
    Load binary Darknet weights
    :param weights_path: Path to weights file
    :param weights_list: A list of weights and biases tensor
    :return: Assign operations to run in a session
    """

    file = open(weights_path, 'rb')

    # The first 5 values are header information
    # 1. Major version number
    # 2. Minor Version Number
    # 3. Subversion number
    # 4,5. Images seen by the network (during training)
    header = np.fromfile(file, dtype=np.int32, count=5)

    weights = np.fromfile(file, dtype=np.float32)
    pointer = 0

    file.close()

    assign_ops = []
    for i in range(len(weights_list)):
        node = weights_list[i]

        tf_node_shape = weights_list[i].get_shape().as_list()

        if len(tf_node_shape) > 1:
            # Convert from TensorFlow HWC filter shape: [H x W x in_C x out_C]
            # to Darknet CHW filter shape: out_C x in_C x H x W
            darknet_node_shape = [tf_node_shape[3],
                                  tf_node_shape[2],
                                  tf_node_shape[0],
                                  tf_node_shape[1]]
        else:
            # Bias, batch norm parameters shape are the same
            darknet_node_shape = tf_node_shape[0]

        num_params = np.prod(darknet_node_shape)

        values = np.reshape(weights[pointer:pointer + num_params], darknet_node_shape)

        # Convert from Darknet filter back to TensorFlow filter
        if len(tf_node_shape) > 1:
            values = np.transpose(values, [2, 3, 1, 0])

        pointer += num_params
        assign_ops.append(tf.assign(node, values))

    return assign_ops


def threshold(detections, object_confidence_threshold=0.25):
    """
    Reduce low confidence bounding boxes (for a batch of images)
    :param detections: detection result
    :param object_confidence_threshold: threshold to reduce cells which has't responsibility to detect objects
    :return: a tensor of size (batch, num_detections, 7)
    which 7 is x1, y1, x2, y2, object_confidence, class_scores, class_indices
    """

    with tf.variable_scope(name_or_scope='thresholding'):
        object_confidence = detections[:, :, 4]
        object_confidence_mask = tf.cast(object_confidence > object_confidence_threshold, tf.float32)
        object_confidence_mask = tf.expand_dims(object_confidence_mask, axis=-1)

        thresholded = tf.multiply(detections, object_confidence_mask)
        bx = thresholded[:, :, 0]
        by = thresholded[:, :, 1]
        bw = thresholded[:, :, 2]
        bh = thresholded[:, :, 3]
        object_confidence = thresholded[:, :, 4]
        class_scores = tf.reduce_max(thresholded[:, :, 5:], axis=-1)
        class_indices = tf.argmax(thresholded[:, :, 5:], axis=-1)
        class_indices = tf.cast(class_indices, tf.float32)

        x1 = tf.expand_dims(bx - bw / 2,
                            axis=-1,
                            name='x1')
        y1 = tf.expand_dims(by - bh / 2,
                            axis=-1,
                            name='y1')
        x2 = tf.expand_dims(bx + bw / 2,
                            axis=-1,
                            name='x2')
        y2 = tf.expand_dims(by + bh / 2,
                            axis=-1,
                            name='y2')

        output = tf.concat(values=[x1,
                                   y1,
                                   x2,
                                   y2,
                                   tf.expand_dims(object_confidence, axis=-1),
                                   tf.expand_dims(class_scores, axis=-1),
                                   tf.expand_dims(class_indices, axis=-1)],
                           axis=-1)

    return output


def numpy_iou(box1, box2):

    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = np.max(b1_x1, b2_x1)
    inter_rect_y1 = np.max(b1_y1, b2_y1)
    inter_rect_x2 = np.min(b1_x2, b2_x2)
    inter_rect_y2 = np.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = np.max(0, inter_rect_x2 - inter_rect_x1 + 1) * np.max(0, inter_rect_y2 - inter_rect_y1 + 1)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def numpy_non_max_suppression(thresholded_detections, num_classes, iou_threshold):
    """
    Perform non-maximum suppression using NumPy
    :param thresholded_detections: thresholded detections
    :param num_classes: the number of classes
    :param iou_threshold: IoU threshold
    :return: a vector which has shape (batch, ?, 7)
    which 7 is [x1, y1, x2, y2, object_confidence, class_scores, class_indices]
    """

    batch_size = thresholded_detections.shape[0]

    for image_index in range(batch_size):

        image_detections = thresholded_detections[image_index]
        nonzero_indices = np.nonzero(image_detections[:, 4])
        image_detections = image_detections[nonzero_indices]
        unique_classes = np.unique(image_detections[:, -1])

        if len(unique_classes) == 0:
            print('There is no object in this image')
            continue

        for class_item in unique_classes:
            class_mask = image_detections * np.expand_dims(image_detections[:, -1] == class_item, axis=-1)
            class_mask_index = np.nonzero(class_mask[:, -2])
            image_detections_class = image_detections[class_mask_index]

            # Sorting image_prediction_class by object confidence DESCENDING
            sorted_confidence_indices = np.argsort(image_detections_class[:, 4])[::-1]
            image_detections_class = image_detections_class[sorted_confidence_indices]

            num_boxes = image_detections_class.shape[0]
            for box_index in range(num_boxes):
                this_box = np.expand_dims(image_detections_class[box_index], axis=0)
                try:
                    ious = numpy_iou(this_box, image_detections_class[box_index + 1:])
                    print(ious)
                except TypeError:
                    break


def torch_non_max_suppression(detections, confidence_threshold, num_classes, nms_conf):
    # Check all boxes which have object confidence less than threshold
    object_confidence_mask = (detections[:, :, 4] > confidence_threshold).float()
    # Add a dimension for multiplying
    object_confidence_mask = object_confidence_mask.unsqueeze(2)
    # Apply mask to detections
    detections = detections * object_confidence_mask

    # Calculate top-left and right-bottom coordinate
    box_corner = detections.new(detections.shape)
    # top-left x-coordinate = centre_x - width / 2
    box_corner[:, :, 0] = detections[:, :, 0] - detections[:, :, 2] / 2
    # top-left y-coordinate = centre_y - height / 2
    box_corner[:, :, 1] = detections[:, :, 1] - detections[:, :, 3] / 2
    # right-bottom x-coordinate = centre_x + width / 2
    box_corner[:, :, 2] = detections[:, :, 0] + detections[:, :, 2] / 2
    # right-bottom y-coordinate = centre_y + height / 2
    box_corner[:, :, 3] = detections[:, :, 1] + detections[:, :, 3] / 2

    # Transform bx, by, bw, bh to top_left_x, top_left_y, right_bottom_x, right_bottom_y
    detections[:, :, :4] = box_corner[:, :, :4]

    batch_size = detections.size(0)
    write = False

    for index in range(batch_size):
        image_prediction = detections[index]  # 10647 x 85

        # max_confidence, max_confidence_class: 10647
        max_confidence, max_confidence_class = torch.max(input=image_prediction[:, 5:5 + num_classes], dim=1)

        # Add a dimension for multiplying
        max_confidence = max_confidence.float().unsqueeze(1)
        max_confidence_class = max_confidence_class.float().unsqueeze(1)

        # Concatenate image_prediction, max_confidence and max_confidence_class
        sequence = (image_prediction[:, :5], max_confidence, max_confidence_class)
        image_prediction = torch.cat(sequence, dim=1)

        # Get rid of bounding-boxes which have object confidence less than threshold
        # Get index of elements which have non-zero value
        non_zero_index = torch.nonzero(image_prediction[:, 4])
        try:
            # 7 is: bx, by, bw, bh, object_confidence, max_confidence, max_confidence_class
            # After this step, our prediction only have some boxes for each class
            # Perform non-max suppression to get rid of boxes which have low IoU
            image_prediction_ = image_prediction[non_zero_index.squeeze(), :].view(-1, 7)
        except:
            # In this case, there is not any detection
            continue

        try:
            image_classes = unique(image_prediction_[:, -1])  # The last index is the class index
        except IndexError:
            print('There is no object in this image')
            continue

        for class_ in image_classes:
            class_mask = image_prediction_ * (image_prediction_[:, -1] == class_).float().unsqueeze(1)
            class_mask_index = torch.nonzero(class_mask[:, -2]).squeeze()
            image_prediction_class = image_prediction_[class_mask_index].view(-1, 7)

            # Sorting image_prediction_class by object confidence
            conf_sort_index = torch.sort(image_prediction_class[:, 4], descending=True)[1]
            image_prediction_class = image_prediction_class[conf_sort_index]

            # Perform IoU
            no_of_boxes = image_prediction_class.size(0)
            for box_index in range(no_of_boxes):
                # Get the IOUs of all boxes that come after the one we are looking at in the loop
                try:
                    ious = bbox_iou(image_prediction_class[box_index].unsqueeze(0),
                                    image_prediction_class[box_index + 1:])
                except ValueError:
                    break
                except IndexError:
                    break

                # Zero out all the detections that have IoU > treshhold
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_prediction_class[box_index + 1:] *= iou_mask

                # Remove the non-zero entries
                non_zero_index = torch.nonzero(image_prediction_class[:, 4]).squeeze()
                image_prediction_class = image_prediction_class[non_zero_index].view(-1, 7)

            batch_index = image_prediction_class.new(image_prediction_class.size(0), 1).fill_(index)
            # Repeat the batch_id for as many detections of the class cls in the image
            seq = batch_index, image_prediction_class

            if not write:
                output = torch.cat(seq, 1)
                write = True
            else:
                out = torch.cat(seq, 1)
                output = torch.cat((output, out))

        try:
            return output
        except:
            return 0


def bbox_iou(box1, box2):
    """
    Returns the IoU of two bounding boxes
    """
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Get the coordinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1,
                                                                                     min=0)

    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area)

    return iou


def unique(tensor):
    tensor_np = tensor.detach().cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)

    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res


def convert_to_original_size(point, original_height, original_width, new_height_canvas, new_width_canvas):
    """
    Convert a point from YOLO input image size to actual image size
    :param point: a tuple which determine the point to convert
    :param original_height: image original height
    :param original_width: image original width
    :param new_height_canvas: new image size (include canvas)
    :param new_width_canvas: new image size (include canvas)
    :return: a tuple which determine the converted point
    """

    new_x, new_y = point[0], point[1]

    if original_width > original_height:
        new_width_no_canvas = new_width_canvas
        stride = new_width_no_canvas / original_width
        new_height_no_canvas = original_height * stride

        original_x = new_x / stride
        canvas = new_height_canvas - new_height_no_canvas
        original_y = (new_y - canvas / 2) / stride
    else:
        new_height_no_canvas = new_height_canvas
        stride = new_height_no_canvas / original_height
        new_width_no_canvas = original_width * stride

        canvas = new_width_canvas - new_width_no_canvas
        original_x = (new_x - canvas / 2) / stride
        original_y = new_y / stride

    original_x = max(0, original_x)
    original_y = max(0, original_y)

    return int(original_x), int(original_y)


def draw_bounding_box_with_text(image, top_left, right_bottom, text, box_color, text_color):

    # Draw the box
    cv2.rectangle(img=image,
                  pt1=top_left,
                  pt2=right_bottom,
                  color=box_color,
                  thickness=2)

    # Draw a rectangle and fill it to make text background
    font_scale = 0.7
    text_size = cv2.getTextSize(text=text,
                                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                                fontScale=font_scale,
                                thickness=1)[0]

    text_left_bottom = [top_left[0], top_left[1] - 4]
    bottom = False
    if text_left_bottom[1] - text_size[1] < 0:
        bottom = True
        text_left_bottom[1] = text_left_bottom[1] + text_size[1] + 4
    text_left_bottom = tuple(text_left_bottom)
    if bottom:
        text_top_left = (text_left_bottom[0] - 1, text_left_bottom[1] - text_size[1] - 1)
    else:
        text_top_left = (text_left_bottom[0] - 1, text_left_bottom[1] - text_size[1] - 4)
    text_right_bottom = (text_left_bottom[0] + text_size[0], text_left_bottom[1] + 4)
    cv2.rectangle(image, text_top_left, text_right_bottom, box_color, thickness=cv2.FILLED)

    # Finally, put text to image
    cv2.putText(img=image,
                text=text,
                org=text_left_bottom,
                fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=font_scale,
                color=text_color)

    return image


def load_classes(names_path):
    """
    Load class names from text file
    Params:
        names_path: path to classname file
    Return:
        classes: a list of class names
    """

    file = open(names_path, 'rt')
    classes = file.read().split('\n')[:-1]
    return classes


def read_label(label_path):
    f = open(label_path, 'rt')
    label = f.read().split('\n')
    lines = [line for line in label if len(line) != 0]
    lines = [line.split(' ') for line in lines]
    lines = [[float(item) for item in line] for line in lines]
    return lines


def _parse_function(example_proto, height, width, channels):
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
    image = tf.reshape(image, [height, width, channels])
    image = tf.cast(image, tf.float32)

    return image / 255., parsed_features['class_indices'], parsed_features['bx'], \
           parsed_features['by'], parsed_features['bw'], parsed_features['bh']


def create_dataset(tfrecord_files, batch, height, width, channels):
    """
    Create a TFRecord Dataset
    :param tfrecord_files: a list of *.tf files
    :param batch: batch size
    :param height: image height
    :param width: image width
    :param channels: number of image channels
    :return: a TFRecord Dataset instance
    """
    dataset = tf.data.TFRecordDataset(tfrecord_files)
    dataset = dataset.map(lambda x: _parse_function(x, height, width, channels))
    dataset = dataset.repeat().batch(batch)
    return dataset


def get_next_batch(dataset):
    """
    Get next batch from TFRecord Dataset
    :param dataset: a TFRecord Dataset instance
    :return: a list of 6 vectors
        images_batch: a batch of images
        dense_indices_batch: a vector which determines class indices in each image
        dense_bxs_batch: a vector which determines bx values in each image
        dense_bys_batch: a vector which determines by values in each image
        dense_bws_batch: a vector which determines bw values in each image
        dense_bhs_batch: a vector which determines bh values in each image
    """
    dataset_iterator = dataset.make_one_shot_iterator()
    images_batch, sparse_indices_batch, sparse_bxs_batch, sparse_bys_batch, sparse_bws_batch, sparse_bhs_batch \
        = dataset_iterator.get_next()
    dense_indices_batch = tf.sparse_tensor_to_dense(sparse_indices_batch, default_value=-1)
    dense_bxs_batch = tf.sparse_tensor_to_dense(sparse_bxs_batch, default_value=-1)
    dense_bys_batch = tf.sparse_tensor_to_dense(sparse_bys_batch, default_value=-1)
    dense_bws_batch = tf.sparse_tensor_to_dense(sparse_bws_batch, default_value=-1)
    dense_bhs_batch = tf.sparse_tensor_to_dense(sparse_bhs_batch, default_value=-1)

    return images_batch, dense_indices_batch, dense_bxs_batch, \
           dense_bys_batch, dense_bws_batch, dense_bhs_batch


def construct_true_vector(true_boxes, num_classes, grid_sizes, anchors, width, height):
    """
    Construct true vector for an image
    :param true_boxes: list of true boxes
    :param anchors: List of list anchors. Each grid size has a list of anchors
    :param grid_sizes: List of grid sizes. There are multiple grid in YOLOv3
    :param num_classes: The number of classes
    :return: True vector with shape [1, num_boxes, 5 + num_classes]
    """

    # Shuffle all true boxes to catch the problem multiple objects in the same grid cell
    random.shuffle(true_boxes)
    class_indices, bxs, bys, bws, bhs = zip(*true_boxes)

    outputs = None

    for grid_index in range(len(grid_sizes)):
        # print('Grid size: {}x{}'.format(grid_sizes[grid_index], grid_sizes[grid_index]))

        num_anchors = len(anchors[grid_index])
        # print('Number of anchors: {}'.format(num_anchors))
        # print('Anchors:', anchors[grid_index])

        output = np.zeros(shape=[grid_sizes[grid_index] * grid_sizes[grid_index] * num_anchors, 5 + num_classes])
        # print('Output shape for this grid: {}'.format(output.shape))

        num_true_boxes = len(bxs)
        # print('Number of true boxes: {}'.format(num_true_boxes))

        for box_index in range(num_true_boxes):

            bx = bxs[box_index]
            by = bys[box_index]
            bw = bws[box_index]
            bh = bhs[box_index]
            class_index = class_indices[box_index]
            # print('Box {}: bx = {}, by = {}, bw = {}, bh = {}, class = {}'
            #       .format(box_index, bx, by, bw, bh, class_index))

            # Choose the grid cell which object belongs to
            col = np.floor((bx * width) / (width / grid_sizes[grid_index]))
            row = np.floor((by * height) / (height / grid_sizes[grid_index]))
            # print('This box belongs to cell ({}, {})'.format(row, col))

            truth_box = yolo_ratios_to_real_voc(bx, by, bw, bh, height, width)
            # print('Ground-truth box:', truth_box)

            # Convert anchor to box format to calculate IoU
            ious = []
            for anchor in anchors[grid_index]:
                anchor_box = yolo_ratios_to_real_voc(bx, by, anchor[0] / height, anchor[1] / width, height, width)
                ious.append(intersection_over_union(truth_box, anchor_box))
            selected_anchor = np.argmax(ious) + 1
            # print('Selected anchor: {}'.format(selected_anchor))

            position = int(row * grid_sizes[grid_index] * num_anchors + col * num_anchors + selected_anchor)
            # print('This box position in true vector: {}'.format(position))

            # Output vector has order bx, by, bh, bw, p, [class_scores]
            output[position, 0] = bx * width
            output[position, 1] = by * height
            output[position, 2] = bh * height
            output[position, 3] = bw * width
            output[position, 4] = 1  # Object confidence
            output[position, 5 + class_index] = 1  # Class score

        output = np.expand_dims(output, axis=0)

        if outputs is None:
            outputs = output
        else:
            outputs = np.concatenate((outputs, output), axis=1)

    return outputs


def construct_batch_true_vector(indices_batch, bxs_batch, bys_batch, bws_batch, bhs_batch,
                                num_classes, grid_sizes, anchors, width, height):

    outputs_batch = None

    batch_size = indices_batch.shape[0]

    # Iterate over all images in this batch
    for index in range(batch_size):

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

        true_boxes = list(zip(indices, bxs, bys, bws, bhs))

        outputs = construct_true_vector(true_boxes=true_boxes,
                                        num_classes=num_classes,
                                        grid_sizes=grid_sizes,
                                        anchors=anchors,
                                        width=width,
                                        height=height)
        if outputs_batch is None:
            outputs_batch = outputs
        else:
            outputs_batch = np.concatenate((outputs_batch, outputs), axis=0)

    return outputs_batch
