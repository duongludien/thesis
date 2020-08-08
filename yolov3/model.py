import utils
import tensorflow as tf


class YOLOv3:

    def __init__(self, cfg_path):
        self.config_blocks = utils.load_config(cfg_path)

        # Parsing network information
        net_info = self.config_blocks[0]
        self.BATCH = int(net_info['batch'])
        self.SUBDIVISIONS = int(net_info['subdivisions'])
        self.WIDTH = int(net_info['width'])
        self.HEIGHT = int(net_info['height'])
        self.CHANNELS = int(net_info['channels'])
        self.MOMENTUM = float(net_info['momentum'])
        self.DECAY = float(net_info['decay'])
        self.ANGLE = int(net_info['angle'])
        self.SATURATION = float(net_info['saturation'])
        self.EXPOSURE = float(net_info['exposure'])
        self.HUE = float(net_info['hue'])
        self.LEARNING_RATE = float(net_info['learning_rate'])
        self.BURN_IN = int(net_info['burn_in'])
        self.MAX_BATCHES = int(net_info['max_batches'])
        self.POLICY = net_info['policy']
        self.STEPS = [int(x) for x in net_info['steps'].split(',')]
        self.SCALES = [float(x) for x in net_info['scales'].split(',')]
        self.GRID_SIZES = []
        self.ANCHORS = []
        self.NUM_CLASSES = 22   # Default number of classes, change to number of classes in YOLO layer later

    def forward(self, inputs):
        # inputs tensor should be defined like this
        # inputs = tf.placeholder(dtype=tf.float32,
        #                         name='input_images',
        #                         shape=[self.BATCH, self.WIDTH, self.HEIGHT, self.CHANNELS])

        # The first layer is input images
        layers = {-1: inputs}
        previous_filters = self.CHANNELS

        # weights list
        weights_list = []

        # YOLO outputs
        outputs = None

        for index, block in enumerate(self.config_blocks[1:]):

            # ====================== convolutional layer ======================
            if block['name'] == 'convolutional':

                filters = int(block['filters'])
                size = int(block['size'])
                stride = int(block['stride'])
                pad = int(block['pad'])
                activation = block['activation']
                try:
                    batch_normalize = int(block['batch_normalize'])
                except KeyError:
                    batch_normalize = 0

                if pad:
                    pad = 'SAME'
                else:
                    pad = 'VALID'

                output, weights = utils.conv_layer(layer_index=index,
                                                   layer_input=layers[index - 1],
                                                   input_filters=previous_filters,
                                                   output_filters=filters,
                                                   size=size,
                                                   stride=stride,
                                                   pad=pad,
                                                   activation=activation,
                                                   batch_normalize=batch_normalize)

                weights_list += weights

                previous_filters = filters

                # Finally, add this layer output to list
                layers[index] = output

            # ====================== max pooling layer ======================
            elif block['name'] == 'maxpool':

                size = int(block['size'])
                stride = int(block['stride'])

                output = tf.layers.max_pooling2d(inputs=layers[index - 1],
                                                 pool_size=[size, size],
                                                 strides=stride,
                                                 padding='same',
                                                 name='{}_maxpool'.format(index))

                # Filters doesn't change, just add this layer output to list
                layers[index] = output

            # ====================== shortcut layer ======================
            elif block['name'] == 'shortcut':

                from_layer = int(block['from'])

                # Just add 2 layers (the previous and the layers from_layer)
                # So the number of filters will not change
                output = tf.add(layers[index - 1], layers[index + from_layer],
                                name='{}_shortcut_add_{}_{}'.format(index, index - 1, index + from_layer))

                # print(output)
                # Finally, add this layer output to list
                layers[index] = output

            # ====================== route layer ======================
            elif block['name'] == 'route':

                routes = block['layers'].split(',')

                start = int(routes[0])
                try:
                    end = int(routes[1])
                except IndexError:
                    end = 0

                # Calculate the number of step from index to start and end
                if start > 0:
                    start = start - index
                if end > 0:
                    end = end - index

                if end < 0:
                    output = tf.concat(values=[layers[index + start], layers[index + end]],
                                       axis=-1,
                                       name='{}_route_concat_{}_{}'.format(index, index + start, index + end))
                else:
                    output = layers[index + start]
                    output = tf.identity(output, name='{}_route_to_{}'.format(index, index + start))

                previous_filters = output.get_shape().as_list()[-1]

                # print(output)
                # Finally, add this layer output to list
                layers[index] = output

            # ====================== upsample layer ======================
            elif block['name'] == 'upsample':
                stride = int(block['stride'])

                # Just increase size
                old_shape = layers[index - 1].get_shape().as_list()
                old_width = old_shape[1]
                old_height = old_shape[2]

                new_width = old_width * stride
                new_height = old_height * stride

                with tf.variable_scope('{}_upsample'.format(index)):
                    output = tf.image.resize_images(images=layers[index - 1],
                                                    size=[new_width, new_height],
                                                    align_corners=True,
                                                    method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

                # print(output)
                # Finally, add this layer output to list
                layers[index] = output

            # ====================== yolo layer ======================
            elif block['name'] == 'yolo':

                mask = block['mask'].split(',')
                mask = [int(x) for x in mask]

                anchors = block['anchors'].split(',')
                anchors = [int(x) for x in anchors]
                anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
                anchors = [anchors[i] for i in mask]

                num_classes = int(block['classes'])
                self.ANCHORS.append(anchors)
                self.GRID_SIZES.append(layers[index - 1].get_shape().as_list()[1])
                self.NUM_CLASSES = num_classes

                output = utils.transform_features_map(input_size=self.WIDTH,
                                                      layer_index=index,
                                                      features_map=layers[index - 1],
                                                      anchors=anchors,
                                                      num_classes=num_classes)

                if outputs is None:
                    outputs = output
                else:
                    outputs = tf.concat(values=[outputs, output], axis=1)

                # Finally, add this layer output to list
                layers[index] = output

        return weights_list, outputs
