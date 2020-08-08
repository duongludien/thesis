from model import YOLOv3
import tensorflow as tf
import utils


CFG_PATH = 'cfg/yolov3_traffic_inference.cfg'
DARKNET_WEIGHTS_PATH = '/home/diendl/tiny_model/yolov3_traffic_train_3100.weights'
FROZEN_MODEL_NAME = './frozen_full_model_3100.pb'

model = YOLOv3(CFG_PATH)

graph = tf.Graph()

with graph.as_default():
    inputs = tf.placeholder(dtype=tf.float32,
                            name='input_images',
                            shape=[model.BATCH, model.WIDTH, model.HEIGHT, model.CHANNELS])

    weights_list, predictions = model.forward(inputs)
    predictions = tf.identity(predictions, name='outputs')
    load_weights_ops = utils.load_darknet_weights(DARKNET_WEIGHTS_PATH, weights_list)
    saver = tf.train.Saver()

with tf.Session(graph=graph) as sess:
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.logging.info('Loading weights from Darknet weights file...')
    sess.run(load_weights_ops)

    output_graph_def = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                                    input_graph_def=graph.as_graph_def(),
                                                                    output_node_names=['outputs'])
    with tf.gfile.GFile(FROZEN_MODEL_NAME, "wb") as f:
        f.write(output_graph_def.SerializeToString())
