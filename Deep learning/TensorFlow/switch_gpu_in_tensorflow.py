import os
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# self.sess = tf.Session(graph=detection_graph)
sess = tf.Session(config=config,graph=detection_graph)