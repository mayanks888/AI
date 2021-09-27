import tarfile
import pickle
import random
import numpy as np
import tensorflow as tf
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tensorflow.keras import backend as K
from tensorflow.summary import FileWriter


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        # Graph -> GraphDef ProtoBuf
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph

def _preprocess_and_save(normalize, one_hot_encode, features, labels, filename):
    features = normalize(features)
    labels = one_hot_encode(labels)

    pickle.dump((features, labels), open(filename, 'wb'))

def conv_net(x, keep_prob):
    conv1_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], mean=0, stddev=0.08))
    conv2_filter = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], mean=0, stddev=0.08))
    conv3_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 128, 256], mean=0, stddev=0.08))
    conv4_filter = tf.Variable(tf.truncated_normal(shape=[5, 5, 256, 512], mean=0, stddev=0.08))

    # 1, 2
    conv1 = tf.nn.conv2d(x, conv1_filter, strides=[1,1,1,1], padding='SAME')
    conv1 = tf.nn.relu(conv1)
    conv1_pool = tf.nn.max_pool(conv1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv1_bn = tf.layers.batch_normalization(conv1_pool)

    # 3, 4
    conv2 = tf.nn.conv2d(conv1_bn, conv2_filter, strides=[1,1,1,1], padding='SAME')
    conv2 = tf.nn.relu(conv2)
    conv2_pool = tf.nn.max_pool(conv2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv2_bn = tf.layers.batch_normalization(conv2_pool)

    # 5, 6
    conv3 = tf.nn.conv2d(conv2_bn, conv3_filter, strides=[1,1,1,1], padding='SAME')
    conv3 = tf.nn.relu(conv3)
    conv3_pool = tf.nn.max_pool(conv3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv3_bn = tf.layers.batch_normalization(conv3_pool)

    # 7, 8
    conv4 = tf.nn.conv2d(conv3_bn, conv4_filter, strides=[1,1,1,1], padding='SAME')
    conv4 = tf.nn.relu(conv4)
    conv4_pool = tf.nn.max_pool(conv4, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
    conv4_bn = tf.layers.batch_normalization(conv4_pool)

    # 9
    flat = tf.contrib.layers.flatten(conv4_bn)

    # 10
    full1 = tf.contrib.layers.fully_connected(inputs=flat, num_outputs=128, activation_fn=tf.nn.relu)
    full1 = tf.nn.dropout(full1, keep_prob)
    full1 = tf.layers.batch_normalization(full1)

    # 11
    full2 = tf.contrib.layers.fully_connected(inputs=full1, num_outputs=256, activation_fn=tf.nn.relu)
    full2 = tf.nn.dropout(full2, keep_prob)
    full2 = tf.layers.batch_normalization(full2)

    # 12
    full3 = tf.contrib.layers.fully_connected(inputs=full2, num_outputs=512, activation_fn=tf.nn.relu)
    full3 = tf.nn.dropout(full3, keep_prob)
    full3 = tf.layers.batch_normalization(full3)

    # 13
    full4 = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=1024, activation_fn=tf.nn.relu)
    full4 = tf.nn.dropout(full4, keep_prob)
    full4 = tf.layers.batch_normalization(full4)

    # 14
    out = tf.contrib.layers.fully_connected(inputs=full3, num_outputs=10, activation_fn=None)
    return out
#
def main():

    # Hyper parameters
    epochs = 10
    batch_size = 128
    keep_probability = 0.7
    learning_rate = 0.001

    # Remove previous weights, bias, inputs, etc..
    tf.reset_default_graph()

    # Inputs
    x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
    y =  tf.placeholder(tf.float32, shape=(None, 10), name='output_y')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')

    # Build model
    logits = conv_net(x, keep_prob)
    model = tf.identity(logits, name='logits') # Name logits Tensor, so that can be loaded from disk after training

    # Loss and Optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Accuracy
    correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

    # Training Phase
    # save_model_path = './image_classification'
    save_model_path = 'saved_model/image_classification'

    print('Training...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())

        # # Training cycle
        # for epoch in range(epochs):
        #     # Loop over all batches
        #     n_batches = 5
        #     for batch_i in range(1, n_batches + 1):
        #         for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
        #             train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        #
        #         print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        #         print_stats(sess, batch_features, batch_labels, cost, accuracy)

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)
############################################################################################
        # saver = tf.train.import_meta_graph('/home/mayank_sati/codebase/python/camera/tensorflow/CIFAR10-img-classification-tensorflow/saved_model/image_classification.meta')
        # saver.restore(sess, tf.train.latest_checkpoint('./'))
        # print("finised loading")
        # ################################################33
        # frozen_graph = freeze_session(sess)
        frozen_graph = freeze_session(sess,output_names=['output_y'])
        # frozen_graph = freeze_session(K.get_session())

        tf.train.write_graph(frozen_graph, "model", "tf_model_ti.pb", as_text=False)
        FileWriter("__tb", sess.graph)
        # @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        # frozen_graph_def = tf.graph_util.convert_variables_to_constants(sess, sess.graph_def, ["output_y"])
        # # Save the frozen graph
        # with open('output_graph.pb', 'wb') as f:
        #     f.write(frozen_graph_def.SerializeToString())
        ##########################################################################3
        # Print all operators in the graph
        for op in sess.graph.get_operations():
            print(op)
        # Print all tensors produced by each operator in the graph
        for op in sess.graph.get_operations():
            print(op.values())
        tensor_names = [[v.name for v in op.values()] for op in sess.graph.get_operations()]
        tensor_names = np.squeeze(tensor_names)
        print(tensor_names)
        ############################################################################
        # graph_def = sess.as_graph_def()
        # all_names = [n.name for n in graph_def.node]
        #
        # print(all_names)
        ############################################################################
        # from tensorflow.python.tools import freeze_graph
        #
        # tf.train.write_graph(sess.graph_def, './pb_model', 'model.pb') # Generate a model file.
        # freeze_graph.freeze_graph(
        # input_graph='./pb_model/model.pb',   # Pass the model file generated using write_graph.
        # input_saver='',
        # input_binary=False,
        # input_checkpoint='/home/mayank_sati/codebase/python/camera/tensorflow/CIFAR10-img-classification-tensorflow/saved_model/image_classification.ckpt.data-00000-of-00001',  # Pass the checkpoint file generated in training.
        # output_node_names='output',  #  Consistent with the output node of the inference network.
        # restore_op_name='save/restore_all',
        # filename_tensor_name='save/Const:0',
        # output_graph='./pb_model/alexnet.pb',   # Set to the name of the inference network to be generated.
        # clear_devices=False,
        # initializer_nodes='')
        # print("done")
        ############################################
if __name__ == "__main__":
    main()
