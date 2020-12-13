import os
import scipy.misc
from scipy.misc import imsave
import tensorflow as tf
import cv2
from tensorflow.contrib.eager.python import tfe
from tensorflow.python.keras.datasets import boston_housing

# enable eager mode
tf.enable_eager_execution()
def save_images_from_event(fn, tag, output_dir='./'):
    output_dir="/home/mayank_sati/Desktop/tensorboard"
    assert(os.path.isdir(output_dir))

    # import tensorflow as tf
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    ######################
    # sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                print(v.tag)
                # if v.tag == tag:
                # if v.tag.find("image-0") != -1:
                im=tf.image.decode_image(v.image.encoded_image_string)
                # im = im_tf.eval({image_str: v.image.encoded_image_string})
                output_fn = os.path.realpath('{}/image_{:05d}.jpg'.format(output_dir, count))
                print("Saving '{}'".format(output_fn))
                scipy.misc.imsave(output_fn, im)
                # cv2.imwrite(output_fn, im)
                count += 1

if __name__ == '__main__':
    save_images_from_event('/home/mayank_sati/codebase/docker_ob/eval/events.out.tfevents.1574943065.3426c80a67ad', 'tag0')

# Specify which GPU(s) to use
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # Or 2, 3, etc. other than 0
#
# # On CPU/GPU placement
# config = tf.compat.v1.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# config.gpu_options.allow_growth = True
# tf.compat.v1.Session(config=config)