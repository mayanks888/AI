import tensorflow as tf


'''input_x = tf.placeholder(tf.float32,shape=[None,224,224,3])
y_true = tf.placeholder(tf.float32,shape=[None,4])
# ### Layers
# x_image = tf.reshape(x,[-1,224,224,3])

# importing vgg base_model
# imput_shape=new_image_input[0].shape#good thing to know the shape of input array
x_image = tf.reshape(input_x,[-1,224,224,3])
# input_tensor = tf.keras.Input(shape=(224, 224, 3))
#
# inputshape=(224,224,3)
base_model=tf.keras.applications.vgg16.VGG16(weights = "imagenet", include_top=True,input_tensor=x_image)#loading vgg16  model trained n imagenet datasets
print (base_model.summary())
last_layer = base_model.get_layer('block5_conv3').output#taking the previous layers from vgg16 officaal model
features=last_layer
rpn= region_proposal_network.RegionProposalNetwork(features, gt_bbox['train'], im_dims['train'],anchor_scale, Mode)'''

#
# def smoothL1(x, sigma):
#     conditional = tf.less(tf.abs(x), 1 / sigma ** 2)
#     close = 0.5 * (sigma * 2) ** 2
#     far = tf.abs(x) - 0.5 / sigma ** 2

def smoothL1(x, sigma):

        k=tf.divide(1,(tf.square(sigma)))
        val=tf.cond( (tf.less(tf.abs(x) < k,(0.5 * k ** 2*x**2),(tf.abs(x) - 0.5 / k))))

        # if  (tf.less(tf.abs(x), (1 / sigma ** 2))):
        #     val = 0.5 * (sigma * 2) ** 2
        # else:
        #     val = tf.abs(x) - 0.5 / sigma ** 2
        return val
def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        sigma2 = sigma * sigma

        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))

        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))

        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul


data1=tf.random_uniform([1,37,62,36],minval=0,maxval=2,dtype=tf.float32)
l
dat=smoothL1(data1,2)

sess=tf.Session()
print(sess.run(dat))
print(dat.shape)