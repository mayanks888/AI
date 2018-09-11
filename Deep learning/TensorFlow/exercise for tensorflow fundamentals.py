import  tensorflow as tf
import numpy as np

with tf.name_scope("variables"):
    update_var=tf.Variable(initial_value=0,name="update_variable",trainable=False,dtype=tf.float32)
    run_count=tf.Variable(initial_value=0,name='total_run',trainable=False,dtype=tf.float32)


with tf.name_scope("Transformation"):
    with tf.name_scope("Initialise"):
        a=tf.placeholder(tf.float32,shape=None,name="input")
    with tf.name_scope("Intermediate"):
        b = tf.reduce_prod(a, name="pord_B")
        c = tf.reduce_sum(a, name='Sum_C')
    with tf.name_scope("Output"):
        d=tf.add(c,b,name='Sum_d')

with tf.name_scope("update"):
    update_var=update_var.assign_add(d)
    run_count=run_count.assign_add(1)



#this section will generate summary
'''with tf.name_scope('summary_report'):
    avg_sumary=tf.div(update_var,tf.cast(run_count,tf.float32),'Average_Summary')
    tf.summary.scalar(update_var, name="output_summary")
    tf.summary.scalar(b'RunCount', run_count, name="runCount")
    tf.summary.scalar(b'AVG', avg_sumary, name="AVG_summary")

merged_summaries = tf.merge_all_summaries()#merging all  summaries'''
init=tf.global_variables_initializer()
sess=tf.Session()
sess.run(init)

def run_graph(input_tensor):
    feed_dict = {a: input_tensor}
    # _, step, summary = sess.run([d, run_count,update_var],feed_dict=feed_dict)
    value_d, step, total_sum = sess.run([d, run_count, update_var], feed_dict=feed_dict)
    # value_d=sess.run(d,feed_dict=feed_dict)
    # value_d, step, total_sum = sess.run([d, run_count, update_var], feed_dict=feed_dict)
    print(value_d)
    print(step)
    print(total_sum)
# print (sess.run(d,feed_dict={a:[2,3]}))

print (run_graph([3,2]))
print (run_graph([3,2]))
print (run_graph([5,2]))
# print (run_graph([4,2]))