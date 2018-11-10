"""
Several improvements have been added within respect to exp01.py :
 -- better loss function for the synthetic gradients model
 -- train, validation and test accuracy evaluations
 -- tensorboard support

This experiment has the same parameters reported in the paper.
The synthetic gradients model takes into account the state of the layer at the same level.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import argparse
import os.path
import sys
import time
import numpy as np

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
#from tensorflow.examples.tutorials.mnist import mnist

# The MNIST dataset has 10 classes, representing the digits 0 through 9.
NUM_CLASSES = 10

# The MNIST images are always 28x28 pixels.
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE

class Flags:
    def __init__(self):
        self.learning_rate = 3e-5 #3*(10**-5)
        self.max_steps = 500*1000
        self.hidden1 = 128
        self.hidden2 = 32
        self.batch_size = 256
        #self.input_data_dir = '/tmp/tensorflow/mnist/input_data'
        self.input_data_dir = '/notebooks/datasets/mnist'
        self.log_dir = '/tmp/tensorflow/mnist/logs/fully_connected_feed'
        self.summaries_dir = "./exp02/tensorboard"
        self.fake_data = False
        

FLAGS = Flags()


tf.reset_default_graph()

if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
if tf.gfile.Exists(FLAGS.summaries_dir):
    tf.gfile.DeleteRecursively(FLAGS.summaries_dir)
    tf.gfile.MakeDirs(FLAGS.summaries_dir)

start_time = time.time()
    
with tf.device("/gpu:0"):
    #Graph(s)
    with tf.Graph().as_default():

        learning_rate_placeholder = tf.placeholder(tf.float32, name="learning_rate_placeholder")

        # global optimizer
        #optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate_placeholder)

        """first half of the net"""
        images_placeholder = tf.placeholder(tf.float32, shape=(FLAGS.batch_size, IMAGE_PIXELS), name="net_input")
        # Hidden 1
        with tf.name_scope('hidden1'):
            #forward pass
            weights1 = tf.Variable(tf.truncated_normal([IMAGE_PIXELS, FLAGS.hidden1], stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
                name='weights')
            biases1 = tf.Variable(tf.zeros([FLAGS.hidden1]), name='biases')
            hidden1 = tf.nn.relu(tf.matmul(images_placeholder, weights1) + biases1)

            #backward pass
            hidden1_synt_grad_placeholder = tf.placeholder(tf.float32, shape=hidden1.get_shape().as_list(), name="synt_grad_input")
            hidden1_grad_vars = optimizer.compute_gradients(hidden1, grad_loss=hidden1_synt_grad_placeholder)
            #hidden1_grad_vars_names = [ (x[1].name, x[0]) for x in hidden1_grad_vars ]
            #print(hidden1_grad_vars_names)
            hidden1_appliable_grads = [ t for t in hidden1_grad_vars if t[0] is not None]
            hidden1_optimize = optimizer.apply_gradients(hidden1_appliable_grads)

        """second half of the net"""
        # Hidden 2
        with tf.name_scope('hidden2'):
            #forward pass
            layer2_input = tf.placeholder(tf.float32, shape=hidden1.get_shape().as_list(), name="layer2_input")

            weights2 = tf.Variable(tf.truncated_normal([FLAGS.hidden1, FLAGS.hidden2],stddev=1.0 / math.sqrt(float(FLAGS.hidden1))), 
                name='weights')
            biases2 = tf.Variable(tf.zeros([FLAGS.hidden2]), name='biases')
            hidden2 = tf.nn.relu(tf.matmul(layer2_input, weights2) + biases2)

        # Linear
        labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name="net_labels")
        with tf.name_scope('softmax_linear'):
            #forward pass
            weights3 = tf.Variable(
                tf.truncated_normal([FLAGS.hidden2, NUM_CLASSES], stddev=1.0 / math.sqrt(float(FLAGS.hidden2))), name='weights')
            biases3 = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
            logits3 = tf.matmul(hidden2, weights3) + biases3

        #backward pass
        labels = tf.to_int64(labels_placeholder)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits3, name='xentropy')
        loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
        summary_loss = tf.summary.scalar('net_loss', loss)

        hidden2_true_gradients_list = tf.gradients(loss, layer2_input)
        hidden2_true_gradients = hidden2_true_gradients_list[0]

        train_op = optimizer.minimize(loss)


        """synthetic gradients model"""
        synthetic_gradients_labels_placeholder = tf.placeholder(tf.int32, shape=(FLAGS.batch_size), name="synt_grad_labels")
        synthetic_gradients_input_message = tf.placeholder(tf.float32, shape=hidden1.get_shape().as_list(), 
                                                           name="synt_grad_input_message")
        synthetic_gradients_input_weights = tf.placeholder(tf.float32, shape=weights2.get_shape().as_list(), 
                                                           name="synt_grad_input_weights")
        synthetic_gradients_input_biases = tf.placeholder(tf.float32, shape=biases2.get_shape().as_list(), 
                                                           name="synt_grad_input_biases")


        with tf.name_scope('synthetic_gradients'):
            #forward pass
            labels_one_hot = tf.one_hot(synthetic_gradients_labels_placeholder, 10)
            input_and_labels = tf.concat([synthetic_gradients_input_message, labels_one_hot], 1)
            """
            hidden_2_flattened_model = tf.concat([
                tf.reshape(weights2,[-1]), # Hidden 2 weights (layer state)
                tf.reshape(biases2,[-1]), # Hidden 2 biases (layer state)
            ],0)
            """
            hidden_2_flattened_model = tf.concat([
                tf.reshape(synthetic_gradients_input_weights,[-1]), # Hidden 2 weights (layer state)
                tf.reshape(synthetic_gradients_input_biases,[-1]), # Hidden 2 biases (layer state)
            ],0)
            hidden_2_replicated_model = tf.concat([
                tf.expand_dims(hidden_2_flattened_model,0) for _ in range(input_and_labels.get_shape().as_list()[0])
            ],0)
            synthetic_gradients_model_input = tf.concat([
                input_and_labels,
                hidden_2_replicated_model
            ],1)

            """
            hidden1_shape = hidden1.get_shape().as_list()
            hidden1_output_length = 1
            for n in hidden1_shape:
                hidden1_output_length *= n
            """

            true_gradients_shape = hidden2_true_gradients.get_shape().as_list()
            true_gradients_length = 1
            for n in true_gradients_shape:
                true_gradients_length *= n

            synthetic_gradients_model_input_width = synthetic_gradients_model_input.get_shape().as_list()[1]
            synthetic_gradients_model_weights = tf.Variable(
                tf.zeros([synthetic_gradients_model_input_width, true_gradients_length]),name='weights')
                #tf.truncated_normal([synthetic_gradients_model_input_width, true_gradients_length]),name='weights')
            synthetic_gradients_model_biases = tf.Variable(tf.zeros([true_gradients_length]), name='biases')
            synthetic_gradients_model_matmul = tf.matmul(synthetic_gradients_model_input, synthetic_gradients_model_weights)
            synthetic_gradients_model_logits = synthetic_gradients_model_matmul + synthetic_gradients_model_biases
            synthetic_gradients_model_output = tf.reshape(tf.reduce_mean(synthetic_gradients_model_logits,0), true_gradients_shape)
                                                #hidden1.get_shape().as_list()) #hidden1_synthetic_gradient

            #backward pass
            true_gradients_placeholder = tf.placeholder(tf.float32, shape=true_gradients_shape, name="hidden2_true_gradients")
            #synthetic_gradients_model_loss = tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(
            #                                                                        true_gradients_placeholder,
            #                                                                        synthetic_gradients_model_output))))

            # Mean squared error 
            synthetic_gradients_model_loss = tf.reduce_sum(
                tf.pow(true_gradients_placeholder - synthetic_gradients_model_output, 2))/(2*FLAGS.batch_size)
            summary_synthetic_gradients_model_loss = tf.summary.scalar('synthetic_gradients_model_loss', synthetic_gradients_model_loss)

            synthetic_gradients_model_optimize_op = optimizer.minimize(synthetic_gradients_model_loss)

        """Evaluate the quality of the logits at predicting the label."""
        correct = tf.nn.in_top_k(logits3, labels_placeholder, 1)
        eval_correct = tf.reduce_sum(tf.cast(correct, tf.int32))
        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
        summary_accuracy = tf.summary.scalar('net_accuracy', accuracy)
        #merged = tf.summary.merge_all()


        # Train(s)
        data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
            train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train', sess.graph)
            eval_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/validation')
            test_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/test')

            sess.run(tf.global_variables_initializer())

            def do_eval(global_step, data_set, writer):
                """Runs one evaluation against the full epoch of data."""
                # And run one epoch of eval.
                true_count = 0  # Counts the number of correct predictions.
                steps_per_epoch = data_set.num_examples // FLAGS.batch_size
                num_examples = steps_per_epoch * FLAGS.batch_size
                for step in xrange(steps_per_epoch):
                    #feed_dict = fill_feed_dict(data_set,images_placeholder,labels_placeholder)
                    images_feed, labels_feed = data_set.next_batch(FLAGS.batch_size, FLAGS.fake_data)

                    layer1_output = sess.run(hidden1, feed_dict={
                        images_placeholder: images_feed
                    })
                    true_count += sess.run(eval_correct, feed_dict={
                        layer2_input: layer1_output, 
                        labels_placeholder: labels_feed
                    })
                    if step == 0:
                        res = sess.run([summary_accuracy, summary_loss], feed_dict={
                            layer2_input: layer1_output, 
                            labels_placeholder: labels_feed
                        })
                        for r in res:
                            writer.add_summary(r, global_step)

                    precision = float(true_count) / num_examples
                print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' % (num_examples, true_count, precision))

            def train_step(print_out=False, learning_rate=FLAGS.learning_rate, step=0):
                images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)

                # Hidden1 forward:
                layer1_output = sess.run(hidden1, feed_dict={
                    images_placeholder: images_feed
                })

                # Hidden 2&3 forward with gradient (w.r.t. layer2_input) computation, and hidden2 state retrival
                layer2_real_gradient, hidden2_weights_state, hidden2_biases_state = sess.run([
                        hidden2_true_gradients, weights2, biases2
                    ], feed_dict={
                        layer2_input: layer1_output, 
                        labels_placeholder: labels_feed
                    })

                # synthetic_gradients_model forward
                layer1_synt_grad = sess.run(synthetic_gradients_model_output, feed_dict={
                    synthetic_gradients_labels_placeholder: labels_feed,
                    synthetic_gradients_input_message: layer1_output,
                    synthetic_gradients_input_weights: hidden2_weights_state,
                    synthetic_gradients_input_biases: hidden2_biases_state
                })

                # Hidden1 backward
                _ = sess.run(hidden1_optimize, feed_dict={
                    learning_rate_placeholder: learning_rate,
                    images_placeholder: images_feed,
                    hidden1_synt_grad_placeholder: layer1_synt_grad
                })

                # Hidden 2&3 backward and loss evaluation
                _, net_loss = sess.run([train_op, loss], feed_dict={
                    learning_rate_placeholder: learning_rate,
                    layer2_input: layer1_output, 
                    labels_placeholder: labels_feed
                })


                # synthetic_gradients_model backward and loss evaluation
                _,  synt_grad_model_loss, ssgml = sess.run([
                    synthetic_gradients_model_optimize_op, 
                    synthetic_gradients_model_loss,
                    summary_synthetic_gradients_model_loss
                    ], 
                    feed_dict={
                        learning_rate_placeholder: learning_rate,
                        synthetic_gradients_labels_placeholder: labels_feed,
                        synthetic_gradients_input_message: layer1_output,
                        synthetic_gradients_input_weights: hidden2_weights_state,
                        synthetic_gradients_input_biases: hidden2_biases_state,
                        true_gradients_placeholder: layer2_real_gradient
                    }
                )

                if print_out:
                    print("net loss:", net_loss)
                    print("synthetic_gradients_model_loss:", synt_grad_model_loss)
                    train_writer.add_summary(ssgml, step)
                    do_eval(step, data_sets.train, train_writer)
                    do_eval(step, data_sets.validation, eval_writer)
                    do_eval(step, data_sets.test, test_writer)



            #train_step()
            def train_loop():
                for step in xrange(FLAGS.max_steps):

                    learning_rate = FLAGS.learning_rate
                    if step > 300*1000:
                        learning_rate/=10
                        if step > 400*1000:
                            learning_rate/=10

                    train_step(print_out=(step % 100 == 0), learning_rate=learning_rate, step=step)
            train_loop()


            def equals(x,y):
                return np.equal(np.asarray(x), np.asarray(y)).all()

            def train_step_with_asserts(print_out=False):
                images_feed, labels_feed = data_sets.train.next_batch(FLAGS.batch_size, FLAGS.fake_data)

                # Hidden1 forward:
                layer1_output = sess.run(hidden1, feed_dict={
                    images_placeholder: images_feed
                })

                # Hidden 2&3 forward with gradient (w.r.t. layer2_input) computation, and hidden2 state retrival
                layer2_real_gradient, hidden2_weights_state, hidden2_biases_state = sess.run([
                        hidden2_true_gradients, weights2, biases2
                    ], feed_dict={
                        layer2_input: layer1_output, 
                        labels_placeholder: labels_feed
                    })

                # synthetic_gradients_model forward
                layer1_synt_grad = sess.run(synthetic_gradients_model_output, feed_dict={
                    synthetic_gradients_labels_placeholder: labels_feed,
                    synthetic_gradients_input_message: layer1_output,
                    synthetic_gradients_input_weights: hidden2_weights_state,
                    synthetic_gradients_input_biases: hidden2_biases_state
                })

                # Hidden1 backward
                _ = sess.run(hidden1_optimize, feed_dict={
                    images_placeholder: images_feed,
                    hidden1_synt_grad_placeholder: layer1_synt_grad
                })

                # Hidden 2&3 backward and loss evaluation
                _, net_loss, layer2_real_gradient2, hidden2_weights_state2, hidden2_biases_state2 = sess.run(
                    [train_op, loss, hidden2_true_gradients, weights2, biases2], feed_dict={
                    layer2_input: layer1_output, 
                    labels_placeholder: labels_feed
                })


                # synthetic_gradients_model backward and loss evaluation
                _,  synt_grad_model_loss = sess.run([synthetic_gradients_model_optimize_op, synthetic_gradients_model_loss], 
                    feed_dict={
                        synthetic_gradients_labels_placeholder: labels_feed,
                        synthetic_gradients_input_message: layer1_output,
                        synthetic_gradients_input_weights: hidden2_weights_state,
                        synthetic_gradients_input_biases: hidden2_biases_state,
                        true_gradients_placeholder: layer2_real_gradient
                    }
                )

                if print_out:
                    print("net loss:", net_loss)
                    print("EQUALS ASSERTS")
                    print("layer2_real_gradient:", equals(layer2_real_gradient, layer2_real_gradient2))
                    print("hidden2_weights_state:", equals(hidden2_weights_state, hidden2_weights_state2))
                    print("hidden2_biases_state:", equals(hidden2_biases_state, hidden2_biases_state2))
                    print("synthetic_gradients_model_loss:", synt_grad_model_loss)

            #train_step_with_asserts()

end_time = time.time() - start_time
print("--- training took %s seconds ---" % end_time)
print("--- training took %s minutes ---" % (end_time/60))
print("--- training took %s hours ---" % (end_time/(60*60)))