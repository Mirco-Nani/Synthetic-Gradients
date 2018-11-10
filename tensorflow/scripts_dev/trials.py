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
        self.learning_rate = 0.01/100.0
        self.max_steps = 2000*100
        self.hidden1 = 128
        self.hidden2 = 32
        self.batch_size = 100
        #self.input_data_dir = '/tmp/tensorflow/mnist/input_data'
        self.input_data_dir = '/notebooks/datasets/mnist'
        self.log_dir = '/tmp/tensorflow/mnist/logs/fully_connected_feed'
        self.fake_data = False
        

FLAGS = Flags()


tf.reset_default_graph()

if tf.gfile.Exists(FLAGS.log_dir):
    tf.gfile.DeleteRecursively(FLAGS.log_dir)
    tf.gfile.MakeDirs(FLAGS.log_dir)
    
#Graph(s)
with tf.Graph().as_default():
    # global optimizer
    optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
    
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
            tf.truncated_normal([synthetic_gradients_model_input_width, true_gradients_length]),name='weights')
        synthetic_gradients_model_biases = tf.Variable(tf.zeros([true_gradients_length]), name='biases')
        synthetic_gradients_model_matmul = tf.matmul(synthetic_gradients_model_input, synthetic_gradients_model_weights)
        synthetic_gradients_model_logits = synthetic_gradients_model_matmul + synthetic_gradients_model_biases
        synthetic_gradients_model_output = tf.reshape(tf.reduce_mean(synthetic_gradients_model_logits,0), true_gradients_shape)
                                            #hidden1.get_shape().as_list()) #hidden1_synthetic_gradient
        
        #backward pass
        true_gradients_placeholder = tf.placeholder(tf.float32, shape=true_gradients_shape, name="hidden2_true_gradients")
        synthetic_gradients_model_loss = tf.sqrt( tf.reduce_sum(tf.square(tf.subtract(
                                                                                true_gradients_placeholder,
                                                                                synthetic_gradients_model_output))))
        synthetic_gradients_model_optimize_op = optimizer.minimize(synthetic_gradients_model_loss)
        
    # Train(s)
    data_sets = input_data.read_data_sets(FLAGS.input_data_dir, FLAGS.fake_data)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        def train_step(print_out=False):
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
            _, net_loss = sess.run([train_op, loss], feed_dict={
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
                print("synthetic_gradients_model_loss:", synt_grad_model_loss)
        #train_step()
        def train_loop():
            for step in xrange(FLAGS.max_steps):
                train_step(step % 100 == 0)
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
    