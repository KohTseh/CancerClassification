# imports
import glob, os, json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

# imaging library
# run `pip install Pillow` if ModuleNotFoundError
from PIL import Image 
from sklearn.model_selection import train_test_split

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.INFO)

def build_dataset():
  MAX_NUMBER_OF_DATA = 10000
  number_of_data = 0

  img_file_names = []
  labels = []
  data = []

  for image_file_path in glob.glob("ISIC-data/resized_images_width_64/*.jpg", recursive=True):
    (image_name, ext) =  os.path.splitext(os.path.basename(image_file_path))
    meta_data_path = os.path.join('ISIC-data', 'Descriptions', image_name)

    if (os.path.isfile(meta_data_path)):
      with open(meta_data_path) as meta: 
        metadata = json.load(meta)
        label = metadata["meta"]["clinical"]["benign_malignant"]
        img = Image.open(image_file_path)
        img = img.resize((56,56))

        img_file_names.append(image_file_path)
        labels.append(1 if label == "malignant" else 0)

        img = np.asarray(img)
        img = np.reshape(img, (56 * 56, 3))
        
        data.append(img)
        
      number_of_data += 1

    if (number_of_data == MAX_NUMBER_OF_DATA):
      break
  
  # returns 2-tuple of data and labels.
  return (np.asarray(data, dtype=np.float32), np.asarray(labels, dtype=np.int32)) 

def cnn_model_fn(features, labels, mode):
  """Model function for CNN."""
  # Input Layer with shape [batch_size, image_width, image_height, channels]
  # -1 for batch size ==> dynamically computed based on input values
  # 28,28 for img width and height
  # 1 channel (monochrome)
  input_layer = tf.reshape(features["x"], [-1, 56, 56, 3])
  print(input_layer)

  # Convolutional Layer #1
  # Applies 32 5x5 filters (extracting 5x5-pixel subregions), with ReLU activation function
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)

  # Pooling Layer #1
  # Performs max pooling with a 2x2 filter and stride of 2
  # pool regions do not overlap
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

  # Convolutional Layer #2 and Pooling Layer #2
  # Applies 64 5x5 filters, with ReLU activation function
  conv2 = tf.layers.conv2d(
      inputs=pool1,
      filters=64,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  # conv2 shape: [batchsize, 14, 14, 64]
  # max pooling with a 2x2 filter and stride of 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

  # Dense Layer (same as fully connected)
  # pool2 width and pool2 height = 7
  # pool2 channels = 64
  pool2_flat = tf.reshape(pool2, [-1, 14 * 14 * 64])
  # 1024 units, ReLU activation
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  # dropout layer has shape [batch_size, 1024]
  dropout = tf.layers.dropout(
    inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN
    )

  # Logits Layer
  # 10 units, one for each digit target class (0–9).
  logits = tf.layers.dense(inputs=dropout, units=2)

  predictions = {
      # Generate predictions (for PREDICT and EVAL mode)
      "classes": tf.argmax(input=logits, axis=1),
      # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
      # `logging_hook`.
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }

  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

  # Calculate Loss (for both TRAIN and EVAL modes)
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

  # Configure the Training Op (for TRAIN mode)
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(
        loss=loss,
        global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

  # Add evaluation metrics (for EVAL mode)
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(
          labels=labels, predictions=predictions["classes"])}
  return tf.estimator.EstimatorSpec(
      mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# Create the Estimator
cs3244_classifier = tf.estimator.Estimator(
    model_fn=cnn_model_fn, model_dir="models/")

# Set up logging for predictions
# uncomment to log probabilities
# tensors_to_log = {"probabilities": "softmax_tensor"}
tensors_to_log = {}
logging_hook = tf.train.LoggingTensorHook(
    tensors=tensors_to_log, every_n_iter=50)

def main():
  NUM_TRAINING_ITERATIONS = 20
  
  # Load training and eval data
  data, labels = build_dataset()
  
  # We oversample positive training sets
  while (sum(labels) / len(labels) < 0.4):
      data = np.concatenate((data, data[labels == 1]))
      labels = np.concatenate((labels, labels[labels == 1]))
  
  training_data, test_data, training_labels, test_labels = train_test_split(
      data, labels, test_size=0.33, shuffle=True)

  # actual training of model
  train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"x": training_data}, # shape (N, w*d)
    y=training_labels, # shape
    batch_size=10,
    num_epochs=2, # num of epochs to iterate over data. If `None` will run forever.
    shuffle=True)
  
  for i in range(0, NUM_TRAINING_ITERATIONS):
      print(round(i / NUM_TRAINING_ITERATIONS * 100, 2), "% done")
      cs3244_classifier.train(
              input_fn=train_input_fn,
              steps=None, # train until input_fn stops
              )
  
  # Evaluate the model and print results
  eval_input_fn = tf.estimator.inputs.numpy_input_fn(
      x={"x": test_data},
      y=test_labels,
      num_epochs=1,
      shuffle=True)
  eval_results = cs3244_classifier.evaluate(input_fn=eval_input_fn)
  print(eval_results)

def debug_print():
  temp_x, temp_y = build_dataset()
  input_layer = tf.reshape(temp_x[0], [-1,56,56,3])
  filter_size = 5
  input_channels = 3
  output_filters = 32
  xx = tf.placeholder(tf.float32, shape=[None, 512, 512, 3])
  yy = tf.nn.conv2d(xx, filter=tf.Variable(tf.truncated_normal([filter_size, filter_size, input_channels, output_filters], stddev=0.5)), strides=[1,1,1,1] , padding='SAME')
  conv1 = tf.layers.conv2d(
      inputs=input_layer,
      filters=32,
      kernel_size=[5, 5],
      padding="same",
      activation=tf.nn.relu)
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

main()