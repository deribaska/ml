from __future__ import print_function
import cx_Oracle
import pandas as pd
import numpy as np
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

connection=cx_Oracle.connect('ml/ml@127.0.0.1:1521/orcl')
sql='SELECT * FROM TITANIC_2'
df1=pd.read_sql(sql,con=connection)
df1.set_index('PASSENGERID')
df1 = df1.reindex(np.random.permutation(df1.index))
print(df1.columns.values)

def preprocess_features(df1):
  selected_features = df1[["PCLASS","SEX","AGE"]]
  processed_features = selected_features.copy()
  return processed_features
def preprocess_targets(df1):
  output_targets = pd.DataFrame()
  output_targets["SURVIVED"] = (
    df1["SURVIVED"] > 0).astype(float)
  return output_targets

training_examples = preprocess_features(df1.head(600))
training_targets = preprocess_features(df1.head(600))
validation_examples = preprocess_features(df1.tail(291))
validation_targets = preprocess_features(df1.tail(291))

print("Training examples summary:")
print(training_examples.describe())
print("Validation examples summary:")
print(validation_examples.describe())
print("Training targets summary:")
print(training_targets.describe())
print("Validation targets summary:")
print(validation_targets.describe())

def construct_feature_columns(input_features):
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
#
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                            
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels
#
predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                  validation_targets["SURVIVED"], 
                                                  num_epochs=1, 
                                                  shuffle=False)


def train_linear_classifier_model(
    learning_rate,
    steps,
    batch_size,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear classification model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    training_examples: A `DataFrame` containing one or more columns from
      `df1` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `df1` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `df1` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `df1` to use as target for validation.
      
  Returns:
    A `LinearClassifier` object trained on the training data.
  """

  periods = 10
  steps_per_period = steps / periods
  
  # Create a linear classifier object.
  my_optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)  
  linear_classifier = tf.estimator.LinearClassifier(
      feature_columns=construct_feature_columns(training_examples),
      optimizer=my_optimizer
  )
  
  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["SURVIVED"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["SURVIVED"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["SURVIVED"], 
                                                    num_epochs=1, 
                                                    shuffle=False)
  
  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("LogLoss (on training data):")
  training_log_losses = []
  validation_log_losses = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_classifier.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.    
    training_probabilities = linear_classifier.predict(input_fn=predict_training_input_fn)
    training_probabilities = np.array([item['probabilities'] for item in training_probabilities])
    
    validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
    validation_probabilities = np.array([item['probabilities'] for item in validation_probabilities])
    
    training_log_loss = metrics.log_loss(training_targets, training_probabilities)
    validation_log_loss = metrics.log_loss(validation_targets, validation_probabilities)
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  print("Model training finished.")
  
  # Output a graph of loss metrics over periods.
  plt.ylabel("LogLoss")
  plt.xlabel("Periods")
  plt.title("LogLoss vs. Periods")
  plt.tight_layout()
  plt.plot(training_log_losses, label="training")
  plt.plot(validation_log_losses, label="validation")
  plt.legend()
  plt.show()

  return linear_classifier
#
#linear_classifier = train_linear_classifier_model(
#    learning_rate=0.000005,
#    steps=500,
#    batch_size=20,
#    training_examples=training_examples,
#    training_targets=training_targets,
#    validation_examples=validation_examples,
#    validation_targets=validation_targets)
#
#evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

#print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
#print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])
#
#validation_probabilities = linear_classifier.predict(input_fn=predict_validation_input_fn)
# Get just the probabilities for the positive class.
#validation_probabilities = np.array([item['probabilities'][1] for item in validation_probabilities])

#false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(
#    validation_targets, validation_probabilities)
#plt.plot(false_positive_rate, true_positive_rate, label="our model")
#plt.plot([0, 1], [0, 1], label="random classifier")
#_ = plt.legend(loc=2)
#
linear_classifier = train_linear_classifier_model(
    learning_rate=0.000003,
    steps=20000,
    batch_size=500,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.2f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.2f" % evaluation_metrics['accuracy'])


