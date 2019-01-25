from __future__ import print_function
import cx_Oracle
import math
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

connection=cx_Oracle.connect('ml/ml@127.0.0.1:1521/XE')
sql='SELECT PCLASS,SEX,SIBSP,PARCH,FARE,A10,A20,A30,A40,A50,A60,A70,A80,ANA,R1,R2,R3,R4,RNA,T1,T2,T3,T4,T5,CA,CB,CC,CD,CE,CF,CG,CN,EMBARKED1 E1,EMBARKED2 E2,EMBARKED3 E3,SURVIVED FROM TITANIC_2'
#sql='SELECT PCLASS,SEX,SURVIVED FROM TITANIC_2'
df1=pd.read_sql(sql,con=connection)
df1 = df1.reindex(np.random.permutation(df1.index))
print(df1.columns.values)

training_examples = df1.drop(['SURVIVED'],axis=1)
training_examples = training_examples.head(600)
training_targets = df1[["SURVIVED"]].head(600)
validation_examples = df1.drop(['SURVIVED'],axis=1)
validation_examples = validation_examples.tail(291)
validation_targets = df1[["SURVIVED"]].tail(291)

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
    print("  period %02d : %0.4f" % (period, training_log_loss))
    # Add the loss metrics from this period to our list.
    training_log_losses.append(training_log_loss)
    validation_log_losses.append(validation_log_loss)
  
  print("learning_rate = %0.5f" % learning_rate)
  print("steps = %0.0f" % steps)
  print("batch_size = %0.0f" % batch_size)
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

linear_classifier = train_linear_classifier_model(
    learning_rate=0.00065,
    steps=20000,
    batch_size=18,
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)

evaluation_metrics = linear_classifier.evaluate(input_fn=predict_validation_input_fn)

print("AUC on the validation set: %0.3f" % evaluation_metrics['auc'])
print("Accuracy on the validation set: %0.3f" % evaluation_metrics['accuracy'])
#
sql2='SELECT PASSENGERID,PCLASS,SEX,SIBSP,PARCH,FARE,A10,A20,A30,A40,A50,A60,A70,A80,ANA,R1,R2,R3,R4,RNA,T1,T2,T3,T4,T5,CA,CB,CC,CD,CE,CF,CG,CN,EMBARKED1 E1,EMBARKED2 E2,EMBARKED3 E3,SURVIVED FROM TITANIC_T_2 ORDER BY PASSENGERID'
df2=pd.read_sql(sql2,con=connection)
print(df2.columns.values)

def eval_input_fn():
    dataset = tf.data.Dataset.from_tensor_slices(dict(df2.drop(['PASSENGERID','SURVIVED'],axis=1)))
    dataset = dataset.batch(1)
    return dataset
  
predictions = linear_classifier.predict(input_fn=lambda:eval_input_fn())
results = list(predictions)
print(len(results))

passengers = {}
i = 892
for x in results:
	passengers[i] = int(x['class_ids'][0])
	i+=1

import csv
csvfile = 'submissions.csv'
with open(csvfile, 'w') as f:
    outcsv = csv.writer(f, delimiter=',')
    header = ['PassengerId','Survived']
    outcsv.writerow(header)
    for k,v in passengers.items():
        outcsv.writerow([k,v])
