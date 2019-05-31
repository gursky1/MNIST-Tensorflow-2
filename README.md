
# Classifying MNIST with Keras in Tensorflow 2.0

For this project we are looking at classifying the classic MNIST dataset using Keras in Tensorflow 2.0.  We will look at using a convolutional network architecture, a tried and true method for image recognition.


```python
# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf
```


```python
# Checking the version of tensorflow
print(tf.__version__)
# Are we running with Eager execution?
print(tf.executing_eagerly())
```

    1.13.1
    False
    

## Importing and Preparing Data
We pull the data for this project from the corresponding Kaggle competition, which represents each pixel as a column, with a target column.  We are going to read the data in using pandas, separate into predictor and target arrays, then convert to numpy.


```python
# Importing our data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Splitting train and test sets
X_train = train.drop('label',axis=1).values.astype('float32')
y_train = train['label'].values
X_test = test.values.astype('float32')
```


```python
# Checking what pixel values look like
print(np.max(X_train))
```

    255.0
    

Next we need to normalize our pixels by dividing the value by 255, in order to have a normalized range for our network:


```python
# Now we need to divide them all by 255
X_train = X_train/255.0
X_test = X_test/255.0
```

Lastly we need to reshape our pixel data from columns of pixels to a three-dimensional set representing the images, as that is how the ConvNet accepts input in two dimensions:


```python
# Next we need to reshape our data for the convolutional network
X_train = X_train.reshape(X_train.shape[0], 28, 28,1)
X_test = X_test.reshape(X_test.shape[0], 28, 28,1)
```


```python
# Checking our unique values
print(y_train)
```

    [1 0 1 ... 7 6 9]
    

## Training our CNN
We are going to use Keras to create a simple CNN architecture using three convolutional layers, max pooling, batch normalization for regularization, flattening, and final dense layers leading to the output:


```python
# Creating our MNIST network
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv2D(32, kernel_size=(3, 3),activation='relu',kernel_initializer='he_normal',input_shape=(28,28,1)))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(128, (3, 3), activation='relu',kernel_initializer='he_normal'))
model.add(tf.keras.layers.MaxPool2D((2, 2)))
model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_initializer='he_normal'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.Dense(10, activation='softmax'))

# Compiling model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Training model
model.fit(X_train, y_train, epochs=5)
```

    WARNING:tensorflow:From C:\Users\Jacob\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\ops\resource_variable_ops.py:435: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.
    WARNING:tensorflow:From C:\Users\Jacob\AppData\Local\Continuum\anaconda3\lib\site-packages\tensorflow\python\ops\math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.
    Epoch 1/5
    42000/42000 [==============================] - 10s 231us/sample - loss: 0.1246 - acc: 0.9616
    Epoch 2/5
    42000/42000 [==============================] - 7s 176us/sample - loss: 0.0536 - acc: 0.9832
    Epoch 3/5
    42000/42000 [==============================] - 7s 177us/sample - loss: 0.0396 - acc: 0.9879
    Epoch 4/5
    42000/42000 [==============================] - 8s 179us/sample - loss: 0.0318 - acc: 0.9901
    Epoch 5/5
    42000/42000 [==============================] - 8s 180us/sample - loss: 0.0251 - acc: 0.9919
    




    <tensorflow.python.keras.callbacks.History at 0x1cb2aca4dd8>



Wow, this model seems to train quite quickly! Although this is a relatively easy task, so we can expect rapid convergence.

## Prediction for Competition
Finally, we need to run the testing set through our trained net and prepare them in the necessary format for the 
Kaggle competition for submission:


```python
# Creating our predictions for submission
preds = pd.DataFrame({'ImageId': list(range(1,test.shape[0]+1)),'Label': model.predict_classes(X_test)})
preds.to_csv('submission.csv', index=False, header=True)
```
