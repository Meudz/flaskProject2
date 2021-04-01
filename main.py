import os

from app import app

os.chdir('/Users/adaml/sport/')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np
import pandas as pd
import PIL
from keras.preprocessing.image import img_to_array, load_img


path = os.chdir('/Users/adaml/sport/')

def getPrediction(filename):
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
    train_gen = train_datagen.flow_from_directory(
        'train',  # path du train
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical'
    )

    val_gen = val_datagen.flow_from_directory(
        'valid',
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical'
    )
    train_batches = tf.keras.preprocessing.image.ImageDataGenerator(
        preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory='/Users/adaml/sport', target_size=(224, 224), class_mode='categorical',
                             batch_size=10)
    inputs = tf.keras.layers.Input(shape=(150, 150, 3))

    x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(inputs)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(32, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Conv2D(64, 3, activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D(2)(x)

    x = tf.keras.layers.Flatten()(x)

    x = tf.keras.layers.Dense(512, activation='relu')(x)

    outputs = tf.keras.layers.Dense(70, activation='softmax')(x)

    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy', metrics=['acc'],
                  optimizer='adam')
    history_model = model.fit(train_gen, epochs=5, steps_per_epoch=10,
                              validation_data=val_gen, validation_steps=len(val_gen))
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    class_name = os.listdir('/Users/adaml/sport/train')
    filename = load_img(img_path, target_size=(224, 224))
    img = filename.resize((150, 150), resample=PIL.Image.BICUBIC)
    arr = img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr /= 255.0

    score = model.predict(arr)
    pd.DataFrame(score, columns=class_name)
    value_max = np.amax(score)
    index_max = np.argmax(score)



    return class_name[index_max]
