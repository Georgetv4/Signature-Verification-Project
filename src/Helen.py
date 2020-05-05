"""
Испортируется всё необходимое:
-Библиотека TensorFlow
-Постоянные значения для алгоритма
"""

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os
import numpy as np
import matplotlib.pyplot as plt

from src.config import batch_size, epochs, IMG_HEIGHT, IMG_WIDTH, PATH

"""
Указывается путь к различным директориям с данными:
"""

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_my_dir = os.path.join(train_dir, 'My Signatures')
train_other_dir = os.path.join(train_dir, 'Other Signatures')
validation_my_dir = os.path.join(validation_dir, 'My Ver Signatures')
validation_other_dir = os.path.join(validation_dir, 'Other Ver Signatures')

num_my_tr = len(os.listdir(train_my_dir))
num_other_tr = len(os.listdir(train_other_dir))

num_my_val = len(os.listdir(validation_my_dir))
num_other_val = len(os.listdir(validation_other_dir))

total_train = num_my_tr + num_other_tr
total_val = num_my_val + num_other_val

"""
Выводится кол-во изображений для тренировки и проверки:
"""

print('total training michael signatures:', num_my_tr)
print('total training other signatures:', num_other_tr)

print('total validation michael signatures:', num_my_val)
print('total validation other signatures:', num_other_val)
print("--")
print("Total training signatures:", total_train)
print("Total validation signatures:", total_val)

"""
Подгтовка данных к обработке:
-Счтение с диска
-Перевод в формат, удобный для обработки
"""

train_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our training data
validation_image_generator = ImageDataGenerator(rescale=1./255)  # Generator for our validation data

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='binary')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='binary')

sample_training_images, _ = next(train_data_gen)

"""
Вывод модели нейронной сети на экран:
"""

model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()

"""
Тренироква нейронной сети:
"""

history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
)

"""
Граф в конце программы:
"""

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()
