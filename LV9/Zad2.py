import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.preprocessing import image_dataset_from_directory
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import datetime

batch_size = 32
img_size = (48, 48)

train_ds = image_dataset_from_directory(
    directory='C:/Users/brnar/Downloads/dataset/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    subset="training",
    seed=123,
    validation_split=0.2,
    image_size=img_size
)

validation_ds = image_dataset_from_directory(
    directory='C:/Users/brnar/Downloads/dataset/train',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    subset="validation",
    seed=123,
    validation_split=0.2,
    image_size=img_size
)

test_ds = image_dataset_from_directory(
    directory='C:/Users/brnar/Downloads/dataset/test',
    labels='inferred',
    label_mode='categorical',
    batch_size=batch_size,
    image_size=img_size
)

model = models.Sequential([
    layers.InputLayer(input_shape=(48, 48, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint = ModelCheckpoint('best_model.h5', save_best_only=True, monitor='val_loss', mode='min', verbose=1)
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(
    train_ds,
    validation_data=validation_ds,
    epochs=20,
    callbacks=[checkpoint, tensorboard]
)

test_loss, test_acc = model.evaluate(test_ds)
print(f'Test Accuracy: {test_acc:.4f}')

y_true = np.concatenate([y for x, y in test_ds], axis=0)
y_true_labels = np.argmax(y_true, axis=1)
y_pred_probs = model.predict(test_ds)
y_pred_labels = np.argmax(y_pred_probs, axis=1)

cm = confusion_matrix(y_true_labels, y_pred_labels)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Matrica zabune")
plt.show()
