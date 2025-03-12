# %%
import tensorflow as tf 
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model 
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
from PIL import Image
     

# %%
# Setting Dataset Directory 
train_dir = r"D:\chest_xray\train"
val_dir = r"D:\chest_xray\val"
test_dir = r"D:\chest_xray\test"

# %%
# Define image size and batch size
IMG_SIZE = (224, 224)  # Smaller size for faster training
BATCH_SIZE = 32

# %%
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    seed=123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)


# %%
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    seed=123,
    image_size = IMG_SIZE,
    batch_size = BATCH_SIZE
)

# %%
class_names = train_ds.class_names
print(class_names)

# %%
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# %%
num_classes = len(class_names)

num_classes

# %%
# Load Pretrained VGG19 Model
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))

# %%
# Freeze all layer in VGG19
for layer in base_model.layers:
    layer.trainable = False


# %%
# Add Custom Layers on top
x = Flatten()(base_model.output)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(1, activation='sigmoid')(x)

# %%
model = Model(inputs=base_model.input, outputs=x)

# %%
model.summary()

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# %%
# ✅ Use ModelCheckpoint to Save Best Model
checkpoint = ModelCheckpoint(
    'best_model.h5',                # Filepath to save the best model
    monitor='val_accuracy',          # Monitor validation accuracy
    save_best_only=True,            # Save only the best model
    mode='max',                     # Maximize validation accuracy
    verbose=1                       # Print when a model is saved
)


# %%
callbacks = [checkpoint]

# %%
epochs = 25
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,callbacks=callbacks
)

# %%
# Loading the best model

# %%



