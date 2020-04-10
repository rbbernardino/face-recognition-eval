# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# SELECTS TENSORFLOW VERSION (soon default will be tf2)
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# %%
###################################
# LOAD LIBRARIES
import itertools
import os

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub

tf.get_logger().propagate = False  # avoid duplicated warnings

###################################
# LOAD CUSTOM LIBRARIES
# assume already linked "mylibs" directory o /content of colab
import mylibs.aiutils as aiutils

###################################
# PRINT DEVICE INFO
aiutils.print_device_info()

# %%
###################################
# OPTIONS (MODEL TO USE, etc.)
module_selection = ("mobilenet_v2_100_224", 224)
# module_selection = ("mobilenet_v2_100_224", 224)
# module_selection = ("inception_v3", 299)

handle_base, pixels = module_selection
MODULE_HANDLE = f"https://tfhub.dev/google/imagenet/{handle_base}/feature_vector/4"
IMAGE_SIZE = (pixels, pixels)
BATCH_SIZE = 32
MODEL_OUTPUT_DIR = '/content/gdrive/My Drive/Colab Notebooks/models_output/ludus/model1/'

print(f"Using {MODULE_HANDLE} with input size {IMAGE_SIZE}")

###################################
# DATASET PATH
data_dir = '/content/gdrive/My Drive/Colab Notebooks/data/ludus'
train_dir = data_dir + "/train_set1/train"
validation_dir = data_dir + "/train_set1/valid"

# %%
###################################
# Setup Dataset
datagen_kwargs = dict(rescale=1. / 255)
dataflow_kwargs = dict(target_size=IMAGE_SIZE, batch_size=BATCH_SIZE,
                       interpolation="bilinear")

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
train_datagen = valid_datagen

valid_generator = valid_datagen.flow_from_directory(
    validation_dir, shuffle=False, **dataflow_kwargs)

train_generator = train_datagen.flow_from_directory(
    train_dir, subset="training", shuffle=True, **dataflow_kwargs)

# Save the classes in order according to the labels indices
class_dict = train_generator.class_indices
class_list = sorted(class_dict.keys(), key=lambda x: class_dict[x])
class_list_str = '\n'.join(list(class_list))
with open(MODEL_OUTPUT_DIR + "retrained_labels.txt", 'w') as f:
    f.write(class_list_str)

# %%
###################################
# ASSEMBLE THE MODEL
print("Building model with", MODULE_HANDLE)
model = tf.keras.Sequential()
model.add(hub.KerasLayer(MODULE_HANDLE, trainable=False,
                         input_shape=(224, 224, 3)))
model.add(tf.keras.layers.Dropout(rate=0.2))
model.add(tf.keras.layers.Dense(
    train_generator.num_classes,
    activation='softmax',
    kernel_regularizer=tf.keras.regularizers.l2(0.0001))
)
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

# %%
###################################
# TRAIN THE MODEL
model.compile(
    optimizer=tf.keras.optimizers.SGD(lr=0.005, momentum=0.9),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy'])

steps_per_epoch = train_generator.samples // train_generator.batch_size
validation_steps = valid_generator.samples // valid_generator.batch_size
hist = model.fit(
    train_generator,
    epochs=5, steps_per_epoch=steps_per_epoch,
    validation_data=valid_generator,
    validation_steps=validation_steps).history

# %%
###################################
# SAVE THE MODEL
# Don't need to save weights, as .save() already include them!
tf.saved_model.save(model, MODEL_OUTPUT_DIR)
model.save(MODEL_OUTPUT_DIR + "ludus_model1.h5")
# model.save_weights(MODEL_OUTPUT_DIR + "ludus_model1_weights.h5")

# %%
###################################
# PLOT RESULTS
plt.figure()
plt.ylabel("Loss (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 2])
plt.plot(hist["loss"])
plt.plot(hist["val_loss"])

plt.figure()
plt.ylabel("Accuracy (training and validation)")
plt.xlabel("Training Steps")
plt.ylim([0, 1])
plt.plot(hist["accuracy"])
plt.plot(hist["val_accuracy"])

# %%
###################################
# EVALUATE
# will give loss = 0.5146 | acc = 1.0
model.evaluate(valid_generator, steps=validation_steps)

# %%
###################################
# OPTMIZE MODEL
################
# @title Optimization settings
optimize_lite_model = False  # @param {type:"boolean"}

# Setting a value greater than zero enables quantization of neural
# network activations. A few dozen is already a useful amount.
# @param {type:"slider", min:0, max:1000, step:1}
num_calibration_examples = 60

representative_dataset = None
if optimize_lite_model and num_calibration_examples:
    # Use a bounded number of training examples without labels for calibration.
    # TFLiteConverter expects a list of input tensors, each with batch size 1.
    def representative_dataset(): return itertools.islice(
        ([image[None, ...]]
         for batch, _ in train_generator for image in batch),
        num_calibration_examples)

# %%
# convert
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_OUTPUT_DIR)
if optimize_lite_model:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_dataset:  # This is optional, see above.
        converter.representative_dataset = representative_dataset
lite_model_content = converter.convert()

# %%
# save the TF model
with open(f"{MODEL_OUTPUT_DIR}/lite_ludus_model1", "wb") as f:
    f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))
