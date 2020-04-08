# %%
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

# %%
###################################
# CHECK TIME LEFT before next reboot
import time
import psutil
uptime = time.time() - psutil.boot_time()
remain = 12 * 60 * 60 - uptime
print(time.strftime('%H:%M:%S', time.gmtime(remain)))

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

# %%
###################################
# PRINT DEVICE INFO
print("TF version:", tf.__version__)
print("Hub version:", hub.__version__)
if tf.config.list_physical_devices('GPU'):
    local_device_protos = tf.python.client.device_lib.list_local_devices()
    gpu_details = [x for x in local_device_protos if x.device_type == 'GPU'][0]
    gpu_name = ''
    for d in gpu_details.physical_device_desc.split(', '):
        if d.split(':')[0] == 'name':
            gpu_name = d.split(': ')[1]
            break
    print(f"GPU is available: {gpu_name}")
else:
    print("GPU is NOT AVAILABLE")

# %%
###################################
# OPTIONS (MODEL TO USE, etc.)

# tensorflow configs
tf.get_logger().propagate = False  # avoid duplicated warnings

module_selection = ("mobilenet_v2_100_224", 224)
# module_selection = ("mobilenet_v2_100_224", 224)
# module_selection = ("inception_v3", 299)

handle_base, pixels = module_selection
IMAGE_SIZE = (pixels, pixels)
BATCH_SIZE = 32
MODEL_OUTPUT_DIR = '/content/gdrive/My Drive/Colab Notebooks/models_output/ludus/model1/'

print(f"Using {MODULE_HANDLE} with input size {IMAGE_SIZE}")
print(f"Data augmentation: {DO_DATA_AUGMENTATION}")

# %%
###################################
# DATASET PATH
data_dir = '/content/gdrive/My Drive/Colab Notebooks/data/ludus'

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
    data_dir + "/valid", shuffle=False, **dataflow_kwargs)

train_generator = train_datagen.flow_from_directory(
    data_dir + "/train", subset="training", shuffle=True, **dataflow_kwargs)

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
tf.saved_model.save(model, MODEL_OUTPUT_DIR)
model.save(MODEL_OUTPUT_DIR + "ludus_model1.h5")

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
# loaded_model_2.evaluate(valid_generator, steps=validation_steps)
model.evaluate(generator=valid_generator, steps=validation_steps)

# %%
###################################
# TEST (predict)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs)
test_generator = test_datagen.flow_from_directory(
    data_dir + "/test", shuffle=False, **dataflow_kwargs)
labels = (train_generator.class_indices)
labels = dict((v, k) for k, v in labels.items())

STEP_SIZE_TEST = test_generator.n // test_generator.batch_size
test_generator.reset()  # TODO confirm it is necessary
pred = model.predict(test_generator,
                     steps=STEP_SIZE_TEST,
                     verbose=1)

# %%
###################################
# SAVE PREDICTIONS
predicted_class_indices = np.argmax(pred, axis=1)
predictions = [labels[k] for k in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename":filenames,
                       "Predictions":predictions})
results.to_csv("results.csv",index=False)

# %%
###################################
# LOAD THE MODEL
# saved model
loaded_model = tf.saved_model.load(MODEL_OUTPUT_DIR)

# models.save
loaded_model_2 = tf.keras.models.load_model(
    MODEL_OUTPUT_DIR + "ludus_model1.h5",
    custom_objects={'KerasLayer': hub.KerasLayer})


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
converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_OUTPUT_DIR)
if optimize_lite_model:
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_dataset:  # This is optional, see above.
        converter.representative_dataset = representative_dataset
lite_model_content = converter.convert()

# %%
with open(f"{MODEL_OUTPUT_DIR}/lite_ludus_model1", "wb") as f:
    f.write(lite_model_content)
print("Wrote %sTFLite model of %d bytes." %
      ("optimized " if optimize_lite_model else "", len(lite_model_content)))
