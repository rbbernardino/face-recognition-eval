# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# SELECTS TENSORFLOW VERSION (soon default will be tf2)
try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

# %%
###################################
from pathlib import Path

import numpy as np
import pandas as pd
# import matplotlib.pylab as plt

import tensorflow as tf
import tensorflow_hub as hub

###################################
tf.get_logger().propagate = False  # avoid duplicated warnings
pd.options.display.float_format = '{:,.2f}'.format

###################################
# LOAD CUSTOM LIBRARIES
if not Path("./mylibs").exists():
    !ln - s "gdrive/My Drive/Colab Notebooks/libs" "mylibs"
import mylibs.aiutils as aiutils

###################################
# PRINT DEVICE INFO
aiutils.print_device_info()

# %% [markdown]
# Model and Data Definitions

# %%
######################
IMAGE_SIZE_MODEL_1 = (224, 224)  # mobilenet_v2_100_224
IMAGE_SIZE_MODEL_2 = (299, 299)  # inception_v3
IMAGE_RESIZE_METHOD = "bilinear"
BATCH_SIZE = 32  # for image directory iterators

BASE_DIR = Path('/content/gdrive/My Drive/Colab Notebooks')
DATA_DIR = BASE_DIR / 'data' / 'ludus'
RESULTS_DIR = BASE_DIR / 'results' / 'ludus'
MODELS_DIR = Path(
    '/content/gdrive/My Drive/Colab Notebooks/models_output/ludus')
MODEL_LUDUS_1 = MODELS_DIR / "model1" / "ludus_model1.h5"
CLASS_LABELS_1 = MODELS_DIR / "model1" / "retrained_labels.txt"

OUTPUT_PRED_CSV = RESULTS_DIR / "pred_mdl1_tr1_tst1.csv"

if not RESULTS_DIR.exists():
    RESULTS_DIR.mkdir(parents=True)

# %% [markdown]
# Prepare for Testing

# %%
# PREPARE FOR TESTING
#####################
# LOAD THE MODEL
model_1 = tf.keras.models.load_model(
    MODEL_LUDUS_1,
    custom_objects={'KerasLayer': hub.KerasLayer})

# LOAD THE IMAGES
datagen_kwargs_mdl1 = dict(rescale=1. / 255)  # mobilenet pixel value: 0~1
dataflow_kwargs = dict(target_size=IMAGE_SIZE_MODEL_1, batch_size=BATCH_SIZE,
                       interpolation=IMAGE_RESIZE_METHOD)

datagen_mdl1 = tf.keras.preprocessing.image.ImageDataGenerator(
    **datagen_kwargs_mdl1)
test_dirIterator = datagen_mdl1.flow_from_directory(
    str(DATA_DIR / "test_set1"), shuffle=False, **dataflow_kwargs)

# LOAD THE LABELS
labels_byName = aiutils.load_class_indices(CLASS_LABELS_1)
train_labels = dict((v, k) for k, v in labels_byName.items())

# %%
###################################
# TEST (predict)
test_labels = dict((v, k) for k, v in test_dirIterator.class_indices.items())
test_dirIterator.reset()  # ensure iteration from the beginning
step_size_mdl1 = test_dirIterator.n // test_dirIterator.batch_size
pred = model_1.predict(test_dirIterator,
                       steps=step_size_mdl1,
                       verbose=1)

# %%
###################################
# SAVE PREDICTIONS
pred_class_indices = np.argmax(pred, axis=1)
pred_confidence = np.max(pred, axis=1)
predictions = [train_labels[k] for k in pred_class_indices]

filenames = [Path(f).name for f in test_dirIterator.filenames]
correct_classes = [test_labels[k] for k in test_dirIterator.classes]
results = pd.DataFrame({"Filename": filenames,
                        "Expected": correct_classes,
                        "Predicted": predictions,
                        "Confidence": pred_confidence})
results.to_csv(OUTPUT_PRED_CSV, index=False)
display(results)

# %% [markdown]
# ## Results

# %%
###################################
# REPORT RESULTS
results_unknown = results[results['Expected'] == "Unknown"]

image_count = len(results)
unknown_count = len(results_unknown)
known_count = image_count - unknown_count
unknown_perc = 100 * unknown_count / image_count

correct_count = sum(results['Expected'] == results['Predicted'])
correct_unknown_count = sum(results_unknown['Predicted'] == 'Unknown')

full_accuracy = 100 * correct_count / image_count
known_accuracy = 100 * correct_count / known_count
unknown_accuracy = 100 * correct_unknown_count / unknown_count

# %%
print(
    f"{'Test Images:':<20}{image_count}\n"
    f"{'Unknown Images:':<20}{unknown_count} ({unknown_perc:.2f}%)\n"
    f"{'Full Accuracy:':<20}{full_accuracy:.2f}%\n"
    f"{'Known Accuracy:':<20}{known_accuracy:.2f}%\n"
    f"{'Unknown Accuracy:':<20}{unknown_accuracy:.2f}%"
)

# %% [markdown]
# ### Classes Distribution

# %%
class_dist = (
    results.groupby("Expected")
    .agg({'Filename': 'count'})
    .reset_index()
    .rename(columns={
        'Expected': 'Label',
        'Filename': 'Count'})
    .assign(Count_p=lambda df: 100 * df['Count'] / image_count)
)
display(class_dist)

# %% [markdown]
# ### Precision by Class

# %%
# TODO horrible code, in R is much simpler, I must be overcomplicating
acc_byClass = (
    results.assign(Correct_n=lambda x: (x['Expected'] == x['Predicted'])
                   .astype(int))
    .groupby("Expected")
    .agg({'Filename': 'count', 'Correct_n': 'sum'})
    .reset_index()
    .rename(columns={
        'Expected': 'Label',
        'Filename': 'n'})
    .assign(n_perc=lambda df: 100 * df['n'] / image_count)
    .assign(Precision=lambda df: 100 * df['Correct_n'] / df['n'])
    .reindex(columns=['Label', 'n', 'n_perc', 'Correct_n', 'Precision'])
)
display(acc_byClass)
