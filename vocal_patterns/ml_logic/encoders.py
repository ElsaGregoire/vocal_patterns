import numpy as np
from sklearn.calibration import LabelEncoder
from tensorflow.keras.utils import to_categorical


def target_encoder(y, num_classes):
    label_encoder = LabelEncoder()
    y_labels = label_encoder.fit_transform(np.ravel(y, order="c"))
    y_cat = to_categorical(y_labels, num_classes=num_classes)
    return y_cat
