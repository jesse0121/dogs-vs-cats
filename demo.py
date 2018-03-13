from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py

gen = ImageDataGenerator()
test_generator = gen.flow_from_directory("row_data/test", (299,299), shuffle=False,
                                         de=None, follow_links=True)batch_size=16, class_mo