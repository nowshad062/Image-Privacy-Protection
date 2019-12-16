#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import division, print_function
import os

import warnings
import numpy as np

from keras import backend as K
from keras.layers import Input
from keras.layers.core import Activation, Dense, Flatten
from keras.layers.pooling import MaxPooling2D
from keras.models import Model
from keras.layers import Conv2D
from keras.regularizers import l2
from keras.layers.core import Dropout
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras_applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras.utils import layer_utils
from keras.preprocessing import image
from keras.applications.imagenet_utils import preprocess_input
from keras.preprocessing.image import load_img

WEIGHTS_PATH = 'vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5'


def VGG16_Places365(include_top=True, 
                    weights='places', 
                    input_tensor=None, 
                    input_shape=None, 
                    pooling=None, 
                    classes=365):

    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten =include_top)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    # Block 1
    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv1')(img_input)

    x = Conv2D(filters=64, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block1_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block1_pool", padding='valid')(x)

    # Block 2
    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv1')(x)

    x = Conv2D(filters=128, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block2_conv2')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block2_pool", padding='valid')(x)

    # Block 3
    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv1')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv2')(x)

    x = Conv2D(filters=256, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block3_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block3_pool", padding='valid')(x)

    # Block 4
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block4_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block4_pool", padding='valid')(x)

    # Block 5
    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv1')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv2')(x)

    x = Conv2D(filters=512, kernel_size=3, strides=(1, 1), padding='same',
               kernel_regularizer=l2(0.0002),
               activation='relu', name='block5_conv3')(x)

    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name="block5_pool", padding='valid')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5, name='drop_fc1')(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5, name='drop_fc2')(x)
        x = Dense(365, activation='softmax', name="predictions")(x)

    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # Create model.
    model = Model(inputs, x, name='vgg16-places365')

    # load weights
    if weights == 'places':
        if include_top:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('vgg16-places365_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')

        model.load_weights(weights_path)

    elif weights is not None:
        model.load_weights(weights)
        
    
    model.save('places_model.h5');

    return model

if __name__ == '__main__':
    
    import csv
    import urllib2
    import numpy as np
    from PIL import Image
    from cv2 import resize

    #TEST_IMAGE_URL = 'http://places2.csail.mit.edu/imgs/demo/15.jpg'
    #image = Image.open(urllib2.urlopen(TEST_IMAGE_URL))
    
    file_name = 'categories_places365.txt'
    image_id_array = np.empty(shape = (0, 1), dtype="S20")
    result_array = np.empty(shape = (1, 5), dtype="S20")
    prob_array = np.empty(shape = (1, 5), dtype="S20")
    
    result_array[0, 0] = "Prediction_1"
    result_array[0, 1] = "Prediction_2"
    result_array[0, 2] = "Prediction_3"
    result_array[0, 3] = "Prediction_4"
    result_array[0, 4] = "Prediction_5"
    
    prob_array[0, 0] = "Probability_1"
    prob_array[0, 1] = "Probability_2"
    prob_array[0, 2] = "Probability_3"
    prob_array[0, 3] = "Probability_4"
    prob_array[0, 4] = "Probability_5"
    
    #with open('output.csv', mode='wa') as output_file:
        #output_file = csv.writer(output_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    
    model = VGG16_Places365(weights='places')
    counter = 0
    
    for filename in os.listdir('images'):
        if filename.endswith(".jpg") or filename.endswith(".JPEG"): 
            print(os.path.join('images/', filename))
            
            image_id = filename.split(".")[0]
            
            image = load_img(os.path.join('images/', filename))
            image = np.array(image, dtype=np.uint8)
            image = resize(image, (224, 224))
            image = np.expand_dims(image, 0)

            
            predictions_to_return = 5
            preds = model.predict(image)[0]
                
            top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
            
            top_probs = preds[np.argsort(preds)[-5:]]
            top_probs = np.sort(top_probs)[::-1]
            
            #print(top_probs[0])
            
            # load the class label
            classes = list()
            with open(file_name) as class_file:
                for line in class_file:
                    classes.append(line.strip().split(' ')[0][3:])
            classes = tuple(classes)

            print('--PREDICTED SCENE CATEGORIES SAVED TO CSV:')
            
            image_id_array = np.append(image_id_array,[image_id])
            result_array = np.append(result_array,[[classes[top_preds[0]],
                                                    classes[top_preds[1]],
                                                    classes[top_preds[2]],
                                                   classes[top_preds[3]],
                                                    classes[top_preds[4]]]],axis = 0)
            
            prob_array = np.append(prob_array,[[top_probs[0],top_probs[1],top_probs[2],top_probs[3],top_probs[4]]],axis = 0)
            
            np.savetxt("scene_image_ids.csv", image_id_array , delimiter=",", fmt="%s")
            np.savetxt("scenes.csv", result_array , delimiter=",", fmt="%s")
            np.savetxt("scene_probability.csv", prob_array , delimiter=",", fmt="%s")
            
            
            counter+=1
            print(counter)
           
            continue
        else:
            continue
    #print(result_array)
    #np.savetxt("output.csv", result_array , delimiter=",", fmt="%s")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




