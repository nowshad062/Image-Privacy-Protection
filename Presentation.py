#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import the necessary packages
from IPython.display import Image
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
from keras.models import load_model

# load the trained model from disk
print("[INFO] loading privacy model...")
privacy_model = load_model('privacy.model')
print("[INFO] loading VGG16 model...")
object_model = VGG16()
print("[INFO] loading Places365 model...")
places_model = load_model('places_model.h5')


# In[2]:


import os
import csv
import numpy as np
import pandas as pd
import numpy as np
import argparse
import cv2
import imutils
pd.options.mode.chained_assignment = None


# In[3]:


# load the input image and then clone it so we can draw on it later
#test1.jpeg
#test2.jpg
#test3.jpg

imgname = 'test/test.jpg'
image = cv2.imread(imgname)
output = image.copy()
output = imutils.resize(output, width=400)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = cv2.resize(image, (224, 224))

# convert the image to a floating point data type and perform mean subtraction
image = image.astype("float32")
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
image -= mean
CLASSES = ['private', 'public']
preds = privacy_model.predict(np.expand_dims(image, axis=0))[0]

i = np.argmax(preds)
#print preds[i]
label = CLASSES[i]

print label
Image(filename=imgname)


# In[4]:


image = load_img(os.path.join('test/', 'test.jpg'), target_size=(224, 224))
image = img_to_array(image)
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
image = preprocess_input(image)
pred = object_model.predict(image)
label = decode_predictions(pred, top=10)[0]
object_list = []
object_prob_list = []
for i in range (0,10):
    object_list.append(str(label[i][1]))
    object_prob_list.append(round(label[i][2],5))

print("[INFO] Object List and Their Probabilities")
print object_list
print object_prob_list


# In[5]:


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
import csv
import urllib2
from PIL import Image
from cv2 import resize

image = load_img(os.path.join('test/', 'test.jpg'))
image = np.array(image, dtype=np.uint8)
image = resize(image, (224, 224))
image = np.expand_dims(image, 0)
predictions_to_return = 5
preds = places_model.predict(image)[0]
top_preds = np.argsort(preds)[::-1][0:predictions_to_return]
top_probs = preds[np.argsort(preds)[-5:]]
top_probs = np.sort(top_probs)[::-1]
file_name = 'categories_places365.txt'
classes = list()
with open(file_name) as class_file:
    for line in class_file:
        classes.append(line.strip().split(' ')[0][3:])
classes = tuple(classes)

scene_list = []
scene_prob_list = []
for i in range (0,5):
    scene_list.append(classes[top_preds[i]])
    scene_prob_list.append(top_probs[i])

print("[INFO] Scene List and Their Probabilities")
print scene_list
print scene_prob_list


# In[6]:


import networkx as nx
G = nx.Graph()
column_names = object_list
row_names    = object_list
no_column = len(column_names)
no_rows = len(row_names)
a = np.zeros(shape=(no_rows, no_column))
matrix = np.reshape(a, (no_rows, no_column))
df = pd.DataFrame(matrix, columns=column_names, index=row_names)

for i in range (0,10):
    for j in range (0,10):
        if i==j:
            df.loc[object_list[i]][object_list[j]] = 0.0;
        else:
            df.loc[object_list[i]][object_list[j]] = (df.loc[object_list[i]][object_list[j]] + (object_prob_list[i]*object_prob_list[j]))
            G.add_node(object_list[i])
            G.add_node(object_list[j])
            G.add_edge(object_list[i],object_list[j])
            
    
print("[INFO] Object-Object Co-Occurrence Matrix")
df


# In[7]:


column_names = object_list
row_names    = scene_list
no_column = len(column_names)
no_rows = len(row_names)
a = np.zeros(shape=(no_rows, no_column))
matrix = np.reshape(a, (no_rows, no_column))
df2 = pd.DataFrame(matrix, columns=column_names, index=row_names)
for i in range (0,10):
    for j in range (0,5):
            df2[object_list[i]][scene_list[j]] = (df2[object_list[i]][scene_list[j]] + (object_prob_list[i]*scene_prob_list[j]))
            G.add_node(object_list[i])
            G.add_node(scene_list[j])
            G.add_edge(object_list[i],scene_list[j])
    
df2


# In[8]:


objects = ['candle', 'altar', 'groom', 'ice_cream', 'confectionery', 'eggnog', 'toyshop', 'torch', 'maraca', 'restaurant']
object_prob = [0.99864, 0.00087, 0.00037, 3e-05, 2e-05, 2e-05, 1e-05, 0.0, 0.0, 0.0]
scenes = ['toyshop', 'candy_store', 'ball_pit', 'gift_shop', 'flea_market/indoor']
scene_prob = [0.40125963, 0.1225396, 0.07077751, 0.06333199, 0.059057415]

column_names = objects
row_names    = objects
no_column = len(column_names)
no_rows = len(column_names)
a = np.zeros(shape=(no_rows, no_column))
matrix = np.reshape(a, (no_rows, no_column))
df = pd.DataFrame(matrix, columns=column_names, index=row_names)
for i in range (0,10):
    for j in range (0,10):
        if i==j:
            df.loc[objects[i]][objects[j]] = 0.0;
        else:
            df.loc[objects[i]][objects[j]] = (df.loc[objects[i]][objects[j]] + (object_prob[i]*object_prob[j]))
            #if(df.loc[objects[i]][objects[j]] > 0.003):
            G.add_edge(objects[i],objects[j])
            
    
column_names = objects
row_names    = scenes
no_column = len(column_names)
no_rows = len(row_names)
a = np.zeros(shape=(no_rows, no_column))
matrix = np.reshape(a, (no_rows, no_column))
df2 = pd.DataFrame(matrix, columns=column_names, index=row_names)
for i in range (0,10):
    for j in range (0,5):
        if i==j:
            df2[objects[i]][scenes[j]] = 0.0;
        else:
            df2[objects[i]][scenes[j]] = (df2[objects[i]][scenes[j]] + (object_prob[i]*scene_prob[j]))
            #if df2[objects[i]][scenes[j]] > 0.004:
            G.add_edge(objects[i],scenes[j])




# In[9]:


list(G.edges)


# In[10]:


#nx.draw(G)
#plt.axis ("off")
#print vertex_cover.min_weighted_vertex_cover(G)


# In[11]:


df_ot = pd.read_csv('object_threshold.csv',sep=',',index_col=0)


# In[12]:


df_st = pd.read_csv('scene_threshold.csv',sep=',',index_col=0)


# In[13]:



#from networkx.algorithms.approximation import vertex_cover
import matplotlib.pyplot as plt

#objects = ['gown', 'swimming_trunks', 'groom', 'bubble', 'dalmatian', 'sunglass', 'bathing_cap', 'whippet', 'hammerhead', 'bikini']
#object_prob = [0.07553, 0.05757999999999999, 0.05012, 0.032639999999999995, 0.0306, 0.0295, 0.0263, 0.02621, 0.01906, 0.01816]
#scenes = ['shower', 'bathroom', 'ice_skating_rink/outdoor', 'hotel_room', 'ice_skating_rink/indoor']
#scene_prob = [0.6863218000000001, 0.2923281, 0.006975433299999999, 0.001655356, 0.0014729362]

objects = object_list
object_prob = object_prob_list
scenes = scene_list
scene_prob = scene_prob_list

column_names = objects
row_names    = objects
no_column = len(column_names)
no_rows = len(column_names)
a = np.zeros(shape=(no_rows, no_column))
matrix = np.reshape(a, (no_rows, no_column))
df = pd.DataFrame(matrix, columns=column_names, index=row_names)
for i in range (0,10):
    for j in range (0,10):
        if i==j:
            df.loc[objects[i]][objects[j]] = 0.0;
        else:
            df.loc[objects[i]][objects[j]] = (df.loc[objects[i]][objects[j]] + (object_prob[i]*object_prob[j]))
            #print objects[i] + '-' + objects[j]  +  ' ' + str(df.loc[objects[i]][objects[j]]) +  ' : ' + str(df_ot.loc[objects[i]][objects[j]])
            if(df.loc[objects[i]][objects[j]] > 0.01):
                print objects[i] + '-' + objects[j]
                #G.add_edge(objects[i],objects[j])
                G.add_node(objects[i])
                G.add_node(objects[j])
            
    
column_names = objects
row_names    = scenes
no_column = len(column_names)
no_rows = len(row_names)
a = np.zeros(shape=(no_rows, no_column))
matrix = np.reshape(a, (no_rows, no_column))
df2 = pd.DataFrame(matrix, columns=column_names, index=row_names)
for i in range (0,10):
    for j in range (0,5):
            df2[objects[i]][scenes[j]] = (df2[objects[i]][scenes[j]] + (object_prob[i]*scene_prob[j]))
            if df2[objects[i]][scenes[j]] > 0.01:
                print objects[i] + '-' + scenes[j]
                G.add_edge(objects[i],scenes[j])
                G.add_node(objects[i])
                G.add_node(scenes[j])
    
#nx.draw_networkx(G)
#plt.axis ("off")
#print vertex_cover.min_weighted_vertex_cover(G)


# In[1]:


nx.draw_networkx(G)
#plt.axis ("off")


# In[ ]:





# In[ ]:




