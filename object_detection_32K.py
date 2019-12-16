#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16

import os
import csv
import numpy as np
# load the model
model = VGG16()

image_id_array = np.empty(shape = (0, 1), dtype="S20")
#image_id_array[0,0] = "image_ids"
result_array = np.empty(shape = (1, 10), dtype="S20")
result_array[0, 0] = "Prediction_1"
result_array[0, 1] = "Prediction_2"
result_array[0, 2] = "Prediction_3"
result_array[0, 3] = "Prediction_4"
result_array[0, 4] = "Prediction_5"
result_array[0, 5] = "Prediction_6"
result_array[0, 6] = "Prediction_7"
result_array[0, 7] = "Prediction_8"
result_array[0, 8] = "Prediction_9"
result_array[0, 9] = "Prediction_10"

prob_array = np.empty(shape = (1, 10), dtype="S20")
prob_array[0, 0] = "Probability_1"
prob_array[0, 1] = "Probability_2"
prob_array[0, 2] = "Probability_3"
prob_array[0, 3] = "Probability_4"
prob_array[0, 4] = "Probability_5"
prob_array[0, 5] = "Probability_6"
prob_array[0, 6] = "Probability_7"
prob_array[0, 7] = "Probability_8"
prob_array[0, 8] = "Probability_9"
prob_array[0, 9] = "Probability_10"

counter = 0

for filename in os.listdir('images'):
        if filename.endswith(".jpg") or filename.endswith(".JPEG"): 
            print(os.path.join('images/', filename))
            image_id = filename.split(".")[0]
            image = load_img(os.path.join('images/', filename), target_size=(224, 224))
            image = img_to_array(image)
            image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
            image = preprocess_input(image)
            pred = model.predict(image)
            label = decode_predictions(pred, top=10)[0]
            #print label
            top1 = label[0][1]
            top2 = label[1][1]
            top3 = label[2][1]
            top4 = label[3][1]
            top5 = label[4][1]
            top6 = label[5][1]
            top7 = label[6][1]
            top8 = label[7][1]
            top9 = label[8][1]
            top10 = label[9][1]
            
           
            image_id_array = np.append(image_id_array,[image_id])
            result_array = np.append(result_array,[[top1,top2,top3,top4,top5,top6,top7,top8,top9,top10]],axis = 0)
            
            
            prob1 = round(label[0][2],5)
            prob2 = round(label[1][2],5)
            prob3 = round(label[2][2],5)
            prob4 = round(label[3][2],5)
            prob5 = round(label[4][2],5)
            prob6 = round(label[5][2],5)
            prob7 = round(label[6][2],5)
            prob8 = round(label[7][2],5)
            prob9 = round(label[8][2],5)
            prob10 = round(label[9][2],5)

            prob_array = np.append(prob_array,[[prob1,prob2,prob3,prob4,prob5,prob6,prob7,prob8,prob9,prob10]],axis = 0)
            
            np.savetxt("object_image_ids.csv", image_id_array , delimiter=",", fmt="%s")
            np.savetxt("objects.csv", result_array , delimiter=",", fmt="%s")
            np.savetxt("object_probability.csv", prob_array , delimiter=",", fmt="%s")
            
            counter+=1
            print counter
            


# In[ ]:




