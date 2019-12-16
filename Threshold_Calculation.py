#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

df_osp = pd.read_csv('object_scene_matrix_private.csv',sep=',',index_col=0)
df_osp.head(10)


# In[2]:


df2 = df_osp[df_osp.columns[0]]
df_osa = pd.read_csv('object_scene_matrix_all.csv',sep=',', index_col = 0)
df_osa.head(10)


# In[3]:


df_osp = df_osp.drop(df_osp.columns[0], axis=1)
df_osa = df_osa.drop(df_osa.columns[0], axis=1)
#df_osp.head(10)
#df_osa.head(10)
df_osp = df_osp/df_osa
df_osp.fillna(0.0,inplace=True)
#df_osp.head(10)
scene_threshold = pd.concat([df2, df_osp], axis=1)
scene_threshold.head(20)


# In[4]:


df_oop = pd.read_csv('object_object_matrix_private.csv',sep=',', index_col = 0)
df_oop.head(10)


# In[5]:


df3 = df_oop[df_oop.columns[0]]
df_ooa = pd.read_csv('object_object_matrix_all.csv',sep=',', index_col = 0)
df_ooa.head(10)


# In[6]:


df_oop = df_oop.drop(df_oop.columns[0], axis=1)
df_ooa = df_ooa.drop(df_ooa.columns[0], axis=1)
#df_osp.head(10)
#df_osa.head(10)
df_oop = df_oop/df_ooa
df_oop.fillna(0.0,inplace=True)
#df_osp.head(10)
object_threshold = pd.concat([df3, df_oop], axis=1)
object_threshold.head(20)


# In[7]:


object_threshold.to_csv('object_threshold.csv', header=True, sep=',', index=True);
scene_threshold.to_csv('scene_threshold.csv', header=True, sep=',', index=True);


# In[8]:


print object_threshold['swimming_trunks']['gown']
print scene_threshold['swimming_trunks']['shower']


# In[ ]:




