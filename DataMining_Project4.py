#!/usr/bin/env python
# coding: utf-8

# In[36]:


# importing python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import *
import csv
import pickle
# create expanding window features
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
import warnings
warnings.filterwarnings('ignore')
from operator import is_not
from functools import partial
from sklearn import preprocessing
from sklearn import decomposition, datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


# In[37]:


def load_csv(i):
    
    insulin_dataset = pd.DataFrame()
    insulin_df = pd.read_csv("DataFolder/InsulinBolusLunchPat{}.csv".format(i), sep = '\t', header = None, skiprows = 1)
    insulin_dataset = insulin_dataset.append(insulin_df, ignore_index = True)    
    insulin_dataset = insulin_dataset[0].str.split(',', expand = True)
    insulin_dataset = insulin_dataset.dropna(how = 'all')
    insulin_dataset = insulin_dataset.fillna(0)
    insulin_dataset = insulin_dataset.replace("NaN", 0)
    insulin_dataset = insulin_dataset.replace('', 0)
    insulin_dataset = insulin_dataset.astype(float)
    # insulin_dataset = insulin_dataset.astype(int)
    insulin_dataset = np.array(insulin_dataset)

    cgm_dataset = pd.DataFrame()
    cgm_df = pd.read_csv("DataFolder/CGMSeriesLunchPat{}.csv".format(i), sep = '\t', header = None, skiprows = 1)
    cgm_dataset = cgm_dataset.append(cgm_df, ignore_index = True)
    
    cgm_dataset = cgm_dataset[0].str.split(',', expand = True)
    cgm_dataset = cgm_dataset.dropna(how = 'all')
    cgm_dataset = cgm_dataset.fillna(0)
    cgm_dataset = cgm_dataset.replace("NaN", 0)
    cgm_dataset = cgm_dataset.replace('', 0)
    cgm_dataset = cgm_dataset.astype(float)
    # cgm_dataset = cgm_dataset.astype(int)
    cgm_dataset = np.array(cgm_dataset)
    
    return [insulin_dataset, cgm_dataset]


# In[38]:


def create_lists(insulin_dataset, cgm_dataset):
    insulin_list = []
    for i in range(len(insulin_dataset)):
        insulin_list.append(max(insulin_dataset[i]))
    insulin_list = np.asarray(insulin_list)

    for i in range(len(insulin_list)):
        insulin_list[i] = round(insulin_list[i])

    cgm_list = []
    for i in range(len(cgm_dataset)):
        cgm_list.append(max(cgm_dataset[i]))
    cgm_list = np.asarray(cgm_list)

    cgm_col6 = cgm_dataset[:,6]
    
    insulin_list= insulin_list.reshape((len(insulin_list),1))
    cgm_list= cgm_list.reshape((len(insulin_list),1))
    cgm_col6 = cgm_col6.reshape((len(insulin_list),1))
    
    return [insulin_list, cgm_list,cgm_col6]


# In[39]:


# Ground Truth
def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins

from collections import Counter
def find_bin(value, bins):
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
                return i
    return -1


# In[40]:


def combine_matrix(cgm_bins, cgm_col6_bins, insulin_list):
    combined_matrix = np.append(cgm_bins, cgm_col6_bins, axis = 1)
    combined_matrix = np.append(combined_matrix, insulin_list, axis = 1)
    return combined_matrix


# In[41]:


def create_dict(combined_matrix):
    _dict = {}
    for i in combined_matrix:
        temp = str(i[0])+','+str(i[1])+','+str(i[2])
        if temp in _dict:
            _dict[temp] += 1
        else:
            _dict[temp] = 1

    _set = set(_dict.values())
    return(_dict)


# In[42]:


def find_frequent_items(_dict):
    freq_items = []
    maximum_frequent = max(_dict.values())
    for i in _dict:
        if _dict[i] == maximum_frequent:
            freq_items.append(i)
    return [freq_items,maximum_frequent]


# In[43]:


def calculate_largest_confidence(combined_matrix, freq_items,maximum_frequent):
    confidence = {}
    for i in freq_items:
        count=0
        temp_list = []
        _list = i.split(',')
        _list[0] = float(_list[0])
        _list[1] = float(_list[1])
        for j in combined_matrix:
            if j[0] == _list[0] and j[1] == _list[1]:
                count+=1
                temp_list.append(j)
        confidence[i] = round(maximum_frequent/count, 2)
    #     print(temp_list)
    
    temp = []
    for i in confidence:
        if confidence[i] != max(confidence.values()):
            temp.append(i)
    for j in temp:
        confidence.pop(j)
    return(confidence)


# In[44]:


def calculate_all_confidence(combined_matrix,_dict):
    confidence = {}
    for i in _dict:
        count = 0
        temp_list = []
        _list = i.split(',')
        _list[0] = float(_list[0])
        _list[1] = float(_list[1])
        for j in combined_matrix:
            if j[0] == _list[0] and j[1] == _list[1]:
                count+=1
                temp_list.append(j)
        confidence[i] = round(_dict[i]/count, 2)
    return confidence


# In[45]:


result_freq_items= {}
result_max_confidence_rules = {}
result_anomalous_rules = {}

for i in range(1,6):
    [insulin_dataset, cgm_dataset]= load_csv(i)
    [insulin_list, cgm_list,cgm_col6] = create_lists(insulin_dataset, cgm_dataset)

    bins = create_bins(lower_bound=0, width=10, quantity=400)

    cgm_bins = []
    for value in cgm_list:
        bin_index = find_bin(value, bins)
        cgm_bins.append(bin_index)
    cgm_bins = np.array(cgm_bins)
    cgm_bins = cgm_bins.reshape((len(cgm_list),1))

    cgm_col6_bins = []
    for value in cgm_col6:
        bin_index = find_bin(value, bins)
        cgm_col6_bins.append(bin_index)
    cgm_col6_bins = np.array(cgm_col6_bins)
    cgm_col6_bins = cgm_col6_bins.reshape((len(insulin_list),1))

    combined_matrix = combine_matrix(cgm_bins, cgm_col6_bins, insulin_list)
    _dict = create_dict(combined_matrix)
    [freq_items,maximum_frequent] = find_frequent_items(_dict)
    largest_confidences = calculate_largest_confidence(combined_matrix, freq_items, maximum_frequent)
  
#     largest_confidence_rules = []
#     max_conf = max(largest_confidences.values())
#     for t in largest_confidences:
#         if largest_confidences[t] == max_conf:
#             largest_confidence_rules.append(t)
    
    anom_confidence = calculate_all_confidence(combined_matrix,_dict)

    anom_rules = []
    for j in anom_confidence:
        if anom_confidence[j] == min(anom_confidence.values()):
            anom_rules.append(j)
            
    result_freq_items[i] = freq_items
    result_max_confidence_rules[i] = largest_confidences
    result_anomalous_rules[i] = anom_rules


# In[47]:


from csv import writer
def append_list_as_row(file_name, list_of_elem):
# Open file in append mode
    with open(file_name, 'ab') as write_obj:
    # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


# In[48]:


for i in range(1,6):
    append_list_as_row("Frequent_items.csv",["\n"])
    append_list_as_row("Frequent_items.csv",["Patient:"+ str(i)])
    for j in result_freq_items[i]:
        append_list_as_row("Frequent_items.csv",[j])
    append_list_as_row("Largest_confidence_rules.csv",["\n"])
    append_list_as_row("Largest_confidence_rules.csv",["Patient:"+str(i)])
    for j in result_max_confidence_rules[i]:
        append_list_as_row("Largest_confidence_rules.csv",[j])
    append_list_as_row("Anomalous_rules.csv",["\n"])
    append_list_as_row("Anomalous_rules.csv",["Patient:"+str(i)])
    for j in result_anomalous_rules[i]:
        append_list_as_row("Anomalous_rules.csv",[j])


# In[ ]:




