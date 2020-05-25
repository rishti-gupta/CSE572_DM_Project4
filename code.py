#!/usr/bin/env python
# coding: utf-8

# In[447]:


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
from apyori import apriori
# from mlxtend.frequent_patterns import apriori
from copy import deepcopy


# In[448]:


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


# In[449]:


def create_lists(insulin_dataset, cgm_dataset):
    insulin_list = []
    for i in range(len(insulin_dataset)):
        insulin_list.append(max(insulin_dataset[i]))
    insulin_list = np.asarray(insulin_list)

#     for i in range(len(insulin_list)):
#         insulin_list[i] = round(insulin_list[i])

    cgm_list = []
    for i in range(len(cgm_dataset)):
        cgm_list.append(max(cgm_dataset[i]))
    cgm_list = np.asarray(cgm_list)

    cgm_col6 = cgm_dataset[:,5]
    
    insulin_list= insulin_list.reshape((len(insulin_list),1))
    cgm_list= cgm_list.reshape((len(insulin_list),1))
    cgm_col6 = cgm_col6.reshape((len(insulin_list),1))
    
    return [insulin_list, cgm_list,cgm_col6]


# In[450]:


# Ground Truth
def create_bins(lower_bound, width, quantity):
    bins = []
    for low in range(lower_bound, 
                     lower_bound + quantity*width + 1, width):
        bins.append((low, low+width))
    return bins

from collections import Counter
def find_bin(value, bins):

    if value == 0:
        return 0
    for i in range(0, len(bins)):
        if bins[i][0] <= value < bins[i][1]:
                return i+1
    return -1

# In[451]:
def combine_matrix(cgm_bins, cgm_col6_bins, insulin_list):
    combined_matrix = np.append(cgm_bins, cgm_col6_bins, axis = 1)
    combined_matrix = np.append(combined_matrix, insulin_list, axis = 1)
    return combined_matrix

# In[452]:
def extract_Association_Rules(association_rules):
    results = []
    for item in association_rules:
        pair=item[0]
        items = [x for x in pair]
        if len(items) != 3:
            continue
        value0 = items
        value1=str(item[1])[:8]
        value2=str(item[2][0][2])[:7]
        value3=str(item[2][0][3])[:7]
        value4 = item[2]
        rows = (value0,value1,value2,value3,value4)
        results.append(rows)
        
    labels = ['Title-1','Support','Confidence', 'Lift','Ordered Set']
    extracted = pd.DataFrame.from_records(results,columns = labels)
    return extracted

# In[453]:

def extract_OrderedSet(fs,os):
    results = []
    for item in os:
        last = len(item)-1
        item_1 = [x for x in item[last][0]]
        item_2 = [y for y in item[last][1]]
        results.append([[item_1[0], item_1[1], item_2[0]], item[last][2]])
    return results

# In[454]:

def extract_max_rules(os):
    results = []
    for item in os:
        last = len(item)-1
        item_1 = [x for x in item[last][0]]
        item_2 = [y for y in item[last][1]]
        results.append([[item_1[0], item_1[1], item_2[0]], item[last][2]])
    max_conf_rules = []
    
    max_conf = 0.0
    for i in results:
        if max_conf < i[1]:
            max_conf = i[1]
    for i in results:
        if i[1] == max_conf:
            max_conf_rules.append(i[0])
    return max_conf_rules

# In[455]:

def extract_min_rules(os):
    results = []
    for item in os:
        last = len(item)-1
        item_1 = [x for x in item[last][0]]
        item_2 = [y for y in item[last][1]]
        results.append([[item_1[0], item_1[1], item_2[0]], item[last][2]])
    min_conf_rules = []
    
    min_conf = 100.0
    for i in results:
        if min_conf > i[1]:
            min_conf = i[1]
    for i in results:
        if i[1] == min_conf:
            min_conf_rules.append(i[0])
    return min_conf_rules

# In[456]:

result_freq_items= {}
result_max_confidence_rules = {}
result_anomalous_rules = {}

for i in range(1,6):
    [insulin_dataset, cgm_dataset]= load_csv(i)
    [insulin_list, cgm_list,cgm_col6] = create_lists(insulin_dataset, cgm_dataset)

    bins = create_bins(lower_bound=40, width=10, quantity=36)

    cgm_bins = []
    for value in cgm_list:
        bin_index = find_bin(value, bins)
        cgm_bins.append(bin_index+1000)
    cgm_bins = np.array(cgm_bins)
    cgm_bins = cgm_bins.reshape((len(cgm_list),1))

    cgm_col6_bins = []
    for value in cgm_col6:
        bin_index = find_bin(value, bins)
        cgm_col6_bins.append(bin_index+100)
    cgm_col6_bins = np.array(cgm_col6_bins)
    cgm_col6_bins = cgm_col6_bins.reshape((len(insulin_list),1))

    combined_matrix = combine_matrix(cgm_bins, cgm_col6_bins, insulin_list)
    combined_matrix = list(combined_matrix)

    # _combined = []
    # for k in range(len(combined_matrix)):
    #     if 999 in combined_matrix[k] or 99 in combined_matrix[k]:
    #         continue
    #     else:
    #         _combined.append(combined_matrix[k])
    #
    # combined_matrix = []
    # combined_matrix = _combined

    association_rules = apriori(combined_matrix, min_support =0.001)
    association_result = list(association_rules)
    extracted_df = extract_Association_Rules(association_result)
#     extracted_OS = extract_OrderedSet(extracted_df['Title-1'],extracted_df['Ordered Set'])
    
    for j in extracted_df['Title-1']:
        j.sort(reverse = True)
        j[0] -= 1000
        j[1] -= 100
    
    result_freq_items[i] = extracted_df['Title-1']
    
#     print("Patient" + str(i))
#     print(extracted_OS)
    
    max_conf_rules = []
    max_conf_rules = extract_max_rules(extracted_df['Ordered Set'])
    for j in max_conf_rules:
        j.sort(reverse = True)
        j[0] -= 1000
        j[1] -= 100
        
    result_max_confidence_rules[i] = max_conf_rules
    
    min_conf_rules=[]
    min_conf_rules = extract_min_rules(extracted_df['Ordered Set'])
    for j in min_conf_rules:
        j.sort(reverse = True)
        j[0] -= 1000
        j[1] -= 100
    result_anomalous_rules[i] = min_conf_rules

#     print("Patient" + str(i))
#     print(max_conf_rules)
#     print("haha")
#     print(min_conf_rules)

# In[457]:

from csv import writer
def append_list_as_row(file_name, list_of_elem):
# Open file in append mode
    with open(file_name, 'ab') as write_obj:
    # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)

# In[458]:
import os
if os.path.exists("Frequent_items.csv"):
    os.remove("Frequent_items.csv")
if os.path.exists("Largest_confidence_rules.csv"):
    os.remove("Largest_confidence_rules.csv")
if os.path.exists("Anomalous_rules.csv"):
    os.remove("Anomalous_rules.csv")

for i in range(1,6):
#     append_list_as_row("Frequent_items.csv",["\n"])
#     append_list_as_row("Frequent_items.csv",["Patient:"+ str(i)])
    for j in result_freq_items[i]:
        append_list_as_row("Frequent_items.csv",["{" + str(j[0])+","+str(j[1])+"," + str(j[2]) + "}"])
#     append_list_as_row("Largest_confidence_rules.csv",["\n"])
#     append_list_as_row("Largest_confidence_rules.csv",["Patient:"+str(i)])
    for j in result_max_confidence_rules[i]:
        append_list_as_row("Largest_confidence_rules.csv",["{" + str(j[0])+","+str(j[1])+"->"+str(j[2]) + "}"])
#     append_list_as_row("Anomalous_rules.csv",["\n"])
#     append_list_as_row("Anomalous_rules.csv",["Patient:"+str(i)])
    for j in result_anomalous_rules[i]:
        append_list_as_row("Anomalous_rules.csv",["{" + str(j[0])+","+str(j[1])+"->"+str(j[2]) + "}"])

