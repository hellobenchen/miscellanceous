# -*- coding: utf-8 -*-
"""
@author: zchen
fit a logitic regression model with L1 regularization
plot the ROC curve for the training data
output the coefficients to a file, so as to remove zero weight variables from the model.
"""

import pyodbc
import numpy as np
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
import sklearn.metrics as met
import sklearn.preprocessing as pre

'''get data from a database'''
conn = pyodbc.connect('DRIVER={SQL Server};SERVER=aubriprbiw02;DATABASE=Claims_Automation;Trusted_Connection=yes;')
cursor = conn.cursor()
cursor.execute('SELECT * FROM [Claims_Automation].[model].[Online_Claim_All_OSN_Python_Training_VW]')
columns = [column[0] for column in cursor.description]
columns_array = np.array(columns[2:]).astype(str)#column names, the 1st and 2nd are OSN and label
rows = cursor.fetchall() #get data rows
rows_array = np.array(rows)

'''stored the data and the label separately, label -> y, data -> X'''
y = rows_array[:,1].astype(int) #2nd column is the label
y = np.reshape(y, (1, len(y)))[0] #convert to a row vector
      
X = rows_array[:, 2:].astype(float) 
X = pre.scale(X) #standarization

'''train the model'''
lr = lm.LogisticRegression(penalty='l1', solver='liblinear', max_iter=200, C=1) #use l1 for pruning feature
lr.fit(X, y)

'''check how well the model works on the training data'''
result = lr.predict(X)
result_prob = lr.predict_proba(X)[:, 1] # the second row is label 1, lr.classes_
compare = np.concatenate((np.reshape(result,(len(y),1)),np.reshape(y, (len(y),1))),axis=1)

'''calculate indicators and print/plot them'''
tp = sum(int(x==1 and y==1) for (x,y) in compare)
fp = sum(int(x==1 and y==0) for (x,y) in compare)
tn = sum(int(x==0 and y==0) for (x,y) in compare)
fn = sum(int(x==0 and y==1) for (x,y) in compare)

recall = tp / (tp + fn)
precision = tp / (tp + fp)
specifity = tn / (tn + fp)

OneMinusSpec, sensitivity, thresholds = met.roc_curve(y, result_prob, pos_label = 1)
auc = met.roc_auc_score(y, result_prob, average='macro')
                                                    
print('true positive {0}, false positive {1} true negative {2}, false negative {3}'.format(tp, fp, tn, fn))
print('recall/sensitivity {0}, precision {1}, specifity {2}'.format(recall, precision, specifity))
print('Area Under ROC Curve {0}'.format(auc))

'''plot the ROC curve'''
plt.figure(figsize=(4,4))
plt.plot(OneMinusSpec, sensitivity)
plt.xlim([0,1])
plt.ylim([0,1])
plt.xlabel('1 - specificity')
plt.ylabel('recall / sensitivity')
plt.show()

'''save variables and coefficients to a txt file'''
output  = np.concatenate((np.reshape(columns_array, (len(columns_array), 1)), np.reshape(lr.coef_[0].astype(float), (len(lr.coef_[0]), 1))), axis=1)
np.savetxt("c:\\temp\\lr_coefficients.csv", output, fmt='%s',delimiter=",") #format = string with width 32
