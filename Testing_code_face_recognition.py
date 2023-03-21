import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import metrics,svm
from sklearn.model_selection import cross_val_score
import joblib # to save and load your trained model 
# load the trained model 
pathTrmodel='/Users/Arsalan/Desktop/face_svm_model_02_06_2022.pkl'
trModel = joblib.load(pathTrmodel)
# load any image from that face folder 
# and consider that as query image 
user_num = 29
samp_num = 10
path = '/Users/Arsalan/Desktop/orl_faces/orl_faces/u(%d)/%d.png'%(user_num,samp_num)
im = mimg.imread(path)

feat = im.reshape(1,-1)

# predict the User number 
op = trModel.predict(feat)


predicted_userNumber = op[0]+1
print(predicted_userNumber)

# fetch the first image of the user number predicted by the tr model 

path = '/Users/Arsalan/Desktop/orl_faces/orl_faces/u(%d)/1.png'%(int(predicted_userNumber))
im_predicted = mimg.imread(path)
plt.figure(1)
plt.subplot(2,1,1)
plt.imshow(im,cmap='gray') 
plt.title('Query Image ')
plt.axis('off')
plt.subplot(2,1,2)
plt.imshow(im_predicted,cmap='gray') 
plt.title('Predicted User Image ')
plt.axis('off')


