# crossvalidated Face recognition model 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mimg
from sklearn import metrics,svm
from sklearn.model_selection import cross_val_score
import joblib # to save and load your trained model 

images  = np.zeros((400,112,92)) 
X = np.zeros((400,10304))
y = np.zeros((400))
count=0
for i in range(40):
    for j in range(10):
        path = '/Users/Arsalan/Desktop/orl_faces/orl_faces/u(%d)/%d.png'%(i+1,j+1)
        im = mimg.imread(path)
        feat = im.reshape(1,-1) # -1 means arrange all values on the columns side 
        print(im.shape)
        images[count,:,:]=im # images
        X[count,:]=feat # data
        y[count] = i
        count=count+1
        # plt.figure(1)
        # plt.imshow(im,cmap = 'gray')
        # plt.title('User Number: '+str(i+1))
        # plt.axis('off')
        # plt.pause(0.3) # 0.1 second


ModelSvm = svm.SVC(kernel = 'rbf',C = 2)

result = cross_val_score(ModelSvm,X,y,cv=10)
print(result)
print('Avg acc: ',result.mean())


# retraining 
ModelSvm = ModelSvm.fit(X,y)
# if the CV acc is above the threshold (85%) save the model 
if(result.mean()>0.85):
    pathTrmodel='/Users/Arsalan/Desktop/face_svm_model_02_06_2022.pkl'
    joblib.dump(ModelSvm, pathTrmodel)
else:
    print('Accuracy is not at Par')





