import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn import svm

labeled_images = pd.read_csv('../input/train.csv')
images = labeled_images.iloc[:,1:]
labels = labeled_images.iloc[:,:1]
images[images>0]=1
train_images, test_images,train_labels, test_labels = train_test_split(images, labels, train_size=0.8, random_state=0)

param_grid = [
    {'C': [7,7.5,8], 'gamma': [0.014,0.015,0.016],'kernel': ['rbf']}
 ]

#7.5 0.015 0.977

#0.980833333333
#{'C': 7, 'gamma': 0.016, 'kernel': 'rbf'}

svc = svm.SVC(C=7,gamma=0.016,kernel='rbf')
# grid_search = GridSearchCV(svc, param_grid, pre_dispatch= '2*n_jobs',n_jobs=3,cv=None,verbose=3)
# grid_search.fit(train_images, train_labels.values.ravel())
# print(grid_search.score(test_images,test_labels))
# print(grid_search.best_params_)


svc.fit(train_images, train_labels.values.ravel())
svc.score(test_images,test_labels)

test_data=pd.read_csv('../input/test.csv')
test_data[test_data>0]=1
results=svc.predict(test_data)
df = pd.DataFrame(results)
df.index+=1
df.index.name='ImageId'
df.columns=['Label']
df.to_csv('results.csv', header=True)