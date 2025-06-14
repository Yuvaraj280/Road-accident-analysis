import pandas as pd
import matplotlib.pyplot as plt
import pickle


data = pd.read_csv('accident.csv')
X = data.iloc[:,0:9]
Y = data.iloc[:,-1]

'''
X, Y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
plt.scatter(X[:,0], X[:,1])

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()'''

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.1)

from sklearn import linear_model
model = linear_model.LinearRegression()
model.fit(X_train,Y_train)

accl = model.score(X, Y)
print("Accuracy: ",accl*100," %.")
clf_acc = accl*100

#Prediction
prediction = model.predict(X_test)  #Final predicton 

from sklearn import svm
SVM = svm.LinearSVC()
SVM.fit(X_train,Y_train)
acc = SVM.score(X, Y)
print("Accuracy: ",acc*100," %.")
svm_acc = acc*100

from sklearn.linear_model import LogisticRegression
clf=LogisticRegression()
clf.fit(X_train,Y_train)
acc1 = clf.score(X, Y)
print("Accuracy: ",acc1*100," %.")
acc1 = acc*100

input_feature=[2,3,1,1,1,2,9,1,2]
infprob=clf.predict_proba([input_feature])[0][1]   



'''print(model.predict([[1,6,1,1,1,1,9,3,2]]))
print(model.predict([[3,3,1,1,1,2,9,1,2]]))'''


data = {'LogisticRegression':clf_acc, 'SVM':svm_acc}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color ='green', 
        width = 0.4)
 
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy of Algorithms")
plt.show()



"""Deployment codes starts here"""

file=open('my_model.pkl','wb')
pickle.dump(model,file,protocol=3)
file.close()