from Dataset import Dataset
import os
from sklearn.model_selection import train_test_split
from sklearn import svm,metrics
import cv2
import matplotlib.pyplot as plt
from joblib import dump,load

#lanjutkan prediksi untuk 1 image

dataset = Dataset()
classes = os.listdir("Dataset")

images, labels, img_names, cls = dataset.load_train("Dataset", 640, classes)
X = images.reshape((len(images),-1))

X_train,X_test,y_train,y_test = train_test_split(X,cls,test_size=30)  #harus di reshape
print(X_test[1].shape)

clf = svm.SVC(gamma='scale')
clf.fit(X_train,y_train)
savedModel = 'saved_modeltest640.joblib'
dump(clf,savedModel)

# print(X_test[1].reshape(1,-1).shape)
#
# y_pred = clf.predict(X_test[1].reshape(1,-1))
y_pred = clf.predict(X_test)
print(y_pred)

# print((X_test[4]))



print("Classification report for classifier %s:\n%s\n"
      % (clf, metrics.classification_report(y_test, y_pred)))

print("Confusion matrix:\n%s" % metrics.confusion_matrix(y_test, y_pred))

# print(X_train)
# print(y_train)




