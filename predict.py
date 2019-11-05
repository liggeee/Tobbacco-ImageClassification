import cv2
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from joblib import dump,load
import json

data = {"error": False, "data": []}
d={}

savedModel = 'saved_modeltest640.joblib'
dir_path = os.path.dirname(os.path.realpath(__file__))
# print(dir_path+'/Dataset/Kualitas-C/IMG_20191028_124902.jpg')
loadedModel = load(savedModel)


image_size = 640
image = cv2.imread(dir_path+'/Testset/Kualitas-B/IMG_20191028_122652.jpg')
image = cv2.resize(image, (image_size, image_size), 0, 0, cv2.INTER_LINEAR)
image = image.astype(np.float32)
image = np.multiply(image, 1.0 / 255.0)
print(image.reshape(1,-1).shape)
X_test = image.reshape(1,-1)

y_pred = loadedModel.predict(X_test)
d["prediksi"] = y_pred[0]
data["data"].append(d)
print(json.dumps(data))

print(y_pred)
print(image.shape)
cv2.putText(image, y_pred[0], (0,100), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,255,255), 3)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()



