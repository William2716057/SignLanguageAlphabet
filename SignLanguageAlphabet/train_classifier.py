#train classifier 
#load the data and train a classifier with it 

import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

data_dict = pickle.load(open('./data.pickle', 'rb'))
#convert to numpy array
data = np.asarray(data_dict['data'])
labels = np.asarray(data_dict['labels'])



#split data into training set and test set
#With n_samples=1, test_size=0.2 and train_size=None, the resulting train set will be empty. Adjust any of the aforementioned parameters.
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

model = RandomForestClassifier()

model.fit(x_train, y_train)

y_predict = model.predict(x_test)

score = accuracy_score(y_predict, y_test)

print('{}% of samples were correctly classified'.format(score * 100))

#save model
f = open('model.p', 'wb') 
pickle.dump({'model': model}, f)
f.close()