from sklearn import linear_model
import pandas as pd
import numpy as np

#read data 
data = pd.read_csv("train.csv", header = 0, index_col = 0)
Y_data = np.array(data['y'])
X_data = np.array(data[['x1', 'x2', 'x3', 'x4', 'x5']])

x = np.concatenate((X_data, np.square(X_data), np.exp(X_data),
                          np.cos(X_data), np.ones_like(X_data[:,0:1])), 
                          axis = 1)
model = linear_model.RidgeCV(alphas = [315.377], fit_intercept = False, 
                             gcv_mode = 'svd')
reg = model.fit(x, Y_data)

print(model.score(x, Y_data))

out_file = open("submission.csv", "w")
for c in reg.coef_:
	out_file.write(str(c) + "\n")
out_file.close()