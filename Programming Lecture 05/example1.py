# an example python script

import numpy as np


data = np.loadtxt('prog_lec_05.txt')
C = data[:, 0]
X = data[:, 1:]

# form training data
X_train = X[0:100, :]
C_train = C[0:100]

X_valid = X[100:, :]
C_valid = X[100:, :]

# generate 20 random points as fixed points
Z = np.random.normal(20, 1)
Phi_train = form_gaussian_rbf_design_matrix(X_train, Z)
Phi_valid = form_gaussian_rbf_design_matrix(X_valid, Z)


# Normalize
mu_Phi = np.mean(Phi_train)
sig_Phi = np.std(Phi_train)
Phi_train = (Phi_train - mu_Phi)/sig_Phi
Phi_valid = (Phi_valid - mu_Phi)/sig_Phi

# logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(Phi_train, C_train)

# compare
C_valid_model = model.predict(Phi_valid)

# calculate performance metrics
p = precision(C_valid_model, C_valid)
r = recall(C_valid_model, C_valid)
F1_score = F_beta(C_valid_model, C_valid, beta=1)