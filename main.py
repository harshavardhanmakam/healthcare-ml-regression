import numpy as np
from sklearn.metrics import r2_score
from scipy import stats

# Dataset 1
los = [1,2,3,4,5,6,7,8,9,10]
cost = [2000,4000,6000,8500,11000,14000,18000,23000,29000,36000]

model = np.poly1d(np.polyfit(los, cost, 2))
print("Cost prediction:", model(8))
print("R²:", r2_score(cost, model(los)))