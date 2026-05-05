import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Dataset 1
los = [1,2,3,4,5,6,7,8,9,10]
cost = [2000,4000,6000,8500,11000,14000,18000,23000,29000,36000]

model = np.poly1d(np.polyfit(los, cost, 2))

line = np.linspace(min(los), max(los), 100)

print("Prediction:", model(8))
print("R²:", r2_score(cost, model(los)))

plt.scatter(los, cost)
plt.plot(line, model(line))
plt.xlabel("Length of Stay")
plt.ylabel("Cost")
plt.title("LOS vs Cost")
plt.show()
