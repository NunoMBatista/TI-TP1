import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_excel('CarDataset.xlsx')
varNames = data.columns.values.tolist()
dataMatrix = data.to_numpy()
#sdfs


#MPG é o index 6 da lista
MPG = dataMatrix[:,6]

for i in range (6):
    plt.subplot(321+i)
    plt.scatter(dataMatrix[:,i], MPG, color = 'purple')
    plt.xlabel(varNames[i])
    plt.ylabel("MPG")
    plt.title("MPG vs. " + varNames[i])
    
plt.tight_layout()
plt.show()

print(np.max(dataMatrix))