import pandas as pd 
import matplotlib.pyplot as plt

data = pd.read_excel('CarDataset.xlsx')
varNames = data.columns.values.tolist()
print(data)
dataMatrix = data.to_numpy()

#MPG é o index 6 da lista
MPG = dataMatrix[:,6]

for i in range (6):
    plt.figure(2)
    plt.subplot(321+i)
    plt.scatter(dataMatrix[:,i], MPG, color = 'purple')
    plt.xlabel(varNames[i])
    plt.ylabel("MPG")
    plt.title("MPG vs. " + varNames[i])
    
plt.show()

print(MPG)