import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def ocorrencias (varIndex):
    contador = alfa
    for i in dataMatrix[:,varIndex]:
        contador[i] += 1
    return contador    

data = pd.read_excel('CarDataset.xlsx')
varNames = data.columns.values.tolist()
dataMatrix = data.to_numpy()

#MPG é o index 6 da lista
MPG = dataMatrix[:,6]

for i in range (6):
    plt.figure(1)
    plt.subplot(321+i)
    plt.scatter(dataMatrix[:,i], MPG, color = 'purple')
    plt.xlabel(varNames[i])
    plt.ylabel("MPG")
    plt.title("MPG vs. " + varNames[i])
    
plt.tight_layout()
plt.show()

dataMatrix = dataMatrix.astype("uint16") 
alfa = {key: 0 for key in range (np.min(dataMatrix), np.max(dataMatrix) + 1)}

#Falta entender o que dizer quanto à relação de MPG com as restantes variáveis