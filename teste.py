import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def ocorrencias (varIndex):
    contador = alfa
    for i in dataMatrix[:,varIndex]:
        contador[i] += 1
    return contador    

def ocorrenciasPlot (varIndex):
    contador = ocorrencias(varIndex)
    xAxis = [x for x in contador.keys() if contador[x] > 0]
    yAxis = []
    for i in xAxis:
        yAxis.append(contador[i])
  
    x_values = np.arange(len(xAxis))
    filtered_yAxis = [y for y in yAxis if y > 0]
    
    print(xAxis)
    print(yAxis)
    
    plt.figure(2)
    plt.bar(x_values, yAxis, color="blue")
    plt.xlabel(varNames[varIndex])
    plt.ylabel("Count")
    plt.xticks(x_values, xAxis)
    plt.axis("tight")

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

dataMatrix = dataMatrix.astype("uint16") 
alfa = {key: 0 for key in range (np.min(dataMatrix), np.max(dataMatrix) + 1)}

ocorrenciasPlot(0)
plt.show()

#Falta entender o que dizer quanto à relação de MPG com as restantes variáveis