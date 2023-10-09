import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np

def compareMPG():
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

def ocorrencias (varIndex, alfa):
    contador = alfa
    for i in dataMatrix[:,varIndex]:
        contador[i] += 1
    return contador    

def ocorrenciasPlot (varIndex, alfa):
    contador = ocorrencias(varIndex, alfa)
    xAxis = [x for x in contador.keys() if contador[x] > 0]
    yAxis = []
    for i in xAxis:
        yAxis.append(contador[i])
        
    # O x_values é uma linha de valores para o eixo X sem espaços vazios
    x_values = np.arange(len(xAxis))

    plt.figure(2)
    plt.bar(x_values, yAxis, color = "red")
    plt.xlabel(varNames[varIndex])
    plt.ylabel("Count")
    # xticks é usado para trocar as labels do x_values pelas do xAxis, tendo assim 
    # uma linha não interrompida de valores em xmas com as labels corretas do xAxis    
    plt.xticks(x_values, xAxis)
    plt.axis("tight")
    plt.tight_layout()

def binning (varIndex, n):
    target = dataMatrix[:,varIndex]
    binningN = len(target) // n
    #Dividir weight em binningN subarrays
    binnings = [target[i * n : (i + 1) * n] for i in range(binningN)]

    for binning in binnings: 
        count = np.bincount(binning)
        replacement = np.argmax(count)
        binning[True] = replacement
     
    target = np.reshape(target, -1)
    return target

data = pd.read_excel('CarDataset.xlsx')
varNames = data.columns.values.tolist()
dataMatrix = data.to_numpy()

dataMatrix = dataMatrix.astype("uint16") 
alfa = {key: 0 for key in range (np.min(dataMatrix), np.max(dataMatrix) + 1)}

ocorrenciasPlot(2, alfa)
compareMPG()

weight = binning(6, 40)
displacement = binning(2, 5)
horsepower = binning(3, 5)

plt.show()
#Falta entender o que dizer quanto à relação de MPG com as restantes variáveis