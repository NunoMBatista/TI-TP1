import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
import huffmancodec as huffc

def pearson(MPG, target):
    return np.corrcoef(target, MPG)[0][1]

def infoMut(target, MPG, alfa):
    tamanhoMPG = len(MPG)
    ocorrMPG = ocorrencias(MPG, alfa)
    
    tamanhoTarget = len(target)
    ocorrTarget = ocorrencias(target, alfa)
    
    entropyMPG = entropy(MPG, alfa)
    entropyTarget = entropy(target, alfa)
    
    return entropyMPG + entropyTarget - entropyConj(target, MPG, alfa)
    

def entropyConj(targetX, targetY, alfa):
    ocorrX = ocorrencias(targetX, alfa)
    ocorrY = ocorrencias(targetY, alfa)
    
    menorX = min(targetX)
    maiorX = max(targetX)
    
    menorY = min(targetY)
    maiorY = max(targetY)
    
    tamanho = len(targetX)
    
    ent = 0
    for x in range(menorX, maiorX):
        for y in range(menorY, maiorY):
            probXY = probConj(targetX, targetY, x, y)
            if probXY > 0:
                ent += probXY * math.log2(probXY)
    return -ent
    
def probConj(target1, target2, x, y):
    size = len(target1)
    ocorrPar = [(target1[i], target2[i]) for i in range(size)]
    return ocorrPar.count((x, y))/size
                  
def entropyHuff (target, alfa, var):
    codec = huffc.HuffmanCodec.from_data(target) 
    symbols, lenghts = codec.get_code_len()

    ocorr = ocorrencias(target, alfa)   
    tamanho = len(target)
    entropy = 0
    for idx in range(len(symbols)):
        prob = (ocorr[symbols[idx]] ** max(2*var, 1)) / tamanho
        entropy += prob * lenghts[idx]
    return entropy

def entropy(target, alfa):
    # H(X) = -ΣP(i)*log2(P(i))
    contador = ocorrencias(target, alfa)
    menor = min(target)
    maior = max(target)
    tamanho = len(target)
    
    ent = 0
    for i in range(menor, maior):
        prob = contador[i]/tamanho
        #log2(0) não é possível
        if prob > 0:
            ent += prob * math.log2(prob)
    return -ent

def mediaBits (target, alfa):
    contador = ocorrencias(target, alfa)
    nAlfa = len([x for x in contador.values() if x > 0])
    return math.log2(nAlfa)

def compareMPG(dataMatrix, varNames):
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

def ocorrencias (target, alfa):
    contador = alfa.copy()
    for i in target:
        contador[i] += 1
    return contador

def ocorrenciasPlot (target, alfa, name, tickInterval):
    contador = ocorrencias(target, alfa)
    xAxis = [x for x in contador.keys() if contador[x] > 0]
    yAxis = []
    for i in xAxis:
        yAxis.append(contador[i]) 
    # O x_values é uma linha de valores para o eixo X sem espaços vazios
    x_values = np.arange(len(xAxis))
    plt.figure(2)
    plt.bar(x_values, yAxis, color = "red")
    plt.xlabel(name)
    plt.ylabel("Count")
    # xticks é usado para   trocar as labels do x_values pelas do xAxis, tendo assim 
    # uma linha não interrompida de valores em xmas com as labels corretas do xAxis    
    
    tickPos = np.arange(0, len(xAxis), tickInterval)
    tickLabels = [xAxis[i] for i in tickPos]
    
    plt.xticks(tickPos, tickLabels)
    plt.axis("tight")
    plt.tight_layout()

def binning (target, n, firstAlfa):
    lastAlfa = np.max(target)
    targetAlfa = {key: 0 for key in range(firstAlfa, lastAlfa + 1)}
    ocorr = ocorrencias(target, targetAlfa)
    binningN = len(list(ocorr.keys())) // n
    for i in range(binningN):
        binn = list(ocorr.keys())[i * n : (i+1) * n]
        replacement = max(ocorr, key = lambda k: ocorr[k] if k in binn else -1)
        mask = (target >= np.min(binn)) & (target <= np.max(binn))
        target[mask] = replacement

    return target

data = pd.read_excel('CarDataset.xlsx')
varNames = data.columns.values.tolist()
dataMatrix = data.to_numpy()
dataMatrix = dataMatrix.astype("uint16") 

# Definir alfabeto
alfa = {key: 0 for key in range (np.min(dataMatrix), np.max(dataMatrix) + 1)}

acceleration = dataMatrix[:,0]
cylinders = dataMatrix[:,1]
distance = dataMatrix[:,2]
horsepower = dataMatrix[:,3]
model = dataMatrix[:,4]
weight = dataMatrix[:,5]

#compareMPG(dataMatrix, varNames)
#ocorrenciasPlot(dataMatrix[:,5], alfa, varNames[5], 10)
ocorrenciasPlot(acceleration, alfa, "acceleration", 1)
#weight = binning(dataMatrix[:,5], 200, np.min(dataMatrix[:,5]))
#displacement = binning(dataMatrix[:,2], 5, np.min(dataMatrix[:,2]))
#horsepower = binning(dataMatrix[:,3], 5, np.min(dataMatrix[:,3]))

MPG = dataMatrix[:,6]
IMarray = []
# for i in range(6):
#     print(varNames[i])
#     print("Nº Médio de bits com símbolos equiprováveis:", mediaBits(dataMatrix[:,i], alfa))
#     print("Entropia normal:", entropy(dataMatrix[:,i], alfa))
#     huffEntropy = entropyHuff(dataMatrix[:,i], alfa, 0)
#     print("Entropia de Huffman:", huffEntropy) 
#     print("Relação de " + varNames[i] + " com MPG:", pearson(MPG, dataMatrix[:,i]))
#     print("Informação mutua com MPG:", infoMut(dataMatrix[:,i], MPG, alfa))
#     print("\n\n")

# for i in range(6):
#     IMarray.append(infoMut(MPG, dataMatrix[:,i], alfa))

# MPGpred = -5.5241 - 0.146 * IMarray[0] - 0.4909 * IMarray[1] + 0.0026 * IMarray[2] - 0.0045 * IMarray[3] + 0.6725 * IMarray[4] - 0.0059 * IMarray[5]
# print(MPGpred)
# MPGpred = -5.5241 - 0.146 * 0 - 0.4909 * IMarray[1] + 0.0026 * IMarray[2] - 0.0045 * IMarray[3] + 0.6725 * IMarray[4] - 0.0059 * IMarray[5]
# print(MPGpred)
# MPGpred = -5.5241 - 0.146 * IMarray[0] - 0.4909 * IMarray[1] + 0.0026 * IMarray[2] - 0.0045 * IMarray[3] + 0.6725 * IMarray[4] - 0.0059 * 0
# print(MPGpred)

plt.show()

#CALCULAR VARIÂNCIA NO EXERCÍCIO 8