import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
import huffmancodec as huffc

def entropyHuff (target, alfa):
    codec = huffc.HuffmanCodec.from_data(target) 
    symbols, lenghts = codec.get_code_len()

    ocorr = ocorrencias(target, alfa)
    tamanho = len(target)
    entropy = 0
    for idx in range(len(symbols)):
        prob = ocorr[symbols[idx]]/tamanho
        entropy += prob * lenghts[idx]
    return entropy

def entropy(target, alfa):
    # H(X) = -ΣP(i)*log2(P(i))

    return math.log2(np.max(target) - np.min(target))

    contador = ocorrencias(target,alfa)
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

def ocorrencias (target, alfa):
    contador = alfa
    for i in target:
        contador[i] += 1
    return contador

def ocorrenciasPlot (target, alfa, name):
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
    # xticks é usado para trocar as labels do x_values pelas do xAxis, tendo assim 
    # uma linha não interrompida de valores em xmas com as labels corretas do xAxis    
    plt.xticks(x_values, xAxis)
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

#compareMPG()
#ocorrenciasPlot(dataMatrix[:,0], alfa, varNames[0])
#weight = binning(dataMatrix[:,5], 200, np.min(dataMatrix[:,5]))
#displacement = binning(dataMatrix[:,2], 5, np.min(dataMatrix[:,2]))
#horsepower = binning(dataMatrix[:,3], 5, np.min(dataMatrix[:,3]))
#ocorrenciasPlot(weight, alfa, "Weight")
 
for i in range(6):
    print("normal", math.log2(np.max(dataMatrix[:,i]) - np.min(dataMatrix[:,i])))
    alfa = {key: 0 for key in range (np.min(dataMatrix), np.max(dataMatrix) + 1)}
    print("huffman", entropyHuff(dataMatrix[:,i], alfa)) 

alfa = {key: 0 for key in range (np.min(dataMatrix), np.max(dataMatrix) + 1)}
print("normal", entropy(np.reshape(dataMatrix, -1), alfa))
print("huffman", entropyHuff(np.reshape(dataMatrix, -1), alfa)) 


#plt.show()
