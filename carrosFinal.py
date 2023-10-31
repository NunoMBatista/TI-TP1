import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import math
import huffmancodec as huffc

def pearson(MPG, target):
    return np.corrcoef(target, MPG)[0][1]

def infoMut(MPG, target, alfa):    
    #ocorrTarget = ocorrencias(target, alfa)
    #ocorrMPG = ocorrencias(MPG, alfa)
    MPGEntropy = entropy(MPG, alfa)
    targetEntropy = entropy(target, alfa)
    tamanho = len(target)
    
    uniqueMPG = np.unique(MPG)
    uniqueTarget = np.unique(target)
    uniqueMPG = np.max(MPG) - np.min(MPG)
    uniqueTarget = np.max(target) - np.min(target)
    probMPGTarget = np.histogram2d(MPG, target, bins=(uniqueMPG, uniqueTarget))[0] / tamanho
    
    nonzeroIdx = probMPGTarget != 0
    probMPGTarget = probMPGTarget[nonzeroIdx]
    
    MPGTargetEntropy = -np.sum(probMPGTarget * np.log2(probMPGTarget))
        
    return MPGEntropy + targetEntropy - MPGTargetEntropy

# def infoMut(MPG, target, alfa):    
#     ocorrTarget = ocorrencias(target, alfa)
#     ocorrMPG = ocorrencias(MPG, alfa)
#     MPG = np.array(MPG)
#     target = np.array(target)
    
#     tamanhoTarget = len(target)    
#     ocorrMPGTarget = {(MPG[i], target[i]): np.sum((MPG == MPG[i]) & (target == target[i])) for i in range(tamanhoTarget)}
    
#     # A informação mútua é dada pela divergência Kullback Leibler de P(MPG, target) e P(MPG)P(Target) = ∑∑P(x, y)log2(P(x, y)/(P(x)*P(y))
#     #DKL = [(ocorrMPGTarget[(i, j)]/tamanhoMPGTarget)*math.log2((ocorrMPGTarget[(i, j)]/tamanhoMPGTarget) / ((ocorrMPG[i]/tamanhoTarget) * (ocorrTarget[j]/tamanhoTarget))) for i in MPG for j in target if (i, j) in ocorrMPGTarget]
#     DKL = [(ocorrMPGTarget[(MPG[i], target[i])]/tamanhoTarget)*math.log2((ocorrMPGTarget[(MPG[i], target[i])]/tamanhoTarget) / ((ocorrMPG[MPG[i]]/tamanhoTarget) * (ocorrTarget[target[i]]/tamanhoTarget))) for i in range(tamanhoTarget) if ocorrMPGTarget[(MPG[i], target[i])] * ocorrMPG[MPG[i]] * ocorrTarget[target[i]] > 0]

#     return sum(DKL)

def entropyHuff (target, alfa):
    codec = huffc.HuffmanCodec.from_data(target) 
    symbols, lengths = codec.get_code_len()

    ocorr = ocorrencias(target, alfa)   
    tamanho = len(target)

    probabilidades = np.array([(ocorr[symbols[x]]/tamanho)*lengths[x] for x in range(len(symbols))])
    entropy = np.sum(probabilidades)
    
    variancia = var(target, symbols, lengths, entropy, ocorr)   
    print("Variância de Comprimentos:", variancia)
    
    return entropy

def var(target, symbols, lengths, entropy, ocorr):
    tamanho = len(target)
    variance = 0
    for idx in range(len(symbols)):
        prob = ocorr[symbols[idx]] / tamanho
        # Calcula a diferença quadrática ponderada entre o comprimento e a entropia
        variance += prob * (lengths[idx] - entropy) ** 2
    return variance

def entropy(target, alfa):
    # H(X) = -ΣP(i)*log2(P(i))
    contador = ocorrencias(target, alfa)
    menor = min(target)
    maior = max(target)
    tamanho = len(target)
    
    probabilidades = np.array([(contador[x]/tamanho)*math.log2(contador[x]/tamanho) for x in range(menor, maior+1) if (contador[x] > 0)])
    
    return -np.sum(probabilidades)  
    
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

def ocorrenciasPlot (target, alfa, name, tickInterval, figura):
    contador = ocorrencias(target, alfa)
    xAxis = [x for x in contador.keys() if contador[x] > 0]
    yAxis = []
    for i in xAxis:
        yAxis.append(contador[i]) 
        
    # O x_values é uma linha de valores para o eixo X sem espaços vazios
    x_values = np.arange(len(xAxis))
    plt.figure(figura)
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
    targetAlfa = {key: 0 for key in range(0, lastAlfa + 1)}
    ocorr = ocorrencias(target, targetAlfa)
    binningN = math.ceil(len(list(ocorr.keys())) / n)
    print(len(list(ocorr.keys())), binningN)
    for i in range(binningN):
        binn = list(ocorr.keys())[i * n : ((i+1) * n)]
        replacement = max(ocorr, key = lambda k: ocorr[k] if k in binn else -1)
        mask = (target >= np.min(binn)) & (target <= np.max(binn))
        target[mask] = replacement
    return target

def MAE (MPG, target):
    target = np.array(target)
    MPG = np.array(MPG)
    return np.mean(np.abs(MPG-target))

data = pd.read_excel('CarDataset.xlsx')
varNames = data.columns.values.tolist()
dataMatrix = data.to_numpy()
dataMatrix = dataMatrix.astype("uint16")

# Definir alfabeto
alfa = {key: 0 for key in range(0, 65535)} #todos os Uint16

acceleration = np.copy(dataMatrix[:,0])
cylinders = np.copy(dataMatrix[:,1])
distance = np.copy(dataMatrix[:,2])
horsepower = np.copy(dataMatrix[:,3])
model = np.copy(dataMatrix[:,4])
weight = np.copy(dataMatrix[:,5])

print("Entropia da matriz inteira antes do binning: ", entropy(np.reshape(dataMatrix, -1), alfa), "\n")

compareMPG(dataMatrix, varNames)
ocorrenciasPlot(acceleration, alfa, "Acceleration", 1, 2)

weight = binning(weight, 40, np.min(weight))
ocorrenciasPlot(weight, alfa, "Weight", 7, 3)
distance = binning(distance, 5, np.min(distance))
ocorrenciasPlot(distance, alfa, "Distance", 5, 4)
horsepower = binning(horsepower, 5, np.min(horsepower))
ocorrenciasPlot(horsepower, alfa, "Horse Power", 3, 5)

MPG = np.copy(dataMatrix[:,6])
dataMatrixBinn = [acceleration, cylinders, distance, horsepower, model, weight, MPG]

IMarray = []
for i in range(7):
    IMarray.append(infoMut(MPG, dataMatrixBinn[i], alfa))
print(IMarray)

for i in range(7):
    print(varNames[i])
    print("Nº Médio de bits com símbolos equiprováveis:", mediaBits(dataMatrixBinn[i], alfa))
    #print("Entropia normal antes do binning:", entropy(dataMatrix[:,i], alfa))
    print("Entropia normal após binning:", entropy(dataMatrixBinn[i], alfa))
    #print("Entropia de Huffman antes do binning:", entropyHuff(dataMatrix[:,i], alfa)) 
    print("Entropia de Huffman após binning:", entropyHuff(dataMatrixBinn[i], alfa)) 
    print("Relação de " + varNames[i] + " com MPG:", pearson(MPG, dataMatrixBinn[i]))
    print("Informação mútua com MPG: ", IMarray[i])
    print("\n\n")

MPGpred = [-5.5241 - 0.146 * acceleration[i] - 0.4909 * cylinders[i] + 0.0026 * distance[i] - 0.0045 * horsepower[i] + 0.6725 * model[i] - 0.0059 * weight[i] for i in range(len(MPG))]
print("Erro de MPGpred com todas as variáveis: ", MAE(MPG, MPGpred), "MPG previsto: ", np.average(MPGpred))

MPGpred = [-5.5241 - 0.146 * acceleration[i] - 0.4909 * cylinders[i] + 0.0026 * distance[i] - 0.0045 * horsepower[i] + 0.6725 * model[i] - 0.0059 * 0 for i in range(len(MPG))]
print("Erro de MPGpred sem variável de maior MI: ", MAE(MPG, MPGpred), "MPG previsto: ", np.average(MPGpred))

MPGpred = [-5.5241 - 0.146 * 0 - 0.4909 * cylinders[i] + 0.0026 * distance[i] - 0.0045 * horsepower[i] + 0.6725 * model[i] - 0.0059 * weight[i] for i in range(len(MPG))]
print("Erro de MPGpred sem variável de menor MI: ", MAE(MPG, MPGpred), "MPG previsto: ", np.average(MPGpred))

#plt.show()
    