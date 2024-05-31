import numpy as np #Util para realizar calculos avanzados
import pandas as pd #Contiene funciones que nos ayudan en el analisis de datos
import matplotlib.pyplot as plt #Para crear gráficos de buena calidad
import scipy.cluster.hierarchy as sch # Contiene funciones para realizar el clustering jerárquico
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

minute = 1;
while minute < 12:
  url = 'https://raw.githubusercontent.com/JairGuzman/datasets/main/minute'+ str(minute) + '.csv' # URL del archivo CSV
  estudiantes=pd.read_csv(url, engine='python', index_col=0) #Leemos el archivo CSV
  lecturas = estudiantes.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values #Seleccionamos las columnas que contienen las lecturas
  clustering = linkage(lecturas, 'ward') #Realizamos el clustering jerárquico con el método de Ward
  dendograma = sch.dendrogram(clustering) #Generamos el dendograma
  plt.title('Dendrograma ' + str(minute)) 
  plt.xlabel('Estudiantes')
  plt.ylabel('Distancia Euclidiana')
  plt.show()

  minute += 1

