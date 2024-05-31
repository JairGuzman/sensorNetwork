import numpy as np #Util para realizar calculos avanzados
import pandas as pd #Contiene funciones que nos ayudan en el analisis de datos
import matplotlib.pyplot as plt #Para crear gráficos de buena calidad
import scipy.cluster.hierarchy as sch # Contiene funciones para realizar el clustering jerárquico
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

# %matplotlib inline
cl_arrays = []
minute = 1;
while minute < 12:
  url = 'https://raw.githubusercontent.com/JairGuzman/datasets/main/minute'+ str(minute) + '.csv'
  estudiantes=pd.read_csv(url, engine='python', index_col=0)
  lecturas = estudiantes.iloc[:, [0,1,2,3,4,5,6,7,8,9,10,11]].values
  clustering = linkage(lecturas, 'ward') #Utilizamos el metodo Ward para agrupar los clusters
  cl = fcluster(clustering, t=770, criterion='distance')
  # clustering = linkage(lecturas, 'ward') #Utilizamos el metodo Ward para agrupar los clusters
  # dendograma = sch.dendrogram(clustering)
  # plt.savefig('dendograma-minute'+ str(minute) +'.png')  # Gua
  # plt.title('Dendrograma')
  # plt.xlabel('Estudiantes')
  # plt.ylabel('Distancias Euclidianas')
  # plt.show()
  #cl = fcluster(clustering, t=1000, criterion='distance')
  cl_arrays.append(cl)
  minute += 1

cl_matrix = np.array(cl_arrays)
print(cl_matrix)

#mostrar como un mapa de calor en base a la matriz de clusters

plt.imshow(cl_matrix, aspect='auto', cmap='viridis')
plt.title('Clusters')
plt.xlabel('Estudiantes')
plt.ylabel('Minutos')
plt.xticks(np.arange(0, len(cl - 1), 1)) 
plt.yticks(np.arange(0, 12, 1))
plt.show()

#plt.colorbar()
#plt.grid(True)

