# Feature-Selection-para-Previs-o-de-Diabetes-Compara-o-de-M-todos-N-o-Supervisionados

## 1. Introdução ao Estudo

# Objetivo: 
Explorar e comparar técnicas de seleção de características não supervisionadas para otimizar a previsão de diabetes.

# Técnicas Analisadas:
Eliminação de Features por Variância.
Análise de Componentes Principais (PCA).

# Conjunto de Dados:
Utilizou-se o Pima Indians Diabetes Database, disponível no Kaggle.

## 2. Motivação para Seleção de Características
Por que Feature Selection?
Reduz a dimensionalidade dos dados, o que pode levar a:
Melhor desempenho computacional.
Redução de ruído.
Melhor interpretabilidade do modelo.
Foco em Técnicas Não Supervisionadas:
Útil para pré-processamento e para aumentar a generalização do modelo.

## 3. Métodos de Seleção de Características Utilizados

# A. Eliminação de Features por Variância
Descrição: 
remove características com baixa variabilidade entre os valores, pois essas características contribuem pouco para a distinção de instâncias.
Parâmetro de Corte: Definido para identificar e remover variáveis com baixa variância (0.02)



```
from sklearn.feature_selection import VarianceThreshold
import pandas as pd

# Definindo o limiar (queremos manter as variáveis que têm uma variância superior ao valor definido)
threshold = 0.02 #definimos 2% de variança 

selector = VarianceThreshold(threshold)
selector.fit_transform(abt_00)

# Colunas selecionadas
selected_features = abt_00.columns[selector.get_support()]
discarded_features = abt_00.columns[~selector.get_support()]
print('Variáveis que serão deletadas: ', discarded_features)

# Mantendo somente as variáveis selecionadas na ABT
abt_01 = abt_00[selected_features]

abt_01.head()

 ```

Variáveis Selecionadas:
Pregnancies, Glucose, BloodPressure, SkinThickness, Age

![image](https://github.com/user-attachments/assets/9f2bef7f-9467-4a22-9b8b-03afa8397070)


## B. Análise de Componentes Principais (PCA) 

Descrição: 
transforma as variáveis originais em componentes principais, mantendo a maior variabilidade possível dos dados.
Número de Componentes: Selecionados de modo a reter a maior parte da variância.

```
from sklearn.decomposition import PCA
import numpy as np
from sklearn.preprocessing import StandardScaler

X = abt_00
features = X.columns

# Padronizando os dados
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Aplicando PCA
pca = PCA(n_components=None) # Se None todos componentes vão ficar mantidos
pca.fit(X_scaled)

# Extraindo as cargas e criando um DataFrame
loadings = pca.components_
loading_df = pd.DataFrame(loadings, columns=features, index=['PC'+str(i) for i in range(1, loadings.shape[0]+1)])
loading_df_transposed = loading_df.transpose()

loading_df

```

Variáveis Selecionadas:
BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age

'''
#Aplicando PCA para selecionar variáveis
pca = PCA(n_components=5) # Se não todos componentes vão ficar mantidos
pca.fit(X_scaled)

#Extraindo as cargas e criando um DataFrame
loadings = pca.components_
loading_df = pd.DataFrame(loadings, columns=features, index=['PC'+str(i) for i in range(1, loadings.shape[0]+1)])
loading_df_transposed = loading_df.transpose()
loading_df_transposed
'''

![image](https://github.com/user-attachments/assets/139e5756-909b-4922-a56c-a996e468feaf)

## 4. Comparação dos Resultados
Método	Variáveis Selecionadas
- Eliminação por Variância:
Pregnancies, Glucose, BloodPressure, SkinThickness, Age

- Análise de Componentes Principais (PCA)
BloodPressure, SkinThickness, BMI, DiabetesPedigreeFunction, Age


# Observação:
Ambos os métodos retiveram variáveis importantes relacionadas ao risco de diabetes.
As variaveis BloodPressure , SkinThickness e age foram selecionadas por ambos os métodos, indicando sua importância.
