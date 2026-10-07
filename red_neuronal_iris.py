import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

# ==========================================
# 1. CARGA Y EXPLORACIÓN
# ==========================================
iris = load_iris()
df_iris = pd.DataFrame(iris.data, columns=iris.feature_names)
df_iris['especie'] = pd.Categorical.from_codes(iris.target, iris.target_names)

print("--- 1. PRIMERAS FILAS DEL DATASET IRIS ---")
print(df_iris.head(), "\n")

# ==========================================
# 2. PREPROCESAMIENTO
# ==========================================
# División 80% entrenamiento / 20% prueba (estratificada)
X_train, X_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, stratify=iris.target, random_state=42
)

# Estandarización (media 0, varianza 1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==========================================
# 3. MODELO: PERCEPTRÓN MULTICAPA (MLP)
# ==========================================
mlp = MLPClassifier(
    hidden_layer_sizes=(10,), 
    activation='relu', 
    solver='adam', 
    max_iter=500, 
    random_state=42
)
mlp.fit(X_train_scaled, y_train)

# ==========================================
# 4. EVALUACIÓN
# ==========================================
y_pred = mlp.predict(X_test_scaled)

print("--- 4. REPORTE DE CLASIFICACIÓN ---")
print(classification_report(y_test, y_pred, target_names=iris.target_names), "\n")

# ==========================================
# 6. INFERENCIA Y ARQUITECTURA (Pesos y Sesgos)
# ==========================================
flor_desconocida = np.array([[5.1, 3.5, 1.4, 0.2]])
flor_escalada = scaler.transform(flor_desconocida)

probabilidades = mlp.predict_proba(flor_escalada)[0]
clase_predicha = iris.target_names[np.argmax(probabilidades)]

print("--- 6. INFERENCIA DE NUEVA FLOR ---")
print(f"Medidas de la flor: {flor_desconocida[0]}")
for especie, prob in zip(iris.target_names, probabilidades):
    print(f"Probabilidad de ser {especie}: {prob*100:.2f}%")
print(f"-> Predicción final: {clase_predicha.upper()}\n")

print("--- DIMENSIONES DE MATRICES INTERNAS ---")
print(f"Pesos Capa Oculta (W1): {mlp.coefs_[0].shape} -> 4 entradas x 10 neuronas")
print(f"Sesgos Capa Oculta (b1): {mlp.intercepts_[0].shape} -> 10 neuronas")
print(f"Pesos Capa Salida (W2): {mlp.coefs_[1].shape} -> 10 neuronas x 3 salidas")
print(f"Sesgos Capa Salida (b2): {mlp.intercepts_[1].shape} -> 3 salidas\n")

# ==========================================
# 5 Y 7. VISUALIZACIÓN Y GRAFICADO DE LA RED
# ==========================================
fig = plt.figure(figsize=(18, 5))

# Gráfico A: Curva de pérdida
ax1 = fig.add_subplot(131)
ax1.plot(mlp.loss_curve_, color='purple', lw=2)
ax1.set_title('Curva de Pérdida durante el Entrenamiento')
ax1.set_xlabel('Iteraciones (Épocas)')
ax1.set_ylabel('Pérdida (Error)')
ax1.grid(True, alpha=0.3)

# Gráfico B: Matriz de Confusión
ax2 = fig.add_subplot(132)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=ax2)
ax2.set_title('Matriz de Confusión')
ax2.set_xlabel('Predicción del Modelo')
ax2.set_ylabel('Valor Real')

# Gráfico C: Arquitectura de la Red Neuronal (NetworkX)
ax3 = fig.add_subplot(133)
G = nx.DiGraph()
capas = [4, 10, 3] # Entradas, Oculta, Salida
posiciones = {}
nodo_actual = 0

for i, num_nodos in enumerate(capas):
    espaciado_y = np.linspace(-1, 1, num_nodos)
    for j in range(num_nodos):
        G.add_node(nodo_actual, layer=i)
        posiciones[nodo_actual] = (i, espaciado_y[j])
        nodo_actual += 1

# Conectar capas
nodos = list(G.nodes(data='layer'))
for i in range(len(capas) - 1):
    nodos_capa_actual = [n for n, d in nodos if d == i]
    nodos_siguiente_capa = [n for n, d in nodos if d == i + 1]
    for n1 in nodos_capa_actual:
        for n2 in nodos_siguiente_capa:
            G.add_edge(n1, n2)

nx.draw(G, posiciones, ax=ax3, node_size=150, node_color=['lightgreen' if d==0 else 'lightblue' if d==1 else 'salmon' for n, d in nodos], 
        edge_color='gray', width=0.5, arrowsize=5)
ax3.set_title('Arquitectura de la Red (4 → 10 → 3)')

plt.tight_layout()
plt.show()