import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

# ==========================================
# FUNCIÓN PARA DIBUJAR LOS GRÁFICOS
# ==========================================
def graficar_frontera(modelo, X, y, ax, titulo):
    xx, yy = np.meshgrid(np.arange(-0.5, 1.5, 0.01), np.arange(-0.5, 1.5, 0.01))
    Z = modelo.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, s=100, edgecolors='k', cmap='coolwarm')
    ax.set_title(titulo)
    ax.set_xlabel('Entrada 1')
    ax.set_ylabel('Entrada 2')
    ax.grid(True, linestyle='--', alpha=0.3)

# ==========================================
# DATASET
# ==========================================
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])
y_or = np.array([0, 1, 1, 1])
y_xor = np.array([0, 1, 1, 0])

# ==========================================
# ENTRENAMIENTO Y RESULTADOS EN TERMINAL
# ==========================================
print("==========================================")
print("1. COMPUERTA AND (Perceptrón)")
print("==========================================")
neurona_and = Perceptron(max_iter=100, random_state=42)
neurona_and.fit(X, y_and)
pred_and = neurona_and.predict(X)

for entrada, objetivo, pred in zip(X, y_and, pred_and):
    print(f"Entrada: {entrada} -> Esperado: {objetivo} | Predicho: {pred}")
print(f"Precisión: {accuracy_score(y_and, pred_and) * 100}%\n")


print("==========================================")
print("2. COMPUERTA OR (Perceptrón)")
print("==========================================")
neurona_or = Perceptron(max_iter=100, random_state=42)
neurona_or.fit(X, y_or)
pred_or = neurona_or.predict(X)

for entrada, objetivo, pred in zip(X, y_or, pred_or):
    print(f"Entrada: {entrada} -> Esperado: {objetivo} | Predicho: {pred}")
print(f"Precisión: {accuracy_score(y_or, pred_or) * 100}%\n")


print("==========================================")
print("3. COMPUERTA XOR (Red Multicapa)")
print("==========================================")
red_xor = MLPClassifier(hidden_layer_sizes=(4,), activation='relu', solver='lbfgs', max_iter=2000, random_state=42)
red_xor.fit(X, y_xor)
pred_xor = red_xor.predict(X)

for entrada, objetivo, pred in zip(X, y_xor, pred_xor):
    print(f"Entrada: {entrada} -> Esperado: {objetivo} | Predicho: {pred}")
print(f"Precisión: {accuracy_score(y_xor, pred_xor) * 100}%\n")

# ==========================================
# GENERACIÓN DE GRÁFICOS
# ==========================================
print("Generando ventana de gráficos...")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

graficar_frontera(neurona_and, X, y_and, ax1, "Compuerta AND\n(Línea Recta - Perceptrón)")
graficar_frontera(neurona_or, X, y_or, ax2, "Compuerta OR\n(Línea Recta - Perceptrón)")
graficar_frontera(red_xor, X, y_xor, ax3, "Compuerta XOR\n(No lineal - Red Multicapa)")

plt.tight_layout()
plt.show()