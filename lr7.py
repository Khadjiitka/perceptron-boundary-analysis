
import sys
import os
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


np.random.seed(42)  # Фіксуємо seed для відтворюваності випадкових точок

# Межі прямокутної області за умовою варіанту
x_min, x_max = -np.pi, np.pi
y_min, y_max = -3.5, 3.5

# Цільова нелінійна функція розділення: y = |x| * sin(x)
def target_function(x):
    return np.abs(x) * np.sin(x)

# Загальний об'єм точок
n_samples = 100

# Рівномірний розподіл точок у заданому прямокутнику
X_coords = np.random.uniform(x_min, x_max, n_samples)
Y_coords = np.random.uniform(y_min, y_max, n_samples)

# Об'єднуємо координати в матрицю ознак X (розмірність 100x2)
X = np.column_stack((X_coords, Y_coords))

# Формуємо вектор цільових класів y за геометричним правилом
y = np.where(Y_coords >= target_function(X_coords), 1, 0)

# Поділ даних на навчальну та тестову (контрольну) вибірки (75% / 25%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42
)

# Масштабування ознак (стандартизація дуже важлива для збіжності ваг MLP)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ПРОВЕДЕННЯ ЧИСЕЛЬНИХ ЕКСПЕРИМЕНТІВ

neurons_list = [2, 5, 10]
activations_list = ['identity', 'logistic', 'tanh', 'relu']

# Словник для накопичення результатів помилок
error_results = {act: [] for act in activations_list}

print(f"{'Кількість нейронів':<22} | {'Функція активації':<20} | {'Помилки класифікації (Test)':<30}")
print("-" * 78)

for activation in activations_list:
    for neurons in neurons_list:
        # Ініціалізуємо багатошаровий персептрон з 1 прихованим шаром
        mlp = MLPClassifier(
            hidden_layer_sizes=[neurons], 
            activation=activation, 
            solver='lbfgs',         # Стабільний квазіньютонівський метод для малих вибірок
            max_iter=20000, 
            random_state=42
        )
        
        # Навчання моделі
        mlp.fit(X_train_scaled, y_train)
        
        # Прогноз для контрольної вибірки
        y_pred = mlp.predict(X_test_scaled)
        
        # Розрахунок кількості помилок
        errors = np.sum(y_test != y_pred)
        
        # Збереження результату для графіка
        error_results[activation].append(errors)
        
        print(f"{neurons:<22} | {activation:<20} | {errors:<30}")

# ПОБУДОВА СІМЕЙСТВА ГРАФІКІВ ЗАЛЕЖНОСТЕЙ

plt.figure(figsize=(10, 6))

for activation in activations_list:
    plt.plot(
        neurons_list, 
        error_results[activation], 
        marker='o', 
        linewidth=2, 
        label=f"Активація: {activation}"
    )

plt.title("Залежність кількості помилок від кількості нейронів та функції активації (Варіант 30)")
plt.xlabel("Кількість нейронів у прихованому шарі")
plt.ylabel("Кількість помилкових класифікацій на тест-вибірці")
plt.xticks(neurons_list)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()


# ВІЗУАЛІЗАЦІЯ КЛАСІВ ТА ІДЕАЛЬНОЇ МЕЖІ РОЗДІЛУ

plt.figure(figsize=(8, 6))
x_grid = np.linspace(x_min, x_max, 400)
plt.plot(x_grid, target_function(x_grid), color='black', linestyle='--', linewidth=2, label='Межа y = |x|*sin(x)')

# Малюємо точки різних класів за допомогою індексації масивів
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='red', edgecolors='k', alpha=0.8, label='Клас 1 (Вище)')
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='blue', edgecolors='k', alpha=0.8, label='Клас 0 (Нижче)')

plt.title("Вихідний розподіл згенерованих точок та межа класів")
plt.xlabel("Координата X")
plt.ylabel("Координата Y")
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()