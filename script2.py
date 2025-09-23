# test_team2.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("--- Тест для окружения Команды 2 (Anaconda) ---")
# Генерируем данные: x - опыт работы в годах, y - зарплата
np.random.seed(0)
x = np.array(range(1, 11)).reshape(-1, 1)
y = 50 + x * 10 + np.random.randn(10, 1) * 5

model = LinearRegression()
model.fit(x, y)

print(f"Коэффициент (наклон прямой): {model.coef_[0][0]:.2f}")
print(f"Пересечение (начальная точка): {model.intercept_[0]:.2f}")

plt.scatter(x, y, color='blue', label='Исходные данные')
plt.plot(x, model.predict(x), color='red', label='Линия регрессии')
plt.title("Модель линейной регрессии")
plt.xlabel("Опыт работы (годы)")
plt.ylabel("Зарплата (тыс.)")
plt.legend()
plt.grid(True)
print("График готов. Закройте окно с графиком для завершения.")
plt.show()
print("--- Тест успешно завершен ---")