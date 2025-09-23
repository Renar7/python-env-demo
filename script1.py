# test_team1.py
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("--- Тест для окружения Команды 1 (pip + venv) ---")
print("Загрузка данных... Это может занять несколько секунд.")

# Генерируем синтетические данные, чтобы не зависеть от интернета во время демонстрации
np.random.seed(42)
salinity = 34.5 + np.random.rand(500) * 0.5
temperature = 15 - (salinity - 34.5) * 20 + np.random.randn(500) * 0.5
df_sample = pd.DataFrame({'Sal': salinity, 'Temp': temperature})

print("Данные сгенерированы. Первые 5 строк:")
print(df_sample.head())

print("\nСтроим график зависимости...")
sns.scatterplot(x="Sal", y="Temp", data=df_sample)
plt.title("Зависимость температуры от солености (сгенерированные данные)")
plt.show()

X = df_sample[['Sal']]
y = df_sample['Temp']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

regr = LinearRegression()
regr.fit(X_train, y_train)
score = regr.score(X_test, y_test)
print(f"\nМодель обучена. Качество модели (R^2 score): {score:.4f}")
print("--- Тест успешно завершен ---")