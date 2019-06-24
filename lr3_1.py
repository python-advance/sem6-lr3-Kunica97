import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

def gradient(x, y, alpha, n):
    m = len(x)
    theta_0, theta_1 = 0, 0
    for i in range(n):
        sum_1 = 0
        for i in range(m):
            sum_1 += theta_0 + theta_1 * x[i] - y[i]
        temp1 = theta_0 - alpha * (1 / m) * sum_1
        sum_2 = 0
        for i in range(m):
            sum_2 += (theta_0 + theta_1 * x[i] - y[i]) * x[i]
        temp2 = theta_1 - alpha * (1 / m) * sum_2
        theta_0, theta_1 = temp1, temp2
    return theta_0, theta_1

data = pd.read_csv('ex1data1.csv', header=None)
x, y = data[0], data[1]
list_x, list_y = list(x), list(y)
np_x1, np_y1 = np.array(list_x), np.array(list_y)

x1, y1 = [1, 25], [0, 0]
theta_0, theta_1 = gradient(x, y, 0.001, len(x))
y1[0], y1[1] = theta_0 + x1[0] * theta_1, theta_0 + x1[1] * theta_1

plt.plot(x1, y1, 'b', label = u'без Polyfit')
plt.scatter(x, y, label = u'данные из csv', color='black')

np_x, np_y = np.array(x), np.array(y)
new_theta1, new_theta0 = (np.polyfit(np_x, np_y, 1)).tolist()

new_y1 = [0, 0]
new_y1[0], new_y1[1] = new_theta0 + x1[0] * new_theta1, new_theta0 + x1[1] * new_theta1
plt.plot(x1, new_y1, 'red', label = u'используя Polyfit')

print(new_theta0, new_theta1)

# Предсказание
model = LinearRegression() # экземпляр класса LinearRegression
n_x = np_x1.reshape(-1, 1) # меняем форму данных n_x
n_y = np_y1.reshape(-1, 1) # аналогично, но для n_y
model.fit(n_x, n_y) # обучаем модель используя данные из csv файла

x_pr = np.array([2.225, 17.5, 25]) # данные из задания
new_x_pr = x_pr.reshape(-1, 1) # меняем форму данных
y_pr = model.predict(new_x_pr) # предсказываем
y_pr = y_pr.flatten() # выравниваем в "плоский" вид

plt.scatter(new_x_pr, y_pr, label = u'предсказание', color='y')

plt.title('Линейная регрессия с одной переменной. Градиентный спуск.')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()