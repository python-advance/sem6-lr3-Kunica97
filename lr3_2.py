import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.linear_model import LinearRegression

def sq_error(x, y, f_x=None):
    '''
    Функция для подсчета среднеквадратической ошибки.
    Возвращает сумму.
    '''
    sq_er = []
    for i in range(len(x)):
        sq_er.append((f_x(x[i]) - y[i])**2)
    return sum(sq_er)

data = pd.read_csv('web_traffic.tsv', sep='\t', header=None)

x, y = data[0], data[1]
list_x, list_y = list(x), list(y)

for i in range(len(list_y)):
    if math.isnan(list_y[i]) or math.isnan(list_x[i]):
        list_y[i] = 0
        list_x[i] = 0
list_y.remove(0), list_x.remove(0)

x1 = [1, 743]

np_x, np_y = np.array(list_x), np.array(list_y)

x2 = list(range(743))

theta1_0, theta1_1 = sp.polyfit(np_x, np_y, 1)
theta2_0, theta2_1, theta2_2 = sp.polyfit(np_x, np_y, 2)
theta3_0, theta3_1, theta3_2, theta3_3 = sp.polyfit(np_x, np_y, 3)
theta4_0, theta4_1, theta4_2, theta4_3, theta4_4 = sp.polyfit(np_x, np_y, 4)
theta5_0, theta5_1, theta5_2, theta5_3, theta5_4, theta5_5 = sp.polyfit(np_x, np_y, 5)

plt.scatter(list_x, list_y, label = u'Исходные данные', color='black')

f1 = sp.poly1d(sp.polyfit(np_x, np_y, 1))
plt.plot(x2, f1(x2), linewidth = 2, label = u'полином 1-ой степени')
f2 = sp.poly1d(sp.polyfit(np_x, np_y, 2))
plt.plot(x2, f2(x2), linewidth = 2, label = u'полином 2-ой степени')
f3 = sp.poly1d(sp.polyfit(np_x, np_y, 3))
plt.plot(x2, f3(x2), linewidth = 2, label = u'полином 3-ей степени')
f4 = sp.poly1d(sp.polyfit(np_x, np_y, 4))
plt.plot(x2, f4(x2), linewidth = 2, label = u'полином 4-ой степени')
f5 = sp.poly1d(sp.polyfit(np_x, np_y, 5))
plt.plot(x2, f5(x2), linewidth = 2, label = u'полином 5-ой степени')

function_1 = lambda x: theta1_0*x + theta1_1
function_2 = lambda x: theta2_0*x**2 + theta2_1*x + theta2_2
function_3 = lambda x: theta3_0*x**3 + theta3_1*x**2 + theta3_2*x + theta3_3
function_4 = lambda x: theta4_0*x**4 + theta4_1*x**3 + theta4_2*x**2 + theta4_3*x + theta4_4
function_5 = lambda x: theta5_5*x**5 + theta5_1*x**4 + theta5_2*x**3 + theta5_3*x**2 + theta5_4*x + theta5_5

result_1 = sq_error(list_x, list_y, function_1)
result_2 = sq_error(list_x, list_y, function_2)
result_3 = sq_error(list_x, list_y, function_3)
result_4 = sq_error(list_x, list_y, function_4)
result_5 = sq_error(list_x, list_y, function_5)

print("Среднекадратическая ошибка (1):", result_1)
print("Среднекадратическая ошибка (2):", result_2)
print("Среднекадратическая ошибка (3):", result_3)
print("Среднекадратическая ошибка (4):", result_4)
print("Среднекадратическая ошибка (5):", result_5)

# Предсказание
model = LinearRegression() # экземпляр класса LinearRegression
np_x = np_x.reshape(-1, 1) # меняем форму данных np_x
np_y = np_y.reshape(-1, 1) # меняем форму данных np_y
model.fit(np_x, np_y) # обучаем модель используя данные из csv файла

x_pr = np.array(list(range(744, 751))) # данные из задания
new_x_pr = x_pr.reshape(-1, 1) # меняем форму данных
y_pr = model.predict(new_x_pr) # предсказываем
y_pr = y_pr.flatten() # выравниваем в "плоский" вид

plt.scatter(new_x_pr, y_pr, label = u'предсказание', color='y')

plt.title('Линейная регрессия.')
plt.legend()
plt.ylabel('y')
plt.xlabel('x')
plt.show()