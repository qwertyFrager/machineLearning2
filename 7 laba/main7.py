import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Загрузка данных из CSV файла
data1 = pd.read_csv('DataSet1_3.csv', sep=',')

# Преобразование столбца 'Date' в формат datetime и преобразование в секунды
data1[["Date"]] = data1[["Date"]].apply(pd.to_datetime)
data1['Date'] = data1['Date'].astype(np.int64) / 1e9  # нс -> с

# Задание индексов для выборки данных
start_idx = 100
end_idx = 300

# Создание нового DataFrame с необходимыми столбцами
data = pd.DataFrame()
data['x'] = data1['Date'][start_idx:end_idx].to_numpy() - data1['Date'][start_idx] + 1
data['y'] = data1['Close'][start_idx:end_idx].to_numpy()
data['p1'] = data1['Open'][start_idx:end_idx].to_numpy()
data['p2'] = data1['Close'][start_idx-1:end_idx-1].to_numpy()

# Разделение данных на обучающую и тестовую выборки
n = round(0.8 * (end_idx - start_idx))
train = data[0:n].reset_index(drop=True)
test = data[n:].reset_index(drop=True)

# Визуализация исходных данных
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train['x'], train['y'], color='green', linestyle='solid')
ax.plot(train['x'], train['p1'], color='red', linestyle='solid')
ax.plot(train['x'], train['p2'], color='blue', linestyle='solid')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed')
ax.plot(test['x'], test['p1'], color='red', linestyle='dashed')
ax.plot(test['x'], test['p2'], color='blue', linestyle='dashed')

ax.set(title="Исходные данные", xlabel="Время", ylabel="Объем купли/продаж")
ax.legend(['Train y', 'Train p1', 'Train p2', 'Test y', 'Test p1', 'Test p2'])
plt.show()

# Подготовка данных для моделей
train_x = train.drop(['y'], axis=1)
train_y = train['y']
test_x = test.drop(['y'], axis=1)
test_y = test['y']

# МНК (Метод наименьших квадратов)
from scipy.optimize import curve_fit

# Определение функции для аппроксимации
def mapping_func(X, a, b, c, b1, b2):
    x = X['x']
    p1 = X['p1']
    p2 = X['p2']
    return a + b * x + c * x**2 + b1 * p1 + b2 * p2

# Применение curve_fit для нахождения параметров модели
args, covar = curve_fit(lambda xdata, a, b, c, b1, b2: mapping_func(xdata, a, b, c, b1, b2), train_x, train_y)

# Извлечение параметров модели
a, b, c, b1, b2 = args

# Предсказание на обучающей и тестовой выборках
mnk_res_y_tr = mapping_func(train_x, a, b, c, b1, b2)
mnk_res_y_ts = mapping_func(test_x, a, b, c, b1, b2)

# Визуализация результатов МНК
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train['x'], train['y'], color='green', linestyle='solid')
ax.plot(train['x'], mnk_res_y_tr, color='red', linestyle='solid')
ax.plot(test['x'], test['y'], color='green', linestyle='dashed')
ax.plot(test['x'], mnk_res_y_ts, color='red', linestyle='dashed')

ax.set(title="МНК", xlabel="Время", ylabel="Объем купли/продаж")
ax.legend(['Train Actual', 'Train Predict', 'Test Actual', 'Test Predict'])
plt.show()

# Вычисление метрик MAE, RMSE, MAPE, MASE для МНК
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Избегаем деления на ноль
    non_zero_idx = y_true != 0
    y_true = y_true[non_zero_idx]
    y_pred = y_pred[non_zero_idx]
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mase(y_true, y_pred, y_train):
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

# Расчет метрик на тестовой выборке
mae_mnk = mean_absolute_error(test_y, mnk_res_y_ts)
rmse_mnk = np.sqrt(mean_squared_error(test_y, mnk_res_y_ts))
mape_mnk = mean_absolute_percentage_error(test_y, mnk_res_y_ts)
mase_mnk = mase(test_y, mnk_res_y_ts, train_y)

print("Метрики для МНК:")
print(f"MAE: {mae_mnk}")
print(f"RMSE: {rmse_mnk}")
print(f"MAPE: {mape_mnk}%")
print(f"MASE: {mase_mnk}\n")

# Построение box-plot диаграмм для МНК
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['Test Actual', 'Test Predict']
colors = ['pink', 'lightblue']

bplot = ax.boxplot([test_y, mnk_res_y_ts], patch_artist=True, labels=labels)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set(title="МНК", ylabel="Объем купли/продаж")
plt.show()

# Критерий хи-квадрат для МНК
# сравнивает наблюдаемые и ожиданыемые частоты
# статистика измеряет расхождение между наблюдаемыми и ожидаемыми значениями
# p - вероятность получения такой же статистики при условии что гипотеза верна
# если p > 0.05 значит нет оснований отвергать гипотезу
chi2_stat, chi2_p = stats.chisquare(test['y'], np.sum(test['y'])/np.sum(mnk_res_y_ts) * mnk_res_y_ts)
print("Критерий хи-квадрат для МНК:")
print(f"Статистика хи-квадрат: {chi2_stat}")
print(f"p-значение: {chi2_p}\n")

# Критерий Фишера для МНК
# сравнение дисперсий (отклонения) выборок
# F - статистика, отношение дисперсий , если значительно отличается от 1, это может указывать на различие дисперсий
var_actual = np.var(test_y, ddof=1)
var_predict = np.var(mnk_res_y_ts, ddof=1)

f_value = var_actual / var_predict
df1 = len(test_y) - 1
df2 = len(mnk_res_y_ts) - 1

# Вычисляем p-значение
p_value = 1 - stats.f.cdf(f_value, df1, df2)

print("Критерий Фишера для МНК:")
print(f"Статистика F: {f_value}")
print(f"Степени свободы: df1={df1}, df2={df2}")
print(f"p-значение: {p_value}\n")

# Критерий Стьюдента для МНК
# сравнивает средние значения двух выборок чтобы определить, статистически ли они различаются
# измеряет величину разницы относительно вариации данных
t_stat, t_p = stats.ttest_ind(test_y, mnk_res_y_ts)
print("Критерий Стьюдента для МНК:")
print(f"t-статистика: {t_stat}")
print(f"p-значение: {t_p}\n")

# ---------------------------------------------
# SVR (Support Vector Regression)
from sklearn.svm import SVR

# Обучение модели SVR
svr_model = SVR(kernel='rbf')
svr_model.fit(train_x, train_y)

# Предсказание на обучающей и тестовой выборках
svr_res_y_tr = svr_model.predict(train_x)
svr_res_y_ts = svr_model.predict(test_x)

# Визуализация результатов SVR
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train['x'], train_y, color='green', linestyle='solid')
ax.plot(train['x'], svr_res_y_tr, color='red', linestyle='solid')
ax.plot(test['x'], test_y, color='green', linestyle='dashed')
ax.plot(test['x'], svr_res_y_ts, color='red', linestyle='dashed')

ax.set(title="SVR", xlabel="Время", ylabel="Объем купли/продаж")
ax.legend(['Train Actual', 'Train Predict', 'Test Actual', 'Test Predict'])
plt.show()

# Вычисление метрик для SVR
mae_svr = mean_absolute_error(test_y, svr_res_y_ts)
rmse_svr = np.sqrt(mean_squared_error(test_y, svr_res_y_ts))
mape_svr = mean_absolute_percentage_error(test_y, svr_res_y_ts)
mase_svr = mase(test_y, svr_res_y_ts, train_y)

print("Метрики для SVR:")
print(f"MAE: {mae_svr}")
print(f"RMSE: {rmse_svr}")
print(f"MAPE: {mape_svr}%")
print(f"MASE: {mase_svr}\n")

# Построение box-plot диаграмм для SVR
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['Test Actual', 'Test Predict']
colors = ['pink', 'lightblue']

bplot = ax.boxplot([test_y, svr_res_y_ts], patch_artist=True, labels=labels)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set(title="SVR", ylabel="Объем купли/продаж")
plt.show()

# Критерий хи-квадрат для SVR
chi2_stat, chi2_p = stats.chisquare(test['y'], np.sum(test['y'])/np.sum(svr_res_y_ts) * svr_res_y_ts)
print("Критерий хи-квадрат для SVR:")
print(f"Статистика хи-квадрат: {chi2_stat}")
print(f"p-значение: {chi2_p}\n")

# Критерий Фишера для SVR
var_actual = np.var(test_y, ddof=1)
var_predict = np.var(svr_res_y_ts, ddof=1)

f_value = var_actual / var_predict
df1 = len(test_y) - 1
df2 = len(svr_res_y_ts) - 1

p_value = 1 - stats.f.cdf(f_value, df1, df2)

print("Критерий Фишера для SVR:")
print(f"Статистика F: {f_value}")
print(f"Степени свободы: df1={df1}, df2={df2}")
print(f"p-значение: {p_value}\n")

# Критерий Стьюдента для SVR
t_stat, t_p = stats.ttest_ind(test_y, svr_res_y_ts)
print("Критерий Стьюдента для SVR:")
print(f"t-статистика: {t_stat}")
print(f"p-значение: {t_p}\n")

# ---------------------------------------------
# Lasso Regression
from sklearn.linear_model import Lasso

# Обучение модели Lasso
lasso_model = Lasso(alpha=1)
lasso_model.fit(train_x, train_y)

# Предсказание на обучающей и тестовой выборках
lasso_res_y_tr = lasso_model.predict(train_x)
lasso_res_y_ts = lasso_model.predict(test_x)

# Визуализация результатов Lasso
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train['x'], train_y, color='green', linestyle='solid')
ax.plot(train['x'], lasso_res_y_tr, color='red', linestyle='solid')
ax.plot(test['x'], test_y, color='green', linestyle='dashed')
ax.plot(test['x'], lasso_res_y_ts, color='red', linestyle='dashed')

ax.set(title="Lasso", xlabel="Время", ylabel="Объем купли/продаж")
ax.legend(['Train Actual', 'Train Predict', 'Test Actual', 'Test Predict'])
plt.show()

# Вычисление метрик для Lasso
mae_lasso = mean_absolute_error(test_y, lasso_res_y_ts)
rmse_lasso = np.sqrt(mean_squared_error(test_y, lasso_res_y_ts))
mape_lasso = mean_absolute_percentage_error(test_y, lasso_res_y_ts)
mase_lasso = mase(test_y, lasso_res_y_ts, train_y)

print("Метрики для Lasso:")
print(f"MAE: {mae_lasso}")
print(f"RMSE: {rmse_lasso}")
print(f"MAPE: {mape_lasso}%")
print(f"MASE: {mase_lasso}\n")

# Построение box-plot диаграмм для Lasso
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['Test Actual', 'Test Predict']
colors = ['pink', 'lightblue']

bplot = ax.boxplot([test_y, lasso_res_y_ts], patch_artist=True, labels=labels)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set(title="Lasso", ylabel="Объем купли/продаж")
plt.show()

# Критерий хи-квадрат для Lasso
chi2_stat, chi2_p = stats.chisquare(test['y'], np.sum(test['y'])/np.sum(lasso_res_y_ts) * lasso_res_y_ts)
print("Критерий хи-квадрат для Lasso:")
print(f"Статистика хи-квадрат: {chi2_stat}")
print(f"p-значение: {chi2_p}\n")

# Критерий Фишера для Lasso
var_actual = np.var(test_y, ddof=1)
var_predict = np.var(lasso_res_y_ts, ddof=1)

f_value = var_actual / var_predict
df1 = len(test_y) - 1
df2 = len(lasso_res_y_ts) - 1

p_value = 1 - stats.f.cdf(f_value, df1, df2)

print("Критерий Фишера для Lasso:")
print(f"Статистика F: {f_value}")
print(f"Степени свободы: df1={df1}, df2={df2}")
print(f"p-значение: {p_value}\n")

# Критерий Стьюдента для Lasso
t_stat, t_p = stats.ttest_ind(test_y, lasso_res_y_ts)
print("Критерий Стьюдента для Lasso:")
print(f"t-статистика: {t_stat}")
print(f"p-значение: {t_p}\n")

# ---------------------------------------------
# PLS Regression
from sklearn.cross_decomposition import PLSRegression

# Обучение модели PLS
pls_model = PLSRegression(n_components=3)
pls_model.fit(train_x, train_y)

# Предсказание на обучающей и тестовой выборках
pls_res_y_tr = pls_model.predict(train_x).flatten()
pls_res_y_ts = pls_model.predict(test_x).flatten()

# Визуализация результатов PLS
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(train['x'], train_y, color='green', linestyle='solid')
ax.plot(train['x'], pls_res_y_tr, color='red', linestyle='solid')
ax.plot(test['x'], test_y, color='green', linestyle='dashed')
ax.plot(test['x'], pls_res_y_ts, color='red', linestyle='dashed')

ax.set(title="PLS", xlabel="Время", ylabel="Объем купли/продаж")
ax.legend(['Train Actual', 'Train Predict', 'Test Actual', 'Test Predict'])
plt.show()

# Вычисление метрик для PLS
mae_pls = mean_absolute_error(test_y, pls_res_y_ts)
rmse_pls = np.sqrt(mean_squared_error(test_y, pls_res_y_ts))
mape_pls = mean_absolute_percentage_error(test_y, pls_res_y_ts)
mase_pls = mase(test_y, pls_res_y_ts, train_y)

print("Метрики для PLS:")
print(f"MAE: {mae_pls}")
print(f"RMSE: {rmse_pls}")
print(f"MAPE: {mape_pls}%")
print(f"MASE: {mase_pls}\n")

# Построение box-plot диаграмм для PLS
fig, ax = plt.subplots(figsize=(10, 6))

labels = ['Test Actual', 'Test Predict']
colors = ['pink', 'lightblue']

bplot = ax.boxplot([test_y, pls_res_y_ts], patch_artist=True, labels=labels)

for patch, color in zip(bplot['boxes'], colors):
    patch.set_facecolor(color)

ax.set(title="PLS", ylabel="Объем купли/продаж")
plt.show()

# Критерий хи-квадрат для PLS
chi2_stat, chi2_p = stats.chisquare(test['y'], np.sum(test['y'])/np.sum(pls_res_y_ts) * pls_res_y_ts)
print("Критерий хи-квадрат для PLS:")
print(f"Статистика хи-квадрат: {chi2_stat}")
print(f"p-значение: {chi2_p}\n")

# Критерий Фишера для PLS
var_actual = np.var(test_y, ddof=1)
var_predict = np.var(pls_res_y_ts, ddof=1)

f_value = var_actual / var_predict
df1 = len(test_y) - 1
df2 = len(pls_res_y_ts) - 1

p_value = 1 - stats.f.cdf(f_value, df1, df2)

print("Критерий Фишера для PLS:")
print(f"Статистика F: {f_value}")
print(f"Степени свободы: df1={df1}, df2={df2}")
print(f"p-значение: {p_value}\n")

# Критерий Стьюдента для PLS
t_stat, t_p = stats.ttest_ind(test_y, pls_res_y_ts)
print("Критерий Стьюдента для PLS:")
print(f"t-статистика: {t_stat}")
print(f"p-значение: {t_p}")
