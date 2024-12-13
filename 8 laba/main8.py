import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Чтение данных из CSV файла
data = pd.read_csv('DataSet5_2.csv', sep = ',')

# Просмотр первых нескольких строк данных для понимания структуры набора данных
data.head()

# Определение индексов для разделения данных на обучающую и тестовую выборки
start_idx = 0
end_idx = len(data)
n = round(0.8*(end_idx - start_idx))  # 80% данных используется для обучения

# Разделение данных на обучающую и тестовую выборки
train = data[0:n]
test = data[n:]

# Визуализация обучающих и тестовых данных
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(train['count'].index, train['count'], color='blue', linestyle = 'solid')  # обучающие данные
ax.plot(test['count'].index, test['count'], color='blue', linestyle = 'dashed')  # тестовые данные

# Настройка графика
ax.set(title="Исходные данные", xlabel="Номер измерения", ylabel="Количество велосипедов")
ax.legend(['Train', 'Test'])
plt.show()

# Подготовка данных для обучения модели
train_x = train.drop(['count'], axis=1, inplace=False)  # все столбцы, кроме целевого, как признаки
train_y = train['count']  # целевой столбец
test_x = test.drop(['count'], axis=1, inplace=False)
test_y = test['count']

# Импорт функции для подбора параметров регрессионной модели
from scipy.optimize import curve_fit

# Определение функции для регрессионной модели. Это линейная комбинация признаков.
def mapping_func(x, a, b, c, d, e, i, j):
    return a + b*x['holiday'] + c*x['humidity'] + d*x['summer'] + e*x['temp'] + i*x['windspeed'] + j*x['workingday']

# Подбор параметров модели
args, covar = curve_fit(mapping_func, train_x, train_y)

# Извлечение параметров из результата подбора
a, b, c, d, e, i, j = args[0], args[1], args[2], args[3], args[4], args[5], args[6]

# Прогнозы для обучающих и тестовых данных
res_y_tr = a + b*train['holiday'] + c*train['humidity'] + d*train['summer'] + e*train['temp'] + i*train['windspeed'] + j*train['workingday']
res_y_ts = a + b*test['holiday'] + c*test['humidity'] + d*test['summer'] + e*test['temp'] + i*test['windspeed'] + j*test['workingday']
res_y = list(res_y_tr)+list(res_y_ts)

# Визуализация прогнозов модели
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(test['count'].index, test['count'], color='blue', linestyle = 'dashed')  # реальные значения тестовых данных
ax.plot(test['count'].index, res_y_ts, color='red', linestyle = 'solid')  # прогнозы модели
ax.set(title="Исходные данные", xlabel="Номер измерения", ylabel="Количество велосипедов")
ax.legend(['Train', 'Test'])
plt.show()

# Сортировка данных для последующего анализа
data2 = data.sort_values(by='count').reset_index()

# Визуализация отсортированных данных
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(data2['count'].index, data2['count'], color='blue', linestyle = 'solid')
ax.set(title="Исходные данные", xlabel="Номер измерения", ylabel="Количество велосипедов")
ax.legend(['Train', 'Test'])
plt.show()

# Прогнозы для отсортированных данных
test_x2 = data2.drop(['count'], axis=1, inplace=False)
res_y_ts2 = a + b*test_x2['holiday'] + c*test_x2['humidity'] + d*test_x2['summer'] + e*test_x2['temp'] + i*test_x2['windspeed'] + j*test_x2['workingday']

# Визуализация прогнозов для отсортированных данных
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(data2['count'].index, data2['count'], color='blue', linestyle = 'solid')
ax.plot(data2['count'].index, res_y_ts2, color='red', linestyle = 'solid')
ax.set(title="Исходные данные", xlabel="Номер измерения", ylabel="Количество велосипедов")
ax.legend(['Train', 'Test'])
plt.show()

# Создание классов для задачи классификации: 1 - эффективный день, 0 - неэффективный
res1 = [1 if data['count'][i] > 100 else 0 for i in range(len(data))]  # реальные классы
res2 = [1 if res_y[i] > 100 else 0 for i in range(len(res_y))]  # классы, предсказанные моделью

# Импорт метрик качества классификации
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

# Вычисление метрик для оценки модели
precision = round(precision_score(res1, res2) * 100, 2)
recall = round(recall_score(res1, res2) * 100, 2)
accuracy = round(accuracy_score(res1, res2) * 100, 2)
f1 = round(f1_score(res1, res2) * 100, 2)
roc_auc = round(roc_auc_score(res1, res2) * 100, 2)

# Вывод метрик
print('precision:', precision)
print('recall:', recall)
print('accuracy:', accuracy)
print('f1:', f1)
print('roc_auc:', roc_auc)
print('\n')

# Построение ROC-кривой
fpr, tpr, _ = roc_curve (res1, res2)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr,tpr, color='black', linestyle = 'solid')
ax.set(title="ROC Кривая", xlabel="TPR", ylabel="FPR")
plt.show()

# Добавление классов в исходные данные
data_cl = data
data_cl['cls'] = [1 if data['count'][i] > 100 else 0 for i in range(len(data))]

# Разделение данных для задачи классификации
train_cl_x = data_cl.drop(['cls'], axis=1, inplace=False)[0:n]
train_cl_y = data_cl['cls'][0:n]

# Построение модели классификации с использованием K-ближайших соседей
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)  # параметр n_neighbors указывает количество ближайших соседей
neigh.fit(train_cl_x, train_cl_y)

# Прогнозирование классов для всей выборки
res_y_cl = neigh.predict(data_cl.drop(['cls'], axis=1, inplace=False))

# Вычисление метрик качества для классификации
precision = round(precision_score(data_cl['cls'], res_y_cl) * 100, 2)
recall = round(recall_score(data_cl['cls'], res_y_cl) * 100, 2)
accuracy = round(accuracy_score(data_cl['cls'], res_y_cl) * 100, 2)
f1 = round(f1_score(data_cl['cls'], res_y_cl) * 100, 2)
roc_auc = round(roc_auc_score(data_cl['cls'], res_y_cl) * 100, 2)

# Вывод метрик
print('precision:', precision)
print('recall:', recall)
print('accuracy:', accuracy)
print('f1:', f1)
print('roc_auc:', roc_auc)

# Построение ROC-кривой для модели классификации
fpr, tpr, _ = roc_curve (data_cl['cls'], res_y_cl)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr,tpr, color='black', linestyle = 'solid')
ax.set(title="ROC Кривая", xlabel="TPR", ylabel="FPR")
plt.show()


# Задача регрессии и классификации:
# Задача регрессии предполагает предсказание числового (непрерывного) значения на основе данных.
# Например, прогнозирование количества сданных в аренду велосипедов или стоимости акций.
# Задача классификации заключается в предсказании категориальных (дискретных) значений или классов.
# Например, предсказать, эффективен ли день аренды велосипедов (1 или 0), или растет/падает цена акций (1 или 0).

# Зачем сводить задачу регрессии к задаче классификации:
# 1. Практическая интерпретация: зачастую важно знать категорию (например, день эффективен или нет), а не точное значение.
# 2. Упрощение задачи: классификационные модели могут быть проще и устойчивее в некоторых сценариях.
# 3. Лучшая интерпретируемость для бизнеса, где решения принимаются на основе категорий, а не точных прогнозов.

# Как сводится задача регрессии к задаче классификации в данном коде:
# Для задачи с велосипедами введен порог (100 аренд). Если значение выше порога - день эффективен (1), иначе - неэффективен (0).
# Это реализовано строкой:
# data_cl['cls'] = [1 if data['count'][i] > 100 else 0 for i in range(len(data))]
# Аналогично можно сделать для изменения стоимости акций: рост (1) или падение (0) на основе соседних значений.

# Описание графиков:
# 1. Исходные данные: синие линии показывают реальные значения (количество аренд велосипедов) для обучающей и тестовой выборок.
# 2. Прогнозы регрессии: красные линии отображают предсказанные значения регрессии, что позволяет увидеть, насколько хорошо модель аппроксимирует реальные данные.
# 3. ROC-кривая: график показывает зависимость между долей истинно положительных (TPR) и ложноположительных (FPR) классификаций.
# Чем ближе кривая к верхнему левому углу, тем лучше модель.

# Как сравнивать качество моделей:
# Основные метрики:
# - Precision (точность): доля истинно положительных среди всех предсказанных положительных.
# - Recall (полнота): доля истинно положительных среди всех реальных положительных.
# - Accuracy (точность): доля всех правильно классифицированных примеров.
# - F1-метрика: гармоническое среднее precision и recall, важна при несбалансированных классах.
# - ROC AUC: площадь под ROC-кривой, общая мера качества классификации.

# Оценка из примера:
# Первый набор метрик (precision: 65.29, recall: 84.22, accuracy: 72.37, f1: 73.56, roc_auc: 73.32):
# - Модель демонстрирует умеренное качество, с некоторым перекосом в сторону recall (высокая полнота, но точность страдает).
# Второй набор метрик (precision: 99.27, recall: 99.34, accuracy: 99.37, f1: 99.31, roc_auc: 99.36):
# - Почти идеальные результаты. Модель хорошо сбалансирована по всем метрикам.
# - Однако, возможно, данные слишком простые или модель переобучена.

# Выводы по выбору модели:
# - Если важна минимизация ложных срабатываний, ориентируйтесь на precision.
# - Если важно не пропустить положительные примеры, выбирайте модель с высоким recall.
# - Для общего баланса F1-метрика подходит лучше всего.
# - ROC-кривая и AUC дают визуальное и интегральное представление о качестве классификации.