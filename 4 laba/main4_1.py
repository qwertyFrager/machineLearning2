import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.optimize import curve_fit
from sklearn.neighbors import KNeighborsRegressor
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error

# Шаг 1: Загрузка данных
# Мы загружаем данные из файла и преобразуем дату в формат времени (в секундах)
data1 = pd.read_csv('DataSet1_3.csv', sep=',')

# Преобразование столбца 'Date' в формат секунд (чтобы было удобно работать с числовыми данными)
data1["Date"] = pd.to_datetime(data1["Date"])
data1['Date'] = data1['Date'].astype(np.int64) / 1000000000  # нс -> c

# Шаг 2: Подготовка данных
# Мы выбрали индексы от 100 до 300 для дальнейшего анализа
start_idx = 100
end_idx = 300

# Получаем x (время) и y (цена закрытия акции) для указанного диапазона данных
x = data1['Date'][start_idx:end_idx].to_numpy() - data1['Date'][start_idx]
y = data1['Close'][start_idx:end_idx].to_numpy()

# Разделяем данные на обучающую (80%) и тестовую (20%) выборки
n = round(0.8 * (end_idx - start_idx))

train_x = x[:n]
train_y = y[:n]

test_x = x[n:]
test_y = y[n:]

# Шаг 3: Визуализация исходных данных
# На этом этапе строим график, где показываем тренировочные и тестовые данные
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train_x,
    y=train_y,
    line=dict(color='green', width=2),
    name='Данные для обучения'
))
fig.add_trace(go.Scatter(
    x=test_x,
    y=test_y,
    line=dict(color='red', width=2),
    name='Данные для верификации'
))

fig.show()

# Шаг 4: Метод наименьших квадратов (МНК)
# Мы используем полиномиальную регрессию второго порядка для прогнозирования

# Определяем функцию для полиномиальной регрессии: y = c1*x^2 + c2*x + c3
def mapping_func(x, c1, c2, c3):
    return c1 * x**2 + c2 * x + c3

# Используем curve_fit для подбора коэффициентов модели
args, covar = curve_fit(mapping_func, train_x, train_y)
c1, c2, c3 = args[0], args[1], args[2]

# Прогнозирование на тренировочных данных
res_y_train = c1 * train_x**2 + c2 * train_x + c3

# Визуализация результата на тренировочных данных для МНК
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train_x,
    y=train_y,
    line=dict(color='green', width=2),
    name='Данные для обучения'
))
fig.add_trace(go.Scatter(
    x=train_x,
    y=res_y_train,
    line=dict(color='red', width=2),
    name='МНК'
))

fig.show()

# Прогноз на тестовой выборке
res_y_test = c1 * test_x**2 + c2 * test_x + c3

# Визуализация результатов МНК на тестовых данных
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_x,
    y=test_y,
    line=dict(color='green', width=2),
    name='Тестовые данные'
))
fig.add_trace(go.Scatter(
    x=test_x,
    y=res_y_test,
    line=dict(color='red', width=2),
    name='МНК'
))

fig.show()

# Шаг 5: Модель kNN (Метод ближайших соседей)
# Используем модель ближайших соседей для прогнозирования цен закрытия

neigh = KNeighborsRegressor(n_neighbors=2)
neigh.fit(train_x.reshape(-1, 1), train_y)

# Прогнозирование на тренировочной выборке
res_y_train_knn = neigh.predict(train_x.reshape(-1, 1))

# Визуализация результатов kNN на тренировочных данных
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=train_x,
    y=train_y,
    line=dict(color='green', width=2),
    name='Данные для обучения'
))
fig.add_trace(go.Scatter(
    x=train_x,
    y=res_y_train_knn,
    line=dict(color='red', width=2),
    name='Прогноз kNN (обучение)'
))

fig.show()

# Прогнозирование на тестовой выборке
res_y_test_knn = neigh.predict(test_x.reshape(-1, 1))

# Визуализация результатов kNN на тестовых данных
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=test_x,
    y=test_y,
    line=dict(color='green', width=2),
    name='Тестовые данные'
))
fig.add_trace(go.Scatter(
    x=test_x,
    y=res_y_test_knn,
    line=dict(color='red', width=2),
    name='Прогноз kNN (тестирование)'
))

fig.show()

# Шаг 6: Модель ARIMA (авторегрессия, интегрированное скользящее среднее)
# Мы строим прогноз с помощью ARIMA модели на основе предыдущих данных
history = [y for y in train_y]
predictions = []

# Прогнозируем шаг за шагом
for i in range(1, len(test_y)):
    model = sm.tsa.arima.ARIMA(history, order=(1, 1, 0))
    model_fit = model.fit()

    # Прогнозируем следующее значение
    yhat = model_fit.forecast()[0]
    predictions.append(yhat)

    # Добавляем наблюдаемое значение в историю
    obs = test_y[i]
    history.append(obs)

# Визуализация прогноза ARIMA
fig = go.Figure()

fig.add_trace(go.Scatter(
    x=np.concatenate((train_x, test_x[1:])),
    y=history,
    line=dict(color='green', width=2),
    name='Реальные значения'
))
fig.add_trace(go.Scatter(
    x=test_x[1:],  # Прогнозы начинаются с 1-го элемента
    y=predictions,
    line=dict(color='red', width=2),
    name='ARIMA'
))

fig.show()

# Шаг 7: Оценка точности моделей (средняя абсолютная ошибка, MAE)
# MAE измеряет среднее отклонение предсказанных значений от реальных

mae_mnk = mean_absolute_error(test_y, res_y_test)
mae_knn = mean_absolute_error(test_y, res_y_test_knn)
mae_arima = mean_absolute_error(test_y[1:], predictions)

# Выводим точность моделей
print(f"Средняя абсолютная ошибка (MAE) для МНК: {mae_mnk:.4f}")
print(f"Средняя абсолютная ошибка (MAE) для kNN: {mae_knn:.4f}")
print(f"Средняя абсолютная ошибка (MAE) для ARIMA: {mae_arima:.4f}")



# Метод наименьших квадратов (МНК) — Полиномиальная регрессия
# Как работает: Полиномиальная регрессия строит
# зависимость между независимой переменной (в нашем случае это время в секундах) и зависимой переменной (цена
# закрытия) с использованием полинома заданной степени.

# МНК будет искать глобальные тенденции в данных. Это означает, что если данные подчиняются плавной и непрерывной
# зависимости, модель будет хорошо справляться. Однако если есть сильные колебания или неожиданные изменения в
# данных, модель может не уловить эти особенности. МНК особенно эффективен, если данные имеют параболическую форму (в
# случае полинома 2-й степени). Например, если цена акций плавно растет или падает.
#
# Плюсы: Хорошо подходит для нахождения гладких трендов в данных.
# Минусы: Плохо работает с данными, которые содержат резкие колебания или имеют локальные зависимости.
# На моих данных: Если цены акций Google в заданном диапазоне не
# имеют значительных скачков и следуют плавному тренду, модель МНК будет хорошо предсказывать этот тренд. Однако,
# если в данных есть внезапные изменения, результат будет смазан, так как модель не сможет их уловить.


# Метод ближайших соседей (kNN)
# Как работает:
# Модель kNN основывается на том, что прогноз для каждой точки строится на основе значений ближайших соседей. Для каждой новой точки модель ищет n ближайших соседей (в данном случае, n=2) и использует их значения для вычисления прогноза.
# Пример: Для точки времени t модель найдет две самые близкие точки по времени (например, те, которые расположены до и после t), и на основе их значений предскажет цену закрытия для t.

# Модель kNN хорошо подходит для случаев, когда данные содержат локальные паттерны или резкие изменения. Например, если
# цена акций резко колеблется в короткие промежутки времени, модель сможет точно воспроизвести эти колебания, так как
# она смотрит на соседние точки и использует их информацию.
#
# Плюсы: Хорошо работает с данными, которые имеют локальные зависимости и колебания.
# Минусы: Модель может быть менее точной, если данные имеют глобальный тренд, так как она фокусируется на ближайших значениях.
# На моих данных:
# Если данные акций Google содержат резкие колебания в цене на небольших временных интервалах, модель kNN будет предсказывать
# эти изменения более точно. Однако, если данные имеют долгосрочные тренды, kNN может игнорировать эти глобальные изменения
# и сосредоточиться на локальных характеристиках.


# ARIMA (AutoRegressive Integrated Moving Average)
# Как работает:
# ARIMA — это модель для временных рядов, которая учитывает три компонента:
#
# Автогрессию (AR): Зависимость текущего значения от предыдущих значений.
# Интегрирование (I): Преобразование данных для их стационарности (убираются тренды или сезонность).
# Скользящее среднее (MA): Учитываются ошибки предыдущих прогнозов для улучшения точности.

# Модель ARIMA наиболее эффективна, если данные имеют временные зависимости, т.е. значение в один момент времени зависит
# от предыдущих значений. Она работает хорошо с данными, которые можно преобразовать в стационарные (после устранения
# трендов или сезонности). Важно отметить, что ARIMA прогнозирует значение на основе предыдущих наблюдений и
# корректируется на основе ошибок.
#
# Плюсы: Эффективно работает с временными рядами, где каждое следующее значение зависит от предыдущих.
# Минусы: Плохо работает, если данные содержат нелинейные или случайные скачки.
# На моих данных:
# Если цены акций Google показывают определённую зависимость от предыдущих значений, ARIMA сможет это уловить и
# предсказать тенденцию. Например, если цена акций обычно растет или падает на основе прошлых значений, ARIMA сможет
# это смоделировать. Однако, если в данных много шума или неожиданных событий, ARIMA может оказаться менее точной.