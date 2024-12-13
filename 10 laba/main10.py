import pandas as pd

#Читаем файл данных
data = pd.read_csv('DataSet2_1.csv', sep='|')
data.head()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data.drop(data.columns[9], axis = 1), data.fraud.values, test_size=0.2, random_state=42)

from sklearn.metrics import roc_auc_score

from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score

model = SVC(kernel = 'linear', probability = True)
model.fit(X_train, y_train)
y_pred_SVM = model.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, y_pred_SVM)
print('auc_roc', res)


from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 3)
model.fit(X_train, y_train)
y_pred_kNN = model.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, y_pred_kNN)
print('auc_roc', res)

from sklearn.naive_bayes import GaussianNB

model = GaussianNB()
model.fit(X_train, y_train)
y_pred_NB = model.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, y_pred_NB)
print('auc_roc', res)

from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
y_pred_DT = model.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, y_pred_DT)
print('auc_roc', res)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

fpr_svm, tpr_svm, _ = roc_curve(y_test, y_pred_SVM)
fpr_knn, tpr_knn, _ = roc_curve(y_test, y_pred_kNN)
fpr_nb, tpr_nb, _ = roc_curve(y_test, y_pred_NB)
fpr_c4, tpr_c4, _ = roc_curve(y_test, y_pred_DT)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr_svm,tpr_svm, color='red', linestyle = 'solid', label = 'SVM')
ax.plot(fpr_knn,tpr_knn, color='black', linestyle = 'solid', label = 'kNN')
ax.plot(fpr_nb,tpr_nb, color='blue', linestyle = 'solid', label = 'NB')
ax.plot(fpr_c4,tpr_c4, color='green', linestyle = 'solid', label = 'DT')
ax.set(title="ROC Кривые", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax.legend()
plt.show()

print('\n')


# Кроссвалидация
from sklearn.model_selection import cross_validate
import numpy as np

model = SVC(kernel = 'linear', probability = True)
cv_SVC = cross_validate(model, X_train, y_train, cv=3, return_estimator=True)
best_SVC = cv_SVC['estimator'][np.where(cv_SVC['test_score'] == max(cv_SVC['test_score']))[0][0]]
res_SVC = best_SVC.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, res_SVC)
print('auc_roc', res)

model = KNeighborsClassifier(n_neighbors = 3)
cv_kNN = cross_validate(model, X_train, y_train, cv=3, return_estimator=True)
best_kNN = cv_kNN['estimator'][np.where(cv_kNN['test_score'] == max(cv_kNN['test_score']))[0][0]]
res_kNN = best_kNN.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, res_kNN)
print('auc_roc', res)

model = GaussianNB()
cv_NB= cross_validate(model, X_train, y_train, cv=3, return_estimator=True)
best_NB = cv_NB['estimator'][np.where(cv_NB['test_score'] == max(cv_NB['test_score']))[0][0]]
res_NB = best_NB.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, res_NB)
print('auc_roc', res)

model = DecisionTreeClassifier()
cv_DT= cross_validate(model, X_train, y_train, cv=3, return_estimator=True)
best_DT = cv_DT['estimator'][np.where(cv_DT['test_score'] == max(cv_DT['test_score']))[0][0]]
res_DT = best_DT.predict_proba(X_test)[:, 1]
res = roc_auc_score(y_test, res_DT)
print('auc_roc', res)

fpr_svm, tpr_svm, _ = roc_curve(y_test, res_SVC)
fpr_knn, tpr_knn, _ = roc_curve(y_test, res_kNN)
fpr_nb, tpr_nb, _ = roc_curve(y_test, res_NB)
fpr_c4, tpr_c4, _ = roc_curve(y_test, res_DT)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr_svm,tpr_svm, color='red', linestyle = 'solid', label = 'SVM')
ax.plot(fpr_knn,tpr_knn, color='black', linestyle = 'solid', label = 'kNN')
ax.plot(fpr_nb,tpr_nb, color='blue', linestyle = 'solid', label = 'NB')
ax.plot(fpr_c4,tpr_c4, color='green', linestyle = 'solid', label = 'DT')
ax.set(title="ROC Кривые", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax.legend()
plt.show()

print('\n')

#Композиции
from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('SVC', best_SVC), ('NB', best_NB), ('DT',best_DT)], voting='hard')
eclf.fit(X_train, y_train)
res_eclf = eclf.predict(X_test)

print('auc_roc', roc_auc_score(y_test, res_eclf))

fpr_svm, tpr_svm, _ = roc_curve(y_test, res_SVC)
fpr_nb, tpr_nb, _ = roc_curve(y_test, res_NB)
fpr_c4, tpr_c4, _ = roc_curve(y_test, res_DT)
fpr_eclf, tpr_eclf, _ = roc_curve(y_test, res_eclf)


fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fpr_svm,tpr_svm, color='red', linestyle = 'solid', label = 'SVM')
ax.plot(fpr_nb,tpr_nb, color='blue', linestyle = 'solid', label = 'NB')
ax.plot(fpr_c4,tpr_c4, color='green', linestyle = 'solid', label = 'DT')
ax.plot(fpr_eclf,tpr_eclf, color='pink', linestyle = 'solid', label = 'ECLF')

ax.set(title="ROC Кривые", xlabel="False Positive Rate", ylabel="True Positive Rate")
ax.legend()
plt.show()


# Что такое кроссвалидация и зачем она тут?
# Кроссвалидация (cross-validation) — это метод оценки качества модели, при котором данные разбиваются на несколько
# частей (фолдов).
# В каждом цикле одна часть используется для тестирования, а остальные — для обучения.
# Зачем?
# Снижает вероятность переобучения (overfitting).
# Оценивает устойчивость модели на разных разбиениях данных.

# Что такое композиция моделей?
# Композиция моделей — это объединение нескольких алгоритмов машинного обучения в один, чтобы улучшить точность
# предсказаний.
# Пример: VotingClassifier — объединяет модели, используя "голосование".
# Hard voting: выбирается класс, за который проголосовало большинство моделей.
# Soft voting: используется усредненная вероятность предсказаний.


# Как работает ROC кривая и зачем она нужна?
# ROC (Receiver Operating Characteristic):
# График, который показывает зависимость:
# TPR (True Positive Rate, чувствительность) — доля правильно предсказанных мошенничеств.
# FPR (False Positive Rate) — доля ложноположительных срабатываний.

# Зачем она нужна?
# Для оценки качества бинарного классификатора.
# Кривая ROC помогает выбрать оптимальный порог вероятности для классификации.
# AUC-ROC (Area Under Curve) — площадь под ROC-кривой, показывающая общую производительность модели.
# 1.0 — идеальная модель.
# 0.5 — случайное угадывание.

# Основные элементы графика
# Ось X (False Positive Rate, FPR):
# Показывает, как часто модель ошибочно предсказывает мошенничество (ложноположительные случаи).
# Чем ближе к нулю, тем меньше ложных тревог.
# Ось Y (True Positive Rate, TPR или Recall):
# Показывает, какую долю настоящих мошенничеств модель правильно определила.
# Чем ближе к 1, тем лучше.
