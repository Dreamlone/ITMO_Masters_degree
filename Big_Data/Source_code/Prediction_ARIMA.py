import os
import numpy as np
import pandas as pd
import gdal, osr
import scipy.stats
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pmdarima as pm
from pmdarima import model_selection
from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

# Функция предсказания для выбранного года значения урожайности
def predict(path, year, country, crop):

    # Доступ к конкретному файлу с датасетом
    files = os.listdir(path)
    for file in files:
        if file.startswith(country):
            file_path = os.path.join(path, file)
    dataframe = pd.read_csv(file_path)
    dataframe['year'] = dataframe['Year']
    dataframe.set_index('Year', inplace=True)

    # Оставляем только те данные, которые имелись на момент предсказания
    test_dataframe = dataframe.loc[year]
    train_dataframe = dataframe.loc[1980:year-1]
    # Действительное значение для нужного нам года
    actual = test_dataframe[crop]

    train_crop = np.array(train_dataframe[crop])
    model = pm.auto_arima(train_crop, max_p=5, max_q=5, max_P=5, max_Q=5, seasonal=True,
                          stepwise=True, suppress_warnings=True, max_D=10, error_action='ignore')

    predicted = model.predict(n_periods = 1, return_conf_int=False)[0]
    return(predicted, actual)

# Графики остатков + информация по метрикам
# Расчет метрики - cредняя абсолютная процентная ошибка
def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.ravel(y_true)
    y_pred = np.ravel(y_pred)
    # У представленной ниже формулы есть недостаток, - если в массиве y_true есть хотя бы одно значение 0.0,
    # то по формуле np.mean(np.abs((y_true - y_pred) / y_true)) * 100 мы получаем inf, поэтому
    zero_indexes = np.argwhere(y_true == 0.0)
    for index in zero_indexes:
        y_true[index] = 0.01
    value = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return (value)


# Зададим функцию, которая будет выводить на экран значения метрик
def print_metrics(y_test, prediction):
    print('Mean absolute error in the test sample -', round(mean_absolute_error(y_test, prediction),2))
    print('Median absolute error in the test sample -', round(median_absolute_error(y_test, prediction),2))
    print('Root mean square error in the test sample -', round(mean_squared_error(y_test, prediction) ** 0.5, 2))
    print('Mean absolute percentage error in the test sample -', round(mean_absolute_percentage_error(y_test, prediction), 2))


# Зададим функцию для отрисовки графиков
def residuals_plots(y_test, prediction, color='blue'):
    prediction = np.ravel(prediction)
    y_test = np.ravel(y_test)
    # Рассчитываем ошибки
    errors = prediction - y_test
    errors = np.ravel(errors)

    plot_data = pd.DataFrame({'Errors': errors,
                              'Prediction': prediction})

    with sns.axes_style("ticks"):
        g = (sns.jointplot('Prediction', 'Errors', height=7, alpha=0.6,
                           data=plot_data, color=color).plot_joint(sns.kdeplot, zorder=0, n_levels=6))
        g.ax_joint.plot([min(prediction) - 0.1, max(prediction) + 0.1], [0, 0], linewidth=1, linestyle='--',
                        color=color)
        g.ax_marg_y.axhline(y=0, linewidth=1, linestyle='--', color=color)
        plt.xlabel('Predicted value', fontsize=15)
        plt.ylabel('Errors, tones/ha', fontsize=15)
        plt.show()

        g = (sns.jointplot('Prediction', 'Errors', kind="kde", data=plot_data, space=0, height=7,
                           color=color, alpha=0.2))
        g.set_axis_labels('Predicted value', 'Errors, tones/ha', fontsize=15)
        g.ax_joint.plot([min(prediction) - 0.1, max(prediction) + 0.1], [0, 0], linewidth=1, linestyle='--',
                        color=color)
        g.ax_marg_y.axhline(y=0, linewidth=1, linestyle='--', color=color)
        plt.show()

# Применение модели
# Wheat (tonnes per hectare)
# Rice (tonnes per hectare)
# Maize (tonnes per hectare)
# Barley (tonnes per hectare)
# France Germany Italy Romania Spain Czech Republic Netherlands Switzerland Austria Poland
# country = 'Poland'
# crop = 'Barley (tonnes per hectare)'
#
# reals = []
# preds = []
# ys = []
# for yEaR in [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
#     pred, real = predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid', yEaR, country, crop)
#     reals.append(real)
#     preds.append(pred)
#     ys.append(yEaR)
#
# ys = np.array(ys)
# reals = np.array(reals)
# preds = np.array(preds)
# print_metrics(reals, preds)
# residuals_plots(reals, preds, color = 'red')
#
# # Переводим все в одномерный массивы
# ys = np.ravel(ys)
# reals = np.ravel(reals)
# preds = np.ravel(preds)
# # Теперь необходимо сохранить предсказания в файл
# df = pd.DataFrame({'Year': ys,
#                    'Prediction': preds,
#                    'Real': reals})
#
# file_name = 'ARIMA_' + country + '_' + crop + '.csv'
# file_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', file_name)
# df.to_csv(file_path, sep=';', encoding='utf-8', index = False)