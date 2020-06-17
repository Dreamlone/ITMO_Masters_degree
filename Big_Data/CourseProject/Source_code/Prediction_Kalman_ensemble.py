import os
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import gdal, osr
import scipy.stats
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
import simdkalman
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from pykalman import KalmanFilter, UnscentedKalmanFilter, AdditiveUnscentedKalmanFilter

from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

# Функция предсказания для выбранного года значения урожайности
def predict(path, country, crop):
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
        print('Mean absolute error in the test sample -', round(mean_absolute_error(y_test, prediction), 2))
        print('Median absolute error in the test sample -', round(median_absolute_error(y_test, prediction), 2))
        print('Root mean square error in the test sample -', round(mean_squared_error(y_test, prediction) ** 0.5, 2))
        print('Mean absolute percentage error in the test sample -',
              round(mean_absolute_percentage_error(y_test, prediction), 2))

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


    # Составляем имена файлов, которые необходимо рассмотреть
    bayesian_name = 'Bayesian_' + country + '_' + crop + '.csv'
    regression_name = 'Regression_' + country + '_' + crop + '.csv'
    distribution_name = 'Distribution_' + country + '_' + crop + '.csv'
    arima_name = 'ARIMA_' + country + '_' + crop + '.csv'

    # Загружаем датафреймы с предсказаниями для выбранной страны и с/х культуры
    bayesian_data = pd.read_csv(os.path.join(path, bayesian_name), sep = ';')
    regression_data = pd.read_csv(os.path.join(path, regression_name), sep=';')
    distribution_data = pd.read_csv(os.path.join(path, distribution_name), sep=';')
    arima_data = pd.read_csv(os.path.join(path, arima_name), sep=';')

    # Действительные значения урожайности
    real = np.array(bayesian_data['Real'])
    # Предсказанные значения по разным моделям
    bayesian_prediction = np.array(bayesian_data['Prediction'])
    regression_prediction = np.array(regression_data['Prediction'])
    distribution_prediction = np.array(distribution_data['Prediction'])
    arima_prediction = np.array(arima_data['Prediction'])

    # Применение фильтров Калмана для ансамблирования прогнозов
    preds = []
    for index in range(0, len(bayesian_prediction)):
        preds.append([bayesian_prediction[index], regression_prediction[index], distribution_prediction[index], arima_prediction[index]])

    # Среднее значение по ряду (нужно для инициализации)
    bayesian_mean = np.mean(bayesian_prediction)
    regression_mean = np.mean(regression_prediction)
    distribution_mean = np.mean(distribution_prediction)
    arima_mean = np.mean(arima_prediction)
    main_mean = (bayesian_mean + regression_mean + distribution_mean + arima_mean) / 4

    # Применение фильтра для оценки действительного значения величины
    kf = UnscentedKalmanFilter(initial_state_mean = main_mean, n_dim_obs = 4)
    estimated_pred = kf.filter(preds)
    estimated_pred = np.array(estimated_pred[0])

    print('             <<< ', country, ' >>>')
    print_metrics(real, estimated_pred)

    # Загружаем данные нужной страны
    crop_data = pd.read_csv('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Crop_yields_info.csv', sep = ',')
    crop_data_country = crop_data[crop_data['Entity'] == country]
    all_yield = crop_data_country[crop]
    all_years = crop_data_country['Year']

    # Нарисуем график
    plt.plot(all_years[:-(len(real)-1)], all_yield[:-(len(real)-1)], c='blue', linewidth=3, alpha = 0.8)
    plt.plot(bayesian_data['Year'], bayesian_prediction, c='orange', linewidth=2, label='Bayesian network', alpha=0.5)
    plt.plot(bayesian_data['Year'], regression_prediction, c='green', linewidth=2, label='Linear regression', alpha=0.5)
    plt.plot(bayesian_data['Year'], distribution_prediction, c='purple', linewidth=2, label='Distribution analysis', alpha=0.5)
    plt.plot(bayesian_data['Year'], arima_prediction, c='yellow', linewidth=2, label='ARIMA prediction', alpha=0.8)
    plt.plot(bayesian_data['Year'], real, '-ok', linestyle='--', c='blue', linewidth=3, label='Actual values', alpha=0.8)
    plt.plot(bayesian_data['Year'], estimated_pred, '-ok', linestyle='--', c='red', linewidth=3, label='Kalman filter estimation', alpha=0.8)
    plt.grid()
    plt.title(country, fontsize=13)
    plt.legend(fontsize=11)
    plt.xlabel('Year', fontsize=13)
    plt.ylabel(crop, fontsize=13)
    plt.show()

    # График остатков для данного предсказания
    res_arima = real - arima_prediction
    res_bayesian = real - bayesian_prediction
    res_regression = real - regression_prediction
    res_distribution = real - distribution_prediction

    size = 60
    plt.scatter(arima_prediction, res_arima, s = size, alpha = 0.7, c='yellow', label = 'ARIMA', edgecolors = {'#CED500'})
    plt.scatter(bayesian_prediction, res_bayesian, s = size, alpha = 0.5, c='orange', label = 'Bayesian network', edgecolors = {'#FFC726'})
    plt.scatter(regression_prediction, res_regression, s = size, alpha = 0.5, c='green', label = 'Linear regression', edgecolors = {'#00B21B'})
    plt.scatter(distribution_prediction, res_distribution, s = size, alpha = 0.5, c='purple', label = 'Distribution analysis', edgecolors = {'#9638FF'})
    plt.legend(fontsize=11)
    plt.grid()
    plt.ylabel('Residuals', fontsize=13)
    plt.xlabel('Predicted value', fontsize=13)
    plt.title(country + ', ' + crop, fontsize=13)
    plt.close()

    # Составляем датафрейм с предсказаниями
    # 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018
    df = pd.DataFrame({'Year': [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018],
                       'Prediction': np.ravel(estimated_pred),
                       'Real': np.ravel(real)})
    # Теперь необходимо сохранить прогнозы
    file_name = 'Kalman_' + country + '_' + crop + '.csv'
    file_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', file_name)
    df.to_csv(file_path, sep=';', encoding='utf-8', index=False)


# Применение модели
# Wheat (tonnes per hectare)
# Rice (tonnes per hectare)
# Maize (tonnes per hectare)
# Barley (tonnes per hectare)
# France Germany Italy Romania Spain Czech Republic Netherlands Switzerland Austria Poland
# crop = 'Barley (tonnes per hectare)'
#
# for cont in ['France', 'Germany', 'Italy', 'Romania', 'Spain', 'Netherlands', 'Switzerland', 'Austria', 'Poland']:
#     if crop == 'Rice (tonnes per hectare)':
#         if any(cont == bad_country for bad_country in ['Germany', 'Czech Republic', 'Netherlands', 'Switzerland', 'Austria', 'Poland']):
#             pass
#         else:
#             predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', cont, crop)
#     else:
#         predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', cont, crop)