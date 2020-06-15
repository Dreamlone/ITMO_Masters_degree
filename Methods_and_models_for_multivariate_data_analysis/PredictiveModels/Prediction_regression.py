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

    # Теперь выбираем предикторов
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_dataframe[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year']])
    y_train = train_dataframe[crop]
    y_train = np.ravel(y_train)

    # Обучение простой линейной регрессии
    # LinearRegression()
    # SVR(kernel = 'rbf', C = 10.0, epsilon=0.2)
    LR = SVR(kernel = 'rbf', C = 100.0, epsilon=0.2)
    LR.fit(X_train, y_train)

    # Подготовка данных для предсказания
    X_test = test_dataframe[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year']]
    X_test = [X_test]
    X_test = scaler.transform(X_test)
    predicted = LR.predict(X_test)
    predicted = predicted[0]

    print('Предсказанное значение урожайности -', predicted)
    print('Действительное значение урожайности -', actual, '\n')

    # Отрисовываем график для проверки качества модели
    if year == 2012:
        print(test_dataframe[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year']], '\n')

        def plot_3d(dataframe, modeled_params, bounds):
            # Получим список параметров
            parameter_1 = modeled_params[0]
            parameter_2 = modeled_params[1]
            # Теперь определим границы для них
            parameter_1_min = bounds.get('1')[0]
            parameter_1_max = bounds.get('1')[1]
            parameter_2_min = bounds.get('2')[0]
            parameter_2_max = bounds.get('2')[1]

            # Теперь зафиксируем некоторые параметры и будем варьировать только 2 из них
            modeled_par_1 = np.linspace(parameter_1_min, parameter_1_max, 100)
            modeled_par_2 = np.linspace(parameter_2_min, parameter_2_max, 100)

            # Теперь совместим наши параметры
            modeled_par_1 = list(modeled_par_1) * 100
            modeled_par_1 = np.array(modeled_par_1)
            # Значения для второго
            modeled_par_2 = np.repeat(modeled_par_2, 100)

            # В строгой последовательности составляем выборку
            new_df = pd.DataFrame()
            for par in ['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year']:
                if par == parameter_1:
                    new_df[par] = modeled_par_1
                elif par == parameter_2:
                    new_df[par] = modeled_par_2
                else:
                    new_df[par] = np.full(len(modeled_par_1), dataframe[par])

            # Добавляем последнюю строчку действительное значение
            data = np.array(new_df[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean','Temperature_min', 'year']])
            last_string = np.array(dataframe[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year']])
            data = np.vstack((data, last_string))
            new_df = pd.DataFrame(data, columns = ['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year'])

            # Производим процедуру стандартизации, теперь значения можно подавать в модель
            data_scaled = scaler.transform(new_df[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean','Temperature_min', 'year']])
            plot_predictions = LR.predict(data_scaled)
            plot_predictions = np.array(plot_predictions)
            plot_predictions[-1] = actual

            # Визуализация результата
            rcParams['figure.figsize'] = 13, 9
            fig = pyplot.figure()
            ax = Axes3D(fig)

            x_vals = np.array(new_df[parameter_1])
            y_vals = np.array(new_df[parameter_2])
            z_vals = np.array(plot_predictions)
            points = np.ravel(z_vals)

            surf = ax.scatter(x_vals, y_vals, z_vals, c=points, cmap='rainbow')
            ax.set_zlabel(crop, fontsize = 13)
            ax.view_init(45, 240)
            #ax.view_init(0, 270)
            ax.text(x_vals[-1], y_vals[-1], z_vals[-1],'Actual value', fontsize = 12)
            fig.colorbar(surf, shrink = 0.5, aspect = 10)
            if parameter_1 == 'Precip_amount':
                name_par1 = 'Total rainfall, mm'
            if parameter_2 == 'Temperature_SAT':
                name_par2 = 'Sum of active temperatures, °C'
            plt.xlabel(name_par1, fontsize = 13)
            plt.ylabel(name_par2, fontsize = 13)
            name = country + ' ' + str(year) + ' ' + 'Simulated surface'
            plt.title(name, fontsize = 13)
            pyplot.show()

        # Отрисовываем график
        # dataframe      --- датафрейм, в котором есть те параметры, которые наблюдались в конкретный год
        # modeled_params --- список из двух параметров, которые будут смоделированы
        # bounds         --- границы для параметров, в пределах которых будет прозводится моделирование
        # test_dataframe['Pressure_mean'] = 850
        plot_3d(dataframe = test_dataframe[['Precip_amount', 'Temperature_SAT', 'Precip_days', 'Pressure_mean', 'Temperature_min', 'year']],
                modeled_params = ['Precip_amount', 'Temperature_SAT'], bounds = {'1': [100, 500], '2': [1000, 1300]})

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
# country = 'France'
# crop = 'Barley (tonnes per hectare)'
#
# reals = []
# preds = []
# ys = []
# # 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018
# for j in [1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
#     pred, real = predict('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid', j, country, crop)
#     reals.append(real)
#     preds.append(pred)
#     ys.append(j)
#
# ys = np.array(ys)
# reals = np.array(reals)
# preds = np.array(preds)
# print_metrics(reals, preds)
# residuals_plots(reals, preds, color = 'green')
#
# # Переводим все в одномерный массивы
# ys = np.ravel(ys)
# reals = np.ravel(reals)
# preds = np.ravel(preds)
# # Теперь необходимо сохранить предсказания в файл
# df = pd.DataFrame({'Year': ys,
#                     'Prediction': preds,
#                     'Real': reals})
#
# file_name = 'Regression_' + country + '_' + crop + '.csv'
# file_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', file_name)
# df.to_csv(file_path, sep=';', encoding='utf-8', index = False)