import os
import numpy as np
import pandas as pd
import gdal, osr
import scipy.stats
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score

from statsmodels.stats.multitest import multipletests
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

# Функция предсказания для выбранного года значения урожайности
def predict(path, year, crop_data, country, crop):
    # Папки, в которых хранятся данные
    all = os.path.join(path, '01_06')
    feb = os.path.join(path, '02')
    apr = os.path.join(path, '04')
    june = os.path.join(path, '06')

    for parameter in ['Tmp', 'Pressure', 'Rainfall']:
        # Присоединяем все параметры к нужным папкам
        all_parameter = os.path.join(all, parameter)
        feb_parameter = os.path.join(feb, parameter)
        apr_parameter = os.path.join(apr, parameter)
        june_parameter = os.path.join(june, parameter)

        # Ищем файл, который будет использоваться в качестве основного
        name = str(year)
        # Первое полугодие в общем
        all_files = os.listdir(all_parameter)
        all_files.sort()
        for file in all_files:
            if file.startswith(name):
                all_parameter_main = os.path.join(all_parameter, file)
        # Распределение парамерта в феврале
        feb_files = os.listdir(feb_parameter)
        feb_files.sort()
        for file in feb_files:
            if file.startswith(name):
                feb_parameter_main = os.path.join(feb_parameter, file)
        # Распределение парамерта в апреле
        apr_files = os.listdir(apr_parameter)
        apr_files.sort()
        for file in apr_files:
            if file.startswith(name):
                apr_parameter_main = os.path.join(apr_parameter, file)
        # Распределение парамерта в июне
        june_files = os.listdir(june_parameter)
        june_files.sort()
        for file in june_files:
            if file.startswith(name):
                june_parameter_main = os.path.join(june_parameter, file)

        # Массивы для указанного года
        ALL_PARAMETER = np.load(all_parameter_main)
        FEB_PARAMETER = np.load(feb_parameter_main)
        APR_PARAMETER  = np.load(apr_parameter_main)
        JUNE_PARAMETER  = np.load(june_parameter_main)

        ##################################################################################################
        # Блок сравнения массивов одного года с распределениями соответствующего параметра в другие годы #
        ##################################################################################################
        result_list = []
        for YEAR in range(1990, year):

            # Ищем файл, который будет использоваться в качестве основного
            name = str(YEAR)
            # Первое полугодие в общем
            all_files = os.listdir(all_parameter)
            all_files.sort()
            for file in all_files:
                if file.startswith(name):
                    all_parameter_main = os.path.join(all_parameter, file)
            # Распределение парамерта в феврале
            feb_files = os.listdir(feb_parameter)
            feb_files.sort()
            for file in feb_files:
                if file.startswith(name):
                    feb_parameter_main = os.path.join(feb_parameter, file)
            # Распределение парамерта в апреле
            apr_files = os.listdir(apr_parameter)
            apr_files.sort()
            for file in apr_files:
                if file.startswith(name):
                    apr_parameter_main = os.path.join(apr_parameter, file)
            # Распределение парамерта в июне
            june_files = os.listdir(june_parameter)
            june_files.sort()
            for file in june_files:
                if file.startswith(name):
                    june_parameter_main = os.path.join(june_parameter, file)

            # Распределение в этот конкретный год
            ALL_PARAMETER_THIS = np.load(all_parameter_main)
            FEB_PARAMETER_THIS = np.load(feb_parameter_main)
            APR_PARAMETER_THIS = np.load(apr_parameter_main)
            JUNE_PARAMETER_THIS = np.load(june_parameter_main)

            # Производим сравнение двух выборок по критерию Крускала-Уоллиса
            # Либо можно использовать ks_2samp - критерий Колмогорова-Смирнова
            all_stat = scipy.stats.kruskal(ALL_PARAMETER, ALL_PARAMETER_THIS)
            feb_stat = scipy.stats.kruskal(FEB_PARAMETER, FEB_PARAMETER_THIS)
            apr_stat = scipy.stats.kruskal(APR_PARAMETER, APR_PARAMETER_THIS)
            june_stat = scipy.stats.kruskal(JUNE_PARAMETER, JUNE_PARAMETER_THIS)

            if parameter == 'Tmp':
                # Записываем значение статистики
                result_list.append([YEAR, all_stat.statistic, all_stat.pvalue, feb_stat.statistic, feb_stat.pvalue,
                                    apr_stat.statistic, apr_stat.pvalue, june_stat.statistic, june_stat.pvalue])
            else:
                result_list.append([all_stat.statistic, all_stat.pvalue, feb_stat.statistic, feb_stat.pvalue,
                                    apr_stat.statistic, apr_stat.pvalue, june_stat.statistic, june_stat.pvalue])

        # Переводим массив в нужный формат
        result_list = np.array(result_list)
        # После того, как таблица по конкретному параметру сформирована - присоединяем
        if parameter == 'Tmp':
            data = result_list
        else:
            data = np.hstack((data, result_list))

    dataframe = pd.DataFrame(data, columns=['Year', 'TMP_All.stat', 'TMP_All.pvalue', 'TMP_Feb.stat', 'TMP_Feb.pvalue',
                                            'TMP_Apr.stat', 'TMP_Apr.pvalue', 'TMP_June.stat', 'TMP_June.pvalue',
                                            'PRS_All.stat', 'PRS_All.pvalue', 'PRS_Feb.stat', 'PRS_Feb.pvalue',
                                            'PRS_Apr.stat', 'PRS_Apr.pvalue', 'PRS_June.stat', 'PRS_June.pvalue',
                                            'RAN_All.stat', 'RAN_All.pvalue', 'RAN_Feb.stat','RAN_Feb.pvalue',
                                            'RAN_Apr.stat', 'RAN_Apr.pvalue', 'RAN_June.stat', 'RAN_June.pvalue'])

    # В подготовленном датасете каждая строка содержит значения статистики + значение pvalue
    # Необходимо ввести поправку на множественное тестирование и сравнить к какому году ближе всего
    local_data1 = np.array(dataframe[['TMP_All.pvalue', 'TMP_Feb.pvalue', 'TMP_Apr.pvalue', 'TMP_June.pvalue',
                                     'PRS_All.pvalue', 'PRS_Feb.pvalue', 'PRS_Apr.pvalue', 'PRS_June.pvalue',
                                     'RAN_All.pvalue', 'RAN_Feb.pvalue', 'RAN_Apr.pvalue', 'RAN_June.pvalue']])
    local_data2 = np.array(dataframe[['TMP_All.stat', 'TMP_Feb.stat', 'TMP_Apr.stat', 'TMP_June.stat',
                                     'PRS_All.stat', 'PRS_Feb.stat', 'PRS_Apr.stat', 'PRS_June.stat',
                                     'RAN_All.stat', 'RAN_Feb.stat', 'RAN_Apr.stat', 'RAN_June.stat']])
    # Сумма превышений статистик над скорректированным pvalue
    sum_vals = []
    for id_string in range(0, len(local_data1)):
        pvals = np.ravel(local_data1[id_string])
        # Поправка на множественное тестирование Бонферронни
        arr = multipletests(pvals, method = 'bonferroni')
        corrected_pvals = arr[1]

        # Теперь берем значения статистик
        statistics = np.ravel(local_data2[id_string])

        # Разность значения статистик и скорректированных pvalue
        diff = statistics - corrected_pvals
        # Суммируем значения превышений, то есть чем меньше значение суммы превышений статистик над уровнем значимости
        # Тем выше более похожи два рассматриваемых года
        sum_vals.append(np.sum(diff))

    sum_vals = np.array(sum_vals)
    # Индекс самого минимального значения в рассчитанных характеристиках
    id_min = np.argmin(sum_vals)
    dataframe['sum_vals'] = sum_vals

    similar_year = str(dataframe['Year'][id_min])
    print('Наиболее "похожий" на', year ,'год -', similar_year)

    # Загружаем датасет с данными по урожайности
    crop_dataframe = pd.read_csv(crop_data, sep=',', dtype={'Entity': str, 'Year': str})
    # Выбираем только те данные, которые есть для конкретной стртаны
    crop_dataframe = crop_dataframe[crop_dataframe.Entity == country]
    crop_dataframe.drop(columns='Code', inplace=True)

    predicted_dataset = crop_dataframe[crop_dataframe.Year == similar_year[:4]]
    prediction = np.array(predicted_dataset[crop])
    print('Предсказанное значение урожайности -', prediction[0])

    actual_dataset = crop_dataframe[crop_dataframe.Year == str(year)]
    actual = np.array(actual_dataset[crop])
    print('Действительное значение урожайности -', actual[0], '\n')

    return(prediction, actual)

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



# Применение модели для Франции
# Wheat (tonnes per hectare)
# Rice (tonnes per hectare)
# Maize (tonnes per hectare)
# Barley (tonnes per hectare)
# France Germany Italy Romania Spain Czech Republic Netherlands Switzerland Austria Poland
# country = 'Poland'
# crop = 'Barley (tonnes per hectare)'
# main_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Distributions_files', country)
#
# reals = []
# preds = []
# ys = []
# # 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018
# for j in [2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018]:
#     pred, real = predict(main_path, j, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Crop_yields_info.csv', country, crop = crop)
#     reals.append(real)
#     preds.append(pred)
#     ys.append(j)
#
# ys = np.array(ys)
# reals = np.array(reals)
# preds = np.array(preds)
# print_metrics(reals, preds)
# residuals_plots(reals, preds, color = 'blue')
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
# file_name = 'Distribution_' + country + '_' + crop + '.csv'
# file_path = os.path.join('/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/RESULTS', file_name)
# df.to_csv(file_path, sep=';', encoding='utf-8', index = False)