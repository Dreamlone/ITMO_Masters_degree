import os
import numpy as np
import pandas as pd
import gdal, osr
import scipy.stats
from netCDF4 import Dataset
from matplotlib import pyplot as plt
from matplotlib import cm
import seaborn as sns


def aggregate_by_country(country, netCDf_dir, crop_data, fertilizer_data, tractor_data, land_data):
    # Подбор идентефикатора для страны
    if country == 'Germany':
        id_country = 979
    elif country == 'France':
        id_country = 969
    elif country == 'Belgium':
        id_country = 908
    elif country == 'Netherlands':
        id_country = 1061
    elif country == 'Switzerland':
        id_country = 1109
    elif country == 'Austria':
        id_country = 901
    elif country == 'Czech Republic':
        id_country = 960
    elif country == 'Poland':
        id_country = 1075
    elif country == 'Spain':
        id_country = 1101
    elif country == 'Italy':
        id_country = 1003
    elif country == 'Romania':
        id_country = 1084
    elif country == 'Belarus':
        id_country = 916
    elif country == 'Sweden':
        id_country = 1106
    elif country == 'Ireland':
        id_country = 953
    elif country == 'Portugal':
        id_country = 1077
    elif country == 'Greece':
        id_country = 983

    # Поиск какие файлы присутствуют в указанной папке
    files = os.listdir(netCDf_dir)

    # Пока датфрейм пустой
    dataframe = pd.DataFrame()
    for file in files:
        # Если интересующий нас файл - netCDF, то
        if file.endswith(".nc"):
            path = os.path.join(netCDf_dir, file)
            dataset = Dataset(path)

            # Достаем вектор с годами
            years = np.array(dataset.variables['time'])
            years = np.ravel(years)
            # Матрицу с границами государств
            countries = np.array(dataset.variables['countries'])
            # Матрица с типами ландшафтов
            land_matrix = np.array(dataset.variables['biome'])
            # В случае типов ландшафтов нас интересуют следующие значения:
            # 20 - Mosaic cropland (50-70%) / vegetation (grassland/shrubland/forest) (20-50%)
            # 30 - Mosaic vegetation (grassland/shrubland/forest) (50-70%) / cropland (20-50%)

            # Многомерная матрица со всеми слоями
            tensor = np.array(dataset.variables['matrices'])

            # Теперь посмотрим для каких индексов нам нужны значения
            indices_c = np.argwhere(countries == id_country)

            # Для каких биомов нам требуется получать значения
            indices_20 = np.argwhere(land_matrix == 20)
            indices_30 = np.argwhere(land_matrix == 30)
            # Объединяем списки с индексами нужных нам биомов
            indices_num = np.vstack((indices_20, indices_30))

            # Сформируем вектора со значениями координат в виде строк
            str_indices_num = []
            for pixel in indices_num:
                str_indices_num.append(str(pixel[0]) + ':'+ str(pixel[1]))
            frame_num = pd.DataFrame({'ID' : str_indices_num})

            # Также поступаем с индексами стран
            str_indices_c = []
            for pixel in indices_c:
                str_indices_c.append(str(pixel[0]) + ':' + str(pixel[1]))
            frame_c = pd.DataFrame({'ID': str_indices_c})

            # Объеденим оба датфрейма в один
            # Так мы оставляем только индексы тех пикселей, которые входят в нужную страну и при этом являются с/х угодиями
            frame = pd.merge(frame_c, frame_num, on = 'ID')

            # Используем название файла для формирования признака
            str_feature = file[:-3]

            # Если мы работаем с признаком - полем давления, то для него подгатавливаем немного другую структуру
            if str_feature == 'Pressure_mean':
                # Разделение на с/х угодия в случае со средним давлением не предусмотрено
                # Рассчитаем среднее значение за первое полугодие для территории данной страны
                agg_data = []
                for index in range(0, len(years)):
                    # Матрица с метеопараметром для данного года
                    matrix = tensor[index]

                    # Для страны производим выборку данных
                    values = []
                    for i in indices_c:
                        row_id = i[0]
                        col_id = i[1]
                        values.append(matrix[row_id, col_id])

                    values = np.array(values)
                    # Берем среднее значение и записываем в массив
                    agg_data.append(np.mean(values))

                # На данном этапе есть сагрегированный по стране показатель + годы
                agg_data = np.array(agg_data)
                # Добавляем признак в датафрейм
                dataframe[str_feature] = agg_data

                # Теперь необходимо повторить данную процедуру для условных "центров действия атмосферы"
                # Получим средние значения давления ещё для 4 стран: Ирландии, Швеции, Португалии, Греции
                for country_center in ['Sweden_pressure', 'Ireland_pressure', 'Portugal_pressure', 'Greece_pressure']:
                    if country_center == 'Sweden_pressure':
                        id_country_pressure = 1106
                    elif country_center == 'Ireland_pressure':
                        id_country_pressure = 953
                    elif country_center == 'Portugal_pressure':
                        id_country_pressure = 1077
                    elif country_center == 'Greece_pressure':
                        id_country_pressure = 983

                    # Теперь посмотрим для каких индексов нам нужны значения
                    indices_c_pressure = np.argwhere(countries == id_country_pressure)

                    # Рассчитаем среднее значение за первое полугодие для территории данной страны
                    agg_data = []
                    for index in range(0, len(years)):
                        # Матрица с метеопараметром для данного года
                        matrix = tensor[index]

                        # Для страны производим выборку данных
                        values = []
                        for i in indices_c_pressure:
                            row_id = i[0]
                            col_id = i[1]
                            values.append(matrix[row_id, col_id])

                        values = np.array(values)
                        # Берем среднее значение и записываем в массив
                        agg_data.append(np.mean(values))

                    # На данном этапе есть сагрегированный по стране показатель + годы
                    agg_data = np.array(agg_data)
                    # Добавляем признак в датафрейм
                    dataframe[country_center] = agg_data

            else:
                # Для каждого года производим агрегацию данных по выбранной стране
                agg_data = []
                for index in range(0, len(years)):
                    # Матрица с метеопараметром для данного года
                    matrix = tensor[index]

                    # Для страны и с/х угодий производим выборку данных
                    values = []
                    for i in frame['ID']:
                        # Разделим координаты
                        i = i.split(':')
                        row_id = int(i[0])
                        col_id = int(i[1])
                        values.append(matrix[row_id, col_id])

                    values = np.array(values)
                    # Берем среднее значение и записываем в массив
                    agg_data.append(np.mean(values))

                # На данном этапе есть сагрегированный по стране показатель + годы
                agg_data = np.array(agg_data)

                # Если речь идет о среднем количестве дней с осадками, то приведем данные в тип int
                if str_feature == 'Precip_days':
                    agg_data = np.round(agg_data, 0)
                # Добавляем признак в датафрейм
                dataframe[str_feature] = agg_data

            # Закрываем все, что открывали раньше
            dataset.close()
        else:
            pass

    # Финальный штрих - добавляем столбец с годами в наш датафрейм
    dataframe['Year'] = years

    # Теперь загружаем данные об урожайности из файла
    crop_dataframe = pd.read_csv(crop_data, sep = ',', dtype = {'Entity': str, 'Year': str})
    # Выбираем только те данные, которые есть для конкретной стртаны
    crop_dataframe = crop_dataframe[crop_dataframe.Entity == country]
    crop_dataframe.drop(columns = 'Code', inplace = True)

    # Теперь сливаем наши датафреймы по столбцу Year
    new_data = pd.merge(dataframe, crop_dataframe, on = 'Year')
    new_data.drop(columns='Entity', inplace=True)

    # Загружаем данные по используемым удобрениям
    fertilizer_dataframe = pd.read_csv(fertilizer_data, sep=',', dtype={'Entity': str, 'Year': str})
    # Выбираем только те данные, которые есть для конкретной стртаны
    fertilizer_dataframe = fertilizer_dataframe[fertilizer_dataframe.Entity == country]
    # Возьмем только некоторые столбцы
    fertilizer_dataframe = fertilizer_dataframe[['Entity', 'Year','Nitrogen fertilizer use (kilograms per hectare)']]

    # Сливаем датафреймы по столбцу Year
    new_data = pd.merge(new_data, fertilizer_dataframe, on='Year')
    new_data.drop(columns='Entity', inplace=True)

    # Добавляем данные об использовании техники
    tractor_dataframe = pd.read_csv(tractor_data, sep=',', dtype={'Entity': str, 'Year': str})
    # Выбираем только те данные, которые есть для конкретной стртаны
    tractor_dataframe = tractor_dataframe[tractor_dataframe.Entity == country]
    # Возьмем только некоторые столбцы
    tractor_dataframe = tractor_dataframe[['Entity', 'Year', 'Tractors per 100 sq km arable land', 'Total population (Gapminder)']]

    # Сливаем датафреймы по столбцу Year
    new_data = pd.merge(new_data, tractor_dataframe, on='Year')
    new_data.drop(columns='Entity', inplace=True)

    # Добавляем данные об количестве используемых земель
    land_dataframe = pd.read_csv(land_data, sep = ',', dtype = {'Entity': str, 'Year': str})
    # Выбираем только те данные, которые есть для конкретной стртаны
    land_dataframe = land_dataframe[land_dataframe.Entity == country]
    # Возьмем только некоторые столбцы
    land_dataframe = land_dataframe[['Entity', 'Year', 'Land used for cereal (hectares)', 'Cereal production (tonnes)', 'Cereal yield (tonnes per hectare)']]

    # Сливаем датафреймы по столбцу Year
    new_data = pd.merge(new_data, land_dataframe, on='Year')

    # Сохраняем в ту же самую папку, где и лежат файлы netCDF
    name = country + '.csv'
    save_path = os.path.join(netCDf_dir, name)
    new_data.to_csv(save_path, index = False)


# Подготовим датасеты по следующим странам
#aggregate_by_country(country = 'Germany', netCDf_dir = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid',
#                     crop_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Crop_yields_info.csv',
#                     fertilizer_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_crop_yield_vs_fertilizer_application.csv',
#                     tractor_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
#                     land_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Index_of_cereal_production_yield_and_land_use.csv')
#
#aggregate_by_country(country = 'France', netCDf_dir = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid',
#                     crop_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Crop_yields_info.csv',
#                     fertilizer_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_crop_yield_vs_fertilizer_application.csv',
#                     tractor_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
#                     land_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Index_of_cereal_production_yield_and_land_use.csv')
#
#aggregate_by_country(country = 'Italy', netCDf_dir = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid',
#                     crop_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Crop_yields_info.csv',
#                     fertilizer_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_crop_yield_vs_fertilizer_application.csv',
#                     tractor_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
#                     land_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Index_of_cereal_production_yield_and_land_use.csv')
#
#aggregate_by_country(country = 'Romania', netCDf_dir = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid',
#                     crop_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Crop_yields_info.csv',
#                     fertilizer_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_crop_yield_vs_fertilizer_application.csv',
#                     tractor_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
#                     land_data = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Index_of_cereal_production_yield_and_land_use.csv')
