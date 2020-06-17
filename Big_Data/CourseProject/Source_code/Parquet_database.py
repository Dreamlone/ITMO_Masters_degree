import pyarrow
import os
import gdal, osr
import pandas as pd
import numpy as np
from netCDF4 import Dataset
import shutil
import dask.array as da

########################################################################################################################
# Скрипт для формирования parquet файла базы данных с информацией по странам об урожайности и климатических параметрах #
#                                                  за разные годы                                                      #
########################################################################################################################
# csv_path      --- папка с файлами csv, где хранятся все необходимые материалы
# rainfall_file --- netCDF файл с информацией об осадках
# pressure_file --- netCDF файл с информацией о давлении
# maxtmp_file   --- netCDF файл с информацией о максимальной температуре
# tmp_file      --- netCDF файл с информацией о среднесуточной температуре воздуха
# mintmp_file   --- netCDF файл с информацией о минимальной температуре воздуха
# countries_tif --- geotiff файл, матрица с границами государств
# land_tif      --- geotiff файл, матрица с типами ландшафтов
# save          --- путь до файла parquet в котором сохраняется результат
def prepare_data(csv_path, rainfall_file, pressure_file, maxtmp_file, tmp_file, mintmp_file, countries_tif, land_tif, save):

    # Загружаем в память необходимые нам матрицы - осадки
    rainfall_nc = Dataset(rainfall_file)
    rainfall_matrix = rainfall_nc.variables['rr']
    # Давление
    pressure_nc = Dataset(pressure_file)
    pressure_matrix = pressure_nc.variables['pp']
    # Максимальная температура
    maxtmp_nc = Dataset(maxtmp_file)
    maxtmp_matrix = maxtmp_nc.variables['tx']
    # Средняя температура
    tmp_nc = Dataset(tmp_file)
    tmp_matrix = tmp_nc.variables['tg']
    # Минимальная суточная температура воздуха
    mintmp_nc = Dataset(mintmp_file)
    mintmp_matrix = mintmp_nc.variables['tn']

    for country in ['France','Germany','Italy','Romania','Spain','Czech Republic','Netherlands','Switzerland','Austria','Poland']:
        print('\n', country ,'\n')
        main_path = os.path.join(csv_path, country)
        main_path = main_path + '.csv'
        crop_data = pd.read_csv(main_path)

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

        ####################################################################################################################
        #       Определение индексов для которых требуется агрегировать данные матриц (нужная страна + с/х угодья)         #
        ####################################################################################################################

        # Загружаем матрицу со странами
        ds = gdal.Open(countries_tif, gdal.GA_ReadOnly)
        rb = ds.GetRasterBand(1)
        country_array = np.array(rb.ReadAsArray())
        country_array = np.flip(country_array, axis=0)

        # Теперь слой с кодами типов ландшафтов
        ds = gdal.Open(land_tif, gdal.GA_ReadOnly)
        rb = ds.GetRasterBand(1)
        land_matrix = np.array(rb.ReadAsArray())
        land_matrix = np.flip(land_matrix, axis=0)

        # Индексы ячеек матрицы для конкретно выбранной страны
        indices_c = np.argwhere(country_array == id_country)

        # Для каких биомов нам требуется получать значения
        indices_11 = np.argwhere(land_matrix == 11)
        indices_14 = np.argwhere(land_matrix == 14)
        indices_20 = np.argwhere(land_matrix == 20)
        indices_30 = np.argwhere(land_matrix == 30)

        # Объединяем списки с индексами нужных нам биомов
        indices_num = np.vstack((indices_11, indices_14, indices_20, indices_30))

        # Сформируем вектора со значениями координат в виде строк
        str_indices_num = []
        for pixel in indices_num:
            str_indices_num.append(str(pixel[0]) + ':' + str(pixel[1]))
        frame_num = pd.DataFrame({'ID': str_indices_num})

        # Также поступаем с индексами стран
        str_indices_c = []
        for pixel in indices_c:
            str_indices_c.append(str(pixel[0]) + ':' + str(pixel[1]))
        frame_c = pd.DataFrame({'ID': str_indices_c})

        # Объеденим оба датфрейма в один
        # Так мы оставляем только индексы тех пикселей, которые входят в нужную страну и при этом являются с/х угодиями
        frame = pd.merge(frame_c, frame_num, on='ID')

        # Генерируем временной ряд (нам нужны только дни до 7го июля 2019го года)
        time_series = pd.date_range(start='01/01/1950', end='07/31/2019', freq='1D')
        years_series = pd.date_range(start='01/01/1950', end='01/01/2020', freq='1Y')
        # Теперь переводим дату в строковый тип и обрезаем
        clipped_time = []
        for date in time_series:
            str_date = str(date)
            # Теперь записываем в массив только месяц и день (без года)
            clipped_time.append(str_date[5:10])
        clipped_time = np.array(clipped_time)
        # Ищем в массиве индексы тех элементов, которые соответствуют 30го июня
        threshold_id_1_Jan = np.ravel(np.argwhere(clipped_time == '01-01'))
        # А также индексы элементов 1го января
        threshold_id_31_Dec = np.ravel(np.argwhere(clipped_time == '12-31'))

        # Читаем матрицы в Dask массив
        tensor_rainfall = da.from_array(rainfall_matrix, chunks=(5000, 465, 705))
        tensor_pressure = da.from_array(pressure_matrix, chunks=(5000, 465, 705))
        tensor_maxtmp = da.from_array(maxtmp_matrix, chunks=(5000, 465, 705))
        tensor_tmp = da.from_array(tmp_matrix, chunks=(5000, 465, 705))
        tensor_mintmp = da.from_array(mintmp_matrix, chunks=(5000, 465, 705))

        main_rainfall = []
        main_pressure = []
        main_maxtmp = []
        main_tmp = []
        main_mintmp = []

        climate_data = pd.DataFrame(np.zeros((len(threshold_id_31_Dec),5)), columns=['RAINFALL', 'PRESSURE', 'MAX_TMP', 'TMP', 'MIN_TMP'])
        climate_data = climate_data.astype('object')
        # Для каждого года мы будем собирать некоторе количество значений определенного промежутка
        for i in range(0, len(threshold_id_31_Dec)):
            print('Calculations for ', time_series[threshold_id_1_Jan[i]], ' till ', time_series[threshold_id_1_Jan[i+1]])
            print('Year -', str(years_series[i])[:4])

            # Для конкретного года получаем значения нужных нам параметров
            local_rainfall_tensor = tensor_rainfall[threshold_id_1_Jan[i]: threshold_id_1_Jan[i+1], 110:450, :630].compute()
            local_pressure_tensor = tensor_pressure[threshold_id_1_Jan[i]: threshold_id_1_Jan[i+1], 110:450, :630].compute()
            local_maxtmp_tensor = tensor_maxtmp[threshold_id_1_Jan[i]: threshold_id_1_Jan[i+1], 110:450, :630].compute()
            local_tmp_tensor = tensor_tmp[threshold_id_1_Jan[i]: threshold_id_1_Jan[i+1], 110:450, :630].compute()
            local_mintmp_tensor = tensor_mintmp[threshold_id_1_Jan[i]: threshold_id_1_Jan[i+1], 110:450, :630].compute()

            # Для каждого интересующего на пикселя - получаем временной ряд за выбранный год
            values_rainfall = []
            values_pressure = []
            values_maxtmp = []
            values_tmp = []
            values_mintmp = []
            for id_pixel in frame['ID']:
                # Разделим координаты
                id_pixel = id_pixel.split(':')
                row_id = int(id_pixel[0])
                col_id = int(id_pixel[1])
                pixel_rainfall = local_rainfall_tensor[:, row_id, col_id]
                pixel_pressure = local_pressure_tensor[:, row_id, col_id]
                pixel_maxtmp = local_maxtmp_tensor[:, row_id, col_id]
                pixel_tmp = local_tmp_tensor[:, row_id, col_id]
                pixel_mintmp = local_mintmp_tensor[:, row_id, col_id]

                # Блок отсева лишних значений (иногда в данных попадаются пропуски)
                if any(pixel < -50 for pixel in pixel_rainfall):
                    pass
                else:
                    values_rainfall.append(np.ravel(pixel_rainfall))

                if any(pixel < -50 for pixel in pixel_pressure):
                    pass
                else:
                    values_pressure.append(np.ravel(pixel_pressure))

                if any(pixel < -50 for pixel in pixel_maxtmp):
                    pass
                else:
                    values_maxtmp.append(np.ravel(pixel_maxtmp))

                if any(pixel < -50 for pixel in pixel_tmp):
                    pass
                else:
                    values_tmp.append(np.ravel(pixel_tmp))

                if any(pixel < -50 for pixel in pixel_mintmp):
                    pass
                else:
                    values_mintmp.append(np.ravel(pixel_mintmp))

            # Осадки
            values_rainfall = np.array(values_rainfall)
            # Осредняем значения по каждому временному ряду - теперь ряд представляет собой ежедневные значения параметра в выбранный год
            values_rainfall = np.ravel(values_rainfall.mean(axis = 0))

            # Давление
            values_pressure = np.array(values_pressure)
            values_pressure = np.ravel(values_pressure.mean(axis=0))

            # Максимальная температура
            values_maxtmp = np.array(values_maxtmp)
            values_maxtmp = np.ravel(values_maxtmp.mean(axis=0))

            # Средняя температура
            values_tmp = np.array(values_tmp)
            values_tmp = np.ravel(values_tmp.mean(axis=0))

            # Минимальная температура
            values_mintmp = np.array(values_mintmp)
            values_mintmp = np.ravel(values_mintmp.mean(axis=0))

            # Заполняем наши агрегированные данные
            main_rainfall.append(values_rainfall)
            main_pressure.append(values_pressure)
            main_maxtmp.append(values_maxtmp)
            main_tmp.append(values_tmp)
            main_mintmp.append(values_mintmp)

        # Теперь каждая строка в данных массивах - год, который содержит 365/366 столбцов, то есть дневных данных
        main_rainfall = np.array(main_rainfall)
        main_pressure = np.array(main_pressure)
        main_maxtmp = np.array(main_maxtmp)
        main_tmp = np.array(main_tmp)
        main_mintmp = np.array(main_mintmp)

        # На этом этапе для конкретной страны уже подготовлен датасет с климатическими параметрами
        climate_data['RAINFALL'] = list(main_rainfall)
        climate_data['PRESSURE'] = list(main_pressure)
        climate_data['MAX_TMP'] = list(main_maxtmp)
        climate_data['TMP'] = list(main_tmp)
        climate_data['MIN_TMP'] = list(main_mintmp)

        # Сохраняем датасет с климатическими параметрами для данной страны в файл
        prepared_dataframe = crop_data.join(climate_data, lsuffix='_st', rsuffix='_nd')
        # Переименовываем столбцы
        prepared_dataframe = prepared_dataframe.rename(columns={'Wheat (tonnes per hectare)': 'Wheat',
                                                                'Rice (tonnes per hectare)': 'Rice',
                                                                'Maize (tonnes per hectare)': 'Maize',
                                                                'Potatoes (tonnes per hectare)': 'Potatoes',
                                                                'Peas (tonnes per hectare)': 'Peas',
                                                                'Barley (tonnes per hectare)': 'Barley',
                                                                'Nitrogen fertilizer use (kilograms per hectare)': 'Nitrogen',
                                                                'Tractors per 100 sq km arable land': 'Tractors',
                                                                'Total population (Gapminder)': 'Population',
                                                                'Land used for cereal (hectares)': 'Land_used',
                                                                'Cereal production (tonnes)': 'Cereal_production',
                                                                'Cereal yield (tonnes per hectare)': 'Cereal_yield',

                                                                'Beans (tonnes per hectare)': 'Beans',
                                                                'Soybeans (tonnes per hectare)': 'Soybeans',
                                                                'Cassava (tonnes per hectare)': 'Cassava',
                                                                'Cocoa beans (tonnes per hectare)': 'Cocoa_beans',
                                                                'Bananas (tonnes per hectare)': 'Bananas'})

        name_of_file = 'Prepared_' + country + '.parquet'
        sv = os.path.join(csv_path, name_of_file)
        prepared_dataframe.to_parquet(sv)

        # Формируем финальный датафрейм
        if country == 'France':
            # Для первой страны первый блок и будет таблицей
            main_data = prepared_dataframe
        else:
            # Для последующих стран мы последовательно присоединяем блоки к начальной таблице
            main_data = pd.concat([main_data, prepared_dataframe])

    # Сохраняем результат в файл parquet формата
    main_data = main_data.drop(columns = ['Beans', 'Soybeans', 'Cassava', 'Cocoa_beans', 'Bananas'])
    main_data.to_parquet(save)

# Папка, в которой есть необходимые файлы
csv_files_path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid'
# Применение алгоритма
prepare_data(csv_files_path,
             rainfall_file = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/rr_ens_mean_0.1deg_reg_v20.0e.nc',
             pressure_file = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/pp_ens_mean_0.1deg_reg_v20.0e.nc',
             maxtmp_file = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/tx_ens_mean_0.1deg_reg_v20.0e.nc',
             tmp_file = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/tg_ens_mean_0.1deg_reg_v20.0e.nc',
             mintmp_file = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/tn_ens_mean_0.1deg_reg_v20.0e.nc',
             countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
             land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif',
             save = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/CropAndClimate.parquet')