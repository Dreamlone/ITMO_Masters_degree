import os
import numpy as np
import gdal, osr
import pandas as pd
from netCDF4 import Dataset
import shutil
import dask.array as da

import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib import pyplot
from mpl_toolkits.mplot3d import Axes3D
from pylab import rcParams

# path        --- папка, в которой необходимо производить всю обработку
# matrix_name --- название NetCDF файла, для которого следует получить информацию
# country     --- название страны, для которой формируется датасет
# where_to_save --- в какую директорию требуется сохранять файлы
def transform_grid(path, matrix_name, country, countries_tif, land_tif, START, STOP, where_to_save):

    if matrix_name == "rr_ens_mean_0.1deg_reg_v20.0e.nc":
        where_to_save = os.path.join(where_to_save, 'Rainfall')
    elif matrix_name == "pp_ens_mean_0.1deg_reg_v20.0e.nc":
        where_to_save = os.path.join(where_to_save, 'Pressure')
    elif matrix_name == "tx_ens_mean_0.1deg_reg_v20.0e.nc":
        where_to_save = os.path.join(where_to_save, 'Max_tmp')
    elif matrix_name == "tg_ens_mean_0.1deg_reg_v20.0e.nc":
        where_to_save = os.path.join(where_to_save, 'Tmp')
    elif matrix_name == "tn_ens_mean_0.1deg_reg_v20.0e.nc":
        where_to_save = os.path.join(where_to_save, 'Min_tmp')

    # Создаем директорию, сели её не существовало
    if os.path.isdir(where_to_save) == False:
        os.makedirs(where_to_save)

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

    # Определим функцию пространственной привязки
    def spatial_reference(lon, lat, matrix, tmp_path):
        # Теперь провернем процедуру создания матриц для пространственной привязки
        lon, lat = np.meshgrid(lon, lat)

        # дружно обратим последовательность строк во всех массивах
        div = np.ma.array(matrix)
        div = np.flip(div, axis=0)
        lats = np.flip(lat, axis=0)
        lons = np.flip(lon, axis=0)

        # выставим настройки типа данных и типа используемого драйвера, а также всех путей:
        dataType = gdal.GDT_Float64
        driver = gdal.GetDriverByName("GTiff")
        latPath = os.path.join(tmp_path, 'lat.tif')
        lonPath = os.path.join(tmp_path, 'lon.tif')
        imagePath = os.path.join(tmp_path, 'image.tif')
        imageVRTPath = os.path.join(tmp_path, 'image.vrt')

        # Создаем растр для широт (..\TEMP\lat.tif):
        dataset = driver.Create(latPath, div.shape[1], div.shape[0], 1, dataType)
        dataset.GetRasterBand(1).WriteArray(lats)

        # Создаем растр для долгот (..\TEMP\lon.tif):
        dataset = driver.Create(lonPath, div.shape[1], div.shape[0], 1, dataType)
        dataset.GetRasterBand(1).WriteArray(lons)

        # Создаем растр для данных (..\TEMP\image.tif)
        dataset = driver.Create(imagePath, div.shape[1], div.shape[0], 1, dataType)
        dataset.GetRasterBand(1).WriteArray(div)

        # Установим СК WGS84
        gcp_srs = osr.SpatialReference()
        gcp_srs.ImportFromEPSG(4326)
        proj4 = gcp_srs.ExportToProj4()

        # На основе tif-а создадим vrt (..\TEMP\image.vrt)
        vrt = gdal.BuildVRT(imageVRTPath, dataset, separate = True, resampleAlg = 'cubic', outputSRS = proj4)
        band = vrt.GetRasterBand(1)

        # Привяжем координаты к виртуальному растру...
        metadataGeoloc = {
            'X_DATASET': lonPath,
            'X_BAND': '1',
            'Y_DATASET': latPath,
            'Y_BAND': '1',
            'PIXEL_OFFSET': '0',
            'LINE_OFFSET': '0',
            'PIXEL_STEP': '1',
            'LINE_STEP': '1'
        }

        # ...записав это все в <Metadata domain='Geolocation'>:
        vrt.SetMetadata(metadataGeoloc, "GEOLOCATION")

        dataset = None
        vrt = None

        # outputBounds = ['minX', 'minY', 'maxX', 'maxY']
        # Закатаем виртуальный растр для получения привязанной сетки (координаты границ обрезки задаем вручную, все равно менять мы их не будем, да и вредно это, координаты менять)
        warpOptions = gdal.WarpOptions(geoloc=True, format='GTiff', dstNodata = -9999.0, srcSRS = proj4, dstSRS = gcp_srs,
                                       outputBounds = [-25, 36, 38, 70], xRes = 0.1, yRes = 0.1,
                                       creationOptions = ['COMPRESS=LZW'])

        geotiff_name = 'matrix.tif'
        geotiff_path = os.path.join(tmp_path, geotiff_name)
        raster = gdal.Warp(geotiff_path, imageVRTPath, dstNodata = -9999.0, options=warpOptions)

        # Теперь обрезанную матрицу представим в виде numpy массива
        matrix = raster.ReadAsArray()
        matrix = np.array(matrix)

        raster = None
        return(matrix)

    # Временная директория, в которую будут сохраняться промежуточные файлы
    tmp_path = os.path.join(path, 'TMP')
    if os.path.isdir(tmp_path) == False:
        os.makedirs(tmp_path)

    # Доступ к указанному файлу
    file_path = os.path.join(path, matrix_name)
    parameter_matrix = Dataset(file_path)

    print('Information about the selected file')
    print(parameter_matrix.variables)

    # Вектор широт
    lat = np.array(parameter_matrix.variables['latitude'])
    # Вектор долгот
    lon = np.array(parameter_matrix.variables['longitude'])
    # Матрица с нужным нам параметром
    if matrix_name == "rr_ens_mean_0.1deg_reg_v20.0e.nc":
        netcdf_matrix = parameter_matrix.variables['rr']           # Осадки
    elif matrix_name == "pp_ens_mean_0.1deg_reg_v20.0e.nc":
        netcdf_matrix = parameter_matrix.variables['pp']           # Давление на уровне моря
    elif matrix_name == "tx_ens_mean_0.1deg_reg_v20.0e.nc":
        netcdf_matrix = parameter_matrix.variables['tx']           # Максимальная суточная температура воздуха
    elif matrix_name == "tg_ens_mean_0.1deg_reg_v20.0e.nc":
        netcdf_matrix = parameter_matrix.variables['tg']           # Средняя суточная температура воздуха
    elif matrix_name == "tn_ens_mean_0.1deg_reg_v20.0e.nc":
        netcdf_matrix = parameter_matrix.variables['tn']           # Минимальная суточная температура воздуха

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

    # Время (начиная с 1950-01-01 00:00)
    time = np.array(parameter_matrix.variables['time'])
    # Генерируем временной ряд (нам нужны только дни до 7го июля 2019го года)
    time_series = pd.date_range(start = '01/01/1950', end = '07/31/2019', freq = '1D')
    years_series = pd.date_range(start='01/01/1950', end='01/01/2020', freq='1Y')
    if len(time) == len(time_series):
        print('Ok, we have no problems with size of matrix')
        # Читаем матрицу в Dask массив
        tensor = da.from_array(netcdf_matrix, chunks = (5000, 465, 705))

        # Теперь переводим дату в строковый тип и обрезаем
        clipped_time = []
        for date in time_series:
            str_date = str(date)
            # Теперь записываем в массив только месяц и день (без года)
            clipped_time.append(str_date[5:10])
        clipped_time = np.array(clipped_time)

        # Ищем в массиве индексы тех элементов, которые соответствуют 30го июня
        threshold_id_30_June = np.ravel(np.argwhere(clipped_time == STOP))
        # А также индексы элементов 1го января
        threshold_id_1_Jan = np.ravel(np.argwhere(clipped_time == START))

        # Для каждого года мы будем собирать некоторе количество значений определенного промежутка
        for i in range(0, len(threshold_id_30_June)):
            print('Calculations for ', time_series[threshold_id_1_Jan[i]], ' till ', time_series[threshold_id_30_June[i]])

            local_tensor = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], 110:450,:630].compute()
            # Для каждого интересующего на пикселя - получаем временной ряд за первое полугодие
            values = []
            for id_pixel in frame['ID']:
                # Разделим координаты
                id_pixel = id_pixel.split(':')
                row_id = int(id_pixel[0])
                col_id = int(id_pixel[1])
                pixel_timesries = local_tensor[:, row_id, col_id]
                if any(pixel < -50 for pixel in pixel_timesries):
                    pass
                else:
                    values.append(np.ravel(pixel_timesries))
            values = np.array(values)
            values = np.ravel(values)

            # with sns.axes_style("darkgrid"):
            #     rcParams['figure.figsize'] = 15, 8
            #     plt.hist(values, 70, density=True, color='blue', alpha=0.2)
            #     sns.kdeplot(values, shade=False, color='blue', kernel='gau', alpha=1.0,label='Distribution for the first half of the year', linewidth=2)
            #     plt.ylabel('Probability density', fontsize=17)
            #     plt.xlabel('Температура, Франция', fontsize=17)
            #     plt.legend(fontsize=13)
            #     plt.show()

            # Временные метки тоже добавляем в нужный список
            str_datetime = str(years_series[i])
            str_datetime = str_datetime[:4]

            # Сохраняем собранную выборку в бинарный файл
            name = str_datetime + '_' + START + '_' + STOP + '.npy'
            save_path = os.path.join(where_to_save, name)
            np.save(save_path, values)

        # Удаляем временную директорию
        shutil.rmtree(tmp_path, ignore_errors=True)

        return(0, 0, START, STOP)

    else:
        print('ATTENTION! There are problems with the dimension of matrices and time series')
        exit()


#                                                                --- --- --- Применение алгоритма --- --- ---                                                                  #

# Осадки - rr_ens_mean_0.1deg_reg_v20.0e.nc
# Среднесуточная температура воздуха - tg_ens_mean_0.1deg_reg_v20.0e.nc
# Давление - pp_ens_mean_0.1deg_reg_v20.0e.nc
# Страны: France Germany Italy Romania Spain Czech Republic Netherlands Switzerland Austria Poland
# for nazvanie in ['rr_ens_mean_0.1deg_reg_v20.0e.nc', 'tg_ens_mean_0.1deg_reg_v20.0e.nc', 'pp_ens_mean_0.1deg_reg_v20.0e.nc']:
#     transformed_matrix, timesteps, start, stop = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = nazvanie,
#                                                                 country = 'Poland',
#                                                                 countries_tif='/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#                                                                 land_tif='/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif',
#                                                                 START='06-01',
#                                                                 STOP = '06-30',
#                                                                 where_to_save = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Distributions_files/Poland')


