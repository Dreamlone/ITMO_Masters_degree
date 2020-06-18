import os
import numpy as np
import gdal, osr
import pandas as pd
from netCDF4 import Dataset
import shutil
import dask.array as da

########################################################################################################################
#           Функция для приведения ежедневных данных с климатическими параметрами в полугодовые агрегаты               #
#                                                                                                                      #
########################################################################################################################
# path           --- папка, в которой необходимо производить всю обработку
# matrix_name    --- название NetCDF файла, для которого следует получить информацию
# calculate_days --- необходимо ли считать дни с определенным значением параметра или нет
def transform_grid(path, matrix_name, calculate_days = False):

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

    # Время (начиная с 1950-01-01 00:00)
    time = np.array(parameter_matrix.variables['time'])
    # Генерируем временной ряд (нам нужны только дни до 7го июля 2019го года)
    time_series = pd.date_range(start = '01/01/1950', end = '07/31/2019', freq = '1D')
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
        threshold_id_30_June = np.ravel(np.argwhere(clipped_time == '06-30'))
        # А также индексы элементов 1го января
        threshold_id_1_Jan = np.ravel(np.argwhere(clipped_time == '01-01'))

        # Данну матрицу будем заполнять обрезанными матрицами
        main_matrix = []
        times = []
        for i in range(0, len(threshold_id_30_June)):
            print('Calculations for ', time_series[threshold_id_1_Jan[i]], ' till ', time_series[threshold_id_30_June[i]])

            # Для различных матриц будут производится различные процедуры
            if matrix_name == "rr_ens_mean_0.1deg_reg_v20.0e.nc":

                if calculate_days == False:
                    # Для осадков мы считаем их сумму
                    result_matrix = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], :, :].compute().sum(axis = 0)
                    # Теперь присвоим всем значениям, которые нам не подходят, значения -100.0
                    result_matrix[result_matrix < -1.0] = -100.0
                else:
                    # Если требуется считать дни, то
                    tmp_matrix = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], :, :].compute()
                    # Всем значениям, которые больше нуля, присваиваем единицу
                    tmp_matrix[tmp_matrix > 0.0] = 1

                    # Считаем количество дней с осадками для каждого пикселя
                    result_matrix = tmp_matrix.sum(axis=0)
                    # Все значения, которые меньше 0, присваиваем значение -100.0
                    result_matrix[result_matrix < 0.0] = -100.0


            elif matrix_name == "pp_ens_mean_0.1deg_reg_v20.0e.nc":
                # Для поля давления - берем средние значения
                result_matrix = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], :, :].compute().mean(axis = 0)
                # Теперь присвоим всем значениям, которые нам не подходят, значения -100.0
                result_matrix[result_matrix < -1.0] = -100.0

            elif matrix_name == "tx_ens_mean_0.1deg_reg_v20.0e.nc":
                # Для максимальной среднесуточной температуры воздуха - берем максимальное значение
                result_matrix = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], :, :].compute().max(axis=0)
                # Теперь присвоим всем значениям, которые нам не подходят, значения -100.0
                result_matrix[result_matrix < -1.0] = -100.0

            elif matrix_name == "tg_ens_mean_0.1deg_reg_v20.0e.nc":
                # Для среднесуточной температуры воздуха - сумму значений выше 10 градусов (Сумма активных температур)
                tmp_matrix = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], :, :].compute()

                # Теперь все значения меньше 10 - обнуляем :)
                tmp_matrix[tmp_matrix < 10.0] = 0

                # Считаем Сумму активных температур
                result_matrix = tmp_matrix.sum(axis = 0)

                # Теперь присвоим всем значениям, которые нам не подходят, значения -100.0
                result_matrix[result_matrix < 0.1] = -100.0

            elif matrix_name == "tn_ens_mean_0.1deg_reg_v20.0e.nc":
                # Для среднесуточной минимальной температуры - вибираем минимальное значение
                result_matrix = tensor[threshold_id_1_Jan[i]: threshold_id_30_June[i], :, :].compute().min(axis=0)
                # Теперь присвоим всем значениям, которые нам не подходят, значения -100.0
                result_matrix[result_matrix < -80.0] = -100.0

            # Произведем привязку полученной матрицы и её обрезку по экстенду
            ref_matrix = spatial_reference(lon, lat, result_matrix, tmp_path)

            # Полученную матрицу добавляем в единую многомерную матрицу
            main_matrix.append(ref_matrix)

            # Временные метки тоже добавляем в нужный список
            str_datetime = str(time_series[threshold_id_30_June[i]])
            str_datetime = str_datetime[:4]
            times.append(str_datetime[:4])

        main_matrix = np.array(main_matrix)
        times = np.array(times)

        # Удаляем временную директорию
        shutil.rmtree(tmp_path, ignore_errors=True)

        return(main_matrix, times)

    else:
        print('ATTENTION! There are problems with the dimension of matrices and time series')
        exit()

# additional_layer --- путь до tif файла, который следует записать как дополнительный слой
def save_netCDF(tensor, timesteps, save_path, countries_tif, land_tif):

    # Сначала загрузим подготовленный заранее дополнительный слой с границами государств
    ds = gdal.Open(countries_tif, gdal.GA_ReadOnly)
    rb = ds.GetRasterBand(1)
    country_array = np.array(rb.ReadAsArray())

    # Теперь слой кодами типов ландшафтов
    ds = gdal.Open(land_tif, gdal.GA_ReadOnly)
    rb = ds.GetRasterBand(1)
    land_array = np.array(rb.ReadAsArray())

    # Формирует netCDF файл
    root_grp = Dataset(save_path, 'w', format='NETCDF4')
    root_grp.description = 'Clipped matrices'

    # Размерности для данных, которые будем записывать в файл
    dim_tensor = tensor.shape
    root_grp.createDimension('time', len(timesteps))
    root_grp.createDimension('row', dim_tensor[1])
    root_grp.createDimension('col', dim_tensor[2])

    # Записываем данные в файл
    time = root_grp.createVariable('time', 'S2', ('time',))
    data = root_grp.createVariable('matrices', 'f4', ('time', 'row', 'col'))
    countries = root_grp.createVariable('countries', 'i2', ('row', 'col'))
    land_matrix = root_grp.createVariable('biome', 'i2', ('row', 'col'))

    data[:] = tensor
    time[:] = timesteps
    countries[:] = country_array
    land_matrix[:] = land_array

    root_grp.close()

#                                                                --- --- --- Применение алгоритма --- --- ---                                                                  #

# Количество дней с осадками за первое полугодие
#transformed_matrix, timesteps = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = "rr_ens_mean_0.1deg_reg_v20.0e.nc", calculate_days = True)
#save_netCDF(transformed_matrix, timesteps, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Precip_days.nc',
#            countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#            land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif')
#
# Сумма осадков за первое полугодие
#transformed_matrix, timesteps = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = "rr_ens_mean_0.1deg_reg_v20.0e.nc")
#save_netCDF(transformed_matrix, timesteps, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Precip_amount.nc',
#            countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#            land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif')
#
# Среднее поле давления за первые полгода
#transformed_matrix, timesteps = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = "pp_ens_mean_0.1deg_reg_v20.0e.nc")
#save_netCDF(transformed_matrix, timesteps, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Pressure_mean.nc',
#            countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#            land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif')
#
# Максимальная температура воздуха, встречавшаяся за первые полгода
#transformed_matrix, timesteps = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = "tx_ens_mean_0.1deg_reg_v20.0e.nc")
#save_netCDF(transformed_matrix, timesteps, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Temperature_max.nc',
#            countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#            land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif')
#
# Сумма активных температур выше 10 градусов
#transformed_matrix, timesteps = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = "tg_ens_mean_0.1deg_reg_v20.0e.nc")
#save_netCDF(transformed_matrix, timesteps, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Temperature_SAT.nc',
#            countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#            land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif')
#
# Минимальная температура воздуха, встречавшаяся за первые полгода
#transformed_matrix, timesteps = transform_grid(path = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe', matrix_name = "tn_ens_mean_0.1deg_reg_v20.0e.nc")
#save_netCDF(transformed_matrix, timesteps, '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Processed_grid/Temperature_min.nc',
#            countries_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Country_bounds.tif',
#            land_tif = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/LandMatrix.tif')
