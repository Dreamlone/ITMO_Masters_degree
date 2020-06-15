# Prediction of yields and Futures price of crops in Europe using climate data

Team MiECa:
- Mikhail Sarafanov
- Egor Turukhanov
- Juan Camilo Diosa E.

## Processing of climate data

Data is used: [E-OBS daily gridded meteorological data for Europe from 1950 to present](https://cds.climate.copernicus.eu/cdsapp#!/dataset/insitu-gridded-observations-europe?tab=overview)

Preprocessing of hydrometeorological information is carried out in 2 stages:
* Transformation of the source data of the reanalysis grid. Creating features
* Combining hydrometeorological information with yield data + adding additional predictors

### Transformation of the source data of the reanalysis grid. Creating features

The source archives obtained from the European space Agency website contain netCDF files. The files have daily fields for the following parameters:
- Mean daily air temperature, ℃ 
- Minimum daily air temperature, ℃
- Maximum daily air temperature, ℃
- Pressure, HPa
- Precipitation, mm

Based on the initial parameters, indicators for the first half of each year were calculated:
- 1) Precip_amount - total rainfall for the first half of the year, mm
- 2) Precip_days - the number of days with precipitation for the first half of the year, days
- 3) Pressure_mean - average pressure, hpa
- 4) Temperature_max - maximum average daily air temperature for the first six months, ℃
- 5) Temperature_min - the minimum average daily temperature for the first six months, ℃
- 6) Temperature_SAT - the sum of active temperatures above 10 degrees Celsius, ℃

An example of an algorithm for calculating the sum of active temperatures above 10 degrees Celsius can be seen below:
![Data_preparation.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_1.png)

As a result of these conversions, fields with attribute values for each year were obtained. An example for the sum of active temperatures above 10 degrees for the first half of the year in 1950 can be seen below.
![SAT_10.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_3.png)

An algorithm is used to implement the above actions:

#### [TransformNetCDF](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/ClimateProcessing/TransformNetCDF.py)

To speed up the runtime, a powerful library for parallelizing calculations is used - Dask.


```python
# Количество дней с осадками за первое полугодие
transformed_matrix, timesteps = transform_grid(path = '...', 
                                               matrix_name = "rr_ens_mean_0.1deg_reg_v20.0e.nc", 
                                               calculate_days = True)
save_netCDF(transformed_matrix, timesteps, '.../Precip_days.nc',
            countries_tif = '.../Country_bounds.tif',
            land_tif = '.../LandMatrix.tif')

# Сумма осадков за первое полугодие
transformed_matrix, timesteps = transform_grid(path = '...', 
                                               matrix_name = "rr_ens_mean_0.1deg_reg_v20.0e.nc")
save_netCDF(transformed_matrix, timesteps, '.../Precip_amount.nc',
            countries_tif = '.../Country_bounds.tif',
            land_tif = '.../LandMatrix.tif')

# Среднее поле давления за первые полгода
transformed_matrix, timesteps = transform_grid(path = '...', 
                                               matrix_name = "pp_ens_mean_0.1deg_reg_v20.0e.nc")
save_netCDF(transformed_matrix, timesteps, '.../Pressure_mean.nc',
            countries_tif = '.../Country_bounds.tif',
            land_tif = '.../LandMatrix.tif')

# Максимальная температура воздуха, встречавшаяся за первые полгода
transformed_matrix, timesteps = transform_grid(path = '...', 
                                               matrix_name = "tx_ens_mean_0.1deg_reg_v20.0e.nc")
save_netCDF(transformed_matrix, timesteps, '.../Temperature_max.nc',
            countries_tif = '.../Country_bounds.tif',
            land_tif = '.../LandMatrix.tif')

# Сумма активных температур выше 10 градусов
transformed_matrix, timesteps = transform_grid(path = '...', 
                                               matrix_name = "tg_ens_mean_0.1deg_reg_v20.0e.nc")
save_netCDF(transformed_matrix, timesteps, '.../Temperature_SAT.nc',
            countries_tif = '.../Country_bounds.tif',
            land_tif = '.../LandMatrix.tif')

# Минимальная температура воздуха, встречавшаяся за первые полгода
transformed_matrix, timesteps = transform_grid(path = '...', 
                                               matrix_name = "tn_ens_mean_0.1deg_reg_v20.0e.nc")
save_netCDF(transformed_matrix, timesteps, '.../Temperature_min.nc',
            countries_tif = '.../Country_bounds.tif',
            land_tif = '.../LandMatrix.tif')
```

The result of the algorithm is netCDF files with parameter fields. The files also include a matrix of landscape types and a vector consisting of timestamps.

### Combining hydrometeorological information with yield data

The received information about climate parameters is aggregated by country. Data is averaged for each country. A specially prepared country matrix is used for the aggregation procedure. The matrix has the same resolution as the reanalysis grid data.
![Countries.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_2.png)

An algorithm is used to create a bitmap with countries:

#### [Rasterizer](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/ClimateProcessing/Rasterizer.py)

The input data is a vector layer with state borders and a matrix to copy the resolution and spatial reference from. The result is a bitmap image in geotiff format.


```python
# Загружаем шейпфайл из папки
world = gpd.read_file("../Reanalysis_grid_Europe/World_bounds/worldbounds.shp")
# Файл, который служит образцом
rst_fn = '../Reanalysis_grid_Europe/Example_SAT1950.tif'
# Файл, в который будем сохранять растровый слой
out_fn = '../Reanalysis_grid_Europe/world_rasterized.tif'
rasterize(world, rst_fn, out_fn)
```

Clearly, not all of the countries' territories are used as agricultural land. On the territory of European countries, there are mountains, urbanized territories, and nature reserves.

In order to take into account only information that relates to agricultural land, a matrix of landscape types was used. All values that were not related to agricultural land were removed from the analysis.

![Land.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_4.png)

An algorithm was used to combine data into tables for analysis

#### [MergeForm](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/ClimateProcessing/MergeForm.py)



```python
# Подготовим датасеты по следующим странам
aggregate_by_country(country = 'Germany', netCDf_dir = '...',
                     crop_data = '.../Crop_yields_info.csv',
                     fertilizer_data = '.../Cereal_crop_yield_vs_fertilizer_application.csv',
                     tractor_data = '.../Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
                     land_data = '.../Index_of_cereal_production_yield_and_land_use.csv')

aggregate_by_country(country = 'France', netCDf_dir = '...',
                     crop_data = '.../Crop_yields_info.csv',
                     fertilizer_data = '.../Cereal_crop_yield_vs_fertilizer_application.csv',
                     tractor_data = '.../Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
                     land_data = '.../Index_of_cereal_production_yield_and_land_use.csv')

aggregate_by_country(country = 'Italy', netCDf_dir = '...',
                     crop_data = '.../Crop_yields_info.csv',
                     fertilizer_data = '.../Cereal_crop_yield_vs_fertilizer_application.csv',
                     tractor_data = '.../Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
                     land_data = '.../Index_of_cereal_production_yield_and_land_use.csv')

aggregate_by_country(country = 'Romania', netCDf_dir = '...',
                     crop_data = '.../Crop_yields_info.csv',
                     fertilizer_data = '.../Cereal_crop_yield_vs_fertilizer_application.csv',
                     tractor_data = '.../Cereal_yields_vs_tractor_inputs_in_agriculture.csv',
                     land_data = '.../Index_of_cereal_production_yield_and_land_use.csv')
```

### Analysis of the distribution of climate parameters 

For one of the predictive models, we need to get distributions of certain climate parameters for the selected periods:
* For the first half of each year
* For February of each year
* For April of each year
* For June of each year

The parameters considered are: average daily air temperature, average air pressure at sea level, and precipitation. Examples of the obtained distributions for 2000 and 2018 for temperature can be seen in the figure below.

![Distribution.png](https://raw.githubusercontent.com/Dreamlone/ITMO_Masters_degree/master/Images/img_5.png)

Algorithm for preparing distributions in binary npy format

#### [TransformNetCDF_distribution](https://github.com/Dreamlone/ITMO_Masters_degree/blob/master/Methods_and_models_for_multivariate_data_analysis/ClimateProcessing/TransformNetCDF_distribution.py)




```python
# path --- the path where the netCDF files are located
# country --- name of the country that parameters are aggregated for
# countries_tif --- matrix with state borders
# land_tif --- matrix with biomes
# START --- start of the aggregation period
# STOP --- end of the aggregation period
# where_to_save --- folder where you want to save the result
for name in ['rr_ens_mean_0.1deg_reg_v20.0e.nc', 'tg_ens_mean_0.1deg_reg_v20.0e.nc', 'pp_ens_mean_0.1deg_reg_v20.0e.nc']:
    transformed_matrix, timesteps, start, stop = transform_grid(path = '.../Reanalysis_grid_Europe', matrix_name = name,
                                                                country = 'Poland',
                                                                countries_tif ='.../Reanalysis_grid_Europe/Country_bounds.tif',
                                                                land_tif ='.../Reanalysis_grid_Europe/LandMatrix.tif',
                                                                START ='06-01',
                                                                STOP = '06-30',
                                                                where_to_save = '.../Reanalysis_grid_Europe/Processed_grid/Distributions_files/Poland')
```
