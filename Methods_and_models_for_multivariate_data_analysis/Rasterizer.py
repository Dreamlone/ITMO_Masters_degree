
import geopandas as gpd
import rasterio
from rasterio import features

# Загружаем шейпфайл из папки
world = gpd.read_file("/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/World_bounds/worldbounds.shp")

rst_fn = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/Example_SAT1950.tif'
out_fn = '/media/mikhail/Data/ITMO/Reanalysis_grid_Europe/worl_rasterized.tif'

rst = rasterio.open(rst_fn)
meta = rst.meta.copy()
meta.update(compress = 'lzw')

with rasterio.open(out_fn, 'w+', **meta) as out:
    out_arr = out.read(1)

    # this is where we create a generator of geom, value pairs to use in rasterizing
    shapes = ((geom,value) for geom, value in zip(world.geometry, world.ID))

    burned = features.rasterize(shapes=shapes, fill=0, out=out_arr, transform=out.transform)
    out.write_band(1, burned)