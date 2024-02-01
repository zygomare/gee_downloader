###download s2 images from GEE for specific region defined by geojson file#####
### Yanqun Pan, 2023-06-03

import os, glob, pathlib
import sys

import geopandas as gpd
import pendulum
import pickle
import numpy as np
import pandas as pd
import shutil
import requests
import pickle
from contextlib import contextmanager

import rasterio

from rasterio.warp import reproject,calculate_default_transform,Resampling
from rasterio.io import MemoryFile
from rasterio.merge import merge

import ee
from shapely.geometry import MultiPolygon

def gen_subcells(cell_geometry: MultiPolygon, x_step=0.1, y_step=0.1, x_overlap=None, y_overlap=None):
    '''
    because the maximum number of pixels and dimensions of extent that can be downloaded from
    GEE are limited, the big extent has be grided by the x_step (longitude) and y_step (latitude) parameters.
    '''
    x_overlap = x_overlap if x_overlap is not None else x_step * 0.2
    y_overlap = y_overlap if y_overlap is not None else y_step * 0.2
    bounds = cell_geometry.bounds
    # print(bounds)
    _x = (bounds[0],)
    x = []
    while True:
        if (_x[0] + x_step) > bounds[2]:
            _x = _x + (bounds[2],)
            x.append(_x)
            break
        else:
            _x = _x + (_x[0] + x_step,)
            x.append(_x)
            _x = (_x[1] - x_overlap,)
    _y = (bounds[1],)
    y = []
    while True:
        if (_y[0] + y_step) >= bounds[3]:
            _y = _y + (bounds[3],)
            y.append(_y)
            break
        else:
            _y = _y + (_y[0] + y_step,)
            y.append(_y)
            _y = (_y[1] - y_overlap,)
    # print(x, y)
    return x, y


def water_pixels_regular(roi_rect, resolution=10, occurrence_threshold=10):
    '''
    obtain number of water pixels based on Global Surface Water
    '''
    import ee
    dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(roi_rect)  ### the resolution is 30 m
    water = dataset.select('occurrence').gt(occurrence_threshold)
    water = water.rename('surface_water')
    img = dataset.addBands(water)
    waterpixels = img.reduceRegion(
        reducer=ee.Reducer.sum(),
        geometry=roi_rect,
        scale=resolution,
        maxPixels=1e11
    ).get('surface_water')
    dims = img.getInfo()['bands'][-1]['dimensions']
    total_pixels = dims[0] * (30.0 / resolution) * dims[1] * (30.0 / resolution)
    img = img.set('water_num', waterpixels)
    water_pixels = ee.Number(img.get('water_num')).getInfo()
    return water_pixels, total_pixels


def add_waterpixelsnumber_s2l2(images: ee.ImageCollection, roi_rect, resolution=10):
    '''
    add number of water pixels based on the band of classification in L2 image
    '''


    def __f(image: ee.Image):
        slc = image.select('SCL')  ## the resolution 20m
        water_mask = slc.eq(6).Or(slc.eq(11)) ## water or (snow or ice)
        water = water_mask.rename('water')
        image = image.addBands(water)
        waterpixels = image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_rect,
            scale=resolution,
            maxPixels=1e11
        ).get('water')
        return image.set('water_num', waterpixels)

    images = images.map(__f)
    return images

def add_surfacewater_cloudprob(cloud_images: ee.ImageCollection, roi_rect, resolution=10):
    '''
    merge surfacewater and cloudprob
    '''
    dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(roi_rect)  ### the resolution is 30 m






def add_cloudpixelsnumber_image(images: ee.ImageCollection, roi_rect, resolution=10, threshold_prob=50):
    dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(roi_rect)
    water_mask = dataset.select('occurrence').gt(0)

    # images = images.updateMask(water_mask)

    def __f(image:ee.Image):
        image = image.updateMask(water_mask)
        prob = image.select('probability')
        cloud_mask = prob.gt(threshold_prob)
        total_mask = prob.gt(0)

        cloud = cloud_mask.rename('cloud')
        total = total_mask.rename('total')

        image = image.addBands(cloud).addBands(total)

        cloudpixels = image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_rect,
            scale=resolution,
            maxPixels=1e11
        ).get('cloud')

        totalpixels = image.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=roi_rect,
            scale=resolution,
            maxPixels=1e11
        ).get('total')


        return image.set('cloud_num', cloudpixels).set('total_num', totalpixels)
    images = images.map(__f)
    return images


def download_images_roi(images: ee.ImageCollection, roi_geo: MultiPolygon, save_dir, bands, resolution=10):
    '''
    @bands, for s2 ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'QA60']
    '''
    xs, ys = gen_subcells(roi_geo, x_step=0.1, y_step=0.1)
    ee_small_cells = [ee.Geometry.Rectangle([x[0], y[0], x[1], y[1]]) for x in xs for y in ys]
    ee_small_cells_box = [([x[0], y[0], x[1], y[1]]) for x in xs for y in ys]
    # print(images)
    img_count = len(images.getInfo()['features'])
    for i in range(img_count):
        _id = images.getInfo()['features'][i]['id']
        img = ee.Image(_id)
        info = img.getInfo()
        _id_name = _id.split('/')[-1]
        with open(os.path.join(save_dir, f'{_id_name}_info.pickle'), 'wb') as f:
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
        # print(_id.split('/')[-1][:8] + '_' + _id.split('_')[-1])
        name = _id_name[:8] + '_' + _id.split('_')[-1]

        for j, (ee_c, ee_c_b) in enumerate(zip(ee_small_cells, ee_small_cells_box)):
            try:
                url = img.getDownloadURL({
                    'name': 'multi_band',
                    'bands': bands,
                    'region': ee_c,
                    'scale': resolution,
                    'filePerBand': False,
                    'format': 'GEO_TIFF'})

            except Exception as e:
                print(ee_c_b, e)
                continue
            # print(url)
            print(f'downloading {url}')
            response = requests.get(url)
            with open(os.path.join(save_dir, name + '_' + str(j) + '.tif'), 'wb') as f:
                f.write(response.content)
    print("downloading finished")


def reproject_raster_dataset(src, dst_crs):
    # reproject raster to project crs
    # return a dataset in memory
    # with rio.open(in_path) as src:
    src_crs = src.crs
    transform, width, height = calculate_default_transform(src_crs, dst_crs, src.width, src.height, *src.bounds)
    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height})
    with MemoryFile() as memfile:
        with memfile.open(**kwargs) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=Resampling.nearest)
        # with memfile.open() as dataset:  # Reopen as DatasetReader
        #     yield dataset  # Note yield not return as we're a contextmanager
        return memfile.open()


def merge_tifs(tif_files, out_file, descriptions, descriptions_meta, obs_geo=None, bandnames=None, dst_crs=None):
    '''
    @obs_geo, None or (theta_z, theta_s, phi)
    @bandnames: None or a list

    '''
    tif_files = sorted(tif_files, key=lambda x: os.path.getsize(x))
    if bandnames is not None:
        bandnames_c = bandnames.copy()
    src_files_to_mosaic = []
    dst_crs_ret = dst_crs
    for i, tif in enumerate(tif_files[::-1]):
        src = rasterio.open(tif, 'r')
        crs = src.meta['crs']
        if dst_crs_ret is None:
            dst_crs_ret = crs
            src_files_to_mosaic.append(src)
            continue
        if crs == dst_crs_ret:
            src_files_to_mosaic.append(src)
            continue
        reproj_src = reproject_raster_dataset(src, dst_crs=dst_crs_ret)
        src_files_to_mosaic.append(reproj_src)
    mosaic, out_trans = merge(src_files_to_mosaic)
    out_meta = src.meta.copy()
    # print(out_meta)

    # print(obs_geo)
    if obs_geo is not None:
        obs_geo_arr = np.full((3,) + mosaic[0].shape, 0, dtype=out_meta['dtype'])
        valid_mask = mosaic[2] > 0
        obs_geo_arr[0, valid_mask] = obs_geo[0]
        obs_geo_arr[1, valid_mask] = obs_geo[1]
        obs_geo_arr[2, valid_mask] = obs_geo[2]
        mosaic = np.vstack([mosaic, obs_geo_arr])
        if bandnames is not None:
            bandnames_c += ['theta_z', 'theta_s', 'phi']
    out_meta.update(
        {"driver": "GTiff",
         "height": mosaic.shape[1],
         "width": mosaic.shape[2],
         "transform": out_trans,
         "count": mosaic.shape[0],
         "crs": dst_crs_ret
         }
    )
    with rasterio.open(out_file, 'w', **out_meta) as dst:
        # if description is not None:
        #     f.descriptions = description
        dst.update_tags(info=descriptions)
        dst.update_tags(info_item=descriptions_meta)
        dst.write(mosaic)
        if bandnames is not None:
            # print(len(bandnames_c),mosaic.shape)
            dst.descriptions = tuple(bandnames_c)
    return 1, dst_crs_ret


class LogRow():

    def __init__(self, roi_name):
        self.roi_name = roi_name
        self.water_percentage = np.nan
        self.waterpixel_regular = np.nan
        self.date = ''
        self.images_l1 = np.nan
        self.images_l2 = np.nan
        self.water_percentage_image = np.nan
        self.cloud_percentage_image = np.nan

    def get_dataframe(self):
        return pd.DataFrame(data=[
            ['', self.roi_name, self.water_percentage, self.waterpixel_regular, self.date, self.images_l2, self.images_l1,
             self.water_percentage_image,self.cloud_percentage_image]],
                            columns=['', 'roi_name', 'water_percentage', 'waterpixel_regular', 'date', 'images_l2','images_l1',
                                     'water_percentage_image','cloud_percentage_image'])


class Log():

    def __init__(self, csvf):
        self.__csvf = csvf
        if not os.path.exists(csvf):
            pd.DataFrame(data=[], columns=['roi_name', 'water_percentage', 'waterpixel_regular', 'date', 'images_l2', 'images_l1',
                                           'water_percentage_image','cloud_percentage_image']).to_csv(self.__csvf)

    def get_water_percentage_regular(self, roi_name):
        if not os.path.exists(self.__csvf):
            return np.nan, np.nan
        df = pd.read_csv(self.__csvf, index_col=0)
        df_filter = df[df['roi_name'] == roi_name]
        if df_filter.shape[0] == 0:
            return np.nan, np.nan
        water_percentage = df_filter.iloc[0]['water_percentage']
        waterpixel_regular = df_filter.iloc[0]['waterpixel_regular']
        return water_percentage, waterpixel_regular

    def get_water_percentage_image(self, roi_name, date):
        if not os.path.exists(self.__csvf):
            return np.nan
        df = pd.read_csv(self.__csvf, index_col=0)
        df_filter = df[(df['roi_name'] == roi_name) & (df['date'] == date)]
        if df_filter.shape[0] == 0:
            return np.nan
        water_percentage = df_filter.iloc[0]['water_percentage_image']
        return water_percentage

    def get_cloud_percentage_image(self, roi_name, date):
        if not os.path.exists(self.__csvf):
            return np.nan
        df = pd.read_csv(self.__csvf, index_col=0)
        df_filter = df[(df['roi_name'] == roi_name) & (df['date'] == date)]
        if df_filter.shape[0] == 0:
            return np.nan
        cloud_percentage = df_filter.iloc[0]['cloud_percentage_image']
        return cloud_percentage

    def get_images_cover_l2(self, roi_name, date):
        if not os.path.exists(self.__csvf):
            return np.nan
        df = pd.read_csv(self.__csvf, index_col=0)
        df_filter = df[(df['roi_name'] == roi_name) & (df['date'] == date)]
        if df_filter.shape[0] == 0:
            return np.nan
        images = int(df_filter.iloc[0]['images_l2'])
        return images

    def get_images_cover_l1(self, roi_name, date):
        if not os.path.exists(self.__csvf):
            return np.nan
        df = pd.read_csv(self.__csvf, index_col=0)
        df_filter = df[(df['roi_name'] == roi_name) & (df['date'] == date)]
        if df_filter.shape[0] == 0:
            return np.nan
        images = int(df_filter.iloc[0]['images_l1'])
        return images

    def insert(self, logrow: LogRow):
        df = logrow.get_dataframe()
        df.to_csv(self.__csvf, mode='a', index=False, header=False)
        return True


class Downloader(object):
    def __init__(self, savedir, water_threshold=60, water_threshold_regular=10, **kwargs):
        '''
        @water_threshold, the ratio of water pxiels from image and water pixels regular,
        it is used to exclude cloudy images
        @water_threshold_regular, the ratio of water pixels and total pixels based on the
        global surface water map
        '''
        ee.Initialize()
        self.resolution = 10
        self.save_dir = savedir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        # self.gee_datasource_dic = {
        #     'msi_l1':'COPERNICUS/S2_HARMONIZED',
        #     'msi_l2':'COPERNICUS/S2_SR_HARMONIZED'
        # }
        self.sensor_info = {
            'msi_l1': {
                'data_source': 'COPERNICUS/S2_HARMONIZED',
                'download_bands': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
            },
            'msi_l2_rgb': {
                'data_source': 'COPERNICUS/S2_SR_HARMONIZED',
                'download_bands': ['TCI_R', 'TCI_G', 'TCI_B']
            },
            'msi_l2': {
                'data_source': 'COPERNICUS/S2_SR_HARMONIZED',
                'download_bands': ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B11', 'B12']
            },
            'msi_cloudprob' : {
                'data_source': 'COPERNICUS/S2_CLOUD_PROBABILITY',
                'download_bands': ['probability']
            }
        }

        self.water_threshold = water_threshold
        self.water_threshold_regular = water_threshold_regular
        self.remove_download_tiles = False if 'remove_download_tiles' not in kwargs else kwargs['remove_download_tiles']
        self.over_write_l1 = False if 'over_write_l1' not in kwargs else kwargs['over_write_l1']
        self.over_write_l2rgb = False if 'over_write_l2rgb' not in kwargs else kwargs['over_write_l2rgb']
        self.over_write_l2 = False if 'over_write_l2' not in kwargs else kwargs['over_write_l2']
        self.over_write_cldprob = False if 'over_write_cldprob' not in kwargs else kwargs['over_write_cldprob']


        self.cloud_prob_threshold = 60.0 if 'cloud_prob_threshold' not in kwargs else float(kwargs['cloud_prob_threshold'])
        self.cloud_percentage_threshold = 40.0 if 'cloud_percentage_threshold' not in kwargs else float(kwargs['cloud_percentage_threshold'])

        ##
        self.__download_log_csv = os.path.join(self.save_dir, 'download_log.csv')
        self.log = Log(self.__download_log_csv)

    def set_roi(self, roi_geo: MultiPolygon, roi_name):
        self.roi_name = roi_name
        self.roi_geo = roi_geo
        self.roi_bounds = self.roi_geo.bounds
        self.roi_rect = ee.Geometry.Rectangle(self.roi_bounds)
        self.l1_savedir = os.path.join(self.save_dir, 'l1_toa', self.roi_name)
        self.l2_savedir = os.path.join(self.save_dir, 'l2_surf', self.roi_name)
        self.l2rgb_savedir = os.path.join(self.save_dir, 'l2_rgb', self.roi_name)
        self.cldprob_savedir = os.path.join(self.save_dir, 'cld_prob', self.roi_name)


    def get_s2_cloudpercentage(self, s_d, e_d):
        images_cloudprob = ee.ImageCollection(self.sensor_info['msi_cloudprob']['data_source']). \
            filterDate(s_d, e_d).filterBounds(self.roi_rect)

        # images = add_surfacewater_cloudprob(images_cloudprob,roi_rect=self.roi_rect)

        img_count_cloudprob = len(images_cloudprob.getInfo()['features'])
        images_cloud = add_cloudpixelsnumber_image(images_cloudprob,roi_rect=self.roi_rect,threshold_prob=self.cloud_prob_threshold,resolution=20)

        images_cloud_list = images_cloud.toList(images_cloud.size())

        total_pixels, cloud_pixels = 0, 0
        for i in range(img_count_cloudprob):
            img_1 = ee.Image(images_cloud_list.get(i)).clip(self.roi_rect)
            cloudnum = ee.Number(img_1.get('cloud_num')).getInfo() ##90:30 60:986
            totalnum = ee.Number(img_1.get('total_num')).getInfo()
            cloud_pixels += cloudnum
            total_pixels += totalnum

        if total_pixels == 0:
            raise Exception('no water pixels')
        return cloud_pixels/total_pixels*100



    def download_s2(self, start_date, end_date, l1=True, l2rgb=True, l2=True, cloud_prob=True):
        insert_flag = False
        if not (l1 or l2rgb or l2):
            print("l1 or l2rgb or l2?")
            return
        start_date = pendulum.from_format(start_date, 'YYYY-MM-DD')
        end_date = pendulum.from_format(end_date, 'YYYY-MM-DD')
        ### check regular water body percentage in this ROI
        water_pixel_resolution = 20
        ### search in the log csv
        water_percentage, waterpixels_regular = self.log.get_water_percentage_regular(self.roi_name)
        logrow = LogRow(roi_name=self.roi_name)
        if np.isnan(water_percentage):
            waterpixels_regular, pixels_total = water_pixels_regular(roi_rect=self.roi_rect,
                                                                     resolution=water_pixel_resolution,
                                                                     occurrence_threshold=10)
            water_percentage = waterpixels_regular / pixels_total * 100.
            insert_flag = True
        logrow.water_percentage = water_percentage
        logrow.waterpixel_regular = waterpixels_regular
        logrow.water_percentage_image = 0
        logrow.images = 0

        if water_percentage < self.water_threshold_regular:
            print(f'water percentage < {self.water_threshold_regular} %, ROI of {self.roi_name} is skiped')
            if insert_flag: self.log.insert(logrow=logrow)
            return

        for _date in (end_date - start_date):
            logrow.water_percentage_image = 0
            logrow.cloud_percentage_image = 0
            logrow.images = 0
            s_d, e_d = _date.format('YYYY-MM-DD'), (_date + pendulum.duration(days=1)).format('YYYY-MM-DD')
            logrow.date = s_d
            l1_save_dir = os.path.join(self.l1_savedir, s_d)
            l2_save_dir = os.path.join(self.l2_savedir, s_d)
            l2rgb_save_dir = os.path.join(self.l2rgb_savedir, s_d)
            cldprob_save_dir = os.path.join(self.cldprob_savedir, s_d)

            l1_of = pathlib.Path(l1_save_dir).parent / f'S2_{s_d}_L1TOA-{self.roi_name}.tif'
            l2rgb_of = pathlib.Path(l2rgb_save_dir).parent / f'S2_{s_d}_L2RGB-{self.roi_name}.tif'
            l2_of = pathlib.Path(l2_save_dir).parent / f'S2_{s_d}_L2SURF-{self.roi_name}.tif'
            cldprob_of = pathlib.Path(cldprob_save_dir).parent / f'S2_{s_d}_CLDPROB-{self.roi_name}.tif'

            img_count_l2 = self.log.get_images_cover_l2(self.roi_name, s_d)
            img_count_l1 = self.log.get_images_cover_l1(self.roi_name, s_d)

            images_l2 = None
            if np.isnan(img_count_l2):
                images_l2 = ee.ImageCollection(self.sensor_info['msi_l2']['data_source']). \
                    filterDate(s_d, e_d).filterBounds(self.roi_rect)
                img_count_l2 = len(images_l2.getInfo()['features'])
                insert_flag = True
            logrow.images_l2 = img_count_l2

            images_l1 = None
            if np.isnan(img_count_l1):
                images_l1 = ee.ImageCollection(self.sensor_info['msi_l1']['data_source']). \
                    filterDate(s_d, e_d).filterBounds(self.roi_rect)
                img_count_l1 = len(images_l1.getInfo()['features'])
                insert_flag = True
            logrow.images_l1 = img_count_l1


            ## no l1 neither l2, do nothing
            if img_count_l1 <1 and img_count_l2<1:
                print(f'no images are found for {s_d}')
                if insert_flag: self.log.insert(logrow)
                continue

            ## no l2, generally for the dates before 2018-12-14
            ## filter the images based on the CLOUD PROBILITY dataset, and do not download l2 or l2rgb
            elif img_count_l2<1:
                cloud_percentage = self.log.get_cloud_percentage_image(roi_name=self.roi_name, date=s_d)
                if cloud_percentage > 100:
                    print(f'no water pixels found in AOI for the current {s_d}')
                    continue
                if np.isnan(cloud_percentage):
                    insert_flag = True
                    try:
                        cloud_percentage = self.get_s2_cloudpercentage(s_d=s_d,e_d=e_d)
                    except Exception as e:
                        print(f'no water pixels found in AOI for the current {s_d}')
                        logrow.cloud_percentage_image = 101
                        if insert_flag: self.log.insert(logrow)
                        continue

                logrow.cloud_percentage_image = cloud_percentage
                if cloud_percentage > self.cloud_percentage_threshold:
                    print(f'cloudy percentage :{cloud_percentage} > {self.cloud_percentage_threshold}')
                    if insert_flag:self.log.insert(logrow)
                    continue

                l2, l2rgb = False,False


            # no_l2_yes_l1 = False
            # if img_count_l2 < 1:
            #     print(f'no images are found for {s_d}')
            #     if insert_flag: self.log.insert(logrow)
            #     if l1:  ## force to download level-1
            #         images_l1 = ee.ImageCollection(self.sensor_info['msi_l2']['data_source']). \
            #             filterDate(s_d, e_d).filterBounds(self.roi_rect)
            #         img_count_l1 = len(images_l1.getInfo()['features'])
            #
            #     else:
            #         continue

            ## both l2 and l1 are available, filter the images based on the l2 classification band
            else:
                ### check water pixels percentage for the current image
                water_per = self.log.get_water_percentage_image(self.roi_name, s_d)
                if np.isnan(water_per):
                    if images_l2 is None:
                        images_l2 = ee.ImageCollection(self.sensor_info['msi_l2']['data_source']). \
                            filterDate(s_d, e_d).filterBounds(self.roi_rect)
                    images_water = add_waterpixelsnumber_s2l2(images=images_l2, roi_rect=self.roi_rect,
                                                              resolution=water_pixel_resolution)
                    # print(images_water.size().getInfo())
                    images_water_list = images_water.toList(images_water.size())
                    total_pixels, waternum_s2 = 0, 0
                    for i in range(img_count_l2):
                        img_1 = ee.Image(images_water_list.get(i)).clip(self.roi_rect)
                        waternum = ee.Number(img_1.get('water_num')).getInfo()
                        dims = img_1.getInfo()['bands'][-1]['dimensions']
                        total_pixels += dims[0] * (water_pixel_resolution / 20.0) * dims[1] * (
                                water_pixel_resolution / 20.0)  ## convert to water_pixel_resolution
                        waternum_s2 += waternum
                    water_per = waternum_s2 / waterpixels_regular * 100.0
                    insert_flag = True
                logrow.water_percentage_image = water_per
                if water_per < self.water_threshold:
                    print(
                        f'water percentage is {round(water_per, 1)} < {self.water_threshold}%, the date {s_d} is skiped')
                    if insert_flag: self.log.insert(logrow)
                    continue



            ###### download l2 rgb#################
            if (images_l2 is None) and (l2rgb or l2):
                images_l2 = ee.ImageCollection(self.sensor_info['msi_l2']['data_source']).filterDate(s_d, e_d).filterBounds(self.roi_rect)

            dst_crs = None
            if l2rgb:
                # l2rgb_of = pathlib.Path(l2_save_dir).parent / f'S2_{s_d}_L2RGB-{self.roi_name}.tif'
                if (not os.path.exists(l2rgb_of)) or self.over_write_l2rgb:
                    print('downloading l2 RGB...')
                    if not os.path.exists(l2rgb_save_dir):
                        os.makedirs(l2rgb_save_dir)

                    download_images_roi(images=images_l2,
                                        roi_geo=self.roi_geo,
                                        save_dir=l2rgb_save_dir,
                                        bands=self.sensor_info['msi_l2_rgb']['download_bands'],
                                        resolution=self.resolution)
                    print("merging l2 rgb....")
                    dst_crs = self.merge_download_dir(l2rgb_save_dir, output_f=l2rgb_of,dst_crs=dst_crs)
                else:
                    with rasterio.open(l2rgb_of) as src:
                        dst_crs = src.meta['crs']

            if dst_crs is None:
                print('dst_crs is None')
                # sys.exit(-1)


            ####### download l2 surface ##############
            if l2:
                if (not os.path.exists(l2_of)) or self.over_write_l2:
                    print('downloading l2')
                    if not os.path.exists(l2_save_dir):
                        os.makedirs(l2_save_dir)

                    download_images_roi(images=images_l2,
                                        roi_geo=self.roi_geo,
                                        save_dir=l2_save_dir,
                                        bands=self.sensor_info['msi_l2']['download_bands'],
                                        resolution=self.resolution)
                    print("merging l2 surface....")
                    self.merge_download_dir(l2_save_dir, output_f=l2_of,dst_crs=dst_crs)

            ####### download l1 ###########################
            if l1:
                if (not os.path.exists(l1_of)) or self.over_write_l1:
                    images_l1 = ee.ImageCollection(self.sensor_info['msi_l1']['data_source']).filterDate(s_d,
                                                                                                         e_d).filterBounds(
                        self.roi_rect)
                    print("downloading l1...")
                    if not os.path.exists(l1_save_dir):
                        os.makedirs(l1_save_dir)

                    download_images_roi(images=images_l1,
                                        roi_geo=self.roi_geo,
                                        save_dir=l1_save_dir,
                                        bands=self.sensor_info['msi_l1']['download_bands'], resolution=self.resolution)
                    print("merging l1....")
                    # self.merge_download_dir_l1(save_dir,bandnames=self.sensor_info['msi_l1']['download_bands'],output_f=l1_of)
                    self.merge_download_dir(l1_save_dir, output_f=l1_of,
                                            bandnames=self.sensor_info['msi_l1']['download_bands'], dst_crs=dst_crs)

            if cloud_prob:
                if (not os.path.exists(cldprob_of)) or self.over_write_cldprob:
                    images_cldprob = ee.ImageCollection(self.sensor_info['msi_cloudprob']['data_source']).filterDate(s_d,
                                                                                                         e_d).filterBounds(
                        self.roi_rect)
                    print("downloading cloud probility...")
                    if not os.path.exists(cldprob_save_dir):
                        os.makedirs(cldprob_save_dir)

                    download_images_roi(images=images_cldprob,
                                        roi_geo=self.roi_geo,
                                        save_dir=cldprob_save_dir,
                                        bands=self.sensor_info['msi_cloudprob']['download_bands'], resolution=self.resolution)

                    print("merging cloud probility....")
                    # self.merge_download_dir_l1(save_dir,bandnames=self.sensor_info['msi_l1']['download_bands'],output_f=l1_of)
                    self.merge_download_dir(cldprob_save_dir, output_f=cldprob_of,
                                            bandnames=self.sensor_info['msi_cloudprob']['download_bands'], dst_crs=dst_crs,include_geo=False)



            if insert_flag: self.log.insert(logrow)

    def merge_download_dir(self, download_dir, output_f,dst_crs=None, bandnames=None, include_geo=True):
        tifs = [_ for _ in glob.glob(os.path.join(download_dir, f'*_*_*.tif')) if
                (os.path.getsize(_) / 1024.0) > 100]
        if len(tifs) < 1:
            print("warn, no tifs found")
            return
        info_pickels = glob.glob(os.path.join(download_dir, "*.pickle"))
        descriptions = []
        for _pf in info_pickels:
            if include_geo:
                _id, theta_v, theta_s, azimuth_v, azimuth_s = self.__extract_geometry_from_info(_pf)
                descriptions.append(','.join([_id, str(theta_v), str(theta_s), str(azimuth_v), str(azimuth_s)]))
                descriptions_meta = 'product_id,theta_v,theta_s,azimuth_v,azimuth_s'
            else:
                descriptions.append(self.__extract_id_from_info(_pf))
                descriptions_meta = 'product_id'



        ret, dst_crs = merge_tifs(tifs, output_f, descriptions=':'.join(descriptions),
                         descriptions_meta='product_id,theta_v,theta_s,azimuth_v,azimuth_s', bandnames=bandnames,dst_crs=dst_crs)
        if ret == 1 and self.remove_download_tiles:
            shutil.rmtree(download_dir)
        return dst_crs

    def __extract_geometry_from_info(self, pickle_file):
        #     print(pickle_file)
        with open(pickle_file, 'rb') as f:
            info = pickle.load(f)
        theta_v, theta_s, azimuth_v, azimuth_s = [], [], [], []
        #     print(info['bands'][0].keys())
        properties = info['properties']
        for key in properties.keys():
            if 'MEAN_INCIDENCE_ZENITH_ANGLE' in key:
                theta_v.append(properties[key])
            if 'MEAN_SOLAR_ZENITH_ANGLE' in key:
                #             print(key, properties[key])
                theta_s.append(properties[key])
            if 'INCIDENCE_AZIMUTH_ANGLE' in key:
                azimuth_v.append(properties[key])
            if 'MEAN_SOLAR_AZIMUTH_ANGLE' in key:
                azimuth_s.append(properties[key])
        theta_v, theta_s, azimuth_v, azimuth_s = np.asarray(theta_v), np.asarray(theta_s), np.asarray(
            azimuth_v), np.asarray(azimuth_s)
        return (properties['PRODUCT_ID'], theta_v.mean(), theta_s.mean(), azimuth_v.mean(), azimuth_s.mean())

    def __extract_id_from_info(self, pickle_file):
        with open(pickle_file,'rb') as f:
            info = pickle.load(f)
        return info['id'].split('/')[-1]



if __name__ == '__main__':
    grid_cells_dir = '/mnt/Ext_8T_andromede/0_ARCTUS_Projects/18_MEI_SDB2/geojson_rename/'
    configdic = {'over_write':False,'remove_download_tiles':True}
    downloader = Downloader(water_threshold=50,
                        water_threshold_regular=10,
                        savedir="/mnt/Ext_8T_andromede/0_ARCTUS_Projects/18_MEI_SDB2/data/s2_gee_HBE/",**configdic)
    # filename = os.listdir(grid_cells_dir)[6]
    # geojson_f = os.path.join(grid_cells_dir, filename)
    def download_cell(downloader, geojson_f):
        basename = os.path.basename(geojson_f)
        gdf = gpd.read_file(geojson_f)
        downloader.set_roi(gdf.geometry[0],roi_name=os.path.splitext(basename)[0])
        print(os.path.splitext(basename)[0])
        # downloader.download_s2l1(start_date='2021-08-01',
        #                      end_date='2021-08-31',
        #                      download_l2_rgb=True)
        # downloader.download_s2(start_date='2021-08-01',end_date='2021-08-31',l1=True,l2rgb=True, l2=True)
        downloader.download_s2(start_date='2022-08-01', end_date='2022-08-31', l1=True, l2rgb=True,l2=True)
        # downloader.download_s2(start_date='2019-08-01', end_date='2019-08-31', l1=False, l2rgb=False,l2=True)
    # download_dir = "C:/Users/pany0/OneDrive/Desktop/geedownload_test/l2_surf/00007_HBE_5150879N7954083W_20KM/2021-08-07"
    # downloader.merge_download_dir(download_dir,level='l2')
    geojson_fs = sorted(glob.glob(os.path.join(grid_cells_dir,'*geojson')))
    for gf in geojson_fs[:]:
        # if '00005_HBE_5143340N7901250W_20KM' not in gf:
        #     continue
        download_cell(downloader, gf)
