import os, glob
from os.path import exists as Pexists
from os.path import split as Psplit
from os.path import basename as Pbasename

import rasterio
import ee
import pendulum
import geopandas as gpd
import logging

# from gee import get_s2cld
import gee
from gee.utils import (gen_subcells,
                       download_images_roi,
                       merge_download_dir)
from gee.exceptions import NoEEImageFoundError, EEImageOverlayError, NoEEIntersectionBandsError

CATALOG_DIC = {'image_collections':
                   {'optical':('s2','lc08','lc09'),
                    'radar':('s1')},
               'image':{}}

class Downloader():
    def __init__(self, **config):
        self._config_dic = config

        self.save_dir = config['global']['save_dir']
        self.cloud_percentage_threshold = float(config['global']['cloud_percentage'])
        self._assets = [str.lower(str.strip(_)) for _ in  config['global']['assets'].split(',')]

        self.assets_dic = {}


        ## water or all
        self._target = str.lower(config['global']['target'])
        self._aoi_f = config['global']['aoi']

        self.aoi_name = os.path.splitext(Pbasename(self._aoi_f))[0]
        self.aoi_geo = gpd.read_file(self._aoi_f).geometry[0]
        self.aoi_bounds = self.aoi_geo.bounds

        self.asset_dic = self.__categroy_assets()

        self.start_date = pendulum.from_format(self._config_dic['global']['start_date'], 'YYYY-MM-DD')
        self.end_date = pendulum.from_format(self._config_dic['global']['end_date'], 'YYYY-MM-DD')


        if not Pexists(self.save_dir):
            os.makedirs(self.save_dir)






    def __categroy_assets(self):
        asset_dic = {}
        imagecoll_dic = {}
        image_dic = {}
        for _ in self._assets:
            if _ not in self._config_dic:
                logging.warning(f'config for {str.upper(_)} not found!')
                continue

            sensor_type = None
            prefix = str.lower(_.split('_')[0])
            _config_dic = self._config_dic[_]
            if prefix not in imagecoll_dic:
                if prefix in CATALOG_DIC['image_collections']['optical']:
                    sensor_type = 'optical'
                elif prefix in CATALOG_DIC['image_collections']['radar']:
                    sensor_type = 'radar'

                elif prefix in CATALOG_DIC['image']:
                    if prefix not in image_dic:
                        image_dic[prefix] = {'config':{_:_config_dic}}
                    else:
                        image_dic[prefix]['config'].update({_: _config_dic})

                imagecoll_dic[prefix] = {'sensor_type':sensor_type, 'config':{_:_config_dic}}

            else:
                imagecoll_dic[prefix]['config'].update({_:_config_dic})

        asset_dic['image_collection'] = imagecoll_dic
        asset_dic['image'] = image_dic
        return asset_dic


    def _insert_record(self):
        pass



class GEEDownloader(Downloader):
    from shapely.geometry import MultiPolygon

    def __init__(self, **config):
        ee.Initialize()
        super(GEEDownloader, self).__init__(**config)
        self.aoi_rect_ee = ee.Geometry.Rectangle(self.aoi_bounds)

    def run(self):
        ## 1. generate small cells
        xs, ys = gen_subcells(self.aoi_geo, x_step=0.1, y_step=0.1)
        self.ee_small_cells = [ee.Geometry.Rectangle([x[0], y[0], x[1], y[1]]) for x in xs for y in ys]
        self.ee_small_cells_box = [([x[0], y[0], x[1], y[1]]) for x in xs for y in ys]

        self.download_imagecollection()

    def download_imagecollection(self):
        '''
        start_date='2021-08-01'
        '''
        image_collection_dic = self.asset_dic['image_collection']

        for prefix in image_collection_dic:
            sensor_type = image_collection_dic[prefix]['sensor_type']
            download_func = getattr(self, f'_download_{sensor_type}')
            download_func(prefix, **image_collection_dic[prefix]['config'])
            # self.download_func = getattr(self, f'_download_{prefix}')
            # self.download_func(image_collection_dic[prefix])


    def download_image(self):
        pass


    def __download_imgcoll_assets(self, date, **config):
        s_d, e_d = date.format('YYYY-MM-DD'), (date + pendulum.duration(days=1)).format('YYYY-MM-DD')
        _s_d = date.format('YYYYMMDD')
        for asset in config:
            if asset in ['extral_info']:
                continue
            rgb = False
            vmin, vmax = None,None
            if asset.find('rgb')>-1:
                vmin = float(config[asset]['vmin'])
                vmax = float(config[asset]['vmax'])
                rgb = True

            data_source = config[asset]['source']

            bands = [str.strip(_) for _ in config[asset]['include_bands'].split(',')]
            resolution = int(config[asset]['resolution'])
            anynom = config[asset]['anonym']
            asset_savedir = config[asset]['save_dir']

            extral_info_dic = config['extral_info'] if 'extral_info' in config else {}
            cloud_per = 0 if 'cloud_percentage' not in extral_info_dic else extral_info_dic['cloud_percentage']

            save_dir = os.path.join(self.save_dir, asset_savedir, anynom)

            ofs = glob.glob(os.path.join(save_dir, f'{str.upper(asset)}_{_s_d}*{self.aoi_name}_{resolution}m.tif'))
            if len(ofs) == 1:
                if 'cloud_percentage' in extral_info_dic:

                    with rasterio.open(ofs[0],'r') as src:
                        tags = src.tags()
                        if 'cloud_percentage' in tags:
                            cloud_per_img = float(tags['cloud_percentage'])
                            if cloud_per_img == cloud_per:
                                print(f'{s_d}, {asset} exist, skip downloading')
                                continue
                else:
                    print(f'{s_d}, {asset} exist, skip downloading')
                    continue

            images = ee.ImageCollection(data_source).filterDate(s_d, e_d).filterBounds(self.aoi_rect_ee)
            if len(images.getInfo()['features']) == 0:
                print(f'{s_d}, {asset},cloud percentage: {cloud_per} No image found!')
                continue
            print(f'{s_d}, {asset}, cloud percentage: {cloud_per} Start downloading ')

            temp_dir = os.path.join(save_dir, s_d)
            try:
                res, bands = download_images_roi(images=images, grids=(self.ee_small_cells,
                                                                    self.ee_small_cells_box),
                                              save_dir=temp_dir,
                                              bands=bands,
                                              resolution=resolution)
            except Exception as e:
                print(f'{s_d},{asset}:{str(e)}')

            if res!=1:
                continue
                # prefix, acquisition_time, des, des_meta = getattr(gee, f'get_descriptions_{asset}')(temp_dir) if hasattr(gee, f'get_descriptions_{asset}') else None
            # else:
            prefix, acquisition_time, des, des_meta = getattr(gee, f'get_descriptions_{asset}')(temp_dir) if hasattr(gee,f'get_descriptions_{asset}') else None
            #
            #
            output_f = os.path.join(save_dir, f'{str.upper(asset)}_{acquisition_time}_{self.aoi_name}_{resolution}m.tif')
            try:
                merge_download_dir(download_dir=temp_dir,
                                   output_f=output_f,
                                   dst_crs=None,
                                   descriptions=des,
                                   descriptions_meta=des_meta,
                                   bandnames=bands,
                                   remove_temp=True,
                                   RGB = rgb,
                                   min_max = (vmin,vmax),
                                   **extral_info_dic)
            except Exception as e:
                print(e)
                continue


    def _download_optical(self, prefix, **config):
        '''
        download optical image
        requires cloud percentage
        '''
        # print(prefix, config)
        if self._target == 'water':
            dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(self.aoi_rect_ee)
            water_mask = dataset.select('occurrence').gt(0)
        else:
            water_mask = None

        for _date in (self.end_date - self.start_date):
            ## 1. obtain cloud percentage

            func_cld = getattr(gee, f'get_{prefix}cld')
            # cld_percentage = self.get_s2_cloudpercentage(s_d='2018-06-10',e_d='2018-06-11')
            try:
                cld_percentage = func_cld(_date, self.aoi_rect_ee,
                                      cloud_prob_threshold=60,
                                      water_mask=water_mask,
                                      resolution=20)
            except NoEEImageFoundError as e:
                print(e)
                continue
            except EEImageOverlayError as e:
                print(e)
                continue

            if cld_percentage > self.cloud_percentage_threshold:
                print(f'{_date}, {prefix}, cloud percentage = {round(cld_percentage,1)} > {self.cloud_percentage_threshold}. skip!')
                continue

            config.update({'extral_info':{'cloud_percentage': round(cld_percentage,1)}})
            self.__download_imgcoll_assets(date=_date, **config)


    def _download_radar(self, prefix, **config):
        '''
        does not require cloud percentage
        '''
        for _date in (self.end_date - self.start_date):
            self.__download_imgcoll_assets(date=_date, **config)


