import os
import pickle
import requests
import glob
import shutil
import numpy as np

import ee
import rasterio
from shapely.geometry import MultiPolygon

from utils import merge_tifs
from .exceptions import DownloadDirIncompleteError, NoEEIntersectionBandsError

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


def download_images_roi(images: ee.ImageCollection, grids, save_dir, bands=None, resolution=10):
    '''
    @grids ee_small_cells and ee_small_cells_box
    @bands, for s2 ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'QA60']
    '''
    # xs, ys = gen_subcells(roi_geo, x_step=0.1, y_step=0.1)
    # ee_small_cells = [ee.Geometry.Rectangle([x[0], y[0], x[1], y[1]]) for x in xs for y in ys]
    # ee_small_cells_box = [([x[0], y[0], x[1], y[1]]) for x in xs for y in ys]
    ee_small_cells, ee_small_cells_box = grids
    bands_c = bands.copy()
    # print(images)
    img_count = len(images.getInfo()['features'])
    info_s = glob.glob(os.path.join(save_dir, '*.pickle'))
    bands_all = None
    download  = True
    if len(info_s) == img_count:
        for _ in info_s:
            with open(_, 'rb') as f:
                info = pickle.load(f)
            bands_all = set([_['id'] for _ in info['bands']]) if bands_all is None \
                else bands_all.intersection(set([_['id'] for _ in info['bands']]))
        download = False

    else:
        ## redownload
        for i in range(img_count):
            _id = images.getInfo()['features'][i]['id']
            img = ee.Image(_id)
            info = img.getInfo()
            bands_all = set([_['id'] for _ in info['bands']]) if bands_all is None \
                else bands_all.intersection(set([_['id'] for _ in info['bands']]))

    for _ in bands:
        if _ not in bands_all:
            bands_c.remove(_)

    if len(bands_c) == 0:
        raise NoEEIntersectionBandsError()

    if not download:
        return 1, bands_c

    ### download
    for i in range(img_count):
        _id = images.getInfo()['features'][i]['id']
        img = ee.Image(_id)
        info = img.getInfo()
        _id_name = _id.split('/')[-1]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        with open(os.path.join(save_dir, f'{_id_name}_info.pickle'), 'wb') as f:
            pickle.dump(info, f, pickle.HIGHEST_PROTOCOL)
        # print(_id.split('/')[-1][:8] + '_' + _id.split('_')[-1])

        name = _id_name[:8] + '_' + _id.split('_')[-1]
        for j, (ee_c, ee_c_b) in enumerate(zip(ee_small_cells, ee_small_cells_box)):
            try:
                url = img.getDownloadURL({
                    'name': 'multi_band',
                    'bands': bands_c,
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
    return 1, bands_c


def extract_id_from_info(self, pickle_file):
    with open(pickle_file,'rb') as f:
        info = pickle.load(f)
    return info['id'].split('/')[-1]


def merge_download_dir(download_dir,
                       output_f,
                       descriptions_meta,
                       descriptions,
                       dst_crs=None,
                       bandnames=None,
                       remove_temp = True,
                       RGB = False,
                       min_max = (None, None),
                       **extra_info):
    tifs = [_ for _ in glob.glob(os.path.join(download_dir, f'*_*_*.tif')) if(os.path.getsize(_) / 1024.0) > 100]
    if len(tifs) < 1:
        raise DownloadDirIncompleteError(download_dir)

    ret, dst_crs = merge_tifs(tifs, output_f, descriptions=':'.join(descriptions),
                         descriptions_meta=descriptions_meta,
                              bandnames=bandnames,
                              dst_crs=dst_crs,
                              RGB = RGB,
                              min_max = min_max,
                              **extra_info)
    if ret == 1 and remove_temp:
        shutil.rmtree(download_dir)
    return dst_crs