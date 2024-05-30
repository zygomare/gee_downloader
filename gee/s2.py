import os, glob
import numpy as np
import pickle


import ee
import pendulum

from .exceptions import NoEEImageFoundError, EEImageOverlayError

def extract_geometry_from_info(pickle_file):
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


def get_descriptions_l1toa(download_dir):
    info_pickels = glob.glob(os.path.join(download_dir, "*.pickle"))
    descriptions = []
    descriptions_meta = 'product_id,theta_v,theta_s,azimuth_v,azimuth_s'
    acquisition_time = ''
    prefix = ''
    for _pf in info_pickels:
        _id, theta_v, theta_s, azimuth_v, azimuth_s = extract_geometry_from_info(_pf)
        if acquisition_time == '':
            acquisition_time = _id.split('_')[2][:13]
        if prefix == '':
            prefix = '_'.join(_id.split('_')[:2])
        descriptions.append(','.join([_id, str(theta_v), str(theta_s), str(azimuth_v), str(azimuth_s)]))

        # descriptions.append(self.__extract_id_from_info(_pf))
        #  descriptions_meta = 'product_id'
    return prefix, acquisition_time, descriptions, descriptions_meta

def get_descriptions_l2rgb(download_dir):
    return get_descriptions_l1toa(download_dir)




def add_cloudpixelsnumber_image(images: ee.ImageCollection, roi_rect, resolution=10, threshold_prob=50, water_mask=None):
    # images = images.updateMask(water_mask)
    # dataset = ee.Image('JRC/GSW1_4/GlobalSurfaceWater').clip(roi_rect)
    # water_mask = dataset.select('occurrence').gt(0)
    def __f(image:ee.Image):
        if water_mask is not None:
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

# def get_s2_cloudpercentage():
#     print('get_s2_cloudpercentage')

def get_s2_cloudpercentage(date, aoi_rect_ee, cloud_prob_threshold=60, water_mask=None,resolution=20):
    s_d, e_d = date.format('YYYY-MM-DD'), (date + pendulum.duration(days=1)).format('YYYY-MM-DD')
    # COPERNICUS / S2_CLOUD_PROBABILITY
    images_cloudprob = ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY').filterDate(s_d, e_d).filterBounds(aoi_rect_ee)

    # images = add_surfacewater_cloudprob(images_cloudprob,roi_rect=self.roi_rect)

    img_count_cloudprob = len(images_cloudprob.getInfo()['features'])
    images_cloud = add_cloudpixelsnumber_image(images_cloudprob,roi_rect=aoi_rect_ee,
                                               threshold_prob=cloud_prob_threshold,
                                               resolution=resolution,
                                               water_mask=water_mask)

    images_cloud_list = images_cloud.toList(images_cloud.size())

    total_pixels, cloud_pixels = 0, 0

    if img_count_cloudprob == 0:
        raise NoEEImageFoundError('COPERNICUS/S2_CLOUD_PROBABILITY',date=s_d)

    for i in range(img_count_cloudprob):
        img_1 = ee.Image(images_cloud_list.get(i)).clip(aoi_rect_ee)
        cloudnum = ee.Number(img_1.get('cloud_num')).getInfo() ##90:30 60:986
        totalnum = ee.Number(img_1.get('total_num')).getInfo()
        cloud_pixels += cloudnum
        total_pixels += totalnum

    if total_pixels == 0:
        raise EEImageOverlayError(ee_source='S2_CLOUD_PROBABILITY',date=s_d)
    return cloud_pixels/total_pixels*100

