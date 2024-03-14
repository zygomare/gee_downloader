sensor = 'S2'
import os, geopandas as gpd

if sensor == 'L9':
    from gee_downloader_L9 import *
elif sensor == 'L8':
    from gee_downloader_L8 import *
else:
    from gee_downloader_S2 import *

save_dir = "C:/Users/pany0/WorkSpace/projects/SDB_2023/geedownload_test2"
grid_cells_dir = "C:/Users/pany0/WorkSpace/projects/SDB_2023/geojson_rename"

geojson_fs = sorted(glob.glob(os.path.join(grid_cells_dir,'*geojson')))

configdic = {'over_write': False, 'remove_download_tiles': True,'cloud_prob_threshold':60, 'cloud_percentage_threshold':40}
downloader = Downloader(water_threshold=50,
                        water_threshold_regular=10,
                        savedir=save_dir, **configdic)


def download_cell(downloader, geojson_f):
    basename = os.path.basename(geojson_f)
    gdf = gpd.read_file(geojson_f)
    downloader.set_roi(gdf.geometry[0], roi_name=os.path.splitext(basename)[0])
    print(os.path.splitext(basename)[0])
    # downloader.download_s2l1(start_date='2021-08-01',
    #                      end_date='2021-08-31',
    #                      download_l2_rgb=True)
    # downloader.download_s2(start_date='2021-08-01',end_date='2021-08-31',l1=True,l2rgb=True, l2=True)
    downloader.download_s2(start_date='2018-08-01', end_date='2018-08-31', l1=True, l2rgb=True, l2=True)

if __name__ == '__main__':
    download_cell(downloader, geojson_f=geojson_fs[3])