# sensor = 'l8_oli'
sensor = 'L8'
import os, geopandas as gpd

if sensor == 'L9':
    from gee_downloader_L9 import *
elif sensor == 'L8':
    from gee_downloader_L8 import *
else:
    from gee_downloader_S2 import *

# save_dir = "C:/Users/pany0/WorkSpace/projects/SDB_2023/geedownload_test2"
save_dir = f'E:/SME-SEPTILES/gee_download/{sensor}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
# grid_cells_dir = "C:/Users/pany0/WorkSpace/projects/SDB_2023/geojson_rename"
grid_cells_dir = 'C:/Users/pany0/WorkSpace/pycharm_proj/sme-chain/data/septiles'
geojson_fs = sorted(glob.glob(os.path.join(grid_cells_dir,'waterboundary_septiles*.geojson')))

configdic = {'over_write': False, 'remove_download_tiles': True,'cloud_prob_threshold':60, 'cloud_percentage_threshold':40}
downloader = Downloader(water_threshold=0,
                        water_threshold_regular=10,
                        savedir=save_dir, **configdic)


def download_cell(downloader, geojson_f):
    basename = os.path.basename(geojson_f)
    gdf = gpd.read_file(geojson_f)
    # print(os.path.splitext(basename)[0])
    downloader.set_roi(gdf.geometry[0], roi_name=os.path.splitext(basename)[0])
    print(os.path.splitext(basename)[0])
    # downloader.download_s2l1(start_date='2021-08-01',
    #                      end_date='2021-08-31',
    #                      download_l2_rgb=True)
    # downloader.download_s2(start_date='2021-08-01',end_date='2021-08-31',l1=True,l2rgb=True, l2=True)

    func = getattr(downloader, f'download_{sensor}')
    for s_d, e_d in zip(['2022-09-01','2023-04-01','2024-04-01'],['2022-10-31','2023-11-01','2024-05-06']):
        func(start_date=s_d, end_date=e_d, l1=True, l2rgb=True, l2=False)
    # downloader.download_s2(start_date='2023-04-01', end_date='2023-10-31', l1=False, l2rgb=True, l2=False)

if __name__ == '__main__':
    download_cell(downloader, geojson_f=geojson_fs[0])