# Python 3 code to rename multiple 
# files in a directory or folder

import os
import glob
from os import rename

folder = "/home/ngalac/gee_downloader-main/sub/"     # absolute path to folder of the images from gee_downloader

for roi in os.listdir(folder):
    nonna = os.path.join(folder, roi, "L2RGB/s2_msi")
    nonna_id = os.listdir(nonna)[-1]                  #extract the nonna id
    nonna_id = nonna_id.split('_')[-2]

    new_id = "aoi_" + nonna_id   #change string to naming convention of your choice. Should not be the same as the old IDs

    old_name = os.path.join(folder, roi)
    rename(old_name, os.path.join(folder, new_id))     #rename and replace files with the same folder