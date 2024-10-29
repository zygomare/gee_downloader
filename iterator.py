import configparser
from tqdm import tqdm
config = configparser.ConfigParser()
#config.sections()
config.read("download.ini")

# import required module
import os
# assign directory
nonna_dir = r'c:\Users\sim-9\MEI_SDB\nonna_ROIs'   #folder of ROIs in the same directory
output_dir = r'c:\Users\sim-9\MEI_SDB\GEE_data_nonna'  #output folder where we download GEE images.

if not os.path.exists(output_dir): os.makedirs(output_dir)
 
# iterate over files in
# that directory
count = 0
for filename in tqdm(os.listdir(nonna_dir)):
    aoi = os.path.join(nonna_dir, filename)
    print(f"Running GEE-downloader on AOI: {aoi}")
    output = os.path.join(output_dir, f"aoi_{count}")   #name of folder to save results
    # checking if it is a file
    if os.path.isfile(aoi):
        config['GLOBAL']['aoi'] = aoi      #update aoi
        config['GLOBAL']['save_dir'] = output       #update name of folder
        with open("download.ini", "w") as file_object:
            config.write(file_object)         #saves changes
            
    os.system("python main.py -c download.ini")
    count +=1           #counter. Folder names may not match aoi ID.