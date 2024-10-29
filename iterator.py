import configparser
config = configparser.ConfigParser()
#config.sections()
config.read("download.ini")

# import required module
import os
# assign directory
nonna = 'nonna_ROIs'   #folder of ROIs in the same directory
 
# iterate over files in
# that directory
count = 0
for filename in os.listdir(nonna):
    aoi = os.path.join(nonna, filename)
    output = "{}/{}".format('Results', count)   #name of folder to save results
    # checking if it is a file
    if os.path.isfile(aoi):
        config['GLOBAL']['aoi'] = aoi      #update aoi
        config['GLOBAL']['save_dir'] = output       #update name of folder
        with open("download.ini", "w") as file_object:
            config.write(file_object)         #saves changes
            
    %run main.py -c download.ini
    count +=1           #counter. Folder names may not match aoi ID.