Download images to local storage from Google Earth Engine in format of geotiff for a given AOI (geojson) and date.
Warns: it can be very slow if AOI is too large because the maximum number of pixels and dimensions of extent that can be downloaded from
GEE are limited.


example of use:
python gee_downloader.py -c download.ini
