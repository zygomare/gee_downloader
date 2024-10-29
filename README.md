Download images to local storage from Google Earth Engine in format of geotiff for a given AOI (geojson) and date.
Warns: it can be very slow if AOI is too large because the maximum number of pixels and dimensions of extent that can be downloaded from
GEE are limited.

requirements:
rasterio, earthengine-api, google-cloud-bigquery, db_dtypes

In the browser, create a new project on google cloud, and enable big query + Google Earth Engine in your gcloud project 
(also register your project to use G.E.E.: https://code.earthengine.google.com/register)

run these commands to authenticate gcloud in the cli (inside the gee-downloader directory)
$gcloud auth login
$gcloud auth application-default login


example of use:
python main.py -c download.ini
