Download images to local storage from Google Earth Engine in format of geotiff for a given AOI (geojson) and date.
Warns: it can be very slow if AOI is too large because the maximum number of pixels and dimensions of extent that can be downloaded from
GEE are limited.

requirements:
rasterio, earthengine-api, google-cloud-bigquery, db_dtypes

run these commands to authenticate gcloud
gcloud auth login
gcloud auth application-default login
, and enable big query in your gcloud project

example of use:
python main.py -c download.ini
