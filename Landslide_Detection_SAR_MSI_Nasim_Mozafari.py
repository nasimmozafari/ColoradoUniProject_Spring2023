#!/usr/bin/env python
# coding: utf-8

# # <center>Detection of landslides using Synthetic Aperature Radar Imagery and Multispectral imagery</center>
# 
# ### <span style="background-color: #green; padding: 5px; border-radius: 5px;"><div style="background-color: #f2f2f2; padding: 10px;"><center>Nasim Mozafari, Mentor: Elsa Culler</center></div></span><span style="background-color: #green; padding: 5px; border-radius: 5px;"><div style="background-color: #f2f2f2; padding: 10px;"><center>August 2023</center></div></span>

# ## Run Google earth Engine (GEE)
# Run the following cell to initialize the API. The output will offer instruction on how to link this notebook with Earth Engine access using your account.

# In[1]:


import ee
# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()


# ## Data source
# 
# 1. [Copernicus_S1_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) and [Copernicus_S2](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2) by European Space Agency(ESA)
# 2. USGS NAIP Imagery NDVI (GEE basemap layer)
# 3. USGS NAIP Imagery False Color
# 4. Verified landslides of North America collected by Earth Lab at the University of Colorado Boulder

# ## Import libraries and packages
# The following python modules provide all required modules to run all the codes in this notebook

# In[2]:


# Required libraries and packages
import os
import json
import earthpy as et
import pandas as pd
import datetime
import pathlib
import folium
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import ee
import geemap

import shapely.geometry as sgeo
import IPython.display as disp
import geemap.foliumap as geemap

from shapely.geometry import Point
# from src.det import det
#from scipy.stats import norm, gamma, f, chi2
from scipy.stats import chi2
get_ipython().run_line_magic('matplotlib', 'inline')


# ### All the functions defined in this section are taken from Google Earth Engine Tutorials on Python API

# In[3]:


# First, in order to add Earth Engine layer to Folium map we need this function below
def add_ee_layer(self, ee_image_object, vis_params, name):
    """
    Adds Earth Engine layers to a folium map.

    Returns
    -------
    Earth Engine Raster Layers to Folium Map
    """

    map_id_dict = ee.Image(ee_image_object).getMapId(vis_params)
    folium.raster_layers.TileLayer(
        tiles=map_id_dict['tile_fetcher'].url_format,
        attr='Map Data &copy; <a href="https://earthengine.google.com/">Google Earth Engine</a>',
        name=name,
        overlay=True,
        control=True).add_to(self)


# In[4]:


def selectvv(current):
    """
    Selects the 'VV' bands from an image

    Returns
    -------
    function
        to select the images with 'VV' bands

    """
    return ee.Image(current).select('VV')


# In[5]:


def omnibus(im_list, m=4.4):
    """Calculates the omnibus test statistic, monovariate case."""
    def log(current):
        return ee.Image(current).log()

    im_list = ee.List(im_list)
    k = im_list.length()
    klogk = k.multiply(k.log())
    klogk = ee.Image.constant(klogk)
    sumlogs = ee.ImageCollection(im_list.map(log)).reduce(ee.Reducer.sum())
    logsum = ee.ImageCollection(im_list).reduce(ee.Reducer.sum()).log()
    return klogk.add(sumlogs).subtract(logsum.multiply(k)).multiply(-2*m)


# In[6]:


def chi2cdf(chi2, df):
    """Calculates Chi square cumulative distribution function for
       df degrees of freedom using the built-in incomplete gamma
       function gammainc().
    """
    return ee.Image(chi2.divide(2)).gammainc(ee.Number(df).divide(2))

def det(im):
    """Calculates determinant of 2x2 diagonal covariance matrix."""
    return im.expression('b(0)*b(1)')


# In[7]:


def sample_vv_imgs(j):
    """Samples the test statistics Rj in the region aoi."""
    j = ee.Number(j)
    # Get the factors in the expression for Rj.
    sj = vv_list.get(j.subtract(1))
    jfact = j.pow(j).divide(j.subtract(1).pow(j.subtract(1)))
    sumj = ee.ImageCollection(vv_list.slice(0, j)).reduce(ee.Reducer.sum())
    sumjm1 = ee.ImageCollection(vv_list.slice(
        0, j.subtract(1))).reduce(ee.Reducer.sum())
    # Put them together.
    Rj = sumjm1.pow(j.subtract(1)).multiply(
        sj).multiply(jfact).divide(sumj.pow(j)).pow(5)
    # Sample Rj.
    sample = (Rj.sample(region=aoi, scale=10, numPixels=1000, seed=123)
              .aggregate_array('VV_sum'))
    return sample


# In[8]:


def log_det_sum(im_list, j):
    """Returns log of determinant of the sum of the first j images in im_list."""
    im_ist = ee.List(im_list)
    sumj = ee.ImageCollection(im_list.slice(0, j)).reduce(ee.Reducer.sum())
    return ee.Image(det(sumj)).log()


def log_det(im_list, j):
    """Returns log of the determinant of the jth image in im_list."""
    im = ee.Image(ee.List(im_list).get(j.subtract(1)))
    return ee.Image(det(im)).log()


def pval(im_list, j, m=4.4):
    """Calculates -2logRj for im_list and returns P value and -2logRj."""
    im_list = ee.List(im_list)
    j = ee.Number(j)
    m2logRj = (log_det_sum(im_list, j.subtract(1))
               .multiply(j.subtract(1))
               .add(log_det(im_list, j))
               .add(ee.Number(2).multiply(j).multiply(j.log()))
               .subtract(ee.Number(2).multiply(j.subtract(1))
               .multiply(j.subtract(1).log()))
               .subtract(log_det_sum(im_list, j).multiply(j))
               .multiply(-2).multiply(m))
    pv = ee.Image.constant(1).subtract(chi2cdf(m2logRj, 2))
    return (pv, m2logRj)


def p_values(im_list):
    """Pre-calculates the P-value array for a list of images."""
    im_list = ee.List(im_list)
    k = im_list.length()

    def ells_map(ell):
        """Arranges calculation of pval for combinations of k and j."""
        ell = ee.Number(ell)
        # Slice the series from k-l+1 to k (image indices start from 0).
        im_list_ell = im_list.slice(k.subtract(ell), k)

        def js_map(j):
            """Applies pval calculation for combinations of k and j."""
            j = ee.Number(j)
            pv1, m2logRj1 = pval(im_list_ell, j)
            return ee.Feature(None, {'pv': pv1, 'm2logRj': m2logRj1})

        # Map over j=2,3,...,l.
        js = ee.List.sequence(2, ell)
        pv_m2logRj = ee.FeatureCollection(js.map(js_map))

        # Calculate m2logQl from collection of m2logRj images.
        m2logQl = ee.ImageCollection(
            pv_m2logRj.aggregate_array('m2logRj')).sum()
        pvQl = ee.Image.constant(1).subtract(
            chi2cdf(m2logQl, ell.subtract(1).multiply(2)))
        pvs = ee.List(pv_m2logRj.aggregate_array('pv')).add(pvQl)
        return pvs

    # Map over l = k to 2.
    ells = ee.List.sequence(k, 2, -1)
    pv_arr = ells.map(ells_map)

    # Return the P value array ell = k,...,2, j = 2,...,l.
    return pv_arr


# In[9]:


def filter_j(current, prev):
    """Calculates change maps; iterates over j indices of pv_arr."""
    pv = ee.Image(current)
    prev = ee.Dictionary(prev)
    pvQ = ee.Image(prev.get('pvQ'))
    i = ee.Number(prev.get('i'))
    cmap = ee.Image(prev.get('cmap'))
    smap = ee.Image(prev.get('smap'))
    fmap = ee.Image(prev.get('fmap'))
    bmap = ee.Image(prev.get('bmap'))
    alpha = ee.Image(prev.get('alpha'))
    j = ee.Number(prev.get('j'))
    cmapj = cmap.multiply(0).add(i.add(j).subtract(1))
    # Check      Rj?            Ql?                  Row i?
    tst = pv.lt(alpha).And(pvQ.lt(alpha)).And(cmap.eq(i.subtract(1)))
    # Then update cmap...
    cmap = cmap.where(tst, cmapj)
    # ...and fmap...
    fmap = fmap.where(tst, fmap.add(1))
    # ...and smap only if in first row.
    smap = ee.Algorithms.If(i.eq(1), smap.where(tst, cmapj), smap)
    # Create bmap band and add it to bmap image.
    idx = i.add(j).subtract(2)
    tmp = bmap.select(idx)
    bname = bmap.bandNames().get(idx)
    tmp = tmp.where(tst, 1)
    tmp = tmp.rename([bname])
    bmap = bmap.addBands(tmp, [bname], True)
    return ee.Dictionary({'i': i, 'j': j.add(1), 'alpha': alpha, 'pvQ': pvQ,
                          'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap':bmap})

def filter_i(current, prev):
    """Arranges calculation of change maps; iterates over row-indices of pv_arr."""
    current = ee.List(current)
    pvs = current.slice(0, -1 )
    pvQ = ee.Image(current.get(-1))
    prev = ee.Dictionary(prev)
    i = ee.Number(prev.get('i'))
    alpha = ee.Image(prev.get('alpha'))
    median = prev.get('median')
    # Filter Ql p value if desired.
    pvQ = ee.Algorithms.If(median, pvQ.focalMedian(2.5), pvQ)
    cmap = prev.get('cmap')
    smap = prev.get('smap')
    fmap = prev.get('fmap')
    bmap = prev.get('bmap')
    first = ee.Dictionary({'i': i, 'j': 1, 'alpha': alpha ,'pvQ': pvQ,
                           'cmap': cmap, 'smap': smap, 'fmap': fmap, 'bmap': bmap})
    result = ee.Dictionary(ee.List(pvs).iterate(filter_j, first))
    return ee.Dictionary({'i': i.add(1), 'alpha': alpha, 'median': median,
                          'cmap': result.get('cmap'), 'smap': result.get('smap'),
                          'fmap': result.get('fmap'), 'bmap': result.get('bmap')})


# In[10]:


def dmap_iter(current, prev):
    """Reclassifies values in directional change maps."""
    prev = ee.Dictionary(prev)
    j = ee.Number(prev.get('j'))
    image = ee.Image(current)
    avimg = ee.Image(prev.get('avimg'))
    diff = image.subtract(avimg)

    # Get positive/negative definiteness.
    posd = ee.Image(diff.select(0).gt(0).And(det(diff).gt(0)))
    negd = ee.Image(diff.select(0).lt(0).And(det(diff).gt(0)))
    bmap = ee.Image(prev.get('bmap'))
    bmapj = bmap.select(j)
    dmap = ee.Image.constant(ee.List.sequence(1, 3))
    bmapj = bmapj.where(bmapj, dmap.select(2))
    bmapj = bmapj.where(bmapj.And(posd), dmap.select(0))
    bmapj = bmapj.where(bmapj.And(negd), dmap.select(1))
    bmap = bmap.addBands(bmapj, overwrite=True)

    # Update avimg with provisional means.
    i = ee.Image(prev.get('i')).add(1)
    avimg = avimg.add(image.subtract(avimg).divide(i))
    # Reset avimg to current image and set i=1 if change occurred.
    avimg = avimg.where(bmapj, image)
    i = i.where(bmapj, 1)
    return ee.Dictionary({'avimg': avimg, 'bmap': bmap, 'j': j.add(1), 'i': i})


# In[11]:


def change_maps(im_list, median=False, alpha=0.01):
    """Calculates thematic change maps."""
    k = im_list.length()
    # Pre-calculate the P value array.
    pv_arr = ee.List(p_values(im_list))
    # Filter P values for change maps.
    cmap = ee.Image(im_list.get(0)).select(0).multiply(0)
    bmap = ee.Image.constant(ee.List.repeat(0, k.subtract(1))).add(cmap)
    alpha = ee.Image.constant(alpha)
    first = ee.Dictionary({'i': 1, 'alpha': alpha, 'median': median,
                           'cmap': cmap, 'smap': cmap, 'fmap': cmap, 'bmap': bmap})
    result = ee.Dictionary(pv_arr.iterate(filter_i, first))

    # Post-process bmap for change direction.
    bmap = ee.Image(result.get('bmap'))
    smap = ee.Image(result.get('smap'))
    fmap = ee.Image(result.get('fmap'))
    avimg = ee.Image(im_list.get(0))
    j = ee.Number(0)
    i = ee.Image.constant(1)
    first = ee.Dictionary({
        'avimg': avimg, 'bmap': bmap, 'smap': smap, 'fmap': fmap,
        'j': j, 'i': i})
    dmap = ee.Dictionary(im_list.slice(
        1).iterate(dmap_iter, first)).get('bmap')
    return ee.Dictionary(result.set('bmap', dmap))


# In[12]:


def plot_change_maps(im_list):
    """Compute and plot change maps"""

    # Run the algorithm with median filter and at 1% significance.
    result = ee.Dictionary(change_maps(im_list, median=True, alpha=0.01))

    # Extract the change maps and export to assets.
    cmap = ee.Image(result.get('cmap'))
    smap = ee.Image(result.get('smap'))
    fmap = ee.Image(result.get('fmap'))
    bmap = ee.Image(result.get('bmap'))
    cmaps = (
        ee.Image
        .cat(cmap, smap, fmap, bmap)
        .rename(['cmap', 'smap', 'fmap']+timestamplist[1:]))
    cmaps = cmaps.updateMask(cmaps.gt(0))
    location = aoi.centroid().coordinates().getInfo()[::-1]

    # create parameters for cmap
    palette = ['black', 'cyan']
    params = {'min': 0, 'max': 1, 'palette': palette}

    # create map with layers

    Map = geemap.Map(location=location, zoom_start=15)

    # Different basemaps. you can select or deselect on image itself
    Map.add_basemap('SATELLITE')
    Map.add_basemap('USGS NAIP Imagery NDVI')
    Map.add_basemap('USGS NAIP Imagery False Color')

    # Our Cmaps layer
    Map.addLayer(cmaps.select(slide_image), params, 'slide_image')

    return Map


# ## Set working directory

# In[13]:


# Change directory to landslide-detect data path
data_path = os.path.join(et.io.HOME, "earth-analytics", "landslide-detect")
if os.path.exists(data_path):
    os.chdir(data_path)
else:
    os.makedirs(data_path)
    print('The new directory is created!')
    os.chdir(data_path)

data_path


# In[14]:


get_ipython().run_cell_magic('bash', '', 'find .\n')


# ## Create dataframe from csv file

# In[15]:


# Create DataFrame and open landslide file of North America
landslide_gdf = gpd.read_file('landslides.verified.csv')
landslide_gdf.head(2)


# In[16]:


# Extract verified large landslides of North America
large_ls = landslide_gdf[landslide_gdf['size'].str.contains
                              ('large')]
large_ls.head()


# ## Folium map of all large verified landslide locations of North America (2015-2017)

# In[17]:


# Display all verified large landslides of North America 
large_ls_map = folium.Map(
    location=[43.0000, -105.0000],
    zoom_start=4,
    width=1000,
    height=600,
    tiles='Stamen terrain')


for index, row in large_ls.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['slide.id'],
        icon=folium.Icon(color="red")
    ).add_to(large_ls_map)

large_ls_map


# ## Extracting Caddy Lake and Big Sur landslides information

# In[18]:


# Extract information of landslides of Caddy Lake, Manitoba, Canada
Caddy_df = landslide_gdf[landslide_gdf['location'].str.contains
                              ('Caddy Lake')]
Caddy_df


# In[19]:


# Extract information of landslides of Big Sur, CA
Big_Sur_df = landslide_gdf[landslide_gdf['location'].str.contains
                              ('Big Sur')]
Big_Sur_df


# ## Area of Interest (AOI)
# #### Create interactive map of landslide Caddy Lake, MB, Canada

# In[20]:


# Display landslide of Caddy Lake location
Caddy_landslide_map = folium.Map(
    location=[49.8063, -95.2098],
    zoom_start=13,
    width=1000,
    height=500,
    tiles='Stamen terrain')


for index, row in Caddy_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['slide.id'],
        icon=folium.Icon(color="red")
    ).add_to(Caddy_landslide_map)

Caddy_landslide_map


# #### Create interactive map of landslide of Big Sur, CA, USA

# In[21]:


# Display landslide of Big Sur location
Big_Sur_landslide_map = folium.Map(
    location=[35.8656, -121.4329],
    zoom_start=13,
    width=1000,
    height=500,
    tiles='Stamen terrain')


for index, row in Big_Sur_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['slide.id'],
        icon=folium.Icon(color="red")
    ).add_to(Big_Sur_landslide_map)

Big_Sur_landslide_map


# In[22]:


# Make split view map half Satellite and half goole maps (Caddy Lake)
# NAIP Imagery False Color seems not to be available for Canada (source??)

Caddy_split_roi = geemap.Map(location=[49.8063, -95.2098], zoom_start=15)
Caddy_split_roi.add_basemap('SATELLITE')
# split_roi.add_basemap('USGS NAIP Imagery False Color')
# Map.add_basemap('USGS NAIP Imagery NDVI')

Caddy_split_roi.split_map(left_layer='SATELLITE',
                    right_layer='Google Maps')

Caddy_split_roi


# In[23]:


# Make split view map half Satellite and half NAIP Imagery False Color (Big Sur)

Big_Sur_split_roi = geemap.Map(location=[35.8656, -121.4329], zoom_start=15)
Big_Sur_split_roi.add_basemap('SATELLITE')
Big_Sur_split_roi.add_basemap('USGS NAIP Imagery False Color')
# Map.add_basemap('USGS NAIP Imagery NDVI')

Big_Sur_split_roi.split_map(left_layer='SATELLITE',
                    right_layer='USGS NAIP Imagery False Color')

Big_Sur_split_roi


# ## An introduction to Sentinel 1 and Sentinel 2 imagery
# 
# Sentinel-1 and Sentinel-2 play pivotal roles within the Copernicus program initiated by the European Space Agency, contributing essential data for diverse Earth observation applications. Sentinel-1, functioning as a Synthetic Aperature Radar, delivers all-weather surveillance and monitoring capabilities by transmitting radar signals and capturing their reflections. This unique capacity permits the observation of Earth's surface regardless of cloud cover or daylight conditions, proving especially valuable for precise monitoring of ground shifts and changes such as deforestation. 
# In contrast, Sentinel-2 employs multispectral sensors to capture intricate imagery of terrestrial and coastal regions. With the capability to collect data across 13 distinct spectral bands, Sentinel-2 is instrumental in monitoring factors such as vegetation health, alterations in land cover, urban growth, and disaster assessment.

# ## SAR Data avalability
# #### Caddy Lake Landslide

# In[24]:


def Caddy_sentinel_1_availability(Caddy_event_date):
    start_date = ee.Date(Caddy_event_date).advance(-180, 'days')
    end_date = ee.Date(Caddy_event_date).advance(180, 'days')

    Caddy_availability = "Sentinel-1 image data range is between {} and {}.".format(
        start_date.format('YYYY-MM-dd').getInfo(),
        end_date.format('YYYY-MM-dd').getInfo()
    )
    
    return Caddy_availability

# Define the event date
Caddy_event_date = '2016-06-25'

Caddy_availability = Caddy_sentinel_1_availability(Caddy_event_date)
print(Caddy_availability)


# In[25]:


def create_Caddy_aoi(Caddy_center_coordinates, width):
    try:
        # Define the center point of the region of interest
        center_point = ee.Geometry.Point(Caddy_center_coordinates)

        # Create an EE AOI using the center coordinates and dimensions
        Caddy_aoi = center_point.buffer(width / 2).bounds()

        # Print the bounding box coordinates with four decimal places
        coords = [[round(x, 4) for x in coord] for coord in Caddy_aoi.
                  coordinates().getInfo()[0]]
        print("Bounding box coordinates: ", coords)

        return Caddy_aoi
    except Exception:
        print("An error occurred while creating the AOI.")
        return None

# Define the center coordinates of the region of interest
Caddy_center_coordinates = [-95.2098, 49.8063]
width = 1000

# Create the AOI using the create_Caddy_aoi function
Caddy_aoi = create_Caddy_aoi(Caddy_center_coordinates, width)

if Caddy_aoi is not None:
    print("AOI successfully created.")
else:
    print("Failed to create AOI. Check the input values and try again.")


# #### Big Sur Landslide

# In[26]:


def Big_Sur_sentinel_1_availability(Big_Sur_event_date):
    start_date = ee.Date(Big_Sur_event_date).advance(-180, 'days')
    end_date = ee.Date(Big_Sur_event_date).advance(180, 'days')

    Big_Sur_availability = "Big_Sur_Sentinel-1 image data range is between {} and {}.".format(
        start_date.format('YYYY-MM-dd').getInfo(),
        end_date.format('YYYY-MM-dd').getInfo()
    )
    
    return Big_Sur_availability

# Define the event date
Big_Sur_event_date = '2017-05-20'

Big_Sur_availability = Big_Sur_sentinel_1_availability(Big_Sur_event_date)
print(Big_Sur_availability)


# In[27]:


def create_Big_Sur_aoi(Big_Sur_center_coordinates, width):
    try:
        # Define the center point of the region of interest
        center_point = ee.Geometry.Point(Big_Sur_center_coordinates)

        # Create an EE AOI using the center coordinates and dimensions
        Big_Sur_aoi = center_point.buffer(width / 2).bounds()

        # Print the bounding box coordinates with four decimal places
        coords = [[round(x, 4) for x in coord] for coord in Big_Sur_aoi.
                  coordinates().getInfo()[0]]
        print("Bounding box coordinates: ", coords)

        return Big_Sur_aoi
    except Exception:
        print("An error occurred while creating the AOI.")
        return None

# Define the center coordinates of the region of interest
Big_Sur_center_coordinates = [-121.4329, 35.8656]
width = 1000

# Create the AOI using the create_Big_Sur_aoi function
Big_Sur_aoi = create_Big_Sur_aoi(Big_Sur_center_coordinates, width)

if Big_Sur_aoi is not None:
    print("AOI successfully created.")
else:
    print("Failed to create AOI. Check the input values and try again.")


# ## Collect SAR Images
# #### 1. Caddy Lake landslide

# In[28]:


# Collect and filter Sentinel-1 images by time and region of interest
Caddy_start_date = '2015-12-28'
Caddy_end_date = '2016-12-22'
Caddy_sentinel_1 = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
              .filterBounds(Caddy_aoi)
              .filterDate(Caddy_start_date, Caddy_end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation'
                                             ,'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation'
                                             ,'VH'))
             )

Caddy_image_collection = Caddy_sentinel_1.filter(ee.Filter.eq('orbitProperties_pass', 
                                                  'ASCENDING'))

orbit_num = (Caddy_image_collection.aggregate_array('relativeOrbitNumber_start')
             .getInfo())
look_angle = Caddy_image_collection.aggregate_array('orbitProperties_pass').getInfo()

if orbit_num and len(orbit_num) > 0:
    orbit_num = orbit_num[0]
else:
    orbit_num = None

if look_angle and len(look_angle) > 0:
    look_angle = look_angle[0]
else:
    look_angle = None

if orbit_num is not None and look_angle is not None:
    print('The Relative Orbit Number for ROI is:', orbit_num)
    print('The orbitology is:', look_angle)
    print('Number of images in the collection:', Caddy_image_collection.size().
          getInfo())
else:
    print('No images found in the collection.')


# In[29]:


# Retrieve acquisition date of each image in the collection as a list
Caddy_timestamplist = (Caddy_image_collection.aggregate_array('system:time_start')
                 .map(lambda t: ee.String('T').cat(ee.Date(t).format(
                     'YYYYMMdd')))
                 .getInfo())
# Check if timestamplist is not empty before retrieving its values
if Caddy_timestamplist:
    print("Caddy Lake Landslide timestamp retrieved.")
else:
    print("No timestamps available.")
Caddy_timestamplist


# #### 2. Big Sur landslide

# In[30]:


# Collect and filter Sentinel-1 images by time and region of interest
Big_Sur_start_date = '2016-11-21'
Big_Sur_end_date = '2017-11-16'   
Big_Sur_sentinel_1 = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
              .filterBounds(Big_Sur_aoi)
              .filterDate(Big_Sur_start_date, Big_Sur_end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation'
                                             ,'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation'
                                             ,'VH'))
             )

Big_Sur_image_collection = Big_Sur_sentinel_1.filter(ee.Filter.eq('orbitProperties_pass', 
                                                  'ASCENDING'))

orbit_num = (Big_Sur_image_collection.aggregate_array('relativeOrbitNumber_start')
             .getInfo())
look_angle = Big_Sur_image_collection.aggregate_array('orbitProperties_pass').getInfo()

if orbit_num and len(orbit_num) > 0:
    orbit_num = orbit_num[0]
else:
    orbit_num = None

if look_angle and len(look_angle) > 0:
    look_angle = look_angle[0]
else:
    look_angle = None

if orbit_num is not None and look_angle is not None:
    print('The Relative Orbit Number for ROI is:', orbit_num)
    print('The orbitology is:', look_angle)
    print('Number of images in the collection:', Big_Sur_image_collection.size().
          getInfo())
else:
    print('No images found in the collection.')


# In[31]:


# Retrieve acquisition date of each image in the collection as a list
Big_Sur_timestamplist = (Big_Sur_image_collection.aggregate_array('system:time_start')
                 .map(lambda t: ee.String('T').cat(ee.Date(t).format(
                     'YYYYMMdd')))
                 .getInfo())
# Check if timestamplist is not empty before retrieving its values
if Big_Sur_timestamplist:
    print("Big Sur Landslide timestamp retrieved.")
else:
    print("No timestamps available.")
Big_Sur_timestamplist


# ## Convert and clip Image collection
# #### Caddy Lake Landslide

# In[32]:


Caddy_im_list = Caddy_image_collection.toList(Caddy_image_collection.size())

# clip our list of images to the aoi geometry
def Caddy_clip_img(img):
    """
    Clips a list of images to our aoi geometry.

    Returns
    -------
    list
        clipped images to aoi

    """
    return ee.Image(img).clip(Caddy_aoi)

Caddy_im_list = ee.List(Caddy_im_list.map(Caddy_clip_img))
Caddy_im_list.get(0)
ee.Image(Caddy_im_list.get(0)).bandNames().getInfo()
# Caddy_im_list.length().getInfo()

if Caddy_im_list.size().getInfo() > 0:
    first_image = ee.Image(Caddy_im_list.get(0))
    band_names = first_image.bandNames().getInfo()
    list_length = Caddy_im_list.length().getInfo()

    print("First image band names:", band_names)
    print("List length:", list_length)
else:
    print("No images available in the list.")


# #### Big Sur Landslide

# In[33]:


Big_Sur_im_list = Big_Sur_image_collection.toList(Big_Sur_image_collection.size())

# clip our list of images to the aoi geometry
def Big_Sur_clip_img(img):
    """
    Clips a list of images to our aoi geometry.

    Returns
    -------
    list
        clipped images to aoi

    """
    return ee.Image(img).clip(Big_Sur_aoi)

Big_Sur_im_list = ee.List(Big_Sur_im_list.map(Big_Sur_clip_img))
Big_Sur_im_list.get(0)
ee.Image(Big_Sur_im_list.get(0)).bandNames().getInfo()
# Big_Sur_im_list.length().getInfo()

if Big_Sur_im_list.size().getInfo() > 0:
    first_image = ee.Image(Big_Sur_im_list.get(0))
    band_names = first_image.bandNames().getInfo()
    list_length = Big_Sur_im_list.length().getInfo()

    print("First image band names:", band_names)
    print("List length:", list_length)
else:
    print("No images available in the list.")


# ## Change detection

# In[34]:


# Add EE drawing method to folium
folium.Map.add_ee_layer = add_ee_layer


# #### Caddy Lake Landslide Change Detection

# In[35]:


# Create a likelihood ratio test statistic and evaluate it for a list of single polarization images
Caddy_vv_list = Caddy_im_list.map(selectvv)


# In[36]:


# Create change map 
alpha = 0.01
location = [49.8063, -95.2098]

Caddy_c_map = ee.Image.constant(1).subtract(chi2cdf(omnibus(Caddy_vv_list), 
                                              len(Caddy_timestamplist)-1))
Caddy_c_map = Caddy_c_map.multiply(0).where(Caddy_c_map.lt(alpha), 1)
Caddy_c_map = Caddy_c_map.updateMask(Caddy_c_map.gt(0))

# Display change map
mp = folium.Map(location=location, zoom_start=15)
mp.add_ee_layer(Caddy_c_map, {'min': 0, 'max': 1, 'palette': ['black', 'cyan']}, 
                'Change map')

# Add layer control
mp.add_child(folium.LayerControl())


# #### Big Sir Landslide Change Detection

# In[37]:


# Create a likelihood ratio test statistic and evaluate it for a list of single polarization images
Big_Sur_vv_list = Big_Sur_im_list.map(selectvv)


# In[38]:


# Create change map 
alpha = 0.01
location = [35.8656, -121.4329]

BS_c_map = ee.Image.constant(1).subtract(chi2cdf(omnibus(Big_Sur_vv_list), 
                                              len(Big_Sur_timestamplist)-1))
BS_c_map = BS_c_map.multiply(0).where(BS_c_map.lt(alpha), 1)
BS_c_map = BS_c_map.updateMask(BS_c_map.gt(0))

# Display change map
mp = folium.Map(location=location, zoom_start=15)
mp.add_ee_layer(BS_c_map, {'min': 0, 'max': 1, 'palette': ['black', 'cyan']}, 
                'Change map')

# Add layer control
mp.add_child(folium.LayerControl())


# ## Multispectral data availability (Sentinel 2)
# #### Caddy Lake Landslide

# In[39]:


# Collecting Caddy Sentinel-2 images
Caddy_start_date = '2015-12-28'
Caddy_end_date = '2016-12-22'
Caddy_sentinel_2 = (ee.ImageCollection('COPERNICUS/S2')
              .filterBounds(Caddy_aoi)
              .filterDate(Caddy_start_date, Caddy_end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
)

Caddy_S2_image_collection = Caddy_sentinel_2.filter(
    ee.Filter.eq('SENSING_ORBIT_DIRECTION', 'DESCENDING'))

if Caddy_image_collection.size().getInfo() > 0:
    orbit_num = Caddy_S2_image_collection.first().get('SENSING_ORBIT_NUMBER')
    orbit_direction = Caddy_S2_image_collection.first(
    ).get('SENSING_ORBIT_DIRECTION')
    print('The Sensing Orbit Number for ROI is:', orbit_num.getInfo())
    print('The Sensing Orbit Direction is:', orbit_direction.getInfo())
    print('Number of images in the collection:', Caddy_S2_image_collection.size(
    ).getInfo())
else:
    print('No images found in the collection.')


# In[40]:


# Get the first image from the collection
Caddy_S2_first_image = Caddy_S2_image_collection.first()

# Get the date of the first image 
date_str = ee.Date(Caddy_S2_first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print('Date of the first image:', date_str)


# ### Big Sur Landslide

# In[41]:


# Collecting Big Sur Sentinel-2 images
Big_Sur_sentinel_2 = (ee.ImageCollection('COPERNICUS/S2')
              .filterBounds(Big_Sur_aoi)
              .filterDate(Big_Sur_start_date, Big_Sur_end_date)
              .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
)

Big_Sur_S2_image_collection = Big_Sur_sentinel_2.filter(ee.Filter.eq('SENSING_ORBIT_DIRECTION', 'DESCENDING'))

if Big_Sur_image_collection.size().getInfo() > 0:
    orbit_num = Big_Sur_image_collection.first().get('SENSING_ORBIT_NUMBER')
    orbit_direction = Big_Sur_image_collection.first().get('SENSING_ORBIT_DIRECTION')
    print('The Sensing Orbit Number for ROI is:', orbit_num.getInfo())
    print('The Sensing Orbit Direction is:', orbit_direction.getInfo())
    print('Number of images in the collection:', Big_Sur_image_collection.size().getInfo())
else:
    print('No images found in the collection.')


# In[42]:


# Get the first image from the collection
Big_Sur_S2_first_image = Big_Sur_S2_image_collection.first()

# Get the date of the first image 
date_str = ee.Date(Big_Sur_S2_first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print('Date of the first image:', date_str)


# ### Multispectral imagery NDWI Calculations 

# In[43]:


# Select necessary bands for NDWI calculation
red_band = 'B3'
nir_band = 'B8'

# Function to calculate NDWI
def calculate_ndwi(image):
    ndwi = image.normalizedDifference([nir_band, red_band])
    return ndwi.gte(0)

# Calculate NDWI for the first image in the collection
Caddy_S2_first_image = Caddy_sentinel_2.first()
Caddy_S2_ndwi_first_image = calculate_ndwi(Caddy_S2_first_image)

# Display the result on the map
Caddy_S2_map = folium.Map(location=[49.8063, -95.2098], zoom_start=15)
folium.TileLayer(
    tiles='SATELLITE',
    attr='Caddy_Sentinel-2 Imagery',
    overlay=True,
    name='NDWI Difference',
    control=True
).add_to(Caddy_S2_map)

# Add the NDWI difference layer to the map
Caddy_S2_map.add_ee_layer(Caddy_S2_ndwi_first_image, {'min': -1, 'max': 1, 'palette': 
                                    ['red', 'yellow', 'green']}, 'NDWI')

# Add a layer control panel to the map
folium.LayerControl().add_to(Caddy_S2_map)

# Display the map
Caddy_S2_map


# In[44]:


# Select necessary bands for NDWI calculation
red_band = 'B3'
nir_band = 'B8'

# Function to calculate NDWI
def calculate_ndwi(image):
    ndwi = image.normalizedDifference([nir_band, red_band])
    return ndwi.gte(0)

# Calculate NDWI for the first image in the collection
Big_Sur_S2_first_image = Big_Sur_sentinel_2.first()
Big_Sur_S2_ndwi_first_image = calculate_ndwi(Big_Sur_S2_first_image)

Big_Sur_S2_map = folium.Map(location=[35.8656, -121.4329], zoom_start=15) 
folium.TileLayer(
    tiles='SATELLITE',
    attr='Caddy_Sentinel-2 Imagery',
    overlay=True,
    name='NDWI Difference',
    control=True
).add_to(Big_Sur_S2_map)

# Add the NDWI difference layer to the map
Big_Sur_S2_map.add_ee_layer(Big_Sur_S2_ndwi_first_image, {'min': -1, 'max': 1, 'palette': 
                                    ['red', 'yellow', 'green']}, 'NDWI')

# Add a layer control panel to the map
folium.LayerControl().add_to(Big_Sur_S2_map)

# Display the map
Big_Sur_S2_map


# ## Land Surface Change with subtracting NDWI

# In[45]:


# Subtract NDWI from the change map
Caddy_LS = Caddy_c_map.subtract(Caddy_S2_ndwi_first_image)

# Display the result on the map
Caddy_map = folium.Map(location=[49.8063, -95.2098], zoom_start=15)
Caddy_map.add_ee_layer(Caddy_LS, {'min': -1, 'max': 1, 'palette': ['black', 'cyan']}, 'Change map with NDWI subtracted')

# Add layer control
Caddy_map.add_child(folium.LayerControl())

Caddy_map


# In[46]:


# Subtract NDWI from the change map
Big_Sur_LS = Big_Sur_S2_ndwi_first_image.subtract(BS_c_map)

# Display the result on the map
Big_Sur_map = folium.Map(location=[35.8656, -121.4329], zoom_start=15)
Big_Sur_map.add_ee_layer(Big_Sur_LS, {'min': -1, 'max': 1, 'palette': ['black', 'cyan']}, 'Change map with NDWI subtracted')

# Add layer control
Big_Sur_map.add_child(folium.LayerControl())

# Display the map
Big_Sur_map


# ## Caddy Lake Landslide (NDWI-Sentinel1)

# ### VH band

# In[48]:


# Function to calculate water mask using only VH band
def Caddy_calculate_water_mask(image):
    vh_band = 'VH'
    Caddy_water_mask = image.select(vh_band).gt(0)
    return image.addBands(Caddy_water_mask.rename('Caddy_water_mask'))

# Load Sentinel-1 collection and filter by date and region of interest
sentinel1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(Caddy_aoi)
    .filterDate(Caddy_start_date, Caddy_end_date)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))
#     .map(Caddy_calculate_water_mask))

# Function to calculate land surface change using VH band
def Caddy_calculate_land_surface_change(image):
    vh_band = 'VH'
    Caddy_land_surface_change = image.filterDate('2016-11-20', Caddy_end_date).first().subtract(image.first())
    return Caddy_land_surface_change

# Calculate land surface change for each image in the collection
Caddy_sentinel1_with_land_surface_change = Caddy_calculate_land_surface_change(sentinel1)

# Display the result on the map
map_vh_Caddy = folium.Map(location=[49.8063, -95.2098], zoom_start=15)

# Add the mosaic image as a layer with Sentinel-1 VH band imagery visualization parameters
vis_params = {
    'min': -20,
    'max': 0
}

# Add the land surface change layer to the map with the specified color palette
map_vh_Caddy.add_ee_layer(Caddy_sentinel1_with_land_surface_change, 
                 {'min': -10, 'max': 10}, 'Caddy Land Surface Change')

# Add a layer control panel to the map
folium.LayerControl().add_to(map_vh_Caddy)

# Display the map
map_vh_Caddy


# In[49]:


# Subtract the outputs
Caddy_subtract_image = Caddy_S2_ndwi_first_image.subtract(Caddy_sentinel1_with_land_surface_change)

# Display the result on the map
map_output = folium.Map(location=[49.8063, -95.2098], zoom_start=15)
folium.TileLayer(
    tiles='SATELLITE',
    attr='Caddy_Sentinel-2 Imagery',
    overlay=True,
    name='NDWI Difference',
).add_to(map_output)

# Add the output layer to the map
map_output.add_ee_layer(Caddy_subtract_image.select([0]), {'min': -1, 'max': 1, 'palette': 
                                    ['red', 'yellow', 'green']}, 'NDWI Difference')

# Add a layer control panel to the map
folium.LayerControl().add_to(map_output)

# Display the map
map_output


# ### VV and VH bands

# In[50]:


# # Function to calculate NDWI based on the first image
# vv_band = 'VV'
# vh_band = 'VH'

# def calculate_ndwi(image):
#     ndwi = image.normalizedDifference([vv_band, vh_band])
#     return image.addBands(ndwi.rename('NDWI'))

# # Calculate NDWI for the first image in the collection
# first_image = Caddy_sentinel_1.map(calculate_ndwi).first()
# ndwi_first_image = first_image.select('NDWI')

# # Display the result on the map
# map_ndwi = folium.Map(location=[49.8063, -95.2098], zoom_start=15)
# folium.TileLayer(
#     tiles='SATELLITE',
#     attr='Caddy_Sentinel-1 Imagery',
#     overlay=True,
#     name='NDWI (First Image)',
#     control=True
# ).add_to(map_ndwi)

# # Add the NDWI layer for the first image to the map
# map_ndwi.add_ee_layer(ndwi_first_image, {'min': -1, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDWI (First Image)')

# # Add a layer control panel to the map
# folium.LayerControl().add_to(map_ndwi)

# # Display the map
# map_ndwi


# ### VV band

# In[51]:


# # Function to calculate Water Mask using VV band
# def create_water_mask_vv(image):
#     # Threshold the VV band to identify water pixels
#     water_mask = image.select('VV').gte(0.5)  
    
#     # Set non-water pixels to 0 and water pixels to 1
#     return water_mask.rename('water_mask')

# # Apply the water mask function to the Sentinel-1 collection
# Caddy_water_mask_vv = Caddy_sentinel_1.map(create_water_mask_vv)

# # Get the water mask from the first image (assuming it's the same for all images)
# water_mask_vv = Caddy_water_mask_vv.first().select('water_mask')

# # Display the water mask
# map_vv = folium.Map(location=[49.8063, -95.2098], zoom_start=14)
# folium.TileLayer(
#     tiles='SATELLITE',
#     attr='Caddy_Sentinel-1 Imagery',
#     overlay=True,
#     name='Sentinel-1 Imagery',
#     control=True
# ).add_to(map_vv)

# # Add the water mask on top of the Sentinel-1 imagery
# map_vv.add_ee_layer(Caddy_sentinel_1.first().visualize(), {}, 'Sentinel-1 Imagery (VV)')
# map_vv.add_ee_layer(water_mask_vv.updateMask(water_mask_vv), {'palette': 'blue'}, 'Water Mask (VV)')  

# # Add a layer control panel to the map
# folium.LayerControl().add_to(map_vv)

# # Display the map
# map_vv


# ## Big Sur Landslide (NDWI-Sentinel 1)

# ### VH band

# In[52]:


# Function to calculate water mask using only VH band
def Big_Sur_calculate_water_mask(image):
    vh_band = 'VH'
    Big_Sur_water_mask = image.select(vh_band).gt(0)
    return image.addBands(Big_Sur_water_mask.rename('Big_Sur_water_mask'))

# Load Sentinel-1 collection and filter by date and region of interest
sentinel1 = (ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(Big_Sur_aoi)
    .filterDate(Big_Sur_start_date, Big_Sur_end_date)
    .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')))

# Function to calculate land surface change using VH band
def Big_Sur_calculate_land_surface_change(image):
    vh_band = 'VH'
    Big_Sur_land_surface_change = image.filterDate('2016-12-20', Big_Sur_end_date).first().subtract(image.first())
    return Big_Sur_land_surface_change

# Calculate land surface change for each image in the collection
Big_Sur_sentinel1_with_land_surface_change = Big_Sur_calculate_land_surface_change(sentinel1)

# Display the result on the map
map_vh_Big_Sur = folium.Map(location=[35.8656, -121.4329], zoom_start=15)

# Add the mosaic image as a layer with Sentinel-1 VH band imagery visualization parameters
vis_params = {
    'min': -20,
    'max': 0
}

# Add the land surface change layer to the map with the specified color palette
map_vh_Big_Sur.add_ee_layer(Big_Sur_sentinel1_with_land_surface_change, 
                 {'min': -10, 'max': 10}, 'Big Sur Land Surface Change')

# Add a layer control panel to the map
folium.LayerControl().add_to(map_vh_Big_Sur)

# Display the map
map_vh_Big_Sur


# ### VV and VH bands

# In[53]:


# Function to calculate NDWI based on the first image
vv_band = 'VV'
vh_band = 'VH'

def calculate_ndwi(image):
    ndwi = image.normalizedDifference([vv_band, vh_band])
    return image.addBands(ndwi.rename('NDWI'))

# Calculate NDWI for the first image in the collection
first_image = Big_Sur_sentinel_1.map(calculate_ndwi).first()
ndwi_first_image = first_image.select('NDWI')

# Display the result on the map
map_ndwi = folium.Map(location=[35.8656, -121.4329], zoom_start=15)
folium.TileLayer(
    tiles='SATELLITE',
    attr='Big_Sur_Sentinel-1 Imagery',
    overlay=True,
    name='NDWI (First Image)',
    control=True
).add_to(map_ndwi)

# Add the NDWI layer for the first image to the map
map_ndwi.add_ee_layer(ndwi_first_image, {'min': -1, 'max': 1, 'palette': ['red', 'yellow', 'green']}, 'NDWI (First Image)')

# Add a layer control panel to the map
folium.LayerControl().add_to(map_ndwi)

# Display the map
map_ndwi


# ### VV band
# 

# In[56]:


# # Function to calculate Water Mask using VV band
# def create_water_mask_vv(image):
#     # Threshold the VV band to identify water pixels
#     water_mask = image.select('VV').gt(0.2)  
    
#     # Set non-water pixels to 0 and water pixels to 1
#     return water_mask.rename('water_mask')

# # Apply the water mask function to the Sentinel-1 collection
# Big_Sur_water_mask_vv = Big_Sur_sentinel_1.map(create_water_mask_vv)

# # Get the water mask from the first image (assuming it's the same for all images)
# water_mask_vv = Big_Sur_water_mask_vv.first().select('water_mask')

# # Display the water mask
# map_vv = folium.Map(location=[35.8656, -121.43298], zoom_start=14)
# folium.TileLayer(
#     tiles='SATELLITE',
#     attr='Big_Sur_Sentinel-1 Imagery',
#     overlay=True,
#     name='Sentinel-1 Imagery',
#     control=True
# ).add_to(map_vv)

# # Add the water mask on top of the Sentinel-1 imagery
# map_vv.add_ee_layer(Big_Sur_sentinel_1.first().visualize(), {}, 'Sentinel-1 Imagery (VV)')
# map_vv.add_ee_layer(water_mask_vv.updateMask(water_mask_vv), {'palette': 'blue'}, 'Water Mask (VV)')  

# # Add a layer control panel to the map
# folium.LayerControl().add_to(map_vv)

# # Display the map
# map_vv


# ## Summary and results

# Land changes due to landslides of interest has been detected. Accordingly, appropriate times series of SAR images are processed to pinpoint land surface changes before and after the landslide events. However, the large water bodies are included in our analysis, so identified as land surface change. To address this issue, we tried to apply a water mask on multispectral imagery for landslides, instead of directly applying a water mask to SAR images (which we found it quite difficult following multiple tries). Finally, we have developed the code using subtraction technique and managed to distinguish between water bodies and land surface changes.
# 
# Once landslide susceptibility of an area is detected, solutions can be developed to predict probable occurrence and mitigate or in case prevent the potential hazards. Preliminary monitoring can provide vital information on how much an area is prone to landslide, which accordingly is essential for emergency response, and catastrophe mitigation in the areas vulnerable to landslides. Accordingly engineered solutions can be performed to stabilize unstable slopes, for instance improving drainage, reducing angle of slope, and building supportive walls at the bottom of the slopes.
