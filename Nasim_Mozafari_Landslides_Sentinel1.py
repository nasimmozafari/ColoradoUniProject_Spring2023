#!/usr/bin/env python
# coding: utf-8

# # Detection of landslide of Caddy Lake, Manitoba, Canada using SAR
# #### Nasim Mozafari, Mentor: Elsa Culler

# ## Run Google earth Engine (GEE)

# In[1]:


import ee
# Trigger the authentication flow.
ee.Authenticate()

# Initialize the library.
ee.Initialize()


# ## Data source
# 
# 1. [Copernicus_S1_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) by European Space Agency(ESA)
# 2. USGS NAIP Imagery NDVI (GEE basemap layer)
# 3. USGS NAIP Imagery False Color
# 4. Verified landslides of North America collected by Earth Lab at the University of Colorado Boulder

# ## Import libraries and packages

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
large_ls.describe()


# In[17]:


# Extract information of landslides of Caddy Lake, Manitoba, Canada
Caddy_df = landslide_gdf[landslide_gdf['location'].str.contains
                              ('Caddy Lake')]
Caddy_df.head(2)


# In[18]:


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


# ## AOI (Area of Interest)
# #### Create interactive map of landslide of AOI (Caddy Lake, MB)

# In[19]:


# Display landslide of Caddy Lake location
landslide_map = folium.Map(
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
    ).add_to(landslide_map)

landslide_map


# In[20]:


# Make split view map half Satellite and half NAIP Imagery False Color 

split_roi = geemap.Map(location=[49.8063, -95.2098], zoom_start=15)
split_roi.add_basemap('SATELLITE')
split_roi.add_basemap('USGS NAIP Imagery False Color')
# Map.add_basemap('USGS NAIP Imagery NDVI')

split_roi.split_map(left_layer='SATELLITE',
                    right_layer='USGS NAIP Imagery False Color')

split_roi


# ## Data avalability

# In[37]:


def get_sentinel_1_availability(event_date):
    start_date = ee.Date(event_date).advance(-180, 'days')
    end_date = ee.Date(event_date).advance(180, 'days')

    availability = "Sentinel-1 image data range is between {} and {}".format(
        start_date.format('YYYY-MM-dd').getInfo(),
        end_date.format('YYYY-MM-dd').getInfo()
    )
    
    return availability

# Example usage:
event_date = '2016-06-25'
availability = get_sentinel_1_availability(event_date)
print(availability)


# In[38]:


def create_aoi(center_coordinates, width):
    # Define the center point of the region of interest
    center_point = ee.Geometry.Point(center_coordinates)

    # Create an Earth Engine AOI using the center coordinates and dimensions
    aoi = center_point.buffer(width / 2).bounds()

    # Print the bounding box coordinates with four decimal places
    coords = [[round(x, 4) for x in coord] for coord in aoi.coordinates().getInfo()[0]]
    print("Bounding box coordinates: ", coords)

    return aoi

# Example usage:
center_coordinates = [-95.2098, 49.8063]
width = 1000
aoi = create_aoi(center_coordinates, width)
aoi


# ## Collect SAR Images

# In[39]:


# Collect and filter Sentinel-1 images by time and region of interest
start_date = '2015-12-28'
end_date = '2016-12-22'
sentinel_1 = (ee.ImageCollection('COPERNICUS/S1_GRD_FLOAT')
              .filterBounds(aoi)
              .filterDate(start_date, end_date)
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation'
                                             ,'VV'))
              .filter(ee.Filter.listContains('transmitterReceiverPolarisation'
                                             ,'VH'))
             )

image_collection = sentinel_1.filter(ee.Filter.eq('orbitProperties_pass', 
                                                  'ASCENDING'))

orbit_num = (image_collection.aggregate_array('relativeOrbitNumber_start')
             .getInfo())
if orbit_num:
    orbit_num = orbit_num[0]

look_angle = (image_collection.aggregate_array('orbitProperties_pass')
              .getInfo())
if look_angle and len(look_angle) > 0:
    look_angle = look_angle[0]

print('The Relative Orbit Number for ROI is: ', orbit_num)
print('The orbitology is: ', look_angle)

image_collection


# In[24]:


# Retrieve acquisition date of each image in the collection as a list
timestamplist = (image_collection.aggregate_array('system:time_start')
                 .map(lambda t: ee.String('T').cat(ee.Date(t).format(
                     'YYYYMMdd')))
                 .getInfo())

timestamplist


# ## Convert and clip Image collection

# In[40]:


im_list = image_collection.toList(image_collection.size())

# clip our list of images to the aoi geometry
def clip_img(img):
    """
    Clips a list of images to our aoi geometry.

    Returns
    -------
    list
        clipped images to aoi

    """
    return ee.Image(img).clip(aoi)

im_list = ee.List(im_list.map(clip_img))
im_list.get(0)
ee.Image(im_list.get(0)).bandNames().getInfo()
im_list.length().getInfo()


# ## Change detection

# In[41]:


# Add EE drawing method to folium
folium.Map.add_ee_layer = add_ee_layer


# In[42]:


# def selectvv(current):
#     return ee.Image(current).select('VV')

# Create a likelihood ratio test statistic and evaluate it for a list of single polarization images
vv_list = im_list.map(selectvv)


# In[43]:


# # Create a likelihood ratio test statistic k = len(timestamplist)//2
# hist = (omnibus(vv_list.slice(0, k))
#         .reduceRegion(ee.Reducer.fixedHistogram(0, 40, 200), geometry=aoi, scale=10)
#         .get('constant')
#         .getInfo())

# a = np.array(hist)
# x = a[:, 0]
# y = a[:, 1]/np.sum(a[:, 1])
# plt.plot(x, y, '.', label='data')
# plt.plot(x, chi2.pdf(x, k-1)/5, '-r', label='chi square')
# plt.legend()
# plt.grid()
# plt.show()


# In[44]:


# Create change map 
alpha = 0.01
location = [49.8063, -95.2098]

c_map = ee.Image.constant(1).subtract(chi2cdf(omnibus(vv_list), 
                                              len(timestamplist)-1))
c_map = c_map.multiply(0).where(c_map.lt(alpha), 1)
c_map = c_map.updateMask(c_map.gt(0))

# Display change map
mp = folium.Map(location=location, zoom_start=15)
mp.add_ee_layer(c_map, {'min': 0, 'max': 1, 'palette': ['black', 'red']}, 
                'Change map')

# Display change map at a higher resolution
c_map_10m = c_map.reproject(c_map.projection().crs(), scale=10)
mp.add_ee_layer(c_map_10m, {'min': 0, 'max': 1, 'palette': ['black', 'cyan']}, 
                'Change map (10m)')

# Add layer control
mp.add_child(folium.LayerControl())


# ## Change map

# In[30]:


# # Sample the first few list indices.
# samples = ee.List.sequence(2, 5).map(sample_vv_imgs)

# # Calculate and display the correlation matrix.
# np.set_printoptions(precision=2, suppress=True)


# In[31]:


# plot_change_maps(im_list)


# ## Summary and results

# A time series including 27 SAR satellite images (Sentinel-1) have been processed across the location of Caddy Lake landslide, Manitoba, Canada to detect land changes occurred following the event of September 2016.
# 
# Our result shows rather clear changes around the verified landslide location confirming high potential of SAR images in order to identify landslides. 
# 
# Once landslide susceptibility of an area is detected, solutions can be developed to predict probable occurrence and mitigate or in case prevent the potential hazards. Preliminary monitoring can provide vital information on how much an area is prone to landslide, which accordingly is essential for emergency response, and catastrophe mitigation in the areas vulnerable to landslides. Accordingly engineered solutions can be performed to stabilize unstable slopes, for instance improving drainage, reducing angle of slope, and building supportive walls at the bottom of the slopes.
