#!/usr/bin/env python
# coding: utf-8

# # <center>Landslide detection using Synthetic Aperture Radar and Multispectral imagery</center>
# 
# ### <span style="background-color: #green; padding: 5px; border-radius: 5px;"><div style="background-color: #f2f2f2; padding: 10px;"><center>Nasim Mozafari Amiri, Mentor: Elsa Culler</center></div></span><span style="background-color: #green; padding: 5px; border-radius: 5px;"><div style="background-color: #f2f2f2; padding: 10px;"><center>August 2023</center></div></span>

# ![Big Dur landslide, May 2015](https://static01.nyt.com/images/2017/05/25/us/25sur-xp/25sur-xp-superJumbo.jpg?quality=75&auto=webp)
# Big Sur landslide, Highway 1, California, USA, Image credit: The New York Times

# <h2 id="Introduction and project-Goals">Project Goals<a class="anchor-link" href="#Project-Goals">&#182;</a></h2><p>As documented by the NASA Global Landslide Catalog, landslides pose a significant geological threat to North America and specifically most states within the United States. Furthermore, data from the US Geological Survey (USGS) suggests an ongoing yearly toll of approximately 25-50 individuals falling victim to landslides across the nation. Recognizing the criticality of landslide exposure assessment and comprehending the geographical context of inhabited areas becomes paramount to effectively manage and minimize the potential impact of these natural disasters.
# <p>Given the significance of studying landslide disasters, this project focused on the examination of two specific landslide incidents. Consequently, we have processed appropriate time series of SAR images to identify changes in the land surface before and after these events: one occurring near Caddy Lake in Manitoba, Canada, and the other at Big Sur in California, USA. To ensure that our analysis excludes water bodies, we calculated the Normalized Difference Water Index (NDWI) using multispectral imagery. This approach was chosen over directly applying a water mask to SAR images, as we encountered challenges in doing so despite multiple attempts. In conclusion, we have developed a codebase capable of distinguishing changes in the land surface by effectively subtracting water bodies.</p>

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

# In[2]:


# Required libraries and packages
import os
import json
import earthpy as et
import pandas as pd
import folium
import geopandas as gpd
import matplotlib.pyplot as plt
import ee
import geemap

import geemap.foliumap as geemap

from scipy.stats import chi2
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Google Earth Engine Modules
# We utilized multiple Python modules provided by Google Earth Engine to process our data.

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


# ## Set working directory

# In[7]:


# Change directory to landslide-detect data path
data_path = os.path.join(et.io.HOME, "earth-analytics", "landslide-detect")
if os.path.exists(data_path):
    os.chdir(data_path)
else:
    os.makedirs(data_path)
    print('The new directory is created!')
    os.chdir(data_path)

data_path


# In[8]:


get_ipython().run_cell_magic('bash', '', 'find .\n')


# ## Create data frame
# Utilizing information curated by Earth Lab at the University of Colorado Boulder (csv file), we have generated an extensive data frame comprising 228 confirmed landslides that occurred in North America from September 2015 to July 2017.

# In[9]:


# Create DataFrame and open landslide file of North America
landslide_gdf = gpd.read_file('landslides.verified.csv')
landslide_gdf.head()


# In[10]:


# Extract verified large landslides of North America
large_ls = landslide_gdf[landslide_gdf['size'].str.contains
                              ('large')]
large_ls.head()


# #### Folium map of all large verified landslide locations of North America (2015-2017)

# In[11]:


# Display all verified large landslides of North America 
large_ls_map = folium.Map(
    location=[43.0000, -105.0000],
    zoom_start=4,
    width=900,
    height=500,
    tiles='Stamen terrain')


for index, row in large_ls.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['slide.id'],
        icon=folium.Icon(color="red")
    ).add_to(large_ls_map)

large_ls_map


# ## Area of Interest (AOI)
# We studied two landslide events namely, Caddy Lake, Manitoba, Canada (June 25, 2016) and Bug Sur, California, USA (May 20, 2017). We opted for these selections to assess the functionality of our code across varying types and sizes of landslides. While the first is characterized by scattered land removal across an area of about 5 square kilometers, the last caused a substantial mass movement across the highway. 
# 
# ## Extracting Caddy Lake and Big Sur landslides data

# In[12]:


# Extract information of landslides of Caddy Lake, Manitoba, Canada
Caddy_df = landslide_gdf[landslide_gdf['location'].str.contains
                              ('Caddy Lake')]
Caddy_df


# In[13]:


# Extract information of landslides of Big Sur, CA
Big_Sur_df = landslide_gdf[landslide_gdf['location'].str.contains
                              ('Big Sur')]
Big_Sur_df


# #### Interactive map of Caddy Lake landslide, MB, Canada, with accuracy of 5 km

# In[14]:


# Display landslide of Caddy Lake location
Caddy_landslide_map = folium.Map(
    location=[49.8063, -95.2098],
    zoom_start=13,
    width=1000,
    height=600,
    tiles='Stamen terrain')


for index, row in Caddy_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['slide.id'],
        icon=folium.Icon(color="red")
    ).add_to(Caddy_landslide_map)

Caddy_landslide_map


# #### Interactive map of landslide of Big Sur, CA, USA

# In[15]:


# Display landslide of Big Sur location
Big_Sur_landslide_map = folium.Map(
    location=[35.8656, -121.4329],
    zoom_start=13,
    width=1000,
    height=600,
    tiles='Stamen terrain')


for index, row in Big_Sur_df.iterrows():
    folium.Marker(
        location=[row['lat'], row['lon']],
        popup=row['slide.id'],
        icon=folium.Icon(color="red")
    ).add_to(Big_Sur_landslide_map)

Big_Sur_landslide_map


# #### <center>Split map of Caddy Lake landslide location</center>

# In[16]:


# Make split view map half Satellite and half goole maps (Caddy Lake)

Caddy_split_roi = geemap.Map(location=[49.8063, -95.2098], zoom_start=15)
Caddy_split_roi.add_basemap('SATELLITE')
# split_roi.add_basemap('USGS NAIP Imagery False Color') # Missing for Canada
# Map.add_basemap('USGS NAIP Imagery NDVI')

Caddy_split_roi.split_map(left_layer='SATELLITE',
                    right_layer='Google Maps')

Caddy_split_roi


# #### <center>Split map of Big Sur landslide location</center>

# In[17]:


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
# Sentinel-1 and Sentinel-2 play pivotal roles within the Copernicus program initiated by the European Space Agency, contributing essential data for diverse Earth observation applications. Sentinel-1, functioning as a Synthetic Aperture Radar, delivers all-weather surveillance and monitoring capabilities by transmitting radar signals and capturing their reflections. This unique capacity permits the observation of Earth's surface regardless of cloud cover or daylight conditions, proving especially valuable for precise monitoring of ground shifts and changes such as deforestation. 
# In contrast, Sentinel-2 employs multispectral sensors to capture intricate imagery of terrestrial and coastal regions. With the capability to collect data across 13 distinct spectral bands, Sentinel-2 is instrumental in monitoring factors such as vegetation health, alterations in land cover, urban growth, and disaster assessment.

# ## SAR Data avalability
# #### Caddy Lake Landslide

# In[18]:


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


# In[19]:


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

# In[20]:


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


# In[21]:


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


# ## Collecting and analyzing SAR images (Sentinel 1)
# We retrieved times series of SAR (Sentinel-1) satellite imagery within a time span of 180 days before and after the event dates to monitor pre- and post- event changes containing 27 and 31 images, respectively, for Caddy Lake and Big Sur locations. By analyzing the time series data, we have the capability to generate change maps, in which the land changes are represented by the red areas in the maps presented below.

# In[22]:


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


# In[23]:


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

# In[24]:


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


# In[25]:


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

# In[26]:


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

# In[27]:


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

# In[28]:


# Add EE drawing method to folium
folium.Map.add_ee_layer = add_ee_layer


# #### Caddy Lake Landslide Change Detection

# In[29]:


# Create a likelihood ratio test statistic and evaluate it for a list of single polarization images
Caddy_vv_list = Caddy_im_list.map(selectvv)


# #### <center>Change map of Caddy Lake landslide based on Sentinel-1 Image time series six months before and after the event date</center>

# In[30]:


# Create change map 
alpha = 0.01
location = [49.8063, -95.2098]

Caddy_c_map = ee.Image.constant(1).subtract(chi2cdf(omnibus(Caddy_vv_list), 
                                              len(Caddy_timestamplist)-1))
Caddy_c_map = Caddy_c_map.multiply(0).where(Caddy_c_map.lt(alpha), 1)

# Display change map
mp = folium.Map(location=location, zoom_start=15)
mp.add_ee_layer(Caddy_c_map, {'min': 0, 'max': 1, 'palette': ['cyan', 'red']}, 
                'Change map')

# Add layer control
mp.add_child(folium.LayerControl())


# #### Big Sir Landslide Change Detection

# In[31]:


# Create a likelihood ratio test statistic and evaluate it for a list of single polarization images
Big_Sur_vv_list = Big_Sur_im_list.map(selectvv)


# #### <center>Change map of Big Sur landslide based on Sentinel-1 Image time series six months before and after the event date</center>

# In[32]:


# Create change map 
alpha = 0.01
location = [35.8656, -121.4329]

BS_c_map = ee.Image.constant(1).subtract(chi2cdf(omnibus(Big_Sur_vv_list), 
                                              len(Big_Sur_timestamplist)-1))
BS_c_map = BS_c_map.multiply(0).where(BS_c_map.lt(alpha), 1)

# Display change map
mp = folium.Map(location=location, zoom_start=15)
mp.add_ee_layer(BS_c_map, {'min': 0, 'max': 1, 'palette': ['cyan', 'red']}, 
                'Change map')

# Add layer control
mp.add_child(folium.LayerControl())


# ## Collecting and analyzing multispectral images (Sentinel 2)
# Upon the creation of change maps, the process of applying the Normalized Difference Water Index (NDWI) involved obtaining multispectral images corresponding to the Caddy Lake and Big Sur areas. However, for the application of the water mask, we exclusively utilized the last image in the time series. The resulting maps showcase water bodies highlighted in yellow.

# In[33]:


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


# In[34]:


# Get the first image from the collection
Caddy_S2_first_image = Caddy_S2_image_collection.first()

# Get the date of the first image 
date_str = ee.Date(Caddy_S2_first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print('Date of the first image:', date_str)


# ### Big Sur Landslide

# In[35]:


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


# In[36]:


# Get the first image from the collection
Big_Sur_S2_first_image = Big_Sur_S2_image_collection.first()

# Get the date of the first image 
date_str = ee.Date(Big_Sur_S2_first_image.get('system:time_start')).format('YYYY-MM-dd').getInfo()
print('Date of the first image:', date_str)


# ### NDWI Calculations using multispectral imagery

# #### <center>NDWI Calculation on Sentinel-2 image, Caddy Lake landslide, Water mask displayed in Yellow</center>

# In[37]:


# Select necessary bands for NDWI calculation
red_band = 'B3'
nir_band = 'B8'

# Function to calculate NDWI
def calculate_ndwi(image):
    ndwi = image.normalizedDifference([nir_band, red_band])
    return ndwi.gte(0)

# Calculate NDWI for the last image in the collection
Caddy_S2_last_image = Caddy_sentinel_2.sort('system:time_start', False).first()
Caddy_S2_ndwi_last_image = calculate_ndwi(Caddy_S2_last_image)

# Display the result on the map
Caddy_S2_map = folium.Map(location=[49.8063, -95.2098], zoom_start=14)
folium.TileLayer(
    tiles='SATELLITE',
    attr='Caddy_Sentinel-2 Imagery',
    overlay=True,
    name='NDWI Difference',
    control=True
).add_to(Caddy_S2_map)

# Add the NDWI difference layer to the map
Caddy_S2_map.add_ee_layer(Caddy_S2_ndwi_last_image, {'min': -1, 'max': 1, 'palette': 
                                    ['red', 'yellow', 'green']}, 'NDWI')

# Add a layer control panel to the map
folium.LayerControl().add_to(Caddy_S2_map)

# Display the map
Caddy_S2_map


# #### <center>NDWI Calculation on Sentinel-2 image, Big Sur landslide, Water mask displayed in Yellow</center>

# In[38]:


# Select necessary bands for NDWI calculation
red_band = 'B3'
nir_band = 'B8'

# Function to calculate NDWI
def calculate_ndwi(image):
    ndwi = image.normalizedDifference([nir_band, red_band])
    return ndwi.gte(0)

# Calculate NDWI for the last image in the collection
Big_Sur_S2_last_image = Big_Sur_sentinel_2.sort('system:time_start', False).first()
Big_Sur_S2_ndwi_last_image = calculate_ndwi(Big_Sur_S2_last_image)

Big_Sur_S2_map = folium.Map(location=[35.8656, -121.4329], zoom_start=14) 
folium.TileLayer(
    tiles='SATELLITE',
    attr='Caddy_Sentinel-2 Imagery',
    overlay=True,
    name='NDWI Difference',
    control=True
).add_to(Big_Sur_S2_map)

# Add the NDWI difference layer to the map
Big_Sur_S2_map.add_ee_layer(Big_Sur_S2_ndwi_last_image, {'min': -1, 'max': 1, 'palette': 
                                    ['red', 'yellow', 'green']}, 'NDWI')

# Add a layer control panel to the map
folium.LayerControl().add_to(Big_Sur_S2_map)

# Display the map
Big_Sur_S2_map


# ## Land Surface Change with applying NDWI
# After completing our statistical analyses and excluding water bodies, the red color layers align with changes in reflectance intensity from satellite images. These areas in red indicate the removal of Earth's surface, highlighting the occurrence of a landslide in our context.

# #### <center>Detected land surface changes displayed in red, Caddy Lake landslide</center>

# In[39]:


# Subtract NDWI from the change map
Caddy_LS = Caddy_S2_ndwi_last_image.And(Caddy_c_map)

# Updating water mask
Caddy_LS = Caddy_LS.updateMask(Caddy_LS.gt(0))

# Display the result on the map
Caddy_LS_map = folium.Map(location=[49.8063, -95.2098], zoom_start=15)
Caddy_LS_map.add_ee_layer(Caddy_LS, {'min': -1, 'max': 1, 'palette': ['black', 'red']}, 'land surface change')

# Add layer control
Caddy_LS_map.add_child(folium.LayerControl())
Caddy_LS_map


# #### <center>Detected land surface changes displayed in red, Big Sur landslide</center>

# In[40]:


# Subtract NDWI from the change map
Big_Sur_LS = Big_Sur_S2_ndwi_last_image.And(BS_c_map)

# Updating water mask
Big_Sur_LS = Big_Sur_LS.updateMask(Big_Sur_LS.gt(0))

# Display the result on the map
Big_Sur_LS_map = folium.Map(location=[35.8656, -121.4329], zoom_start=15)
Big_Sur_LS_map.add_ee_layer(Big_Sur_LS, {'min': 0, 'max': 1, 'palette': ['black', 'red']}, 'land surface change')

# Add layer control
Big_Sur_LS_map.add_child(folium.LayerControl())

# Display the map
Big_Sur_LS_map


# ## Summary and results
# <p>In the preliminary phase of the project, land changes due to landslide of Caddy Lake, Manitoba, Canada, on June 25, 2016 using SAR (Sentinel-1) satellite images has been detected. Accordingly, appropriate times series of SAR images within a time window of six months before and after the event date are processed to pinpoint pre- and post- land surface changes due to landslide. Unlike a large landslide with a significant mass movement, the reported land surface changes were not the result of a major shift. Instead, they involved smaller slope movements scattered across an area of about 5 square kilometers.</p>
# <p>To further test our methods, we also looked into the Big Sur landslide that took place on May 20, 2017. This event involved a substantial mass movement and we used a similar approach to our previous analysis. We were able to accurately detect downward slope movement in this case. However, there is a twist, since both of these locations are close to large bodies of water. This led to water areas being included within our analysis and defined as land changes. To address this, we tried a different tactic. Instead of directly applying a water mask to SAR images (which we found it quite difficult following multiple attempts), we applied the water mask to multispectral images (Sentinel-2). Then, by combining the various steps we have developed a method to distinguish between water areas and changes in the land surface effectively.</p>
# <p>Our study underscores the remarkable and substantial capability of the synergistic integration of Synthetic Aperture Radar (SAR) and multispectral imagery in landslide study. Satellite images from Sentinel-1 SAR are really handy to spot changes on the ground, no matter how big or small the area is. When we use Sentinel-2 images, they help us tell apart water and land changes. By putting together SAR and multispectral images, we have a strong way to look into land changes while leaving out water bodies. This combo is a great tool to keep an eye on land shifts, like possible landslides. This capability besides enhancing our understanding of land alterations opens doors to broader applications in fields such as disaster management, urban planning, agricultural monitoring, and ecological assessments.</p>

# In[ ]:




