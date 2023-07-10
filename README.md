# Landslide detection using SAR (Sentinel-1) images

Landslides can cause significant damages to society, destroy ecosystem, destruct environment and economical losses (eg. Damages to infrastructure, agriculture, water resource, people health, etc).

![Caddy Lake landslide, June 2016](https://i.cbc.ca/1.3654072.1467042154!/fileImage/httpImage/image.jpg_gen/derivatives/original_1180/grant-fisette-s-home.jpg)
## DOI
[![DOI](https://zenodo.org/badge/634037575.svg)](https://zenodo.org/badge/latestdoi/634037575)

## Motivations and goals:
- What are the project motivations?
According to NASA Global Landslide Catalog, landslides are a serious geological hazard to nearly all the States in the United States. In addition, based on USGS, landslides will annually remain approximately 25-50 victims across the country. Though, landslide exposure assessment and understanding the nature of neighborhood to live in it is important to mitigate catastrophe. Preliminary monitoring can provide vital information on how much an area is prone to landslide, which accordingly is essential for emergency response, and catastrophe mitigation in the areas vulnerable to landslides.

- What are the goals?
At the preliminary phase, land changes due to landslide of Caddy Lake, Manitoba, Canada, 2016-06-25 using SAR (Sentinel-1) satellite images has been detected. Accordingly, appropriate times series of SAR images are processed to detect land surface changes before and after the verified landslides. We will attribute the similar code to identify landslides globally. 

## Environment Requirements
How to install your environment?
* [Start with instructions for installing the earth-analytics-python] (https://www.earthdatascience.org/workshops/setup-earth-analytics-python/)
* Access to Goole Earth Engine account

## Include **code** for installing any additional packages
%%bash
pip install geemap

## Data access 
1. [Copernicus_S1_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) by European Space Agency(ESA)
2. USGS NAIP Imagery NDVI (GEE basemap layer)
3. USGS NAIP Imagery False Color
4. Verified landslides of North America collected by Earth Lab at the University of Colorado Boulder (Main source: [NASA Global Landslide Catalog (GLC)] Link: https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog/h9d8-neg4 

## Running the workflow
The python code works by activating the earth-analytics environment in terminal typing conda activate earth-analytics. Then, open a Jupyter notebook browser typing jupyter notebook in terminal. Navigate to the Nasim_Mozafari_Landslides_Project_ColoradoUni.ipynb [here] (https://github.com/nasimmozafari/Landslide_Detection_SAR). Run the ipynb file from the Kernel tab selecting Restart & Run All. The code contains EarthExplorer pre-written syntax

Citations:

Project Contacts
@Nasim Mozafari Amiri

Acknowledgements
This project was inspired by Dr. Elsa Culler, CU Boulder Earth Lab

License:
This project is open source and available under the MIT License.
