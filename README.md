# Landslide detection using SAR (Sentinel-1) and Multispectral (Sentinel-2) images

Landslides can cause significant damages to society, destroy ecosystem, destruct environment and economical losses (eg. Damages to infrastructure, agriculture, water resource, people health, etc). In this study we investigated two landslides, namely Caddy Lake, Canada and Big Sur, USA. 

![Caddy Lake landslide, June 2016](https://i.cbc.ca/1.3654072.1467042154!/fileImage/httpImage/image.jpg_gen/derivatives/original_1180/grant-fisette-s-home.jpg)
Recarious situation of a home after the weekend downpour around Caddy Lake, Manitoba, Canada, June 2016 ©cbc.ca News

![Big Dur landslide, May 2015](https://static01.nyt.com/images/2017/05/25/us/25sur-xp/25sur-xp-superJumbo.jpg?quality=75&auto=webp)
Big Sur landslide, Highway 1, California, USA, May 2015 ©The New York Times
 
## DOI
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.8248086.svg)](https://doi.org/10.5281/zenodo.8248086)

## Motivations and goals:
- What are the project motivations?
According to NASA Global Landslide Catalog, landslides are a serious geological hazard to nearly all the States in the United States. In addition, based on USGS, landslides will annually remain approximately 25-50 victims across the country. Though, landslide exposure assessment and understanding the nature of neighborhood to live in it is important to mitigate catastrophe. Preliminary monitoring can provide vital information on how much an area is prone to landslide, which accordingly is essential for emergency response, and catastrophe mitigation in the areas vulnerable to landslides.

- What are the goals?
In the preliminary phase, land changes due to landslide of Caddy Lake, Manitoba, Canada, on June 25, 2016 using SAR (Sentinel-1) satellite images has been detected. Accordingly, appropriate times series of SAR images are processed to pinpoint land surface changes before and after the landslide event. Unlike a large landslide with a significant mass movement, the reported land surface changes were not the result of a major shift. Instead, they involved smaller slope movements scattered across an area of about 5 square kilometers. To further test our methods, we also looked into the Big Sur landslide that took place on May 20, 2017. This event involved a substantial mass movement and we used a similar approach to our previous analysis. We were able to accurately detect downward slope movement in this case.
However, there is a twist, since both of these locations are close to large bodies of water. This led to water areas being included within our analysis and defined as land changes. To address this, we tried a different tactic. Instead of directly applying a water mask to SAR images (which we found it quite difficult following multiple tries), we applied the water mask to multispectral images (Sentinel-2). Then, by combining the various steps we have developed and using a subtraction method, we managed to distinguish between water areas and changes in the land surface effectively. 

## Environment Requirements
How to install your environment?
* [Start with instructions for installing the earth-analytics-python] (https://www.earthdatascience.org/workshops/setup-earth-analytics-python/)
* Access to Goole Earth Engine account
* environment.yml (provided in repository)

## Include **code** for installing any additional packages
%%bash
pip install geemap

## Data access 
1. [Copernicus_S1_GRD](https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_S1_GRD) and [Copernicus_S2](https://developers.google.com/earth-engine/datasets/catalog/sentinel-2) by European Space Agency(ESA)
2. USGS NAIP Imagery NDVI (GEE basemap layer)
3. USGS NAIP Imagery False Color
4. Verified landslides of North America collected by Earth Lab at the University of Colorado Boulder (Main source: [NASA Global Landslide Catalog (GLC)] Link: https://data.nasa.gov/Earth-Science/Global-Landslide-Catalog/h9d8-neg4 

## Running the workflow
The python code works by activating the earth-analytics environment in terminal typing conda activate earth-analytics. Then, open a Jupyter notebook browser typing jupyter notebook in terminal. Navigate to the Landslide_Detection_SAR_MSI_Nasim_Mozafari.ipynb [here](https://github.com/nasimmozafari/Landslide_Detection_SAR). Run the ipynb file from the Kernel tab selecting Restart & Run All. The code contains some EarthExplorer pre-written syntax.

Project Contacts
Nasim.MozafariAmiri@colorado.edu

Acknowledgements
This project was inspired by Dr. Elsa Culler, CU Boulder Earth Lab

License:
This project is open source and available under the MIT License.
