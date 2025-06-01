# CBLW00-20-Group-9

For running the code first download the data from https://data.police.uk/data/ download all the data from 2010 onwards and put it in a folder called data, 
to get it in our format. Run data_load.py to create a csv that only contains burglary data from the MPS.


To download our external datasets, we advise you to download this zip file and but all the files into the data folder.
Using this you can first run main.py. This will train the model and then save the weights.

After that it's possbile to run prediction.py which will give you the needed csv files to run our website and create the visualisations.



Extra information on other files in the repository
pop_estimates, creates a more useful file than the original to decrease long loading times. Uses the dataset from here:
https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/lowersuperoutputareamidyearpopulationestimates
To then create the population_summary file that is used in our preprocessing.

Filter_wards.py uses the most recent ward info from this dataset:
https://geoportal.statistics.gov.uk/datasets/ons::wards-may-2024-boundaries-uk-bgc-1/explore
To filter it purely on a smaller amount of wards just around London

Shapefiles.py generates a LSOA shapefile for london, by taking all the shapefiles from each london borough in this zip.
https://data.london.gov.uk/download/statistical-gis-boundary-files-london/2a5e50ac-c22e-4d68-89e2-85f1e0ff9057/LB_LSOA2021_shp.zip
If you then put it in a boundaries/london/(files) than it will run the code and generate 1 shapefile called london_lsoa