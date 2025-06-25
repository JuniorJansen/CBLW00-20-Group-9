# CBLW00-20-Group-9
Firstly we suggest making a conda environment and downloading our requirements.txt file to get all the packages needed to run our code by running the command: 
```bash
pip install -r requirements.txt
```
For running the code first download the data from https://data.police.uk/data/ download all the data from April 2021 onwards and put it in a folder called data, 
to get it in our format. Run data_load.py to create a csv that only contains burglary data from the MPS.

To run our website, use the command:
```bash
streamlit run website/Home.py
```
To download our external datasets, we advise you to download this zip file and put all the files into the data folder.
Using this you can first run main.py. This will allow you to run the main.py file and train the model.
https://drive.google.com/file/d/1Q0jjHlCjyyd9ude2LTwQIAAa5qVoajTf/view?usp=drive_link



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