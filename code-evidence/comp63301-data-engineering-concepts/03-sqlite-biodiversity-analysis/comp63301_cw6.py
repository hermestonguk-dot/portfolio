#!/usr/bin/env python
# coding: utf-8

# In this assignment we'll using a real dataset from the [UK Government data portal](https://www.data.gov.uk/).
# 
# You should do this exercise by augmenting this Jupyter Notebook, either using Google Colab or a Conda-installed Jupyter as shown in the workshop.
# 
# Do not remove code in this notebook that writes out the intermediate results to files, as these are used by the autograder. Do test that running the notebook works from top cell to bottom cell.
# 
# 
# ## Dataset
# 
# The dataset [Sites of Biological Importance (SBI), also known as LWS, in Greater Manchester](https://www.data.gov.uk/dataset/81cbf1a0-6304-470c-ade8-60272be0d219/sites-of-biological-importance-sbi-in-greater-manchester) by the Greater Manchester Ecology Unit lists geographical nature areas that are important to protect for biodiversity.
# 
# This dataset is provided as three elements:
# 
# * [Excel spreadsheet with tabular listing of sites](https://gmlrc.org/resources/downloads/gmeu_sbi_tabular.xlsx)
# * [Interactive map](https://gmlrc.org/mapapp/?basemap=terrain)
# * [GIS files (ZIP)](https://gmlrc.org/resources/downloads/sbi_gis_resources.zip) -- map data underlying above
# * [Selection guidelines (PDF)](https://gmlrc.org/resources/downloads/sbi_selection_guidelines.pdf)
# 
# _More details are available on the [Greater Manchester Record Centre](https://gmlrc.org/)_
# 
# 

# ## Ingestion
# 
# In this exercise we're not focusing on ingestion, so we'll quickly download the Excel file as-is using the `curl` command:
# 
# If you are running this Notebook on Windows you probably don't have the `curl` command, in which case you need to download <https://gmlrc.org/resources/downloads/gmeu_sbi_tabular.xlsx> manually and save it to the same folder as this notebook.

# In[ ]:


import os
os.system("curl -L -o gmeu_sbi_tabular.xlsx https://gmlrc.org/resources/downloads/gmeu_sbi_tabular.xlsx")


# You should use use Python and [pandas](https://pandas.pydata.org/) to load and analyse the tabular data, see [chapter 6 in Python for Data Analysis](https://wesmckinney.com/book/accessing-data) for details.
# 
# We'll also need the `openpyxl` module to load Excel files in Pandas

# In[ ]:


os.system("pip install pandas openpyxl")


# We can use the pandas `read_excel` function to import the spreadsheet as a DataFrame. Note that in most cases you will also need to provide a second parameter with the sheet name. If we inspect manually in Excel, this spreadsheet only has a single sheet _Sheet1_.

# In[2]:


import pandas as pd
gmeu = pd.read_excel("gmeu_sbi_tabular.xlsx", "Sheet1")


# Import the table into SQLite for further processing as SQL.

# In[3]:


import sqlite3
import pathlib

dbfile = pathlib.Path("gemu.db")
if dbfile.exists():
     dbfile.unlink()

db = sqlite3.connect("gmeu.db")
gmeu.to_sql("Sites", db, if_exists='replace', index=False)


# For the rest of the exercise, use the Pandas SQL functions and modify `db`. We can check what has been imported:

# In[4]:


pd.read_sql_query("SELECT * FROM Sites", db)


# ## Question 1
# 
# Which preservation sites were recognised in which districts **after** year 2000? (e.g. from 2001 onwards)
# 
# Write a filtering query that return two columns `district` and `site_name`, sorted alphabetically on `district`, then `site_name`.
# 
# | district |	site_name |
# | -- | -- |
# | Bolton |	Seven Acres |
# |	Bolton |	Smith Road Reservoirs & Raikes Clough |
# | ..  |
# 
# Make sure the outputs are written to the file `districts.csv`

# In[5]:


districts = pd.read_sql_query("""
SELECT district, site_name
FROM Sites
WHERE Date(date_est) > '2000-12-31'
ORDER BY district, site_name
""", db)
districts.to_csv("districts.csv")
districts


# # Question 2
# 
# How many sites and total of hectars (`area_ha`) are recognised in each district? What are their statistical distributions?
# 
# Write an aggregate query that return these named aggregate columns for each district. Order by number of sites descending:
# 
# |    | district | sites | total | smallest | biggest | average |
# | -- | -------- | ----- | ----- | -------- | ------- | ------- |
# |    | Bolton  | 67 | 1189.78 | 0.60 | 436.43 | 17.757910 |
# | ... |
# 

# In[8]:


hectars = pd.read_sql_query("""
SELECT district, 
COUNT(site_name) AS sites, 
SUM(area_ha) AS total, 
MIN(area_ha) AS smallest,
MAX(area_ha) AS biggest, 
AVG(area_ha) AS average
FROM Sites
GROUP BY district
ORDER BY COUNT(site_name) DESC
""", db)
hectars.to_csv("hectars.csv")
hectars


# # Question 3a
# 
# The table `Sites` is not normalised, so it is hard to summarise how many hectars that are preserved to protect which feature, as the column `features` has freetext strings, with coding like `Br2` embedded.
# 
# Extract out a list of named features and a list of feature code, per site. You should assume that the codes follow the pattern `Ab1` where `A` is any Upper-case letter, `b` is any letter, and `1` is any single digit. For consistency, convert each extracted code to uppercase.
# 
# **Note**: The data do include some miscoding e.g. `Wd`, do **not** perform data cleaning, ignore such values and ensure only codes matching above pattern is included.

# In[6]:


features = pd.read_sql_query("""
SELECT site_id, features
FROM Sites
""", db)
features


# Augment the above pandas `features` table to add the parsed category and codes and write it to CSV. Implement each of the functions `feature_cat` and `feature_code` for the data tranformations.
# 
# The table in SQL should be flattened with possibly multiple codes per `site_id` (or none), e.g.:
# 
# | site_id  |	feature_code |
# | -------- | ------------- |
# | D28 |	GR2 |
# | A2  |	WD1 |
# |	A2 	| FW3 |

# In[7]:


import re
def feature_cat(feature: str):
  return [m.lstrip(" ,:").strip() for m in re.findall(r"\s*([^();]+?)\s*\(", feature)]
                                    

def feature_code(feature: str):
  return [m.strip().upper() for m in re.findall(r'\b([A-Z][a-zA-Z]\d)\b', feature)]
    
# Remember to test the functions with a good example
feature = "Reedbed (Fw1); Birds (Br3, Br4, Br6)"
print( feature_cat(feature))
print( feature_code(feature))
assert feature_cat(feature) == ["Reedbed", "Birds"]
assert feature_code(feature) == ["FW1", "BR3", "BR4", "BR6"]


# Use the functions to add two new panda columns
features["feature_cat"] = features["features"].apply(feature_cat) ## Replace with YOUR CODE HERE
features["feature_code"] = features["features"].apply(feature_code) ## Replace with YOUR CODE HERE


features.to_csv("features.csv")
features


# From your augmented `features` table, create a new SQL table `SiteFeatureCodes` of all the feature codes per site, with columns `site_id` and `feature_code`. You may use Pandas functions like `DataFrame.explode` and `to_sql`, or regular Python `for` loops etc.
# 
# The table in SQL should be flattened with possibly multiple rows per `site_id` (or no rows if that site had no codes, don't add `NULL` as this is a feature table), e.g.:
# 
# | site_id  |	feature_code |
# | -------- | ------------- |
# | D28 |	GR2 |
# | A2  |	WD1 |
# |	A2 	| FW3 |

# In[40]:


# Save to a new SQL table
sfc = features[["site_id", "feature_code"]].explode("feature_code")

sfc = sfc.dropna(subset=["feature_code"])

sfc.to_sql("SiteFeatureCodes", db, if_exists="replace", index=False)

# Display the new table
fc = pd.read_sql_query("SELECT * FROM SiteFeatureCodes", db)
fc.to_csv("featurecodes.csv")
fc


# Similarly, create a `SiteFeatures` table of the free-text feature categories for
# each site.
# 
# | site_id | feature_cat |
# | ------- | ----------- |
# |	D28 | Grassland |
# | A2 | Woodland |
# | A2 | Open Water |

# In[50]:


## YOUR CODE HERE
sfc = features[["site_id", "feature_cat"]].explode("feature_cat")

sfc = sfc.dropna(subset=["feature_cat"])

sfc.to_sql("SiteFeatures", db, if_exists="replace", index=False)


# Save and display the new table
sitefeatures = pd.read_sql_query("SELECT * FROM SiteFeatures", db)
sitefeatures.to_csv("sitefeatures.csv")
sitefeatures


# # Question 3b
# 
# Assuming the `SiteFeatureCodes` table from above joined with the `Sites` table, what is the total area protected for each feature code, and when was it first recognised?
# 
# With your query, return a table with columns `feature_code`, `total`, `earliest` ordered by `feature_code`.
# 

# In[10]:


protected = pd.read_sql_query("""
SELECT sfc.feature_code,
    SUM(s.area_ha) AS total,
    DATE(MIN(s.date_est)) AS earliest
    FROM SiteFeatureCodes AS sfc
    JOIN Sites As s ON sfc.site_id = s.site_id
    GROUP BY sfc.feature_code
    ORDER BY sfc.feature_code
""", db)
protected.to_csv("protected.csv")
protected


# # Question 3c
# 
# Assuming the `SiteFeatures`, `SiteFeatureCodes` and `Sites` tables, which features describe which feature code across how many sites? How many sites per feature?
# 
# Output should be shaped as:
# 
# | feature_code |	feature_cat    | sites |
# | ------------ | --------------- | ------|
# | None         | Woodland 	     |  1    |
# | AI1          | Amphbiians      |	1    |
# | AI1          | Amphibians      |  7    |
# 
# **Note**: there may be more than one feature per code, and some features without code, as the data is not fully cleaned.

# In[18]:


features_per_code = pd.read_sql_query("""
SELECT
  sfc.feature_code AS feature_code,
  sft.feature_cat,
  COUNT(DISTINCT sft.site_id) AS sites
FROM SiteFeatures AS sft
JOIN SiteFeatureCodes AS sfc
  ON sfc.site_id = sft.site_id
GROUP BY
  sfc.feature_code,
  sft.feature_cat
""", db)

features_per_code.to_csv("features_per_code.csv")
features_per_code




# 
# # Question 4
# 
# Create a new View `BirdSites` of sites larger than 100 hectars, that contain birds with a correct code of `BR1`, `BR2`, ..., `BR7` (ignoring case, but not permitting miscoding like `BR`); do not include  `BR5` unless combined with other bird codes.
# 
# The view should be structured with columns as in:
# 
# | site_id | district | site_name                    | area_ha | date_est |
# | ------- |--------- |----------------------------- | ------- | -------- |
# |	G41     | Oldham   | Dark Peak Moors              | 2378.77 | 2000-04-07 00:00:00  |
# | D47     | Rochdale | South Pennine Moors (North)  | 1323.28 | 2000-05-31 00:00:00  |
# | ... |
# 
# The order returned does not matter, as a view should generally not add ordering constraints for performance reasons.

# In[76]:


db.execute("DROP VIEW IF EXISTS BirdSites")

db.execute("""
CREATE VIEW BirdSites AS
SELECT
  s.site_id,
  s.district,
  s.site_name,
  s.area_ha,
  s.date_est
FROM Sites AS s
WHERE s.area_ha >= 100
  AND EXISTS (
    SELECT 1
    FROM SiteFeatureCodes AS sfc
    WHERE sfc.site_id = s.site_id
      AND UPPER(sfc.feature_code) IN ('BR1','BR2','BR3','BR4','BR6','BR7')
  );
""")

birdsites = pd.read_sql_query("""
SELECT * FROM BirdSites
""", db)
birdsites.to_csv("birdsites.csv")
birdsites


# Now query the view to find the name of oldest and newest large bird site. Return a row with columns as shown below:
# 
# | oldest | oldest_since | newest | newest_since |
# | -- | -- | -- | -- |
# | Old Tom | 1978-01-01 | New Bury | 2022-12-24 |

# In[77]:


old_new = pd.read_sql_query("""
SELECT
  (SELECT site_name
   FROM BirdSites
   WHERE date_est = (SELECT MIN(date_est) FROM BirdSites)
   LIMIT 1) AS oldest,
  (SELECT DATE(MIN(date_est)) FROM BirdSites) AS oldest_since,
  (SELECT site_name
   FROM BirdSites
   WHERE date_est = (SELECT MAX(date_est) FROM BirdSites)
   LIMIT 1) AS newest,
  (SELECT DATE(MAX(date_est)) FROM BirdSites) AS newest_since
""", db)

old_new.to_csv("old_new.csv")
old_new



# # Submitting
# 
# Finally, to submit this notebook to the autograder, use **File** -> **Download** -> **Download.py** to download the Python `.py` version (not the `.ipynb` version)
