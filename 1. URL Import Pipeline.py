# Databricks notebook source
# MAGIC %md
# MAGIC # Import New URLs And Store Them In The Bronze Table

# COMMAND ----------

import requests
import zipfile
import io
import pandas as pd
import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, monotonically_increasing_id, lit

# Step 1: Get the URL of the latest GKG CSV zip file from the provided link
response = requests.get("http://data.gdeltproject.org/gdeltv2/lastupdate.txt")
latest_gkg_zip_url = None
for line in response.text.split("\n"):
    if line.endswith(".gkg.csv.zip"):
        latest_gkg_zip_url = line.split()[-1]
        break

# Step 2: Download the zip file
zip_response = requests.get(latest_gkg_zip_url)
zip_file = zipfile.ZipFile(io.BytesIO(zip_response.content))

# Step 3: Extract the CSV file from the zip
csv_file_name = zip_file.namelist()[0]  # Since there is only one CSV file in the zip
csv_file = zip_file.read(csv_file_name)

# Step 4: Convert the CSV content to a DataFrame
df = pd.read_csv(io.BytesIO(csv_file), delimiter='\t', header=None)

urls = df[4]

url_df = spark.createDataFrame(urls.to_frame(name='url'))
url_df.createOrReplaceTempView("temp_table")

current_max_batch = spark.sql("SELECT MAX(batch) AS max_batch FROM infosys.kamikaze_rtsa.bronze_table_raw").first().max_batch
next_batch = current_max_batch + 1 if current_max_batch is not None else 1
df_spark_with_sno = url_df.withColumn("SNo", monotonically_increasing_id()+1)
df_spark_with_sno.withColumn("batch", lit(next_batch)).createOrReplaceTempView("temp")

spark.sql("""
INSERT INTO infosys.kamikaze_rtsa.bronze_table_raw
SELECT 
  SNo, batch, CURRENT_TIMESTAMP() AS Timestamp, url, 0, 0
FROM temp
""")

# COMMAND ----------

# s3bucket = s3a://workspaces-enb-us-west-2/oregon-prod/7846089482645/FileStore/tables/

# COMMAND ----------

# MAGIC %sql
# MAGIC --drop table infosys.kamikaze_rtsa.bronze_table_raw

# COMMAND ----------

#%fs rm -r /FileStore/tables/bronze_table

# COMMAND ----------



# COMMAND ----------

# MAGIC %sql 
# MAGIC /*
# MAGIC CREATE TABLE infosys.kamikaze_rtsa.bronze_table_raw(
# MAGIC     SNo BIGINT,
# MAGIC     batch BIGINT,
# MAGIC     Timestamp TIMESTAMP,
# MAGIC     url STRING,
# MAGIC     processed SMALLINT, --0 implies article is yet to processed; 1 implies processed
# MAGIC     status  SMALLINT -- 0 is default; 1 is not downloaded ; 2 downloaded
# MAGIC   )USING DELTA
# MAGIC */

# COMMAND ----------

# MAGIC %sql
# MAGIC /*
# MAGIC CREATE EXTERNAL TABLE bronze_table (
# MAGIC     SNo INT,
# MAGIC     batch INT,
# MAGIC     Timestamp TIMESTAMP,
# MAGIC     url STRING,
# MAGIC     processed SMALLINT, --0 implies article is yet to processed; 1 implies processed
# MAGIC     status  SMALLINT -- 0 is default; 1 is not downloaded ; 2 downloaded
# MAGIC   )USING DELTA
# MAGIC location '/FileStore/tables/bronze_table';
# MAGIC */
# MAGIC

# COMMAND ----------

#spark.sql("OPTIMIZE bronze_table")

# COMMAND ----------

# MAGIC %sql
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC /*
# MAGIC select processed, status, count(1) 
# MAGIC from infosys.kamikaze_rtsa.bronze_table_raw
# MAGIC group by processed, status
# MAGIC */

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from infosys.kamikaze_rtsa.silver_table_processed 
# MAGIC where article_text  == '' and article_title != ''
# MAGIC --where processed==0 and status==1
# MAGIC  
