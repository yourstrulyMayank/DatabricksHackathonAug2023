# Databricks notebook source
# MAGIC %pip install --upgrade pip

# COMMAND ----------

#!pip install newspaper3
#!pip install -e https://github.com/codelucas/newspaper/archive/master.zip
%pip install -e git+https://github.com/codelucas/newspaper.git@master#egg=newspaper3k

# COMMAND ----------

# MAGIC %pip install beautifulsoup4

# COMMAND ----------

# MAGIC %pip install nltk

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %sql
# MAGIC --drop table infosys.kamikaze_rtsa.silver_table_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC /*
# MAGIC CREATE TABLE infosys.kamikaze_rtsa.silver_table_processed (
# MAGIC   SNo integer,
# MAGIC   batch integer,
# MAGIC   Timestamp timestamp,
# MAGIC   url string,
# MAGIC   article_title string,
# MAGIC   article_text string,
# MAGIC   transformed_text string
# MAGIC )USING DELTA
# MAGIC --location '/FileStore/tables/silver_table';
# MAGIC */

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, IntegerType, StringType, TimestampType, ShortType, DataType
from newspaper import Article
import nltk
import re
nltk.download('punkt')

# COMMAND ----------

spark = SparkSession.builder.appName("RTSA_app").getOrCreate()

# COMMAND ----------

def getArticle(url):
    try:
        processed = 0
        article = Article(url)
        article.download()
        ##Article was downloaded
        if article.download_state == 2:
            processed = 1
            article.parse()
    except article.download_exception_msg as e:
        if article.download_state==1:
            print(article.download_exception_msg)
        else:
            raise e
    return(article.title,article.text,processed,article.download_state)

# COMMAND ----------

def insert_into_silver():
    out = spark.sql("""
                    INSERT INTO infosys.kamikaze_rtsa.silver_table_processed
                    SELECT distinct SNo, batch, CURRENT_TIMESTAMP() AS Timestamp, urls, title, text, clean_txt
                    FROM   temp_silver
                    WHERE  title != '' 
                    AND    text  != '') 
                    """)
    for row in out.collect():
        if(row[0])>0 : 
            print("Successfully inserted {} records".format(row[0]))
    return(out)

# COMMAND ----------

def update_bronze():
    out = spark.sql("""
                    UPDATE infosys.kamikaze_rtsa.bronze_table_raw t1
                    SET processed = (SELECT max(processed) FROM temp_bronze WHERE urls = t1.url),
                        status = (SELECT max(status) FROM temp_bronze WHERE urls = t1.url)
                    WHERE EXISTS (SELECT 1 
                                FROM  temp_bronze t2
                                WHERE t2.urls = t1.url)
                    """)
    return(out)

# COMMAND ----------

def update_msg(out):
    for row in out.collect():
        if(row[0])>0 : 
            print("Successfully updated {} records".format(row[0]))
        else:
            print("No updates performed")

# COMMAND ----------

def remove_url(input_text):
    pattern=r'(?i)\b((?:[a-z][\w-]+:(?:/{1,3}|[a-z0-9%])|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’]))';
    match = re.findall(pattern, input_text)
    for m in match:
        url = m[0]
        input_text = input_text.replace(url, '')
    return(input_text)

# COMMAND ----------

## Insert records in to Silver table and update bronze table for processed records
def process_urls(df):
    #bronze_table = spark.sql("""SELECT * FROM infosys.kamikaze_rtsa.bronze_table_raw""")

    #df = bronze_table.select("url").filter((bronze_table["processed"] == 0))
    #df = df.limit(50)

    urls = [row[0] for row in df.collect()]
    title, text, clean_txt, processed, status = [], [], [], [], []
    for url in urls:
        tit, txt, procd,stat  = getArticle(url)
        title.append(tit)
        text.append(txt)
        clean_txt.append(remove_url(txt))
        processed.append(procd)
        status.append(stat)
    data = list(zip(urls, title, text, clean_txt))
    data1 = list(zip(urls, processed, status))
    df = spark.createDataFrame(data, ["urls", "title", "text", "clean_txt"])
    df1 = spark.createDataFrame(data1, ["urls", "processed", "status"])
    
    current_max_batch = spark.sql("""SELECT MAX(batch) AS max_batch FROM infosys.kamikaze_rtsa.silver_table_processed""").first().max_batch
    next_batch = current_max_batch + 1 if current_max_batch is not None else 1
    df = df.withColumns({"SNo"  : F.monotonically_increasing_id()+1, 
                        "batch": F.lit(next_batch)})
    df = df.withColumn("SNo",df.SNo.cast('integer'))

    df.createOrReplaceTempView('temp_silver')
    df1.createOrReplaceTempView('temp_bronze')
    
    out = insert_into_silver()
    #update_msg(out)
    out = update_bronze()
    update_msg(out)

# COMMAND ----------

def load_data():
    bronze_table = spark.sql("""SELECT * FROM infosys.kamikaze_rtsa.bronze_table_raw""")
    df = bronze_table.select("url").filter((bronze_table["processed"] == 0) & ((bronze_table["status"] == 0)))
    chunks = chunkify(df,500)
    return(chunks)

# COMMAND ----------

def chunkify(df: F.DataFrame, chunk_size: int):
    start = 0
    length = df.count()

    # If DF is smaller than the chunk, return the DF
    if length <= chunk_size:
        yield df
        return

    # Yield individual chunks
    while start + chunk_size <= length:
        yield df.take(chunk_size)
        start = start + chunk_size

    # Yield the remainder chunk, if needed
    if start < length:
        yield df.take(length - start)

# COMMAND ----------

def main():
    spark = SparkSession.builder.appName("RTSA_app").getOrCreate()
    chunks = load_data()
    for chunk in chunks:
        df = spark.createDataFrame(data=chunk)
        process_urls(df)

# COMMAND ----------

if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %sql
# MAGIC --select count(*) from infosys.kamikaze_rtsa.silver_table_processed

# COMMAND ----------

# MAGIC %sql
# MAGIC /*
# MAGIC select processed, status, count(1) 
# MAGIC from infosys.kamikaze_rtsa.bronze_table_raw
# MAGIC group by processed, status
# MAGIC */
# MAGIC

# COMMAND ----------

# MAGIC %sql
# MAGIC --select * from bronze_table where processed == 1
# MAGIC
