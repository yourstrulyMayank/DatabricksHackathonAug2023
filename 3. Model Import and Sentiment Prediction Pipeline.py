# Databricks notebook source
# MAGIC %sql
# MAGIC   -- drop table infosys.kamikaze_rtsa.gold_table_streaming

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC --  CREATE TABLE infosys.kamikaze_rtsa.gold_table_partitioned (
# MAGIC --    id STRING,
# MAGIC --    url STRING,
# MAGIC --    article STRING,
# MAGIC --   sentiment STRING,
# MAGIC --   emotion STRING
# MAGIC -- )
# MAGIC -- USING DELTA
# MAGIC -- partitioned by (batch int);
# MAGIC
# MAGIC  CREATE OR REPLACE TABLE infosys.kamikaze_rtsa.gold_table_streaming (
# MAGIC    id STRING,
# MAGIC    url STRING,
# MAGIC    article STRING,
# MAGIC   sentiment STRING,
# MAGIC   emotion STRING
# MAGIC )
# MAGIC USING DELTA
# MAGIC partitioned by (batch int);

# COMMAND ----------

# MAGIC %pip install transformers torch xformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

"""Function Definitions and Import"""
from pyspark.sql import SparkSession, Row
from pyspark.sql.functions import col, md5, udf, lit, monotonically_increasing_id, expr, pandas_udf
import pyspark.sql.functions as F
from pyspark.sql.types import StringType
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import torch
from scipy import stats
import numpy as np
import warnings
import pandas as pd
spark = SparkSession.builder.appName("SentimentEmotionPipeline").getOrCreate()
spark.conf.set("spark.databricks.delta.autoOptimize.enabled", "true")
warnings.filterwarnings("ignore")

# Load sentiment analysis model and tokenizer
sentiment_tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
sentiment_model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")

# Load emotion analysis model and tokenizer
emotion_tokenizer = AutoTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")
emotion_model = AutoModelForSequenceClassification.from_pretrained("SamLowe/roberta-base-go_emotions")


def get_sentiment(text, tokenizer=sentiment_tokenizer, model=sentiment_model):
    chunk_size = 512

    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)

    # Split the tokens into chunks
    num_chunks = (len(tokens['input_ids'][0]) - 1) // chunk_size + 1
    sentiments = []

    # Process each chunk
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_tokens = {key: value[:, start:end] for key, value in tokens.items()}

        # Perform sentiment analysis on the current chunk
        with torch.no_grad():
            outputs = model(**chunk_tokens)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1)
            sentiments.append(predicted_class.item())

    # Map sentiment IDs to labels
    sentiment_labels = [
        "negative", "neutral", "positive"
    ]

    sentiment_labels = np.array(sentiment_labels)
    mapped_sentiments = sentiment_labels[sentiments]

    # Calculate the mode sentiment
    mode_sentiment = stats.mode(mapped_sentiments)[0][0]

    return str(mode_sentiment)


def get_emotion(text, tokenizer=emotion_tokenizer, model=emotion_model):
    chunk_size = 512

    # Tokenize the text
    tokens = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)

    # Split the tokens into chunks
    num_chunks = (len(tokens['input_ids'][0]) - 1) // chunk_size + 1
    emotions = []

    # Process each chunk
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk_tokens = {key: value[:, start:end] for key, value in tokens.items()}

        # Perform sentiment analysis on the current chunk
        with torch.no_grad():
            outputs = model(**chunk_tokens)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=1)
            emotions.append(predicted_class.item())

    # Map sentiment IDs to labels
    emotion_labels = [
        "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
        "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
        "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
        "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
    ]

    emotion_labels = np.array(emotion_labels)
    mapped_emotions = emotion_labels[emotions]

    # Calculate the mode sentiment
    mode_emotion = stats.mode(mapped_emotions)[0][0]
    return str(mode_emotion)

# COMMAND ----------

""" Partioned, Un-Batched Original Pipeline (Slow Prediction, works with Fast-Preferred Cluster)"""
silver_table_df = spark.sql("SELECT batch, url, transformed_text as article from infosys.kamikaze_rtsa.silver_table_processed \
                            --where batch = (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed) \
                            ")\
        .withColumn("id", md5(col("url")))\
        .select(*['id','batch','url','article'])
sentiments = []
emotions = []

for row in silver_table_df.collect():
    sentiment_prediction = get_sentiment(row["article"])
    emotion_prediction = get_emotion(row["article"])
    sentiments.append(sentiment_prediction)
    emotions.append(emotion_prediction)    

predictions_df = spark.createDataFrame(zip(sentiments, emotions), ["sentiment", "emotion"])
# Add a sequential index column to both DataFrames to use for joining
silver_table_df = silver_table_df.withColumn("join_key", lit(1))
predictions_df = predictions_df.withColumn("join_key", lit(1))

# Join the two DataFrames on the temporary join_key column
merged_df = silver_table_df.join(predictions_df, on="join_key").drop("join_key")
merged_df.select("id", "url", "article", "sentiment", "emotion","batch").write.mode("append").format("delta").partitionBy("batch").saveAsTable("infosys.kamikaze_rtsa.gold_table")

# COMMAND ----------

""" Partitioned, Batched UDF Pipeline (Slow Write, works with Custom Cluster)"""
get_sentiment_udf = udf(get_sentiment, StringType())
get_emotion_udf = udf(get_emotion, StringType())
silver_table_df = spark.sql("SELECT batch, url, transformed_text as article from infosys.kamikaze_rtsa.silver_table_processed \
                             where batch = (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed) \
                             ")\
        .withColumn("id", md5(col("url")))\
        .select(*['id','batch','url','article'])
unique_batches = silver_table_df.select("batch").distinct().collect()
for batch_row in unique_batches:    
    batch_number = batch_row["batch"]    
    batch_df = silver_table_df.filter(col("batch") == batch_number)
    print(f"Executing Batch {batch_number} with {batch_df.count()} rows")
    # Process sentiment and emotion
    batch_df = batch_df.withColumn("sentiment", lit(get_sentiment_udf(batch_df.article)))
    batch_df = batch_df.withColumn("emotion", lit(get_emotion_udf(batch_df.article)))    
    
    
    # Write the processed data to the gold table
    batch_df.select("id", "url", "article", "sentiment", "emotion","batch").write.mode("append").format("delta").partitionBy("batch").saveAsTable("infosys.kamikaze_rtsa.gold_table_partitioned")

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

# """Batched Original Pipeline (Slow Prediction, works with Fast-Preferred Cluster)"""
# """ Faulty, Creates Duplicates """
# silver_table_df = spark.sql("SELECT batch, url, transformed_text as article from infosys.kamikaze_rtsa.silver_table_processed \
#                             --where batch = (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed) \
#                             ")\
#         .withColumn("id", md5(col("url")))\
#         .select(*['id','batch','url','article'])
# unique_batches = silver_table_df.select("batch").distinct().collect()
# for batch_row in unique_batches:
#     batch_number = batch_row["batch"]
#     print(f"Executing Batch {batch_number}")
#     batch_df = silver_table_df.filter(col("batch") == batch_number)
#     sentiments = []
#     emotions = []

#     for row in batch_df.collect():
#         sentiment_prediction = get_sentiment(row["article"])
#         emotion_prediction = get_emotion(row["article"])
#         sentiments.append(sentiment_prediction)
#         emotions.append(emotion_prediction)    

#     predictions_df = spark.createDataFrame(zip(sentiments, emotions), ["sentiment", "emotion"])
#     # Add a sequential index column to both DataFrames to use for joining
#     batch_df = batch_df.withColumn("join_key", lit(1))
#     predictions_df = predictions_df.withColumn("join_key", lit(1))

#     # Join the two DataFrames on the temporary join_key column
#     merged_df = batch_df.join(predictions_df, on="join_key").drop("join_key")
#     merged_df.select("id", "url", "article", "sentiment", "emotion").write.mode("append").format("delta").saveAsTable("infosys.kamikaze_rtsa.gold_table")

# COMMAND ----------

""" Streaming Pipeline (Requires UDF support)"""
"""POC"""
from pyspark.sql.functions import lit, udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import md5, col
from pyspark.sql import SparkSession

# Create a Spark session
spark = SparkSession.builder \
    .appName("StreamingUDFPipelineWithProgress") \
    .getOrCreate()


# Define UDFs
get_sentiment_udf = udf(get_sentiment, StringType())
get_emotion_udf = udf(get_emotion, StringType())

# Read streaming data using structured streaming
streaming_df = spark.readStream.table("infosys.kamikaze_rtsa.silver_table_processed") \
    .select("batch", "url", "transformed_text") \
    .withColumn("id", md5(col("url"))) \
    .select(*['id', 'batch', 'url', 'transformed_text']).where(col('batch')==74)

# Process streaming data
processed_streaming_df = streaming_df \
    .withColumn("sentiment", get_sentiment_udf(streaming_df["transformed_text"])) \
    .withColumn("emotion", get_emotion_udf(streaming_df["transformed_text"]))

# Write processed data to Delta Lake in append mode
query = processed_streaming_df.writeStream \
    .outputMode("append") \
    .foreachBatch(lambda batch_df, batch_id: batch_df
                  .select("id", "url", "transformed_text", "sentiment", "emotion", "batch")
                  .write
                  .mode("append")
                  .option("mergeSchema", "true")
                  .format("delta").partitionBy("batch")  
                  .saveAsTable("infosys.kamikaze_rtsa.gold_table_streaming")) \
    .start()

# Wait for the query to terminate
query.awaitTermination()
      

# COMMAND ----------

# """ Unpartitioned Batched UDF Pipeline (Slow Write, works with Custom Cluster)"""
# get_sentiment_udf = udf(get_sentiment, StringType())
# get_emotion_udf = udf(get_emotion, StringType())
# silver_table_df = spark.sql("SELECT batch, url, transformed_text as article from infosys.kamikaze_rtsa.silver_table_processed \
#                              --where batch = (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed) \
#                              ")\
#         .withColumn("id", md5(col("url")))\
#         .select(*['id','batch','url','article'])
# unique_batches = silver_table_df.select("batch").distinct().collect()
# for batch_row in unique_batches:    
#     batch_number = batch_row["batch"]    
#     batch_df = silver_table_df.filter(col("batch") == batch_number)
#     print(f"Executing Batch {batch_number} with {batch_df.count()} rows")
#     # Process sentiment and emotion
#     batch_df = batch_df.withColumn("sentiment", lit(get_sentiment_udf(batch_df.article)))
#     batch_df = batch_df.withColumn("emotion", lit(get_emotion_udf(batch_df.article)))    
    
    
#     # Write the processed data to the gold table
#     batch_df.select("id", "url", "article", "sentiment", "emotion").write.mode("append").format("delta").saveAsTable("infosys.kamikaze_rtsa.gold_table")

# COMMAND ----------

# """Chunkify Pipeline (Faulty as it creates a lot of dupicates)"""
# def chunkify(df: F.DataFrame, chunk_size: int):
#     start = 0
#     length = df.count()

#     # If DF is smaller than the chunk, return the DF
#     if length <= chunk_size:
#         yield df
#         return

#     # Yield individual chunks
#     while start + chunk_size <= length:
#         yield df.take(chunk_size)
#         start = start + chunk_size

#     # Yield the remainder chunk, if needed
#     if start < length:
#         yield df.take(length - start)

# def load_data_score():
    
#     urlarticledf = spark.sql("SELECT url, transformed_text as article from infosys.kamikaze_rtsa.silver_table_processed \
#                             --where batch = (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed) \
#                             ")\
#         .withColumn("id", md5(col("url")))\
#         .select(*['id','url','article'])
#     chunks = chunkify(urlarticledf,500)
#     return(chunks)

# def score_emotion_sentiment(urlarticledf):
#     """Original Pipeline"""
#     sentiments = []
#     emotions = []

#     for row in urlarticledf.collect():
#         sentiment_prediction = get_sentiment(row["article"])
#         emotion_prediction = get_emotion(row["article"])
#         sentiments.append(sentiment_prediction)
#         emotions.append(emotion_prediction)    

#     predictions_df = spark.createDataFrame(zip(sentiments, emotions), ["sentiment", "emotion"])
#     # Add a sequential index column to both DataFrames to use for joining
#     urlarticledf = urlarticledf.withColumn("join_key", lit(1))
#     predictions_df = predictions_df.withColumn("join_key", lit(1))

#     # Join the two DataFrames on the temporary join_key column
#     merged_df = urlarticledf.join(predictions_df, on="join_key").drop("join_key")
#     merged_df.write.mode("append").format("delta").saveAsTable("infosys.kamikaze_rtsa.gold_table")

# chunks = load_data_score()
# for i,chunk in enumerate(chunks):
#     print(i)
#     df = spark.createDataFrame(data=chunk)
#     score_emotion_sentiment(df)
