# Databricks notebook source
# MAGIC %md
# MAGIC # Predict Sentiments And Store Them In Gold Table

# COMMAND ----------

# MAGIC %sql
# MAGIC   -- drop table infosys.kamikaze_rtsa.gold_table

# COMMAND ----------

# MAGIC %sql
# MAGIC
# MAGIC  CREATE OR REPLACE TABLE infosys.kamikaze_rtsa.gold_table_classed (
# MAGIC    id STRING,
# MAGIC    url STRING,
# MAGIC    article STRING,
# MAGIC   sentiment STRING,
# MAGIC   emotion STRING
# MAGIC )
# MAGIC USING DELTA
# MAGIC partitioned by (batch int);
# MAGIC /*
# MAGIC  CREATE OR REPLACE TABLE infosys.kamikaze_rtsa.gold_table_streaming (
# MAGIC    id STRING,
# MAGIC    url STRING,
# MAGIC    article STRING,
# MAGIC   sentiment STRING,
# MAGIC   emotion STRING
# MAGIC )
# MAGIC USING DELTA
# MAGIC partitioned by (batch int);
# MAGIC */

# COMMAND ----------

# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.5.0" transformers torch xformers mlflow torchvision tqdm

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
import torchvision
from scipy import stats
import numpy as np
import warnings
import pandas as pd
import mlflow
mlflow.set_registry_uri("databricks-uc-rtsa")
spark = SparkSession.builder.appName("SentimentEmotionPipeline").getOrCreate()
spark.conf.set("spark.databricks.delta.autoOptimize.enabled", "true")
warnings.filterwarnings("ignore")

# COMMAND ----------

# Load sentiment analysis model and tokenizer
sentiment_architecture = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_architecture)
sentiment_model = AutoModelForSequenceClassification.from_pretrained(sentiment_architecture)

# Load emotion analysis model and tokenizer
emotion_architecture = "SamLowe/roberta-base-go_emotions"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_architecture)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_architecture)


# COMMAND ----------

import shutil
shutil.rmtree('dbfs/kamikaze_rtsa/')

# COMMAND ----------

files = shutil.os.listdir("dbfs/kamikaze_rtsa/emotion")
for file in files:
    print(file)

# COMMAND ----------

### Not correctly registering the model
## path = dbfs/my_project_models/
with mlflow.start_run():
    components = {
        "model":  emotion_model,
        "tokenizer": emotion_tokenizer
    }
    mlflow.transformers.save_model(
        transformers_model=components,
        path="dbfs/kamikaze_rtsa/emotion/emotion_model.v1/"
    )
#emotion_model.config.to_json_file("dbfs/kamikaze_rtsa/emotion/config.json")


# COMMAND ----------

with mlflow.start_run():
    components = {
        "model":  sentiment_model,
        "tokenizer": sentiment_tokenizer
    }
    mlflow.transformers.save_model(
        transformers_model=components,
        path="dbfs/kamikaze_rtsa/sentiment/sentiment_model.v1"
    )
#sentiment_model.config.to_json_file("dbfs/my_project_models/sentiment/config.json")

# COMMAND ----------

##Another method I tried
# Register sentiment analysis model and tokenizer
# with mlflow.start_run() as run:
#     mlflow.pytorch.log_model(sentiment_model, "sentiment_model")
#     sentiment_tokenizer.save_pretrained("sentiment_tokenizer")

# # Register emotion analysis model and tokenizer
# with mlflow.start_run() as run:
#     mlflow.pytorch.log_model(emotion_model, "emotion_model")
#     emotion_tokenizer.save_pretrained("emotion_tokenizer")


# COMMAND ----------

class SentimentAnalyzer:
    def __init__(self, model_name):
        self.model = mlflow.pyfunc.load_model(model_name)
        self.sentiment_labels = ["negative", "neutral", "positive"]
        self.sentiment_labels = np.array(self.sentiment_labels)

    def analyze_sentiment(self, text):
        chunk_size = 512
        sentiment_architecture = "cardiffnlp/twitter-roberta-base-sentiment-latest"
        sentiment_tokenizer = AutoTokenizer.from_pretrained(sentiment_architecture)
        tokens = sentiment_tokenizer(text, return_tensors="pt", padding=True, 
                                     truncation=True, max_length=chunk_size)
        num_chunks = (len(tokens['input_ids'][0]) - 1) // chunk_size + 1
        sentiments = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk_tokens = {key: value[:, start:end] for key, value in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**chunk_tokens)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1)
                sentiments.append(predicted_class.item())

        mapped_sentiments = self.sentiment_labels[sentiments]
        mode_sentiment = stats.mode(mapped_sentiments)[0][0]
        return str(mode_sentiment)

# COMMAND ----------

class EmotionAnalyzer:
    def __init__(self, model_name):
        self.model = mlflow.pyfunc.load_model(model_name)
        self.emotion_labels = [
            "admiration", "amusement", "anger", "annoyance", "approval", "caring", "confusion",
            "curiosity", "desire", "disappointment", "disapproval", "disgust", "embarrassment",
            "excitement", "fear", "gratitude", "grief", "joy", "love", "nervousness", "optimism",
            "pride", "realization", "relief", "remorse", "sadness", "surprise", "neutral"
        ]
        self.emotion_labels = np.array(self.emotion_labels)

    def analyze_emotion(self,text):
        chunk_size = 512
        emotion_architecture = "SamLowe/roberta-base-go_emotions"
        emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_architecture)
        tokens = emotion_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=chunk_size)
        num_chunks = (len(tokens['input_ids'][0]) - 1) // chunk_size + 1
        emotions = []

        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size
            chunk_tokens = {key: value[:, start:end] for key, value in tokens.items()}

            with torch.no_grad():
                outputs = self.model(**chunk_tokens)
                logits = outputs.logits
                predicted_class = torch.argmax(logits, dim=1)
                emotions.append(predicted_class.item())

        mapped_emotions = self.emotion_labels[emotions]
        mode_emotion = stats.mode(mapped_emotions)[0][0]
        return str(mode_emotion)

# COMMAND ----------

emotion_analyzer = mlflow.pyfunc.load_model('dbfs/kamikaze_rtsa/emotion/emotion_model.v1/')
emotion_analyzer.predict('This is bad article')

# COMMAND ----------

sentiment_model = mlflow.pyfunc.load_model("dbfs/kamikaze_rtsa/sentiment/sentiment_model.v1/")
mlflow.pyfunc.save_model(sentiment_model, "my_sentiment_model")
sentiment_analyzer = SentimentAnalyzer(my_sentiment_model)

#emotion_model = mlflow.pyfunc.load_model("dbfs/kamikaze_rtsa/emotion/emotion_model.v1/")


# COMMAND ----------

sentiment_model.predict()

# COMMAND ----------

sentiment_analyzer = SentimentAnalyzer(str(sentiment_model))
emotion_analyzer = EmotionAnalyzer(str(emotion_model))

# COMMAND ----------

from tqdm import tqdm 
def main():
    spark = SparkSession.builder.appName("SentimentEmotionAnalysis").getOrCreate()
    
    sentiment_model = mlflow.pyfunc.load_model("dbfs/kamikaze_rtsa/sentiment/sentiment_model.v1/")
    emotion_model = mlflow.pyfunc.load_model("dbfs/kamikaze_rtsa/emotion/emotion_model.v1/")
    sentiment_analyzer = SentimentAnalyzer(sentiment_model)
    emotion_analyzer = EmotionAnalyzer(emotion_model)
    
    silver_table_df = spark.sql("SELECT batch, url, transformed_text as article from infosys.kamikaze_rtsa.silver_table_processed \
                                 where batch = (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed)")
    silver_table_df = silver_table_df.withColumn("id", md5(col("url"))).select(*['id','batch','url','article'])
    
    unique_batches = silver_table_df.select("batch").distinct().collect()
    
    for batch_row in tqdm(unique_batches):  # Use tqdm for the loop
        batch_number = batch_row["batch"]    
        batch_df = silver_table_df.filter(col("batch") == batch_number)
        print(f"Executing Batch {batch_number} with {batch_df.count()} rows")
        
        sentiments = []
        emotions = []
       
        for row in batch_df.collect():
            sentiment_prediction = sentiment_analyzer.analyze_sentiment(row["article"])
            emotion_prediction = emotion_analyzer.analyze_emotion(row["article"])
            sentiments.append(sentiment_prediction)
            emotions.append(emotion_prediction)
            
        batch_df = batch_df.withColumn("sentiment", lit(sentiments))
        batch_df = batch_df.withColumn("emotion", lit(emotions))
        
        batch_df.select("id", "url", "article", "sentiment", "emotion", "batch") \
            .write.mode("append").format("delta").partitionBy("batch") \
            .saveAsTable("infosys.kamikaze_rtsa.gold_table_classed")
    
    spark.stop()

# COMMAND ----------

if __name__ == "__main__":
    main()

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from infosys.kamikaze_rtsa.gold_table_classed

# COMMAND ----------


