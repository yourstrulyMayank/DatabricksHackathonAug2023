# Databricks notebook source
# MAGIC %pip install --upgrade "mlflow-skinny[databricks]>=2.5.0" tensorflow

# COMMAND ----------

# MAGIC %pip install torchvision

# COMMAND ----------

# MAGIC %pip install transformers torch xformers

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc-rtsa")


# COMMAND ----------

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




# COMMAND ----------

from transformers import pipeline
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification 
import torchvision
# Load sentiment analysis model and tokenizer


# Load emotion analysis model and tokenizer
emotion_architecture = "SamLowe/roberta-base-go_emotions"
emotion_tokenizer = AutoTokenizer.from_pretrained(emotion_architecture)
emotion_model = AutoModelForSequenceClassification.from_pretrained(emotion_architecture)

with mlflow.start_run():
    components = {
        "model":  emotion_model,
        "tokenizer": emotion_tokenizer
    }
    mlflow.transformers.save_model(
        transformers_model=components,
        path="/model/emotion_model.v1"
    )

# COMMAND ----------

def load_emotion_model():
    emotion_architecture = "SamLowe/roberta-base-go_emotions"
    #model = mlflow.load_model("/model/emotion_model.v1")
    loaded = mlflow.pyfunc.load_model("/model/emotion_model.v1")
    tokenizer = AutoTokenizer.from_pretrained(emotion_architecture)
    return model, tokenizer

# COMMAND ----------

text = "This article is of great importance to mankind!"
loaded = mlflow.pyfunc.load_model("/model/emotion_model.v1")
loaded.predict(text)

# COMMAND ----------

def predict_emotion(text):
    model, tokenizer = load_emotion_model()
    prediction = get_emotion(text,model,tokenizer)
    return prediction

# COMMAND ----------

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



# COMMAND ----------

inference = spark.sql("""select transformed_text from infosys.kamikaze_rtsa.silver_table_processed""")
#inference = inference.select("transformed_text").filter((inference["url"] == 'https://kathmandupost.com/food/2023/08/17/a-symphony-of-flavours') )
for row in inference.collect():
    print(predict_emotion(row[0]))

# COMMAND ----------

# MAGIC %sql
# MAGIC select url, transformed_text from infosys.kamikaze_rtsa.silver_table_processed
# MAGIC where  batch in (select max(batch) from infosys.kamikaze_rtsa.silver_table_processed)
# MAGIC and url = 'https://kathmandupost.com/food/2023/08/17/a-symphony-of-flavours'
# MAGIC
