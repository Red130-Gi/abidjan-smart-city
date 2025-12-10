from pyspark.sql import SparkSession
import cassandra
print("Cassandra module imported successfully")
spark = SparkSession.builder.appName("Test").getOrCreate()
print("Spark Session Created Successfully")
spark.stop()
