"""
Spark Streaming Job for Traffic Data Processing
Speed Layer implementation for real-time traffic analysis.
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, window, avg, count, max as spark_max, 
    min as spark_min, stddev, when, lit, current_timestamp,
    to_timestamp, expr
)
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, 
    BooleanType, IntegerType, TimestampType
)

# Traffic data schema matching the producer output
TRAFFIC_SCHEMA = StructType([
    StructField("vehicle_id", StringType(), True),
    StructField("vehicle_type", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("speed", DoubleType(), True),
    StructField("heading", DoubleType(), True),
    StructField("segment_id", StringType(), True),
    StructField("segment_name", StringType(), True),
    StructField("acceleration", DoubleType(), True),
    StructField("is_stopped", BooleanType(), True),
    StructField("occupancy", IntegerType(), True),
    StructField("fuel_level", DoubleType(), True),
])

# Weather data schema
WEATHER_SCHEMA = StructType([
    StructField("station_id", StringType(), True),
    StructField("station_name", StringType(), True),
    StructField("timestamp", StringType(), True),
    StructField("latitude", DoubleType(), True),
    StructField("longitude", DoubleType(), True),
    StructField("temperature", DoubleType(), True),
    StructField("humidity", DoubleType(), True),
    StructField("precipitation", DoubleType(), True),
    StructField("wind_speed", DoubleType(), True),
    StructField("wind_direction", DoubleType(), True),
    StructField("visibility", DoubleType(), True),
    StructField("condition", StringType(), True),
    StructField("pressure", DoubleType(), True),
    StructField("uv_index", DoubleType(), True),
])

def create_spark_session():
    """Create Spark session with Kafka and database connectors."""
    return (SparkSession.builder
        .appName("AbidjanTrafficStreaming")
        .master("spark://spark-master:7077")
        .config("spark.jars.packages", 
                "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0,"
                "org.postgresql:postgresql:42.6.0")
        .config("spark.sql.streaming.checkpointLocation", "/tmp/checkpoints")
        .config("spark.streaming.backpressure.enabled", "true")
        .config("spark.streaming.kafka.maxRatePerPartition", "1000")
        .getOrCreate())

def read_kafka_stream(spark, topic, schema):
    """Read stream from Kafka topic."""
    return (spark
        .readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", "kafka:29092")
        .option("subscribe", topic)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
        .selectExpr("CAST(key AS STRING)", "CAST(value AS STRING)")
        .select(from_json(col("value"), schema).alias("data"))
        .select("data.*")
        .withColumn("event_time", to_timestamp(col("timestamp"))))

def compute_segment_aggregations(traffic_df):
    """
    Compute real-time aggregations per road segment.
    Uses 1-minute tumbling windows with 10-second updates.
    """
    return (traffic_df
        .withWatermark("event_time", "30 seconds")
        .groupBy(
            window(col("event_time"), "1 minute", "10 seconds"),
            col("segment_id"),
            col("segment_name")
        )
        .agg(
            avg("speed").alias("avg_speed"),
            spark_max("speed").alias("max_speed"),
            spark_min("speed").alias("min_speed"),
            stddev("speed").alias("speed_stddev"),
            count("*").alias("vehicle_count"),
            avg("acceleration").alias("avg_acceleration"),
            count(when(col("is_stopped"), 1)).alias("stopped_vehicles"),
            count(when(col("vehicle_type") == "gbaka", 1)).alias("gbaka_count"),
            count(when(col("vehicle_type") == "bus", 1)).alias("bus_count"),
            count(when(col("vehicle_type") == "taxi", 1)).alias("taxi_count"),
        )
        .withColumn("congestion_level", 
            when(col("avg_speed") < 10, "severe")
            .when(col("avg_speed") < 25, "heavy")
            .when(col("avg_speed") < 40, "moderate")
            .otherwise("light"))
        .withColumn("processed_at", current_timestamp()))

def compute_vehicle_type_stats(traffic_df):
    """Compute statistics per vehicle type."""
    return (traffic_df
        .withWatermark("event_time", "30 seconds")
        .groupBy(
            window(col("event_time"), "1 minute", "10 seconds"),
            col("vehicle_type")
        )
        .agg(
            avg("speed").alias("avg_speed"),
            count("*").alias("count"),
            avg("occupancy").alias("avg_occupancy"),
        )
        .withColumn("processed_at", current_timestamp()))

def detect_anomalies(traffic_df):
    """
    Detect traffic anomalies in real-time.
    - Sudden speed drops (potential accident)
    - High stopped vehicle concentration
    - Unusual congestion patterns
    """
    return (traffic_df
        .withWatermark("event_time", "30 seconds")
        .groupBy(
            window(col("event_time"), "30 seconds"),
            col("segment_id"),
            col("segment_name")
        )
        .agg(
            avg("speed").alias("avg_speed"),
            count("*").alias("vehicle_count"),
            count(when(col("is_stopped"), 1)).alias("stopped_count"),
            stddev("speed").alias("speed_variance"),
        )
        .withColumn("stopped_ratio", col("stopped_count") / col("vehicle_count"))
        .withColumn("is_anomaly",
            when((col("avg_speed") < 5) & (col("vehicle_count") > 10), True)
            .when(col("stopped_ratio") > 0.7, True)
            .when((col("speed_variance").isNotNull()) & (col("speed_variance") > 30), True)
            .otherwise(False))
        .filter(col("is_anomaly") == True)
        .withColumn("anomaly_type",
            when((col("avg_speed") < 5) & (col("vehicle_count") > 10), "severe_congestion")
            .when(col("stopped_ratio") > 0.7, "mass_stop")
            .otherwise("high_variance"))
        .withColumn("detected_at", current_timestamp()))

def write_to_postgres(df, table_name, mode="append"):
    """Write streaming data to PostgreSQL."""
    return (df.writeStream
        .foreachBatch(lambda batch_df, batch_id: 
            batch_df.write
                .format("jdbc")
                .option("url", "jdbc:postgresql://postgres:5432/smart_city")
                .option("dbtable", table_name)
                .option("user", "admin")
                .option("password", "password")
                .option("driver", "org.postgresql.Driver")
                .mode(mode)
                .save()
        )
        .outputMode("update")
        .option("checkpointLocation", f"/tmp/checkpoints/{table_name}")
        .start())

def write_to_cassandra(df, keyspace, table):
    """Write streaming data to Cassandra using the DataStax driver."""
    def process_batch(batch_df, batch_id):
        if batch_df.isEmpty():
            return
            
        # Collect data to driver (for this simulation scale it's fine, 
        # for real production we'd use the Spark-Cassandra connector)
        rows = batch_df.collect()
        
        from cassandra.cluster import Cluster
        from cassandra.query import BatchStatement
        
        try:
            cluster = Cluster(['cassandra'], port=9042)
            session = cluster.connect(keyspace)
            
            insert_stmt = session.prepare(f"""
                INSERT INTO {table} (segment_id, timestamp, vehicle_id, speed, latitude, longitude, vehicle_type)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """)
            
            batch = BatchStatement()
            count = 0
            
            for row in rows:
                batch.add(insert_stmt, (
                    row.segment_id, 
                    row.event_time, 
                    row.vehicle_id, 
                    row.speed,
                    row.latitude,
                    row.longitude,
                    row.vehicle_type
                ))
                count += 1
                
                # Execute batch every 100 rows
                if count >= 100:
                    session.execute(batch)
                    batch = BatchStatement()
                    count = 0
            
            if count > 0:
                session.execute(batch)
                
            cluster.shutdown()
        except Exception as e:
            print(f"Error writing to Cassandra: {e}")

    return (df.writeStream
        .foreachBatch(process_batch)
        .outputMode("append")
        .option("checkpointLocation", f"/tmp/checkpoints/cassandra_{table}")
        .start())

def main():
    """Main entry point for Spark Streaming job."""
    print("Starting Abidjan Traffic Streaming Job...")
    
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("WARN")
    
    # Read traffic stream
    traffic_stream = read_kafka_stream(spark, "traffic_data", TRAFFIC_SCHEMA)
    
    # Compute aggregations
    segment_stats = compute_segment_aggregations(traffic_stream)
    vehicle_stats = compute_vehicle_type_stats(traffic_stream)
    anomalies = detect_anomalies(traffic_stream)
    
    # Write to PostgreSQL (Aggregated Data)
    queries = [
        write_to_postgres(segment_stats, "traffic_segment_stats"),
        write_to_postgres(anomalies, "traffic_anomalies"),
        # Write RAW data to Cassandra (Big Data Storage)
        write_to_cassandra(traffic_stream, "smart_city", "traffic_data")
    ]

    # Process Weather Data
    weather_stream = read_kafka_stream(spark, "weather_data", WEATHER_SCHEMA)
    # Rename timestamp to recorded_at to match Postgres schema
    weather_stream = weather_stream.withColumnRenamed("timestamp", "recorded_at")
    # We can write raw weather data to Postgres for now (low volume)
    queries.append(write_to_postgres(weather_stream, "weather_data"))
    
    print("Streaming queries started. Waiting for termination...")
    
    for query in queries:
        query.awaitTermination()

if __name__ == "__main__":
    main()

