#this code was sent by aditya and it groups species based on habitat
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql import functions as F

# Step 1: Create Spark session and load dataset
spark = SparkSession.builder.appName("Species-Habitat-Clustering").getOrCreate()
data = spark.read.csv("dataset/output.csv/part-00000-f8f7d0bc-a5bc-40e3-a806-dc5dfb385e15-c000.csv", header=True, inferSchema=True)

# Step 2: Create bins for latitude and longitude
num_bins = 10
lat_min, lat_max = data.select(F.min("decimalLatitude"), F.max("decimalLatitude")).first()
lon_min, lon_max = data.select(F.min("decimalLongitude"), F.max("decimalLongitude")).first()
lat_bin_size = (lat_max - lat_min) / num_bins
lon_bin_size = (lon_max - lon_min) / num_bins

# Create latitude and longitude groups using actual lat/lon bin boundaries
data = data.withColumn(
    "lat_group", 
    (F.floor((F.col("decimalLatitude") - lat_min) / lat_bin_size) * lat_bin_size + lat_min)  # Start value of lat bin
).withColumn(
    "lon_group", 
    (F.floor((F.col("decimalLongitude") - lon_min) / lon_bin_size) * lon_bin_size + lon_min)  # Start value of lon bin
)

# Step 3: Prepare features for clustering
features = ["decimalLatitude", "decimalLongitude", "individualCount"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
data_vector = assembler.transform(data)

# Step 4: Train K-Means Model
k = 10  # Set number of clusters to 10
kmeans = KMeans(k=k, seed=1)
model = kmeans.fit(data_vector)
predictions = model.transform(data_vector)

# Step 5: Evaluate Clustering
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")

# Step 6: Count the number of original data points in each cluster
cluster_counts = (
    predictions.groupBy("prediction")
    .agg(
        F.count("*").alias("count"),
        F.first("decimalLatitude").alias("example_latitude"),
        F.first("decimalLongitude").alias("example_longitude"),
        F.first("lat_group").alias("latitude_group"),
        F.first("lon_group").alias("longitude_group")
    )
)

# Show the aggregated results
cluster_counts.show()

# Step 7: Count distinct species in each latitude and longitude group
distinct_species_counts = (
    data.groupBy("lat_group", "lon_group")
    .agg(
        F.countDistinct("species").alias("unique_species_count"),
        F.collect_set("species").alias("species_names")
    )
)

# Show distinct species counts by latitude and longitude groups
distinct_species_counts.show()

# Step 8: Count distinct species in each K-Means cluster
species_in_clusters = (
    predictions.groupBy("prediction", "lat_group", "lon_group")
    .agg(
        F.countDistinct("species").alias("unique_species_count"),
        F.collect_set("species").alias("species_names")
    )
    .withColumn("species_names_str", F.array_join("species_names", ","))
    .drop("species_names")
)

# Show the species count within each K-Means cluster
species_in_clusters.show(truncate=False)

# Step 9: Export the results to CSV with overwrite mode
output_path = 'dataset/kmeans_clusters.csv'
species_in_clusters.write.csv(output_path, header=True, mode="overwrite")

print(f"CSV file saved to: {output_path}")