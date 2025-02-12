Steps:
1.  make sure you have all the necessary depenencies installed like python and pyspark
2.  run preprocessing file using "python wildlife_monitoring.py" command
3.  output for this is generated at "dataset\output.csv\part-00000-f8f7d0bc-a5bc-40e3-a806-dc5dfb385e15-c000.csv"
4.  run kmeans file:
                    a.  python kmeans.py   :this groups species together in clusters
                    b.  python kmeans_specieswise.py    :this keeps species individually
                    use as per your requirement. Arunima preferred (b)
5.  output is at "dataset\kmeans_clusters.csv\part-00000-65ff40ef-4949-439f-9c88-305fa90a6ab7-c000.csv" and 
    "dataset\kmeans_specieswise.csv\part-00000-74c90456-98ea-49ee-860c-9cd3b7104beb-c000.csv" respectively
6.  run dt code using "python tree.py"
7.  output at "dataset\max_individual_count_per_species.csv\part-00000-c19f3bab-e2b9-459f-a095-ff733bc4943a-c000.csv"
8.  use the csv files for visualisation
7.  first visualisation plots all species on map based on cluster. filters can be applied based on cluster number and species name
8.  second visualisation plots suitable habitat(the coordinates) for each species based upon the maximum count of the species.