# Machine Learning with Apache Spark - Lab Notebooks

This repo contains hands-on labs from a course on machine learning with Apache Spark. These aren't just theoretical exercises—each notebook walks through real ML problems using actual datasets. If you're learning Spark or need to refresh your memory on how to build ML pipelines, this is what you'll find here.

## What's Inside

### Module 1: Getting Started with Apache Spark
**The basics—using scikit-learn and pandas before jumping into Spark**

- **Build_a_classifier_using_Logistic_Regression.ipynb**
  - Classifies iris flower species (the classic ML dataset)
  - Also builds a cancer tumor classifier using breast cancer data
  - Uses logistic regression with pandas/sklearn
  - Walks through the full pipeline: load data → identify features → train → evaluate

- **Building_and_training_a_model_using_Linear_Regression.ipynb**
  - Predicts car mileage from the Auto MPG dataset
  - Also predicts diamond prices
  - Linear regression basics with pandas/sklearn
  - Good intro to regression problems before doing it in Spark

### Module 2: Machine Learning with Apache Spark
**Now we're cooking with Spark—same ML tasks but distributed**

- **Classification_using_SparkML.ipynb**
  - Bean variety classification using SparkML
  - Learn the Spark way: DataFrames → feature vectors → train/test split → model evaluation
  - About 30 mins if you follow along

- **Regression_using_SparkML.ipynb**
  - Car mileage prediction again, but this time with Spark
  - Shows how to do linear regression on a Spark cluster
  - Same problem as Module 1 but scaled up

- **Clustering_using_SparkML.ipynb**
  - Unsupervised learning with K-means clustering
  - Shows how to group data when you don't have labels
  - Includes cluster center analysis

- **Connecting_to_spark_cluster_using_Skills_Network_labs.ipynb**
  - Setup/config notebook for connecting to Spark in the Skills Network environment
  - Probably just boilerplate if you're running locally

### Module 3: Data Engineering For Machine Learning With Spark
**The real-world stuff—ETL, pipelines, and production concerns**

- **ETL_using_Spark.ipynb**
  - Extract, transform, load workflows
  - CSV ↔ Parquet conversions
  - How to read/write data in different formats
  - Includes file consolidation (combining partition files)

- **Feature_Extraction_and_Transformation_using_Spark.ipynb**
  - Text processing: tokenization, stop word removal, TF-IDF
  - Vectorization with CountVectorizer
  - Data prep tools: StringIndexer, StandardScaler
  - All the pre-processing you need before training

- **ML_Pipelines_using_SparkML.ipynb**
  - Chain everything together into a single pipeline
  - Define stages → build pipeline → fit → evaluate
  - This is how you'd actually do it in production

- **Model_persistance_using_SparkML.ipynb**
  - Save trained models to disk
  - Load them back later for predictions
  - Essential for deploying models to production

- **Analyse_a_dataset_using_SparkSQL.ipynb**
  - Use SQL queries on Spark DataFrames
  - Create temp views and run analytics
  - For when you prefer SQL over DataFrame API

- **StructuredStreaming.ipynb**
  - Real-time data processing example
  - Simulates HVAC sensor data from smart buildings
  - Process streaming data with Spark Structured Streaming
  - Monitor temperature/humidity in real-time

## What You'll Learn

- How to build ML models in Spark (classification, regression, clustering)
- Feature engineering and data transformation at scale
- Creating ML pipelines that combine multiple steps
- ETL workflows for ML data prep
- Model persistence (saving/loading trained models)
- Real-time streaming analytics
- When to use Spark vs. pandas/sklearn

## Running These

Most notebooks expect you to have:
- PySpark installed
- Access to datasets (usually downloaded in the notebooks)
- A Spark session (created at the start of each notebook)

The Module 1 notebooks use pandas/sklearn and work without Spark. Everything else needs Spark set up.

## Typical Workflow in Each Notebook

1. Install dependencies
2. Load dataset
3. Explore/transform data
4. Build and train model
5. Evaluate with metrics
6. Make predictions
7. Save results

Each one takes about 30 minutes if you're actively coding along.

## Why This Matters

If you're doing data engineering or ML engineering, knowing Spark is pretty much required when datasets get big. These labs show you the practical side—not just theory, but actual code for building, training, and deploying ML models on distributed systems.

The progression makes sense: start with familiar tools (sklearn), then translate those concepts to Spark, then learn the production stuff (pipelines, streaming, persistence).

## Note

This is coursework from a Data Engineer Professional Specialization. The notebooks are structured as tutorials with tasks and exercises. They're meant to be worked through, not just read.
