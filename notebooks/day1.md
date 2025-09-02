
# Day 1: Environment Setup & Data Exploration
# Smart City IoT Analytics Pipeline

"""
🎯 LEARNING OBJECTIVES:
- Configure Spark cluster and development environment
- Understand IoT data characteristics and challenges  
- Implement basic data ingestion patterns
- Explore PySpark DataFrame operations

📅 SCHEDULE:
Morning (4 hours):
1. Environment Setup (2 hours)
2. Data Exploration (2 hours)

Afternoon (4 hours):  
3. Basic Data Ingestion (2 hours)
4. Initial Data Transformations (2 hours)

✅ DELIVERABLES:
- Working Spark cluster with all services running
- Data ingestion notebook with basic EDA
- Documentation of data quality findings  
- Initial data loading pipeline functions
"""

# =============================================================================
# SECTION 1: ENVIRONMENT SETUP (Morning - 2 hours)
# =============================================================================

print("🚀 Welcome to the Smart City IoT Analytics Pipeline!")
print("=" * 60)

# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import json
import warnings
warnings.filterwarnings('ignore')

# Import PySpark libraries
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import pyspark.sql.functions as F

# =============================================================================
# TODO 1.1: Initialize Spark Session (15 minutes)
# =============================================================================

"""
🎯 TASK: Create a Spark session configured for local development
💡 HINT: Use SparkSession.builder with appropriate configurations
📚 DOCS: https://spark.apache.org/docs/latest/sql-getting-started.html
"""

# TODO: Create Spark session with the following configurations:
# - App name: "SmartCityIoTPipeline-Day1"
# - Master: "local[*]" (use all available cores)
# - Memory: "4g" for driver
# - Additional configs for better performance

spark = (SparkSession.builder
         .appName("YOUR_APP_NAME_HERE")  # TODO: Add your app name
         .master("YOUR_MASTER_CONFIG")   # TODO: Add master configuration
         .config("spark.driver.memory", "YOUR_MEMORY_CONFIG")  # TODO: Set memory
         .config("spark.sql.adaptive.enabled", "true")
         .config("spark.sql.adaptive.coalescePartitions.enabled", "true")
         .getOrCreate())

# TODO: Verify Spark session is working
print("✅ Spark Session Details:")
print(f"   App Name: {spark.sparkContext.appName}")
print(f"   Spark Version: {spark.version}")
print(f"   Master: {spark.sparkContext.master}")
print(f"   Default Parallelism: {spark.sparkContext.defaultParallelism}")

# =============================================================================
# TODO 1.2: Verify Infrastructure (15 minutes) 
# =============================================================================

"""
🎯 TASK: Check that all infrastructure services are running
💡 HINT: Test database connectivity and file system access
"""

# TODO: Test PostgreSQL connection
def test_database_connection():
    """Test connection to PostgreSQL database"""
    try:
        # Database connection parameters
        db_properties = {
            "user": "postgres",
            "password": "password", 
            "driver": "org.postgresql.Driver"
        }
        
        # TODO: Replace with actual connection test
        # Test query - should create a simple DataFrame from database
        test_df = spark.read.jdbc(
            url="jdbc:postgresql://localhost:5432/smartcity",
            table="(SELECT 1 as test_column) as test_table",
            properties=db_properties
        )
        
        # TODO: Collect and display result
        result = test_df.collect()
        print("✅ Database connection successful!")
        return True
        
    except Exception as e:
        print(f"❌ Database connection failed: {str(e)}")
        print("💡 Make sure PostgreSQL container is running: docker-compose up -d")
        return False

# TODO: Run the database connection test
db_connected = test_database_connection()

# TODO: Check Spark UI accessibility
print("\n🌐 Spark UI should be accessible at: http://localhost:4040")
print("   (Open this in your browser to monitor Spark jobs)")

# =============================================================================
# TODO 1.3: Generate Sample Data (30 minutes)
# =============================================================================

"""
🎯 TASK: Run the data generation script to create sample IoT data
💡 HINT: Use the provided data generation script or run it manually
"""

# TODO: Run data generation (if not already done)
import subprocess
import os

def generate_sample_data():
    """Generate sample IoT data for the lab"""
    try:
        # TODO: Check if data already exists
        data_dir = "data/raw"
        if os.path.exists(f"{data_dir}/data_summary.json"):
            print("✅ Sample data already exists!")
            return True
            
        print("🔄 Generating sample data... (this may take a few minutes)")
        
        # TODO: Run the data generation script
        # (In practice, students would run: python scripts/generate_data.py)
        print("   Run: python scripts/generate_data.py")
        print("   This creates ~30 days of sensor data across 5 different sensor types")
        
        return True
        
    except Exception as e:
        print(f"❌ Data generation failed: {str(e)}")
        return False

# TODO: Generate or verify sample data exists
data_ready = generate_sample_data()

# =============================================================================
# SECTION 2: DATA EXPLORATION (Morning - 2 hours)
# =============================================================================

print("\n" + "=" * 60)
print("📊 SECTION 2: EXPLORATORY DATA ANALYSIS")
print("=" * 60)

# =============================================================================
# TODO 2.1: Load and Examine Data Sources (45 minutes)
# =============================================================================

"""
🎯 TASK: Load each data source and examine its structure
💡 HINT: Use appropriate Spark readers for different file formats
📚 CONCEPTS: Schema inference, file formats, data types
"""

# Define data directory
data_dir = "data/raw"

# TODO: Load city zones reference data
print("📍 Loading City Zones Reference Data...")
try:
    zones_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{data_dir}/city_zones.csv")
    
    # TODO: Display basic information about zones
    print(f"   📊 Records: {zones_df.count()}")
    print(f"   📋 Schema:")
    zones_df.printSchema()
    
    # TODO: Show sample data
    print(f"   🔍 Sample Data:")
    zones_df.show(5, truncate=False)
    
except Exception as e:
    print(f"❌ Error loading zones data: {str(e)}")

# TODO: Load traffic sensors data  
print("\n🚗 Loading Traffic Sensors Data...")
try:
    # TODO: Load CSV file with proper options
    traffic_df = spark.read.option("YOUR_OPTIONS_HERE").csv(f"{data_dir}/traffic_sensors.csv")
    
    # TODO: Display basic information
    print(f"   📊 Records: {traffic_df.count()}")
    print(f"   📋 Schema:")
    traffic_df.printSchema()
    
    # TODO: Show sample data
    print(f"   🔍 Sample Data:")
    traffic_df.show(5)
    
except Exception as e:
    print(f"❌ Error loading traffic data: {str(e)}")

# TODO: Load air quality data (JSON format)
print("\n🌫️ Loading Air Quality Data...")
try:
    # TODO: Load JSON file - note different file format!
    air_quality_df = spark.read.json(f"{data_dir}/air_quality.json")
    
    # TODO: Display basic information
    print(f"   📊 Records: {air_quality_df.count()}")
    print(f"   📋 Schema:")
    air_quality_df.printSchema()
    
    # TODO: Show sample data
    print(f"   🔍 Sample Data:")
    air_quality_df.show(5)
    
except Exception as e:
    print(f"❌ Error loading air quality data: {str(e)}")

# TODO: Load weather data (Parquet format)
print("\n🌤️ Loading Weather Data...")
try:
    # TODO: Load Parquet file - another different format!
    weather_df = spark.read.parquet(f"{data_dir}/weather_data.parquet")
    
    # TODO: Display basic information
    print(f"   📊 Records: {weather_df.count()}")
    print(f"   📋 Schema:")
    weather_df.printSchema()
    
    # TODO: Show sample data
    print(f"   🔍 Sample Data:")
    weather_df.show(5)
    
except Exception as e:
    print(f"❌ Error loading weather data: {str(e)}")

# TODO: Load energy meters data
print("\n⚡ Loading Energy Meters Data...")
try:
    # TODO: Load CSV file
    energy_df = spark.read.option("header", "true").option("inferSchema", "true").csv(f"{data_dir}/energy_meters.csv")
    
    # TODO: Display basic information
    print(f"   📊 Records: {energy_df.count()}")
    print(f"   📋 Schema:")
    energy_df.printSchema()
    
    # TODO: Show sample data
    print(f"   🔍 Sample Data:")
    energy_df.show(5)
    
except Exception as e:
    print(f"❌ Error loading energy data: {str(e)}")

# =============================================================================
# TODO 2.2: Basic Data Quality Assessment (45 minutes)
# =============================================================================

"""
🎯 TASK: Assess data quality across all datasets
💡 HINT: Check for missing values, duplicates, data ranges
📚 CONCEPTS: Data profiling, quality metrics, anomaly detection
"""

def assess_data_quality(df, dataset_name):
    """
    Perform basic data quality assessment on a DataFrame
    
    Args:
        df: Spark DataFrame to assess
        dataset_name: Name of the dataset for reporting
    """
    print(f"\n📋 Data Quality Assessment: {dataset_name}")
    print("-" * 50)
    
    # TODO: Basic statistics
    total_rows = df.count()
    total_cols = len(df.columns)
    print(f"   📊 Dimensions: {total_rows:,} rows × {total_cols} columns")
    
    # TODO: Check for missing values
    print(f"   🔍 Missing Values:")
    for col in df.columns:
        missing_count = df.filter(F.col(col).isNull()).count()
        missing_pct = (missing_count / total_rows) * 100
        if missing_count > 0:
            print(f"      {col}: {missing_count:,} ({missing_pct:.2f}%)")
    
    # TODO: Check for duplicate records
    duplicate_count = total_rows - df.dropDuplicates().count()
    if duplicate_count > 0:
        print(f"   🔄 Duplicate Records: {duplicate_count:,}")
    else:
        print(f"   ✅ No duplicate records found")
    
    # TODO: Numeric column statistics
    numeric_cols = [field.name for field in df.schema.fields 
                   if field.dataType in [IntegerType(), DoubleType(), FloatType(), LongType()]]
    
    if numeric_cols:
        print(f"   📈 Numeric Columns Summary:")
        # Show basic statistics for numeric columns
        df.select(numeric_cols).describe().show()

# TODO: Assess quality for each dataset
datasets = [
    (zones_df, "City Zones"),
    (traffic_df, "Traffic Sensors"), 
    (air_quality_df, "Air Quality"),
    (weather_df, "Weather Stations"),
    (energy_df, "Energy Meters")
]

for df, name in datasets:
    try:
        assess_data_quality(df, name)
    except Exception as e:
        print(f"❌ Error assessing {name}: {str(e)}")

# =============================================================================
# TODO 2.3: Temporal Analysis (30 minutes)
# =============================================================================

"""
🎯 TASK: Analyze temporal patterns in the IoT data
💡 HINT: Look at data distribution over time, identify patterns
📚 CONCEPTS: Time series analysis, temporal patterns, data distribution
"""

print("\n" + "=" * 60) 
print("⏰ TEMPORAL PATTERN ANALYSIS")
print("=" * 60)

# TODO: Analyze traffic patterns by hour
print("\n🚗 Traffic Patterns by Hour:")
try:
    # TODO: Extract hour from timestamp and analyze vehicle counts
    traffic_hourly = (traffic_df
                     .withColumn("hour", F.hour("timestamp"))
                     .groupBy("hour")
                     .agg(F.avg("vehicle_count").alias("avg_vehicles"),
                          F.count("*").alias("readings"))
                     .orderBy("hour"))
    
    # TODO: Show the results
    traffic_hourly.show(24)
    
    # TODO: What patterns do you notice? Add your observations here:
    print("📝 OBSERVATIONS:")
    print("   - Rush hour patterns: [YOUR ANALYSIS HERE]")
    print("   - Off-peak periods: [YOUR ANALYSIS HERE]")
    print("   - Peak traffic hours: [YOUR ANALYSIS HERE]")
    
except Exception as e:
    print(f"❌ Error analyzing traffic patterns: {str(e)}")

# TODO: Analyze air quality patterns by day of week
print("\n🌫️ Air Quality Patterns by Day of Week:")
try:
    # TODO: Extract day of week and analyze PM2.5 levels
    air_quality_daily = (air_quality_df
                        .withColumn("day_of_week", F.dayofweek("timestamp"))
                        .groupBy("day_of_week")
                        .agg(F.avg("pm25").alias("avg_pm25"),
                             F.avg("no2").alias("avg_no2"))
                        .orderBy("day_of_week"))
    
    # TODO: Show results
    air_quality_daily.show()
    
    # TODO: Add your observations
    print("📝 OBSERVATIONS:")
    print("   - Weekday vs weekend patterns: [YOUR ANALYSIS HERE]")
    print("   - Pollution trends: [YOUR ANALYSIS HERE]")
    
except Exception as e:
    print(f"❌ Error analyzing air quality patterns: {str(e)}")

# =============================================================================
# SECTION 3: BASIC DATA INGESTION (Afternoon - 2 hours)
# =============================================================================

print("\n" + "=" * 60)
print("📥 SECTION 3: DATA INGESTION PIPELINE")
print("=" * 60)

# =============================================================================
# TODO 3.1: Create Reusable Data Loading Functions (60 minutes)
# =============================================================================

"""
🎯 TASK: Create reusable functions for loading different data formats
💡 HINT: Handle schema validation and error handling
📚 CONCEPTS: Function design, error handling, schema enforcement
"""

def load_csv_data(file_path, expected_schema=None):
    """
    Load CSV data with proper error handling and schema validation
    
    Args:
        file_path: Path to CSV file
        expected_schema: Optional StructType for schema enforcement
        
    Returns:
        Spark DataFrame or None if error
    """
    try:
        # TODO: Implement CSV loading with options
        df = spark.read.option("YOUR_OPTIONS_HERE").csv(file_path)
        
        # TODO: Add schema validation if provided
        if expected_schema:
            # Validate schema matches expected
            pass
            
        print(f"✅ Successfully loaded CSV: {file_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading CSV {file_path}: {str(e)}")
        return None

def load_json_data(file_path):
    """
    Load JSON data with error handling
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Spark DataFrame or None if error
    """
    try:
        # TODO: Implement JSON loading
        df = spark.read.json(file_path)
        
        print(f"✅ Successfully loaded JSON: {file_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading JSON {file_path}: {str(e)}")
        return None

def load_parquet_data(file_path):
    """
    Load Parquet data with error handling
    
    Args:
        file_path: Path to Parquet file
        
    Returns:
        Spark DataFrame or None if error
    """
    try:
        # TODO: Implement Parquet loading
        df = spark.read.parquet(file_path)
        
        print(f"✅ Successfully loaded Parquet: {file_path}")
        return df
        
    except Exception as e:
        print(f"❌ Error loading Parquet {file_path}: {str(e)}")
        return None

# TODO: Test your loading functions
print("🧪 Testing Data Loading Functions:")

test_files = [
    (f"{data_dir}/city_zones.csv", "CSV", load_csv_data),
    (f"{data_dir}/air_quality.json", "JSON", load_json_data), 
    (f"{data_dir}/weather_data.parquet", "Parquet", load_parquet_data)
]

for file_path, file_type, load_func in test_files:
    print(f"\n   Testing {file_type} loader...")
    test_df = load_func(file_path)
    if test_df:
        print(f"      Records loaded: {test_df.count():,}")

# =============================================================================
# TODO 3.2: Schema Definition and Enforcement (60 minutes)
# =============================================================================

"""
🎯 TASK: Define explicit schemas for data consistency
💡 HINT: Use StructType and StructField for schema definition
📚 CONCEPTS: Schema design, data types, schema enforcement
"""

from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, TimestampType

# TODO: Define schema for traffic sensors
traffic_schema = StructType([
    StructField("sensor_id", StringType(), False),
    StructField("timestamp", TimestampType(), False),
    StructField("location_lat", DoubleType(), False),
    StructField("location_lon", DoubleType(), False),
    # TODO: Add remaining fields
    # StructField("vehicle_count", ???, ???),
    # StructField("avg_speed", ???, ???),
    # StructField("congestion_level", ???, ???),
    # StructField("road_type", ???, ???),
])

# TODO: Define schema for air quality data
air_quality_schema = StructType([
    # TODO: Define all fields for air quality data
    # Hint: Look at the JSON structure and define appropriate types
])

# TODO: Define schema for weather data
weather_schema = StructType([
    # TODO: Define all fields for weather data
])

# TODO: Define schema for energy data
energy_schema = StructType([
    # TODO: Define all fields for energy data
])

# TODO: Test schema enforcement
print("\n🔍 Testing Schema Enforcement:")

def load_with_schema(file_path, schema, file_format="csv"):
    """Load data with explicit schema enforcement"""
    try:
        if file_format == "csv":
            df = spark.read.schema(schema).option("header", "true").csv(file_path)
        elif file_format == "json":
            df = spark.read.schema(schema).json(file_path)
        elif file_format == "parquet":
            df = spark.read.schema(schema).parquet(file_path)
        
        print(f"✅ Schema enforcement successful for {file_path}")
        return df
        
    except Exception as e:
        print(f"❌ Schema enforcement failed for {file_path}: {str(e)}")
        return None

# TODO: Test with one of your schemas
test_schema_df = load_with_schema(f"{data_dir}/traffic_sensors.csv", traffic_schema, "csv")
if test_schema_df:
    print("   Schema enforcement test passed!")
    test_schema_df.printSchema()

# =============================================================================
# SECTION 4: INITIAL DATA TRANSFORMATIONS (Afternoon - 2 hours)
# =============================================================================

print("\n" + "=" * 60)
print("🔄 SECTION 4: DATA TRANSFORMATIONS")
print("=" * 60)

# =============================================================================
# TODO 4.1: Timestamp Standardization (45 minutes)
# =============================================================================

"""
🎯 TASK: Standardize timestamp formats across all datasets
💡 HINT: Some datasets may have different timestamp formats
📚 CONCEPTS: Date/time handling, format standardization, timezone handling
"""

def standardize_timestamps(df, timestamp_col="timestamp"):
    """
    Standardize timestamp column across datasets
    
    Args:
        df: Input DataFrame
        timestamp_col: Name of timestamp column
        
    Returns:
        DataFrame with standardized timestamps
    """
    try:
        # TODO: Convert timestamps to standard format
        standardized_df = (df
                          .withColumn("timestamp_std", F.to_timestamp(F.col(timestamp_col)))
                          .drop(timestamp_col)
                          .withColumnRenamed("timestamp_std", timestamp_col))
        
        # TODO: Add derived time columns
        result_df = (standardized_df
                    .withColumn("year", F.year(timestamp_col))
                    .withColumn("month", F.month(timestamp_col))
                    .withColumn("day", F.dayofmonth(timestamp_col))
                    .withColumn("hour", F.hour(timestamp_col))
                    .withColumn("day_of_week", F.dayofweek(timestamp_col))
                    .withColumn("is_weekend", F.when(F.dayofweek(timestamp_col).isin([1, 7]), True).otherwise(False)))
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error standardizing timestamps: {str(e)}")
        return df

# TODO: Test timestamp standardization
print("⏰ Testing Timestamp Standardization:")

# Test with traffic data
traffic_std = standardize_timestamps(traffic_df)
print("   Traffic data timestamp standardization:")
traffic_std.select("timestamp", "year", "month", "day", "hour", "day_of_week", "is_weekend").show(5)

# =============================================================================
# TODO 4.2: Geographic Zone Mapping (45 minutes)
# =============================================================================

"""
🎯 TASK: Map sensor locations to city zones
💡 HINT: Join sensor coordinates with zone boundaries
📚 CONCEPTS: Spatial joins, geographic data, coordinate systems
"""

def map_to_zones(sensor_df, zones_df):
    """
    Map sensor locations to city zones
    
    Args:
        sensor_df: DataFrame with sensor locations (lat, lon)
        zones_df: DataFrame with zone boundaries
        
    Returns:
        DataFrame with zone information added
    """
    try:
        # TODO: Create join condition for geographic mapping
        # A sensor is in a zone if its coordinates fall within zone boundaries
        join_condition = (
            (sensor_df.location_lat >= zones_df.lat_min) &
            (sensor_df.location_lat <= zones_df.lat_max) &
            (sensor_df.location_lon >= zones_df.lon_min) &
            (sensor_df.location_lon <= zones_df.lon_max)
        )
        
        # TODO: Perform the join
        result_df = (sensor_df
                    .join(zones_df, join_condition, "left")
                    .select(sensor_df["*"], 
                           zones_df.zone_id, 
                           zones_df.zone_name, 
                           zones_df.zone_type))
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error mapping to zones: {str(e)}")
        return sensor_df

# TODO: Test zone mapping
print("\n🗺️ Testing Geographic Zone Mapping:")

# Test with traffic sensors
traffic_with_zones = map_to_zones(traffic_std, zones_df)
print("   Traffic sensors with zone mapping:")
traffic_with_zones.select("sensor_id", "location_lat", "location_lon", "zone_id", "zone_type").show(10)

# TODO: Verify mapping worked correctly
zone_distribution = traffic_with_zones.groupBy("zone_type").count().orderBy(F.desc("count"))
print("   Sensors by zone type:")
zone_distribution.show()

# =============================================================================
# TODO 4.3: Data Type Conversions and Validations (30 minutes)
# =============================================================================

"""
🎯 TASK: Ensure proper data types and add validation columns
💡 HINT: Cast columns to appropriate types, add data quality flags
📚 CONCEPTS: Data type conversion, validation rules, data quality flags
"""

def add_data_quality_flags(df, sensor_type):
    """
    Add data quality validation flags to DataFrame
    
    Args:
        df: Input DataFrame
        sensor_type: Type of sensor for specific validations
        
    Returns:
        DataFrame with quality flags added
    """
    try:
        result_df = df
        
        # TODO: Add general quality flags
        result_df = result_df.withColumn("has_missing_values", 
                                        F.when(F.col("sensor_id").isNull(), True).otherwise(False))
        
        # TODO: Add sensor-specific validations
        if sensor_type == "traffic":
            # Traffic-specific validations
            result_df = (result_df
                        .withColumn("valid_speed", 
                                   F.when((F.col("avg_speed") >= 0) & (F.col("avg_speed") <= 100), True).otherwise(False))
                        .withColumn("valid_vehicle_count",
                                   F.when(F.col("vehicle_count") >= 0, True).otherwise(False)))
        
        elif sensor_type == "air_quality":
            # Air quality specific validations
            result_df = (result_df
                        .withColumn("valid_pm25",
                                   F.when((F.col("pm25") >= 0) & (F.col("pm25") <= 500), True).otherwise(False))
                        .withColumn("valid_temperature",
                                   F.when((F.col("temperature") >= -50) & (F.col("temperature") <= 50), True).otherwise(False)))
        
        # TODO: Add more sensor-specific validations
        
        return result_df
        
    except Exception as e:
        print(f"❌ Error adding quality flags: {str(e)}")
        return df

# TODO: Test data quality flags
print("\n🏷️ Testing Data Quality Flags:")

# Test with traffic data
traffic_with_flags = add_data_quality_flags(traffic_with_zones, "traffic")
print("   Traffic data with quality flags:")
traffic_with_flags.select("sensor_id", "avg_speed", "vehicle_count", "valid_speed", "valid_vehicle_count").show(10)

# TODO: Check quality flag distribution
quality_stats = (traffic_with_flags
                .agg(F.sum(F.when(F.col("valid_speed"), 1).otherwise(0)).alias("valid_speed_count"),
                     F.sum(F.when(F.col("valid_vehicle_count"), 1).otherwise(0)).alias("valid_vehicle_count_count"),
                     F.count("*").alias("total_records")))

print("   Quality statistics:")
quality_stats.show()

# =============================================================================
# DAY 1 DELIVERABLES & CHECKPOINTS
# =============================================================================

print("\n" + "=" * 60)
print("📋 DAY 1 COMPLETION CHECKLIST")
print("=" * 60)

# TODO: Complete this checklist by running the validation functions

def validate_day1_completion():
    """Validate that Day 1 objectives have been met"""
    
    checklist = {
        "spark_session_created": False,
        "database_connection_tested": False,
        "data_loaded_successfully": False,
        "data_quality_assessed": False,
        "loading_functions_created": False,
        "schemas_defined": False,
        "timestamp_standardization_working": False,
        "zone_mapping_implemented": False,
        "quality_flags_added": False
    }
    
    # TODO: Add validation logic for each item
    try:
        # Check Spark session
        if spark and spark.sparkContext._jsc:
            checklist["spark_session_created"] = True
            
        # Check if data exists
        if 'traffic_df' in locals() and traffic_df.count() > 0:
            checklist["data_loaded_successfully"] = True
            
        # TODO: Add more validation checks
        
    except Exception as e:
        print(f"❌ Validation error: {str(e)}")
    
    # Display results
    print("✅ COMPLETION STATUS:")
    for item, status in checklist.items():
        status_icon = "✅" if status else "❌"
        print(f"   {status_icon} {item.replace('_', ' ').title()}")
    
    completion_rate = sum(checklist.values()) / len(checklist) * 100
    print(f"\n📊 Overall Completion: {completion_rate:.1f}%")
    
    if completion_rate >= 80:
        print("🎉 Great job! You're ready for Day 2!")
    else:
        print("📝 Please review incomplete items before proceeding to Day 2.")
    
    return checklist

# TODO: Run the validation
completion_status = validate_day1_completion()

# =============================================================================
# NEXT STEPS
# =============================================================================

print("\n" + "=" * 60)
print("🚀 WHAT'S NEXT?")
print("=" * 60)

print("""
📅 DAY 2 PREVIEW: Data Quality & Cleaning Pipeline

Tomorrow you'll work on:
1. 🔍 Comprehensive data quality assessment
2. 🧹 Advanced cleaning procedures for IoT sensor data  
3. 📊 Missing data handling and interpolation strategies
4. 🚨 Outlier detection and treatment methods
5. 📏 Data standardization and normalization

📚 RECOMMENDED PREPARATION:
- Review PySpark DataFrame operations
- Read about time series data quality challenges
- Familiarize yourself with statistical outlier detection methods

💾 SAVE YOUR WORK:
- Commit your notebook to Git
- Document any issues or questions for tomorrow
- Save any custom functions you created

🤝 QUESTIONS?
- Post in the class discussion forum
- Review Spark documentation for any unclear concepts
- Prepare questions for tomorrow's Q&A session
""")

# TODO: Save your progress
print("\n💾 Don't forget to save your notebook and commit your changes!")

# Clean up (optional)
# spark.stop()

