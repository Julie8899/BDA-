import sys
import tempfile


import requests
from pyspark import StorageLevel
from pyspark.ml import Pipeline
from pyspark.ml.feature import CountVectorizer
from pyspark.sql import SparkSession
from split_column_transform import SplitColumnTransform


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

    # Load Data directly from MariaDB

    t_rolling_lookup_df = spark.read \
        .format("jdbc") \
        .option("url", "jdbc:mysql://localhost:3306/baseball") \
        .option("driver", "org.mariadb.jdbc.Driver") \
        .option("query",
                '''
                SELECT 
                    g.game_id
                    , local_date
                    , batter
                    , atBat
                    , Hit
                FROM batter_counts bc
                JOIN game g ON g.game_id = bc.game_id
                WHERE atBat > 0
                ORDER BY batter, local_date
                '''
                ) \
        .option("user", "root") \
        .option("password", "bda696") \
        .load()

    # t_rolling_lookup_df.show()
    # Create 100 days rolling query

    t_rolling_lookup_df.createOrReplaceTempView("t_rolling_lookup_v")
    t_rolling_lookup_df.persist(StorageLevel.DISK_ONLY)

    results = spark.sql(
        """ 
        SELECT 
            rl1.batter
            , rl1.game_id
            , rl1.local_date
            , format_number(SUM(rl2.Hit) / SUM(rl2.atBat), 4) AS BA
        FROM t_rolling_lookup_v rl1
        JOIN t_rolling_lookup_v rl2 ON rl1.batter = rl2.batter
        AND rl2.local_date BETWEEN DATE_SUB(rl1.local_date, 100) AND rl1.local_date
        GROUP BY rl1.batter, rl1.game_id, rl1.local_date;

        """
    )

    results.createOrReplaceTempView("results_v")
    results.persist(StorageLevel.DISK_ONLY)

    results.show()

# TRANSFORMATION
    # Split Column Transform
    split_column_transform = SplitColumnTransform(
    inputCols=["batter", "game_id", "local_date"], outputCol="categorical"
    )
    count_vectorizer = CountVectorizer(
    inputCol="categorical", outputCol="categorical_vector"
    )

    # Pipeline Setup
    pipeline = Pipeline(
        stages=[split_column_transform, count_vectorizer]
    )

    # Fit the pipeline
    model = pipeline.fit(results)
    results = model.transform(results)
    results.show()
    return


if __name__ == "__main__":
    sys.exit(main())
