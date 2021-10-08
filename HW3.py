import sys

from pyspark.sql import SparkSession


def main():
    spark = SparkSession.builder.master("local[*]").getOrCreate()

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

    t_rolling_lookup_df.createOrReplaceTempView("t_rolling_lookup_v")

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

    results.show()


if __name__ == "__main__":
    sys.exit(main())
