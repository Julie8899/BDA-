-- --------------------------------------------------------------------
-- Use an in memory engine for the temporary table (TIME: 51 seconds)
--
-- ENGINE=MEMORY tables have a default of ~16 meg limit
-- "max_heap_table_size" value needs to be increased in mysql.cnf
-- --------------------------------------------------------------------
-- Create a "master" table
CREATE TABLE IF NOT EXISTS t_rolling_lookup AS
SELECT
		g.game_id
		, local_date
		, batter
		, atBat
		, Hit
    FROM batter_counts bc
    JOIN game g ON g.game_id = bc.game_id
    WHERE atBat > 0
	ORDER BY batter, local_date;

CREATE INDEX rolling_lookup_game_id_idx ON t_rolling_lookup (game_id);
CREATE INDEX rolling_lookup_local_date_idx ON t_rolling_lookup (local_date);
CREATE INDEX rolling_lookup_batter_idx ON t_rolling_lookup (batter);

-- Create the rolling 100 days table
SELECT
		rl1.batter
		, rl1.game_id
		, rl1.local_date
		, SUM(rl2.Hit) / SUM(rl2.atBat) AS BA
	FROM t_rolling_lookup rl1
	JOIN t_rolling_lookup rl2 ON rl1.batter = rl2.batter
		AND rl2.local_date BETWEEN DATE_SUB(rl1.local_date, INTERVAL 100 DAY) AND rl1.local_date
    WHERE rl1.game_id = 12560
	GROUP BY rl1.batter, rl1.game_id, rl1.local_date;
