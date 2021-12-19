CREATE OR REPLACE TABLE temp_rolled_20
SELECT 
	present.game_id AS game_id,
	present.game_date AS game_date, 
	present.team_id AS team_id,
	AVG(past.final_score) AS avg_final_score,
	
	AVG(past.off_single) AS avg_off_single,
	AVG(past.def_single) AS avg_def_single,
	
	AVG(past.off_double) AS avg_off_double,
	AVG(past.def_double) AS avg_def_double,
	
	AVG(past.off_triple) AS avg_off_triple,
	AVG(past.def_triple) AS avg_def_triple,
	
	SUM(past.off_walk) / NULLIF(SUM(past.off_strikeout), 0) AS off_walk_to_strikeout_avg,
	SUM(past.def_walk) / NULLIF(SUM(past.def_strikeout), 0) AS def_walk_to_strikeout_avg,
	
	SUM(past.off_ground) / NULLIF(SUM(past.off_flyout), 0) AS off_ground_to_flyout_avg,
	SUM(past.def_ground) / NULLIF(SUM(past.def_flyout), 0) AS def_ground_to_flyout_avg,
	
	SUM(past.off_hit) / NULLIF(SUM(past.off_at_bat), 0) AS off_batting_avg,
	SUM(past.def_hit) / NULLIF(SUM(past.def_at_bat), 0) AS def_batting_avg,
	
	SUM(past.off_plate_appr) / NULLIF(SUM(past.off_strikeout), 0) AS off_pa_so,
	SUM(past.def_plate_appr) / NULLIF(SUM(past.def_strikeout), 0) AS def_pa_so,
	
	AVG(past.off_on_base_percentage) AS off_on_base_percentage_avg,
	AVG(past.def_on_base_percentage) AS def_on_base_percentage_avg,
	
	AVG(past.off_slug_percentage) AS off_slug_percentage_avg,
	AVG(past.def_slug_percentage) AS def_slug_percentage_avg,
	
	AVG(past.off_on_base_plus_slug) AS off_on_base_plus_slug_avg,
	AVG(past.def_on_base_plus_slug) AS def_on_base_plus_slug_avg,
	
	SUM(past.win) / COUNT(past.game_id) AS win_percentage,
	SUM(past.home_win) / NULLIF(SUM(past.is_home), 0) AS home_win_percentage,
	SUM(past.away_win) / NULLIF(SUM(1 - past.is_home), 0) AS away_win_percentage
	
FROM temp_for_rolling present
JOIN temp_for_rolling past ON 
	present.team_id = past.team_id
	AND past.game_date BETWEEN DATE_SUB(present.game_date, INTERVAL 20 DAY) AND present.game_date
	GROUP BY present.game_id, present.game_date, present.team_id
;

CREATE INDEX idx_game_id ON temp_rolled_20 (game_id);
CREATE INDEX idx_team_id ON temp_rolled_20 (team_id);