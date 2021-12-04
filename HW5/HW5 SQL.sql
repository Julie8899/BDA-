CREATE OR REPLACE TABLE temp_for_rolling 
SELECT 
	g.game_id AS game_id,
	tbc.team_id AS team_id,
	g.local_date AS game_date,
	tbc.homeTeam AS is_home,
	tbc.finalScore AS final_score,
	tbc.atBat AS off_at_bat,
	tbc.Hit AS off_hit,
	tbc.plateApperance AS off_plate_appr,
	tbc.Strikeout AS off_strikeout,
	tbc.Single as off_single,
	tbc.Double as off_double,
	tbc.Triple as off_triple,
	tbc.Walk as off_walk,
	tbc.Ground_Out + tbc.Groundout as off_ground,
	tbc.Fly_Out + tbc.Flyout as off_flyout,
	tpc.atBat AS def_at_bat,
	tpc.Hit AS def_hit,
	tpc.plateApperance AS def_plate_appr,
	tpc.Strikeout AS def_strikeout,
	tpc.Single as def_single,
	tpc.Double as def_double,
	tpc.Triple as def_triple,
	tbc.Walk as def_walk,
	tpc.Ground_Out + tpc.Groundout as def_ground,
	tpc.Fly_Out + tpc.Flyout as def_flyout
	
FROM 
	game g,
	team_batting_counts tbc,
	team_pitching_counts tpc
WHERE 1=1
	AND g.game_id = tbc.game_id
	AND g.game_id = tpc.game_id
	AND tbc.team_id = tpc.team_id
	AND tbc.homeTeam != tbc.awayTeam
	AND tpc.homeTeam != tpc.awayTeam
;
CREATE UNIQUE INDEX unique_game_id_game_date_team_id ON temp_for_rolling (game_id, game_date, team_id);
CREATE UNIQUE INDEX unique_game_id_game_date_team_id_is_home ON temp_for_rolling (game_id, game_date, team_id, is_home);
CREATE INDEX idx_game_id ON temp_for_rolling (game_id);
CREATE INDEX idx_game_date ON temp_for_rolling (game_date);
CREATE INDEX idx_team_id ON temp_for_rolling (team_id);

CREATE OR REPLACE TABLE temp_rolled
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
	SUM(past.def_plate_appr) / NULLIF(SUM(past.def_strikeout), 0) AS def_pa_so
FROM temp_for_rolling present
JOIN temp_for_rolling past ON 
	present.team_id = past.team_id
	AND past.game_date BETWEEN DATE_SUB(present.game_date, INTERVAL 100 DAY) AND present.game_date
	GROUP BY present.game_id, present.game_date, present.team_id
;

CREATE OR REPLACE TABLE temp_for_training
SELECT
	home.game_id AS game_id,
	home.game_date AS game_date,
	
	home.team_id AS home_team_id,
	home.avg_final_score AS home_final_score_avg,
	home.off_batting_avg AS home_off_batting_avg,
	home.def_batting_avg AS home_def_batting_avg,
	home.off_pa_so AS home_off_plateapperance_strikeout,
	home.def_pa_so AS home_def_plateapperance_strikeout,
	home.avg_off_single AS home_off_single_avg,
	home.avg_def_single AS home_def_single_avg,
	home.avg_off_double AS home_off_double_avg,
	home.avg_def_double AS home_def_double_avg,
	home.avg_off_triple AS home_off_triple_avg,
	home.avg_def_triple AS home_def_triple_avg,
	home.off_walk_to_strikeout_avg AS home_off_walk_to_strikeout_avg,
	home.def_walk_to_strikeout_avg AS home_def_walk_to_strikeout_avg,
	home.off_ground_to_flyout_avg AS home_off_ground_to_flyout_avg,
	home.def_ground_to_flyout_avg AS home_def_ground_to_flyout_avg,
	
	
	away.team_id AS away_team_id,
	away.avg_final_score AS away_final_score_avg,
	away.off_batting_avg AS away_off_batting_avg,
	away.def_batting_avg AS away_def_batting_avg,
	away.off_pa_so AS away_off_plateapperance_strikeout,
	away.def_pa_so AS away_def_plateapperance_strikeout,
	away.avg_off_single AS away_off_single_avg,
	away.avg_def_single AS away_def_single_avg,
	away.avg_off_double AS away_off_double_avg,
	away.avg_def_double AS away_def_double_avg,
	away.avg_off_triple AS away_off_triple_avg,
	away.avg_def_triple AS away_def_triple_avg,
	away.off_walk_to_strikeout_avg AS away_off_walk_to_strikeout_avg,
	away.def_walk_to_strikeout_avg AS away_def_walk_to_strikeout_avg,
	away.off_ground_to_flyout_avg AS away_off_ground_to_flyout_avg,
	away.def_ground_to_flyout_avg AS away_def_ground_to_flyout_avg,
	
	tbc.win AS home_team_win
FROM 
	temp_rolled home, 
	temp_rolled away,
	team_batting_counts tbc
WHERE 1=1
	AND home.team_id = tbc.team_id
	AND away.team_id = tbc.opponent_team_id
	AND tbc.homeTeam = 1
	AND home.game_id = away.game_id
	AND home.game_id = tbc.game_id
;


