CREATE OR REPLACE TABLE temp_for_rolling_1
SELECT 
	g.game_id AS game_id,
	tbc.team_id AS team_id,
	g.local_date AS game_date,
	tbc.finalScore AS final_score,
	tbc.homeTeam AS is_home,
	tbc.win AS win,
	IF(tbc.homeTeam = 1 && tbc.win = 1, 1, 0) AS home_win,
	IF(tbc.awayTeam = 1 && tbc.win = 1, 1, 0) AS away_win,
	
	tbc.atBat AS off_at_bat,
	tpc.atBat AS def_at_bat,

	tbc.Hit AS off_hit,
	tpc.Hit AS def_hit,

	tbc.plateApperance AS off_plate_appr,
	tpc.plateApperance AS def_plate_appr,

	tbc.Strikeout AS off_strikeout,
	tpc.Strikeout AS def_strikeout,

	tbc.Single as off_single,
	tpc.Single as def_single,

	tbc.Double as off_double,
	tpc.Double as def_double,

	tbc.Triple as off_triple,
	tpc.Triple as def_triple,

	tbc.Walk as off_walk,
	tbc.Walk as def_walk,

	tbc.Ground_Out + tbc.Groundout as off_ground,
	tpc.Ground_Out + tpc.Groundout as def_ground,

	tbc.Fly_Out + tbc.Flyout as off_flyout,
	tpc.Fly_Out + tpc.Flyout as def_flyout,

	tbc.Hit_By_Pitch AS off_hit_by_pitch,
	tpc.Hit_By_Pitch AS def_hit_by_pitch,

	tbc.Sac_Fly AS off_sac_fly,
	tpc.Sac_Fly AS def_sac_fly,

	tbc.Home_Run AS off_home_run,
	tpc.Home_Run AS def_home_run,

	(tbc.Hit + tbc.Walk + tbc.Hit_By_Pitch) / NULLIF((tbc.atBat + tbc.Walk + tbc.Hit_By_Pitch + tbc.Sac_Fly), 0) AS off_on_base_percentage,
	(tpc.Hit + tpc.Walk + tpc.Hit_By_Pitch) / NULLIF((tpc.atBat + tpc.Walk + tpc.Hit_By_Pitch + tpc.Sac_Fly), 0) AS def_on_base_percentage,

	(tbc.Single + tbc.Double * 2 + tbc.Triple * 3 + tbc.Home_Run * 4) / NULLIF(tbc.atBat, 0) AS off_slug_percentage,
	(tpc.Single + tpc.Double * 2 + tpc.Triple * 3 + tpc.Home_Run * 4) / NULLIF(tpc.atBat, 0) AS def_slug_percentage
		
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

CREATE OR REPLACE TABLE temp_for_rolling
SELECT 
	tfr1.*,
	tfr1.off_on_base_percentage / NULLIF(LOG(10, NULLIF(tfr1.off_on_base_percentage, 0)), 0) + tfr1.off_slug_percentage / NULLIF(LOG(10, NULLIF(tfr1.off_slug_percentage, 0)), 0) - 1 AS off_on_base_plus_slug,
	tfr1.def_on_base_percentage / NULLIF(LOG(10, NULLIF(tfr1.def_on_base_percentage, 0)), 0) + tfr1.def_slug_percentage / NULLIF(LOG(10, NULLIF(tfr1.def_slug_percentage, 0)), 0) - 1 AS def_on_base_plus_slug
FROM temp_for_rolling_1 tfr1
;

CREATE UNIQUE INDEX unique_game_id_date_team_id ON temp_for_rolling (game_id, game_date, team_id);
CREATE UNIQUE INDEX unique_game_id_date_team_id_is_home ON temp_for_rolling (game_id, game_date, team_id, is_home);
CREATE INDEX idx_game_id ON temp_for_rolling (game_id);
CREATE INDEX idx_game_date ON temp_for_rolling (game_date);
CREATE INDEX idx_team_id ON temp_for_rolling (team_id);

CREATE OR REPLACE TABLE temp_rolled_100
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
	AND past.game_date BETWEEN DATE_SUB(present.game_date, INTERVAL 100 DAY) AND present.game_date
	GROUP BY present.game_id, present.game_date, present.team_id
;

CREATE INDEX idx_game_id ON temp_rolled_100 (game_id);
CREATE INDEX idx_team_id ON temp_rolled_100 (team_id);

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

CREATE INDEX idx_game_id ON team_batting_counts (game_id);
CREATE INDEX idx_team_id ON team_batting_counts (team_id);
CREATE INDEX idx_opponent_team_id ON team_batting_counts (opponent_team_id);

CREATE OR REPLACE TABLE temp_for_training
SELECT
	home_100.game_id AS game_id,
	home_100.game_date AS game_date,
	
	home_100.team_id AS home_team_id,
	away_100.team_id AS away_team_id,

	home_100.avg_final_score AS home_final_score_avg_100,
	away_100.avg_final_score AS away_final_score_avg_100,
	
	home_100.off_batting_avg AS home_off_batting_avg_100,
	away_100.off_batting_avg AS away_off_batting_avg_100,
	
	home_100.def_batting_avg AS home_def_batting_avg_100,
	away_100.def_batting_avg AS away_def_batting_avg_100,
	
	home_100.off_pa_so AS home_off_plateapperance_strikeout_100,
	away_100.off_pa_so AS away_off_plateapperance_strikeout_100,
	
	home_100.def_pa_so AS home_def_plateapperance_strikeout_100,
	away_100.def_pa_so AS away_def_plateapperance_strikeout_100,
	
	home_100.avg_off_single AS home_off_single_avg_100,
	away_100.avg_off_single AS away_off_single_avg_100,
	
	home_100.avg_def_single AS home_def_single_avg_100,
	away_100.avg_def_single AS away_def_single_avg_100,
	
	home_100.avg_off_double AS home_off_double_avg_100,
	away_100.avg_off_double AS away_off_double_avg_100,
	
	home_100.avg_def_double AS home_def_double_avg_100,
	away_100.avg_def_double AS away_def_double_avg_100,
	
	home_100.avg_off_triple AS home_off_triple_avg_100,
	away_100.avg_off_triple AS away_off_triple_avg_100,
	
	home_100.avg_def_triple AS home_def_triple_avg_100,
	away_100.avg_def_triple AS away_def_triple_avg_100,
	
	home_100.off_walk_to_strikeout_avg AS home_off_walk_to_strikeout_avg_100,
	away_100.off_walk_to_strikeout_avg AS away_off_walk_to_strikeout_avg_100,
	
	home_100.def_walk_to_strikeout_avg AS home_def_walk_to_strikeout_avg_100,
	away_100.def_walk_to_strikeout_avg AS away_def_walk_to_strikeout_avg_100,
	
	home_100.off_ground_to_flyout_avg AS home_off_ground_to_flyout_avg_100,
	away_100.off_ground_to_flyout_avg AS away_off_ground_to_flyout_avg_100,
	
	home_100.def_ground_to_flyout_avg AS home_def_ground_to_flyout_avg_100,
	away_100.def_ground_to_flyout_avg AS away_def_ground_to_flyout_avg_100,
	
	home_100.off_on_base_percentage_avg AS home_off_on_base_percentage_avg_100,
	away_100.off_on_base_percentage_avg AS away_off_on_base_percentage_avg_100,
	
	home_100.def_on_base_percentage_avg AS home_def_on_base_percentage_avg_100,
	away_100.def_on_base_percentage_avg AS away_def_on_base_percentage_avg_100,
	
	home_100.off_slug_percentage_avg AS home_off_slug_percentage_avg_100,
	away_100.off_slug_percentage_avg AS away_off_slug_percentage_avg_100,
	
	home_100.def_slug_percentage_avg AS home_def_slug_percentage_avg_100,
	away_100.def_slug_percentage_avg AS away_def_slug_percentage_avg_100,
	
	home_100.off_on_base_plus_slug_avg AS home_off_on_base_plus_slug_avg_100,
	away_100.off_on_base_plus_slug_avg AS away_off_on_base_plus_slug_avg_100,
	
	home_100.def_on_base_plus_slug_avg AS home_def_on_base_plus_slug_avg_100,
	away_100.def_on_base_plus_slug_avg AS away_def_on_base_plus_slug_avg_100,
	
	home_100.home_win_percentage AS home_home_win_percentage_100,
	away_100.home_win_percentage AS away_home_win_percentage_100,
	
	home_100.away_win_percentage AS home_away_win_percentage_100,
	away_100.away_win_percentage AS away_away_win_percentage_100,
	
	home_100.win_percentage AS home_win_percentage_100,
	away_100.win_percentage AS away_win_percentage_100,
	
	home_20.avg_final_score AS home_final_score_avg_20,
	away_20.avg_final_score AS away_final_score_avg_20,
	
	home_20.off_batting_avg AS home_off_batting_avg_20,
	away_20.off_batting_avg AS away_off_batting_avg_20,
	
	home_20.def_batting_avg AS home_def_batting_avg_20,
	away_20.def_batting_avg AS away_def_batting_avg_20,
	
	home_20.off_pa_so AS home_off_plateapperance_strikeout_20,
	away_20.off_pa_so AS away_off_plateapperance_strikeout_20,
	
	home_20.def_pa_so AS home_def_plateapperance_strikeout_20,
	away_20.def_pa_so AS away_def_plateapperance_strikeout_20,
	
	home_20.avg_off_single AS home_off_single_avg_20,
	away_20.avg_off_single AS away_off_single_avg_20,
	
	home_20.avg_def_single AS home_def_single_avg_20,
	away_20.avg_def_single AS away_def_single_avg_20,
	
	home_20.avg_off_double AS home_off_double_avg_20,
	away_20.avg_off_double AS away_off_double_avg_20,
	
	home_20.avg_def_double AS home_def_double_avg_20,
	away_20.avg_def_double AS away_def_double_avg_20,
	
	home_20.avg_off_triple AS home_off_triple_avg_20,
	away_20.avg_off_triple AS away_off_triple_avg_20,
	
	home_20.avg_def_triple AS home_def_triple_avg_20,
	away_20.avg_def_triple AS away_def_triple_avg_20,
	
	home_20.off_walk_to_strikeout_avg AS home_off_walk_to_strikeout_avg_20,
	away_20.off_walk_to_strikeout_avg AS away_off_walk_to_strikeout_avg_20,
	
	home_20.def_walk_to_strikeout_avg AS home_def_walk_to_strikeout_avg_20,
	away_20.def_walk_to_strikeout_avg AS away_def_walk_to_strikeout_avg_20,
	
	home_20.off_ground_to_flyout_avg AS home_off_ground_to_flyout_avg_20,
	away_20.off_ground_to_flyout_avg AS away_off_ground_to_flyout_avg_20,
	
	home_20.def_ground_to_flyout_avg AS home_def_ground_to_flyout_avg_20,
	away_20.def_ground_to_flyout_avg AS away_def_ground_to_flyout_avg_20,
	
	home_20.off_on_base_percentage_avg AS home_off_on_base_percentage_avg_20,
	away_20.off_on_base_percentage_avg AS away_off_on_base_percentage_avg_20,
	
	home_20.def_on_base_percentage_avg AS home_def_on_base_percentage_avg_20,
	away_20.def_on_base_percentage_avg AS away_def_on_base_percentage_avg_20,
	
	home_20.off_slug_percentage_avg AS home_off_slug_percentage_avg_20,
	away_20.off_slug_percentage_avg AS away_off_slug_percentage_avg_20,
	
	home_20.def_slug_percentage_avg AS home_def_slug_percentage_avg_20,
	away_20.def_slug_percentage_avg AS away_def_slug_percentage_avg_20,
	
	home_20.off_on_base_plus_slug_avg AS home_off_on_base_plus_slug_avg_20,
	away_20.off_on_base_plus_slug_avg AS away_off_on_base_plus_slug_avg_20,
	
	home_20.def_on_base_plus_slug_avg AS home_def_on_base_plus_slug_avg_20,
	away_20.def_on_base_plus_slug_avg AS away_def_on_base_plus_slug_avg_20,
	
	home_20.home_win_percentage AS home_home_win_percentage_20,
	away_20.home_win_percentage AS away_home_win_percentage_20,
	
	home_20.away_win_percentage AS home_away_win_percentage_20,
	away_20.away_win_percentage AS away_away_win_percentage_20,
	
	home_20.win_percentage AS home_win_percentage_20,
	away_20.win_percentage AS away_win_percentage_20,

	tbc.win AS home_team_win
FROM 
	temp_rolled_100 home_100, 
	temp_rolled_100 away_100,
	temp_rolled_20 home_20, 
	temp_rolled_20 away_20,
	team_batting_counts tbc
WHERE 1=1
	AND home_100.team_id = tbc.team_id
	AND away_100.team_id = tbc.opponent_team_id
	AND home_20.team_id = tbc.team_id
	AND away_20.team_id = tbc.opponent_team_id
	AND tbc.homeTeam = 1
	AND home_100.game_id = away_100.game_id
	AND home_100.game_id = tbc.game_id
	AND home_20.game_id = away_20.game_id
	AND home_20.game_id = tbc.game_id
;
