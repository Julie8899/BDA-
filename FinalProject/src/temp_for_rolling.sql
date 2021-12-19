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
CREATE INDEX idx_game_id ON temp_for_rolling (game_id);
CREATE INDEX idx_game_date ON temp_for_rolling (game_date);
CREATE INDEX idx_team_id ON temp_for_rolling (team_id);