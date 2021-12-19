CREATE INDEX IF NOT EXISTS idx_game_id ON team_batting_counts (game_id);
CREATE INDEX IF NOT EXISTS idx_team_id ON team_batting_counts (team_id);
CREATE INDEX IF NOT EXISTS idx_opponent_team_id ON team_batting_counts (opponent_team_id);

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