# Rows in new csv { ValueRaw, ValueShrunk, OffenseScore, CreationScore, DefenseScore, DisciplinePenalty , Minutes and GamesPlayed }


# playerName,season,team,position,role,GamesPlayed,Minutes,Goals,Assists,Shots,GoalInvolvements,
#
# hotsOnTarget,KeyPasses,Tackles,FoulsCommitted,FoulsSuffered,Offsides,YellowCards,RedCards,PassPct,
# 
# AA,SavePct,G_per_game,A_per_game,S_per_game,SOT_per_game,KP_per_game,Minutes_per_game

# player stat contribution = (player_stat - mean) / std
# player value = (player stats - posision average) std

# adjusted value = playervalue * minutes + somethingCP


# Positives: G_per_game, A_per_game, SOT_per_game, KP_per_game, FoulsSuffered, GoalInvolvements_game
# Negatives: Offsides per game

# Forwards = 



# Positives: Assistspergame, key passes per game, shots on target pg, goals pg
# Negatives: Fouls commited per game, offsides per game

# Midfielder = 




# Positives: Assists, key passes, shots on target, shots, tackle percentage
# Negatives: Fouls commited per game

# Defender = 




# Positives: SavePct, 
# Negatives: GAA

# Goalkeeper = 