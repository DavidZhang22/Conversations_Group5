import random

import numpy as np

from core.engine import Engine
from models.player import Player
from players.pause_player import PausePlayer
from players.player_0.player import Player0
from players.player_1.player import Player1
from players.player_2.player import Player2
from players.player_3.player import Player3
from players.player_4.player import Player4
from players.player_5.player import Player5
from players.player_6.player import Player6
from players.player_7.player import Player7
from players.player_8.player import Player8
from players.player_9.player import Player9
from players.player_10 import Player10
from players.player_11.player import Player11
from players.random_pause_player import RandomPausePlayer
from players.random_player import RandomPlayer
from ui.gui import run_gui


def main_edited_for_opt(total_players, subjects, memory_size, length, params, args):
	random.seed(91)
	np.random.seed(91)

	players: list[type[Player]] = (
		[RandomPlayer] * args['pr']
		+ [PausePlayer] * args['pp']
		+ [RandomPausePlayer] * args['prp']
		+ [Player0] * args['p0']
		+ [Player1] * args['p1']
		+ [Player2] * args['p2']
		+ [Player3] * args['p3']
		+ [Player4] * args['p4']
		+ [Player5] * args['p5']
		+ [Player6] * args['p6']
		+ [Player7] * args['p7']
		+ [Player8] * args['p8']
		+ [Player9] * args['p9']
		+ [Player10] * args['p10']
		+ [Player11] * args['p11']
	)

	engine = Engine(
		players=players,
		player_count=total_players,
		subjects=subjects,
		memory_size=memory_size,
		conversation_length=length,
		params=params,
	)

	simulation_results = engine.run(players)
	return simulation_results
