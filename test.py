import random
from collections import defaultdict

from main_opt import main_edited_for_opt


def make_player_counts(num_players, available_players=tuple(range(1, 10)), pr=False):
	"""
	Returns a defaultdict(int) with counts, e.g. {"p1": x, "p2": y, ..., "pr": z}.
	Guarantees at least one 'p5' if available; optionally one 'pr'.
	"""
	if not available_players:
		available_players = tuple(range(1, 10))

	labels = []

	# guarantee p5
	if 5 in available_players and num_players > 0:
		labels.append('p5')
	remaining = num_players - len(labels)

	# optional 'pr'
	if pr and remaining > 0:
		labels.append('prp')
		remaining -= 1

	# fill remaining slots
	for _ in range(max(0, remaining)):
		p = random.choice(available_players)
		labels.append('p5')

	counts = defaultdict(int)
	for label in labels[:num_players]:
		counts[label] += 1
	return counts


data = main_edited_for_opt(
	args=make_player_counts(10, pr=False),
	total_players=10,
	subjects=20,
	memory_size=10,
	length=50,
	params=[0.5, 1, 2, 0.5, 2],
)

print("Data", data)
