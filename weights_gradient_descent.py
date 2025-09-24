import json
import random
from functools import lru_cache

import numpy as np
import optuna
from collections import defaultdict

from main_opt import main_edited_for_opt

# ----- Parameter search space (length 5 weights) -----
SEARCH_SPACE = [(0.0, 2.0)] * 5

REPLICATES_BASE = 3
REPLICATES_TOPUP = 5
SAVE_EVERY = 5
OUTFILE = "simulation_results2.json"

import math
import numpy as np
import optuna

class PlateauStopper:
	"""
	Stop the study if best_value hasn't improved by at least `min_delta`
	for `patience` completed trials.

	mode='max' for maximization (default), 'min' for minimization.
	If `percentage=True`, interpret min_delta as a relative fraction.
	"""
	def __init__(self, patience=30, min_delta=0.02, mode='max', percentage=False):
		self.patience = patience
		self.min_delta = min_delta
		self.mode = mode
		self.percentage = percentage
		self._last_best = None
		self._since = 0

	def _improved(self, new, old):
		if old is None or not np.isfinite(new):
			return False
		if self.mode == 'max':
			diff = new - old
		else:
			diff = old - new
		if self.percentage and np.isfinite(old) and old != 0:
			return diff / abs(old) >= self.min_delta
		return diff >= self.min_delta

	def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
		# Ignore failed/pruned trials
		if trial.value is None or not np.isfinite(trial.value):
			return
		best = study.best_value if study.best_trial is not None else None

		# Initialize on first valid best
		if self._last_best is None and best is not None and np.isfinite(best):
			self._last_best = best
			self._since = 0
			return

		# Check improvement
		if best is not None and self._improved(best, self._last_best):
			self._last_best = best
			self._since = 0
		else:
			self._since += 1
			if self._since >= self.patience:
				print(f"[PlateauStopper] No improvement ≥ {self.min_delta}"
					  f" for {self.patience} trials. Stopping.")
				study.stop()


# ----- Player sampling helpers -----
def make_player_counts(num_players, available_players=tuple(range(0, 12)), pr=False, pr_label="prp"):
	"""
	Return defaultdict(int) like {'p5': 1, 'p2': 3, 'prp': 1}, length == num_players.
	Guarantees one 'p5' if available; optionally 1 'prp'.
	"""
	if not available_players:
		available_players = tuple(range(0, 12))

	labels = []

	# guarantee p5
	if 5 in available_players and num_players > 0:
		labels.append("p5")
	remaining = num_players - len(labels)

	# optional 'prp'
	if pr and remaining > 0:
		labels.append(pr_label)
		remaining -= 1

	# fill remaining slots at random from available numeric players
	for _ in range(max(0, remaining)):
		p = random.choice(range(1, 9))
		labels.append(f"p{p}")

	counts = defaultdict(int)
	for lab in labels[:num_players]:
		counts[lab] += 1
	return counts


def all_p5_counts(num_players):
	d = defaultdict(int)
	d["p5"] = num_players
	return d


# ----- Score extraction from run result -----
def score_from_result(data):
	"""
	Average 'total' for Player5 across its speaker ids.
	Expects:
	  data['turn_impact'] and data['scores']['player_scores']
	"""
	player_ids = [t["speaker_id"] for t in data["turn_impact"] if t.get("speaker_name") == "Player5"]
	if not player_ids:
		raise ValueError("Player5 speaker_id(s) not found in turn_impact.")
	vals = []
	for p in data["scores"]["player_scores"]:
		if p["id"] in player_ids:
			vals.append(p["scores"]["total"])
	if not vals:
		raise ValueError("No totals found for Player5.")
	return float(sum(vals) / len(vals))


# ----- One simulation run -----
def run_once(player_counts, total_players, subjects, memory_size, length, params):
	"""
	Call your simulator directly (no subprocess). We pass params here.
	Assumes main_edited_for_opt signature: (players, total_players, subjects, memory_size, length, params)
	  - players: dict/defaultdict of counts per label (e.g., {'p5': 3, 'prp': 1, ...})
	"""
	data = main_edited_for_opt(
		args=player_counts,          # <-- was 'args=' previously; use a clear 'players='
		total_players=total_players,
		subjects=subjects,
		memory_size=memory_size,
		length=length,
		params=params,                  # <-- pass the trial params
	)
	return score_from_result(data)


# ----- Cache runs (include params and player_counts in key!) -----
def _freeze_counts(d):
	# sort for stable hashing
	return tuple(sorted(d.items()))  # e.g., (('p2', 3), ('p5', 1), ('prp', 1))

def _freeze_params(params, ndigits=6):
	# round to avoid cache misses on tiny float noise
	return tuple(round(x, ndigits) for x in params)

@lru_cache(maxsize=4096)
def eval_cached(frozen_counts, total_players, subjects, memory_size, length, frozen_params):
	counts = defaultdict(int, dict(frozen_counts))
	params = list(frozen_params)
	try:
		return run_once(counts, total_players, subjects, memory_size, length, params)
	except Exception as e:
		print(f"[eval_cached] failed: {e}")
		return float("nan")


# ----- Evaluate three policies and average -----
def eval_policy_mean(total_players, subjects, memory_size, length, params, repeats):
	"""
	Policies:
	  - random with 'prp'
	  - random without 'prp'
	  - all p5
	We re-sample random policies each repeat (better exploration).
	"""
	vals = []
	for policy in ("rand_pr", "rand_no", "all_p5"):
		policy_scores = []
		for _ in range(repeats):
			if policy == "rand_pr":
				counts = make_player_counts(total_players, pr=True)
			elif policy == "rand_no":
				counts = make_player_counts(total_players, pr=False)
			else:
				counts = all_p5_counts(total_players)

			s = eval_cached(
				_freeze_counts(counts),
				total_players,
				subjects,
				memory_size,
				length,
				_freeze_params(params),
			)
			policy_scores.append(s)
		vals.append(np.nanmean(policy_scores))
	return float(np.nanmean(vals))


# ----- Objective builder -----
def build_objective(total_players, subjects, memory_size, length):
	def objective(trial: optuna.trial.Trial) -> float:
		# trial params (the weights)
		params = [trial.suggest_float(f"p{i}", lo, hi) for i, (lo, hi) in enumerate(SEARCH_SPACE)]

		# cheap pass
		score = eval_policy_mean(total_players, subjects, memory_size, length, params, repeats=REPLICATES_BASE)

		# report for pruning
		trial.report(score, step=0)
		if trial.should_prune():
			raise optuna.TrialPruned()

		# top-up if promising
		best_so_far = None
		try:
			if trial.study.best_trial:
				best_so_far = trial.study.best_value
		except Exception:
			best_so_far = None

		if best_so_far is None or (np.isfinite(score) and score >= 0.95 * best_so_far):
			score = eval_policy_mean(
				total_players, subjects, memory_size, length, params, repeats=REPLICATES_TOPUP
			)
			trial.report(score, step=1)
			if trial.should_prune():
				raise optuna.TrialPruned()

		return score
	return objective


# ----- Main grid w/ per-setting optimization -----
def main():
	results = []
	points = []
	counter = 0

	points = []
	for length in [10, 30, 50, 100, 200]:
		# players: 2–6 for short convos, up to 8 for long ones
		max_players = 6 if length < 100 else 8
		for players in range(2, max_players + 1):
			# subjects: roughly 1–2 per 10 turns
			max_subjects = max(5, length // 5)
			step_subjects = max(5, max_subjects // 5)
			for subjects in range(players + 2, max_subjects + 1, step_subjects):
				# memory: at least 1/4 of subjects, up to subjects
				for memory in range(max(5, subjects // 4), subjects + 1, max(1, subjects // 4)):
					points.append((length, players, subjects, memory))

	for p in points:       

		print(f"=== Running length={p[0]}, players={p[1]}, subjects={p[2]}, memory={p[3]} ===")
		study = optuna.create_study(
			direction="maximize",
			sampler=optuna.samplers.TPESampler(
				seed=42, consider_prior=True, n_startup_trials=20
			),
			pruner=optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=0),
		)

		objective = build_objective(
			total_players=p[1],
			subjects=p[2],
			memory_size=p[3],
			length=p[0],
		)
		callbacks = [PlateauStopper(patience=25, min_delta=0.02, percentage=False)]

		study.optimize(objective, n_trials=100, n_jobs=10, callbacks=callbacks)

		entry = {
			"length": p[0],
			"players": p[1],
			"subjects": p[2],
			"memory": p[3],
			"best_score": study.best_value,
			"best_params": study.best_params,
		}
		results.append(entry)

		counter += 1
		if counter % SAVE_EVERY == 0:
			with open(OUTFILE, "w") as f:
				json.dump(results, f, indent=2)
			print(f"Checkpoint saved at {counter} iterations.")

		with open(OUTFILE, "w") as f:
			json.dump(results, f, indent=2)
		print("All results saved.")


if __name__ == "__main__":
	main()
