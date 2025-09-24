import random
from collections import Counter

from models.player import GameContext, Item, Player, PlayerSnapshot

class ScoreEngine:
    def __init__(self, history=None):
        self.history: list[Item | None] = history or []

    def set_history(self, history):
        self.history = history

    def _freshness(self, i: int, current_item: Item) -> float:
        if i == 0 or self.history[i - 1] is not None:
            return 0.0
        prior_items = (x for x in self.history[max(0, i - 6): i - 1] if x is not None)
        prior_subjects = {s for it in prior_items for s in it.subjects}
        novel_subjects = [s for s in current_item.subjects if s not in prior_subjects]
        return float(len(novel_subjects))

    def _coherence(self, i: int, current_item: Item) -> float:
        ctx = []
		
        # look back up to 3
        for j in range(i-1, max(-1, i - 4), -1):
            if j < 0 or self.history[j] is None:
                break
            ctx.append(self.history[j])
            

        counts = Counter(s for it in ctx for s in it.subjects)
        score = 0.0
        if not all(s in counts for s in current_item.subjects):
            score -= 1.0
        if all(counts.get(s, 0) >= 2 for s in current_item.subjects):
            score += 1.0
        return score

    def _nonmono(self, i: int, current_item: Item, repeated: bool) -> float:
        if repeated:
            return -1.0
        if i < 3:
            return 0.0
        last3 = [self.history[j] for j in range(i - 3, i)]
        if any(x is None for x in last3):
            return 0.0
        for s in current_item.subjects:
            if all(s in it.subjects for it in last3):
                return -1.0
        return 0.0

    def compute(self, i: int, current_item: Item) -> dict:
        repeated = any(x.id == current_item.id for x in self.history[:i] if x is not None)

        coherence = self._coherence(i, current_item)
        freshness = self._freshness(i, current_item)
        nonmono   = self._nonmono(i, current_item, repeated)
        return {
            "coherence": float(0 if repeated else coherence),
            "freshness": float(freshness),
            "importance": float(0 if repeated else current_item.importance),
            "nonmonotonousness": float(nonmono),
        }


class Player5(Player):
    """
    Weights order (length == 5):
      [individual, coherence, freshness, importance, nonmonotonousness]
    """

    MIN_CANDIDATES_COUNT = 10
    CANDIDATE_FRACTION = 0.5

    def __init__(self, snapshot: PlayerSnapshot, ctx: GameContext, params: list[float]) -> None:
        super().__init__(snapshot, ctx)
        self.params = params
        self.preferences = snapshot.preferences
        self.score_engine = ScoreEngine(history=[])

        # sort memory by importance (desc)
        self.memory_bank.sort(key=lambda x: x.importance, reverse=True)

    def individual_score(self, item: Item) -> float:
        prefs = self.preferences
        if not prefs:
            return 0.0
        bonuses = [
            1 - prefs.index(s) / len(prefs)
            for s in item.subjects
            if s in prefs
        ]
        return float(sum(bonuses) / len(bonuses)) if bonuses else 0.0

    def _combine(self, pairs: list[tuple[Item, dict]], w: list[float]) -> Item | None:
        assert len(w) == 5, "Expected 5 weights"

        def score(entry: tuple[Item, dict]) -> float:
            s = entry[1]
            return (
                w[0] * float(s.get("individual", 0.0)) +
                w[1] * float(s.get("coherence", 0.0)) +
                w[2] * float(s.get("freshness", 0.0)) +
                w[3] * float(s.get("importance", 0.0)) +
                w[4] * float(s.get("nonmonotonousness", 0.0))
            )

        if not pairs:
            return None

        # sort candidates
        pairs.sort(key=score, reverse=True)


        if score(pairs[0]) < 0.1:
            return None
        return pairs[0][0]

    def propose_item(self, history: list[Item]) -> Item | None:
        if not self.memory_bank:
            return None

        self.score_engine.set_history(history)

        n = max(self.MIN_CANDIDATES_COUNT, int(len(self.memory_bank) * self.CANDIDATE_FRACTION))
        candidates = self.memory_bank[:n]

        ranked: list[tuple[Item, dict]] = []
        for item in candidates:
            new_hist = history + [item]
            self.score_engine.set_history(new_hist)
            i = len(new_hist) - 1

            scores = self.score_engine.compute(i, item)
            scores["individual"] = round(self.individual_score(item), 4)

            for k in ("coherence", "freshness", "importance", "nonmonotonousness"):
                scores[k] = round(scores[k], 4)

            ranked.append((item, scores))

        return self._combine(ranked, self.params)

