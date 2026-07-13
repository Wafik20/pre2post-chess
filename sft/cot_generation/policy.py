import random
from typing import List, Optional
import chess
from .generator import TreeNode
import math
import chess.engine
import chess.pgn
from tqdm import tqdm
from typing import List, Optional, Dict, Callable


def random_sampling_policy(board: chess.Board, legal_moves: List[chess.Move], seed: Optional[int] = None) -> Optional[chess.Move]:
    if not legal_moves:
        return None
    if seed is not None:
        random.seed(seed)
    return random.choice(legal_moves)


def _softmax(xs: List[float], temp: float = 1.0) -> List[float]:
    if not xs:
        return []
    scaled = [x / temp for x in xs]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def _entropy(probs: List[float], eps: float = 1e-12) -> float:
    h = 0.0
    for p in probs:
        p = max(p, eps)
        h -= p * math.log(p)
    return h


def _choose_width_from_entropy(h: float, max_width: int) -> int:
    """Linear mapping from entropy to branch width in [1, max_width]."""
    if max_width <= 1:
        return 1
    h_max = math.log(max_width)
    frac = 0.0 if h_max <= 0.0 else h / h_max
    return max(1, min(max_width, int(round(1 + frac * (max_width - 1)))))


def stockfish_sampling_policy(
    board: chess.Board,
    legal_moves: List[chess.Move],
    engine: chess.engine.SimpleEngine,
    time_limit: float = 0.01,
    temperature: float = 1.0,
    seed: Optional[int] = None,
    depth: Optional[int] = None,
    top_k: int = 6,
) -> Optional[chess.Move]:
    """
    Sampling policy using Stockfish evaluation scores.

    When `depth` is given, uses a single multipv depth-based analyse call to
    fetch up to `top_k` candidates (K_max).  The effective number of candidates
    actually considered is then shrunk to W = choose_width_from_entropy(H, K_max)
    where H is the entropy of the softmax distribution over all K_max scores.
    Finally, one move is sampled from the re-normalised distribution over the
    top-W candidates.

    When `depth` is None, falls back to the original behaviour: evaluates every
    legal move with a time-limited search and softmax-samples from all of them.
    """
    if not legal_moves:
        return None

    if depth is not None:
        # ── depth-based multipv path ──────────────────────────────────────── #
        k = min(top_k, len(legal_moves))
        info_list = engine.analyse(
            board,
            chess.engine.Limit(depth=depth),
            multipv=k,
        )
        candidates: List[chess.Move] = []
        scores: List[float] = []
        seen: set = set()
        for entry in info_list:
            pv = entry.get("pv", [])
            sc = entry.get("score")
            if not pv or sc is None:
                continue
            mv = pv[0]
            uci = mv.uci()
            if uci in seen:
                continue
            seen.add(uci)
            candidates.append(mv)
            scores.append(float(sc.pov(board.turn).score(mate_score=100000)))

        if not candidates:
            if seed is not None:
                random.seed(seed)
            return random.choice(legal_moves)

        if temperature <= 0:
            return candidates[0]  # top-1 by Stockfish score

        # Compute entropy over full K_max distribution, then shrink to width W
        probs_full = _softmax(scores, temp=temperature)
        h = _entropy(probs_full)
        width = _choose_width_from_entropy(h, len(candidates))

        # Keep only the top-W candidates (multipv returns them in score order)
        candidates = candidates[:width]
        scores = scores[:width]

    else:
        # ── time-based path (original behaviour) ─────────────────────────── #
        candidates = list(legal_moves)
        scores = []
        for mv in candidates:
            b = board.copy()
            b.push(mv)
            info = engine.analyse(b, chess.engine.Limit(time=time_limit))
            score = info["score"].pov(board.turn).score(mate_score=100000)
            scores.append(float(score))

        if temperature <= 0:
            idx = max(range(len(scores)), key=lambda i: scores[i])
            return candidates[idx]

    # Sample from the (possibly width-trimmed) candidate set
    probs = _softmax(scores, temp=temperature)
    if seed is not None:
        random.seed(seed)
    r = random.random()
    c = 0.0
    for mv, p in zip(candidates, probs):
        c += p
        if r <= c:
            return mv
    return candidates[-1]


def model_sampling_policy(
    board: chess.Board,
    legal_moves: List[chess.Move],
    model: Callable[[chess.Board, List[chess.Move]], List[float]],
    temperature: float = 1.0,
) -> Optional[chess.Move]:
    if not legal_moves:
        return None
    logits = model(board, legal_moves)
    if len(logits) != len(legal_moves):
        raise ValueError("Model must return one logit per legal move")
    if temperature <= 0:
        idx = max(range(len(logits)), key=lambda i: logits[i])
        return legal_moves[idx]
    scaled = [logit / temperature for logit in logits]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    probs = [x / z for x in exps]
    r = random.random()
    c = 0.0
    for mv, p in zip(legal_moves, probs):
        c += p
        if r <= c:
            return mv
    return legal_moves[-1]


def stockfish_eval(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    time_limit: float = 0.01,
    root_color: chess.Color = chess.WHITE,
) -> float:
    info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = info["score"].pov(root_color).score(mate_score=100000)
    return float(score)


