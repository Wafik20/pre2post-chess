"""Stockfish-based move-quality metrics.

For each chess position (FEN) we ask Stockfish to score every legal move,
turning the centipawn evaluations into a normalized rank:

    norm_rank(move) = (rank - 1) / (n_legal - 1)   ∈ [0, 1]

where `rank` is 1 for Stockfish's top move and `n_legal` for the worst.
Lower norm_rank = better move from Stockfish's perspective.

The model's reasoning emits many candidate moves at various tree depths.
For each rollout we report:

  Per-role (depth) means over LEGAL moves only:
    mean_candidate_norm_rank   depth 1 (player's first-move candidates)
    mean_opponent_norm_rank    depth 2 (imagined opponent replies)
    best_candidate_norm_rank   best of depth 1
    best_opponent_norm_rank    best of depth 2

  Per-role illegal rates:
    candidate_illegal_rate, opponent_illegal_rate

  Committed-move-quality components:
    selected_norm_rank_self        rank of the move the model committed to
    best_seen_norm_rank_self       best root candidate the model considered
    selection_rank_gap_self        selected minus best_seen
        (>0 = model considered a better move but didn't pick it.)

  Target presence:
    target_in_root_candidates   ground-truth move is a depth-1 candidate
    target_in_any_depth         ground-truth move appears anywhere in tree
"""
from __future__ import annotations

import multiprocessing as mp
from typing import Iterable

import chess
import chess.engine

from tree import build_tree_from_row


# =============================================================================
# Move-string parsing
# =============================================================================

def custom_move_to_uci(move: str | None) -> str | None:
    """Convert the model's custom move format to a UCI from-to string.

    The model emits moves like 'Pe2e4' (pawn e2→e4), 'Bb2xc3+' (bishop
    takes), 'Qh5f7#' (queen mate), or castling 'O-O' / 'O-O-O'. The piece
    letter and 'x' / '+' / '#' decorations are stripped to yield the
    plain from-to UCI (e.g. 'e2e4'). Castling is returned as a sentinel.
    """
    if move is None:
        return None
    m = str(move).strip().replace("+", "").replace("#", "")
    if m in {"O-O", "0-0"}:
        return "castle_kingside"
    if m in {"O-O-O", "0-0-0"}:
        return "castle_queenside"
    m = m.replace("x", "")

    promotion = ""
    if "=" in m:
        left, promo = m.split("=", 1)
        m, promotion = left, promo.lower()

    # strip leading piece letter (P/N/B/R/Q/K) — present in our format
    if len(m) >= 5 and m[0] in "PNBRQK":
        m = m[1:]

    return m + promotion


def resolve_move_to_legal_uci(board: chess.Board, move_str: str) -> tuple[str | None, bool]:
    """Resolve a move-string against the current board.

    Returns (uci_in_ranking_form, is_legal). is_legal is False when the
    move can't be parsed or isn't in board.legal_moves. The returned UCI
    can still be non-None for an illegal-but-parsed move (useful for
    debugging).
    """
    raw = custom_move_to_uci(move_str)
    if raw is None:
        return None, False
    if raw == "castle_kingside":
        raw = "e1g1" if board.turn == chess.WHITE else "e8g8"
    elif raw == "castle_queenside":
        raw = "e1c1" if board.turn == chess.WHITE else "e8c8"
    try:
        m = chess.Move.from_uci(raw)
    except Exception:
        return None, False
    if m in board.legal_moves:
        return raw, True
    # promotion fallback: model often omits =Q
    if len(raw) == 4:
        for promo in ("q", "r", "b", "n"):
            try:
                mp_ = chess.Move.from_uci(raw + promo)
            except Exception:
                continue
            if mp_ in board.legal_moves:
                return raw + promo, True
    return raw, False


def board_from_input(input_text: str) -> chess.Board | None:
    """Replay the move history given in the prompt to get the root board.

    The prompt format is:  <move> <move> ... <move> <T>   (with <T> as a
    delimiter before the model's reasoning). We split, drop <T>, and apply
    each move via custom_move_to_uci.
    """
    prefix = input_text.split("<T>")[0].strip()
    moves = prefix.split()
    board = chess.Board()
    for tok in moves:
        if not _push_custom_move(board, tok):
            return None
    return board


def _push_custom_move(board: chess.Board, move: str) -> bool:
    uci = custom_move_to_uci(move)
    if uci is None:
        return False
    if uci == "castle_kingside":
        uci = "e1g1" if board.turn == chess.WHITE else "e8g8"
    if uci == "castle_queenside":
        uci = "e1c1" if board.turn == chess.WHITE else "e8c8"
    try:
        m = chess.Move.from_uci(uci)
    except Exception:
        return False
    if m in board.legal_moves:
        board.push(m); return True
    if len(uci) == 4:
        for promo in ("q", "r", "b", "n"):
            try:
                m2 = chess.Move.from_uci(uci + promo)
            except Exception:
                continue
            if m2 in board.legal_moves:
                board.push(m2); return True
    return False


# =============================================================================
# Stockfish ranking of all legal moves at one position
# =============================================================================

def rank_all_legal_moves(engine: chess.engine.SimpleEngine,
                          board: chess.Board, depth: int) -> dict[str, dict]:
    """One multipv search → {uci: {rank, norm_rank, cp}} for every legal move.

    rank=1 is Stockfish's top move. cp is from side-to-move's perspective.
    """
    legal = list(board.legal_moves)
    if not legal:
        return {}
    info_list = engine.analyse(
        board, chess.engine.Limit(depth=depth), multipv=len(legal),
    )
    scored: list[tuple[str, int]] = []
    for info in info_list:
        pv = info.get("pv")
        if not pv:
            continue
        score = info["score"].pov(board.turn)
        if score.is_mate():
            mate = score.mate() or 0
            cp = 100_000 if mate > 0 else (-100_000 if mate < 0 else 0)
        else:
            cp = score.score(mate_score=100_000)
        scored.append((pv[0].uci(), cp))
    if not scored:
        return {}
    scored.sort(key=lambda x: x[1], reverse=True)
    denom = max(len(scored) - 1, 1)
    return {
        uci: dict(rank=i + 1, norm_rank=(i) / denom, cp=cp)
        for i, (uci, cp) in enumerate(scored)
    }


# =============================================================================
# FEN cache (one Stockfish call per unique position, in parallel)
# =============================================================================

def collect_fens_for_quality(rows: Iterable[dict]) -> set[str]:
    """All FENs needed to compute candidate / opponent / commit quality:

      - the root board of every rollout                 (depth-0)
      - the board after each legal depth-1 candidate    (depth-1; needed
        only when we want to evaluate depth-2 opponent replies under them)
    """
    fens: set[str] = set()
    for row in rows:
        root = board_from_input(row.get("input", ""))
        if root is None:
            continue
        fens.add(root.fen())
        # expand legal depth-1 children — those FENs power opponent quality
        tree, _, _ = build_tree_from_row(row)
        for move_str in tree.nodes[tree.root_id].children:
            uci, ok = resolve_move_to_legal_uci(root, move_str)
            if not ok:
                continue
            child = root.copy()
            child.push(chess.Move.from_uci(uci))
            fens.add(child.fen())
    return fens


# Each worker process keeps ONE Stockfish engine alive across many FENs.
# Spawning a fresh process per FEN would be 100× slower (UCI startup is
# dominant relative to a single multipv search at depth 8).
# If a call throws (Stockfish crash, broken pipe, timeout), we transparently
# respawn the engine and retry the same FEN once.
_WORKER_ENGINE: chess.engine.SimpleEngine | None = None
_WORKER_STOCKFISH: str = ""
_WORKER_DEPTH: int = 0


def _worker_init(stockfish_path: str, depth: int) -> None:
    global _WORKER_ENGINE, _WORKER_STOCKFISH, _WORKER_DEPTH
    _WORKER_STOCKFISH = stockfish_path
    _WORKER_DEPTH = depth
    _WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(stockfish_path)


def _respawn_engine() -> None:
    global _WORKER_ENGINE
    try:
        if _WORKER_ENGINE is not None:
            _WORKER_ENGINE.close()
    except Exception:
        pass
    _WORKER_ENGINE = chess.engine.SimpleEngine.popen_uci(_WORKER_STOCKFISH)


def _rank_one_fen(fen: str) -> tuple[str, dict]:
    global _WORKER_ENGINE
    for attempt in range(2):
        try:
            return fen, rank_all_legal_moves(_WORKER_ENGINE, chess.Board(fen),
                                              _WORKER_DEPTH)
        except Exception as e:
            if attempt == 0:
                _respawn_engine()
                continue
            return fen, {"_error": str(e)}
    return fen, {"_error": "unreachable"}


def _rank_serial(fens: list[str], depth: int, stockfish_path: str) -> dict:
    out: dict[str, dict] = {}
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    try:
        for fen in fens:
            for attempt in range(2):
                try:
                    out[fen] = rank_all_legal_moves(engine, chess.Board(fen), depth)
                    break
                except Exception as e:
                    if attempt == 0:
                        try: engine.close()
                        except Exception: pass
                        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
                        continue
                    out[fen] = {"_error": str(e)}
    finally:
        try: engine.close()
        except Exception: pass
    return out


def build_position_cache(fens: Iterable[str], depth: int, n_workers: int,
                          stockfish_path: str) -> dict[str, dict]:
    """Rank every FEN once via a pool of long-lived Stockfish processes.

    Any FEN whose first attempt errors gets a second shot with a fresh
    engine. Anything still failing after that is recorded as `_error`.
    Finally we retry the residual failures serially with a fresh engine
    — this catches cases where a worker's engine was wedged for many FENs
    in a row.
    """
    fens = list(fens)
    if n_workers <= 1:
        return _rank_serial(fens, depth, stockfish_path)
    cache: dict[str, dict] = {}
    with mp.Pool(processes=n_workers,
                  initializer=_worker_init,
                  initargs=(stockfish_path, depth)) as pool:
        for fen, result in pool.imap_unordered(_rank_one_fen, fens, chunksize=32):
            cache[fen] = result

    failed = [f for f, v in cache.items() if isinstance(v, dict) and "_error" in v]
    if failed:
        print(f"  retrying {len(failed)} failed FENs serially...", flush=True)
        cache.update(_rank_serial(failed, depth, stockfish_path))
    return cache


# =============================================================================
# Per-rollout quality metrics
# =============================================================================

def compute_role_quality(row: dict, cache: dict[str, dict]) -> dict:
    """Walk the tree and bucket each move into 'candidate' (depth 1) or
    'opponent' (depth 2). For each bucket we report mean and best
    norm_rank over LEGAL moves, plus illegal-move counts.
    """
    out = dict(
        n_candidate_legal=0, n_candidate_illegal=0,
        mean_candidate_norm_rank=float("nan"),
        best_candidate_norm_rank=float("nan"),
        n_opponent_legal=0, n_opponent_illegal=0,
        mean_opponent_norm_rank=float("nan"),
        best_opponent_norm_rank=float("nan"),
        candidate_illegal_rate=float("nan"),
        opponent_illegal_rate=float("nan"),
    )
    root = board_from_input(row.get("input", ""))
    if root is None:
        return out

    tree, _, _ = build_tree_from_row(row)

    cand_ranks: list[float] = []
    cand_illegal = 0
    opp_ranks: list[float] = []
    opp_illegal = 0

    # depth 1 children of root → candidates
    root_fen = root.fen()
    root_cache = cache.get(root_fen, {})
    for move_str, child_id in tree.nodes[tree.root_id].children.items():
        uci, ok = resolve_move_to_legal_uci(root, move_str)
        if not ok:
            cand_illegal += 1
            continue
        if root_cache and "_error" not in root_cache and uci in root_cache:
            cand_ranks.append(root_cache[uci]["norm_rank"])
        # walk into depth 2 to score opponent replies
        child_board = root.copy()
        child_board.push(chess.Move.from_uci(uci))
        child_cache = cache.get(child_board.fen(), {})
        for opp_str in tree.nodes[child_id].children:
            opp_uci, opp_ok = resolve_move_to_legal_uci(child_board, opp_str)
            if not opp_ok:
                opp_illegal += 1
                continue
            if child_cache and "_error" not in child_cache and opp_uci in child_cache:
                opp_ranks.append(child_cache[opp_uci]["norm_rank"])

    out["n_candidate_legal"]   = len(cand_ranks)
    out["n_candidate_illegal"] = cand_illegal
    out["n_opponent_legal"]    = len(opp_ranks)
    out["n_opponent_illegal"]  = opp_illegal
    if cand_ranks:
        out["mean_candidate_norm_rank"] = sum(cand_ranks) / len(cand_ranks)
        out["best_candidate_norm_rank"] = min(cand_ranks)
    if opp_ranks:
        out["mean_opponent_norm_rank"]  = sum(opp_ranks) / len(opp_ranks)
        out["best_opponent_norm_rank"]  = min(opp_ranks)
    cand_total = len(cand_ranks) + cand_illegal
    opp_total  = len(opp_ranks)  + opp_illegal
    if cand_total:
        out["candidate_illegal_rate"] = cand_illegal / cand_total
    if opp_total:
        out["opponent_illegal_rate"]  = opp_illegal / opp_total
    return out


def get_selected_move(row: dict) -> str | None:
    """The committed move = first entry of `extracted_moves` (the model's
    final answer extracted from the rollout)."""
    ex = row.get("extracted_moves")
    if isinstance(ex, str):
        ex = [s for s in ex.split(",") if s.strip()]
    if isinstance(ex, list) and ex:
        return ex[0]
    return None


def compute_commit_quality_components(row: dict, cache: dict[str, dict]) -> dict:
    """Where the model's committed move ranks at the root, and how much
    rank it left on the table relative to the best move it considered."""
    out = dict(
        selected_norm_rank_self=float("nan"),
        best_seen_norm_rank_self=float("nan"),
        selection_rank_gap_self=float("nan"),
    )
    root = board_from_input(row.get("input", ""))
    if root is None:
        return out
    root_cache = cache.get(root.fen(), {})
    if not root_cache or "_error" in root_cache:
        return out

    # Committed move
    sel = custom_move_to_uci(get_selected_move(row))
    if sel and sel in root_cache:
        out["selected_norm_rank_self"] = root_cache[sel]["norm_rank"]

    # Best of the depth-1 candidates that actually appeared in the tree
    tree, _, _ = build_tree_from_row(row)
    seen_ranks: list[float] = []
    for move_str in tree.nodes[tree.root_id].children:
        uci, ok = resolve_move_to_legal_uci(root, move_str)
        if ok and uci in root_cache:
            seen_ranks.append(root_cache[uci]["norm_rank"])
    if seen_ranks:
        out["best_seen_norm_rank_self"] = min(seen_ranks)
        if out["selected_norm_rank_self"] == out["selected_norm_rank_self"]:  # not NaN
            out["selection_rank_gap_self"] = (
                out["selected_norm_rank_self"] - out["best_seen_norm_rank_self"]
            )
    return out


# =============================================================================
# Target presence (no Stockfish — string match against ground truth)
# =============================================================================

def _target_first_uci(row: dict, root: chess.Board | None) -> str | None:
    target = row.get("target_moves")
    if isinstance(target, str):
        parts = [m.strip() for m in target.split(",") if m.strip()]
        first = parts[0] if parts else None
    elif isinstance(target, list) and target:
        first = target[0]
    else:
        first = None
    if not first:
        return None
    if root is None:
        uci = custom_move_to_uci(first)
        return None if uci in {"castle_kingside", "castle_queenside", None} else uci
    uci, _ok = resolve_move_to_legal_uci(root, first)
    return uci


def compute_target_presence(row: dict) -> dict:
    """Two flags per rollout:

      target_in_root_candidates: ground-truth first move is one of the
        depth-1 candidates the model wrote (strict, semantically correct).
      target_in_any_depth: target's UCI appears as a move label anywhere
        in the tree (loose — UCI-string match without checking board state).
    """
    out = dict(target_uci=None,
               target_in_root_candidates=float("nan"),
               target_in_any_depth=float("nan"))
    root = board_from_input(row.get("input", ""))
    t_uci = _target_first_uci(row, root)
    if t_uci is None:
        return out
    out["target_uci"] = t_uci

    tree, _, _ = build_tree_from_row(row)

    # strict
    root_candidates = set()
    for move_str in tree.nodes[tree.root_id].children:
        uci, ok = resolve_move_to_legal_uci(root, move_str) if root else (custom_move_to_uci(move_str), True)
        if ok and uci is not None and uci not in {"castle_kingside", "castle_queenside"}:
            root_candidates.add(uci)
    out["target_in_root_candidates"] = int(t_uci in root_candidates)

    # loose
    any_match = False
    for node in tree.nodes.values():
        if node.is_root():
            continue
        uci = custom_move_to_uci(node.move)
        if uci is None:
            continue
        if uci == "castle_kingside":
            uci = "e1g1" if (root and root.turn == chess.WHITE) else "e8g8"
        elif uci == "castle_queenside":
            uci = "e1c1" if (root and root.turn == chess.WHITE) else "e8c8"
        if uci == t_uci:
            any_match = True
            break
    out["target_in_any_depth"] = int(any_match)
    return out
