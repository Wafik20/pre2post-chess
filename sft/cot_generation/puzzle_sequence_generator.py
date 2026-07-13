#!/usr/bin/env python
"""
Puzzle sequence generator for post-SFT training.

Generates training data in the format:
    {pgn_context} <T> [thinking trace] </T> move2 <call_env> move3 move4 <call_env> ...

Where:
  - The thinking trace is a minimax tree-search CoT (using subtree_minimax_generator internals)
  - After </T>, model moves (1-indexed even positions) and env moves (odd) alternate
  - <call_env> is emitted after every model move

Depth modes for thinking trace rollouts:
  --depth_mode fixed   : always use --trajectory_depth (existing behaviour)
  --depth_mode random  : sample depth per trajectory from Lognormal(mu, sigma) clipped
                         to [depth_min, trajectory_depth], where mu is set so E[X]=depth_mean

Answer injection (--inject_answer):
  - Guarantees the ground-truth move appears among the candidates
  - Adds the full puzzle-solution line as one trajectory for that candidate
  - Generates num_trajectories_answer (default 2×num_trajectories) rollouts from it
    so it is well-represented in the tree, not isolated
"""

import sys
import io
import re
import json
import math
import random
import logging
import traceback
import multiprocessing
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chess
import chess.pgn
import chess.engine
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# ── Path setup ──────────────────────────────────────────────────────────────── #
parent_dir = Path(__file__).parent.parent        # sft/ (cot_generation package lives here)
shared_root = parent_dir.parent                   # repo root (holds shared llm_tokens/)
for _p in (shared_root, parent_dir):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from cot_generation.policy import (
    random_sampling_policy,
    stockfish_sampling_policy,
)
from cot_generation.subtree_minimax_generator import (
    move_to_lan,
    ModelSamplingPolicy,
    load_hf_model_and_tokenizer,
    TrajectoryTreeNode,
    annotate_tree_with_labels,
    annotate_tree_with_labels_parallel,
    compute_candidate_minimax_values,
    select_best_move_by_minimax,
    sample_candidates_with_policy,
    generate_single_trajectory,
    build_tree_from_trajectories,
    compute_trajectory_tree_stats,
    collect_leaf_values_from_trajectory_tree,
    collect_leaf_labels_from_trajectory_tree,
    dfs_policy_order,
    dfs_verifier_order,
    bfs_policy_order,
    bfs_verifier_order,
    order_moves_by_stockfish,
    trajectory_to_string_with_label,
    _init_sf_worker,
)

logger = logging.getLogger(__name__)


# ── Lognormal depth sampling ─────────────────────────────────────────────────── #

def sample_depth_lognormal(
    depth_min: int = 1,
    mean: float = 8.0,
    sigma: float = 0.4,
    rng: Optional[np.random.Generator] = None,
) -> int:
    """
    Sample a depth from Lognormal(mu, sigma) where mu = log(mean) - 0.5*sigma^2,
    so that E[X] ≈ mean.  Result is rounded and clipped to depth_min from below.

    Example – distribution that averages ~8:
        mu = log(8) - 0.5*(0.8)^2 ≈ 1.815
        E[lognormal(mu, 0.8)] = exp(mu + 0.5*0.8^2) = 8
    """
    if rng is None:
        rng = np.random.default_rng()
    mu = np.log(mean) - 0.5 * sigma ** 2
    raw = rng.lognormal(mean=mu, sigma=sigma)
    return max(depth_min, int(round(raw)))


def _sample_depth(args, rng: Optional[np.random.Generator]) -> int:
    """Return the actual depth for one trajectory rollout (fixed or random).

    In random mode the result is clipped to [depth_min, trajectory_depth] so
    trajectory_depth acts as a hard upper bound for the whole batch.
    """
    if getattr(args, "depth_mode", "fixed") == "random":
        raw = sample_depth_lognormal(
            depth_min=getattr(args, "depth_min", 1),
            mean=getattr(args, "depth_mean", 8.0),
            sigma=getattr(args, "depth_sigma", 0.8),
            rng=rng,
        )
        return min(raw, args.trajectory_depth)
    return args.trajectory_depth


def _sample_num_trajectories(
    args, rng: Optional[np.random.Generator], is_gt: bool = False
) -> int:
    """Return the number of trajectories for one candidate (fixed or random).

    In random mode, samples from Lognormal(mu, sigma) clipped to
    [num_trajectories_min, num_trajectories] (or num_trajectories_answer for GT).
    """
    n_max = args.num_trajectories_answer if is_gt else args.num_trajectories
    if getattr(args, "depth_mode", "fixed") == "random":
        n_min = getattr(args, "num_trajectories_min", 1)
        mean = getattr(args, "num_trajectories_mean", float(n_max))
        sigma = getattr(args, "num_trajectories_sigma", 0.8)
        if rng is None:
            rng = np.random.default_rng()
        mu = np.log(mean) - 0.5 * sigma ** 2
        raw = rng.lognormal(mean=mu, sigma=sigma)
        return max(n_min, min(int(round(raw)), n_max))
    return n_max


# ── Budget allocation helpers ────────────────────────────────────────────────── #

def _forced_move_value_cp(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    move: chess.Move,
    depth: int,
    mate_score: int = 100000,
    cache: Optional[dict] = None,
) -> Optional[float]:
    """Evaluate child position after forcing `move`, returned in parent-side POV (cp)."""
    if cache is not None:
        key = (board.fen(), move.uci(), depth)
        if key in cache:
            return cache[key]
    child = board.copy()
    child.push(move)
    info = engine.analyse(child, chess.engine.Limit(depth=depth))
    score = info.get("score")
    if score is None:
        result = None
    else:
        cp_white = score.white().score(mate_score=mate_score)
        # parent-side POV: White-to-move parent => cp_white; Black-to-move parent => -cp_white
        result = float(cp_white) if board.turn == chess.WHITE else float(-cp_white)
    if cache is not None:
        cache[key] = result
    return result


def _std_of(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def compute_uncertainty_weights(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    candidate_moves: List[chess.Move],
    work_depth: int,
    ref_depth: int,
    eps: float = 1e-6,
    cache: Optional[dict] = None,
) -> List[float]:
    """Return uncertainty-based weights for each candidate move.

    weight(a) = sqrt(U(s, a) + eps)
    U(s, a)   = |V_ref(s, a) - V_work(s, a)| / (std(V_ref values) + eps)

    Candidates with larger discrepancy between working-depth and reference-depth
    evaluations receive higher weights and are therefore allocated more budget.
    """
    vals_work: List[Optional[float]] = []
    vals_ref: List[Optional[float]] = []
    for move in candidate_moves:
        vals_work.append(_forced_move_value_cp(engine, board, move, work_depth, cache=cache))
        vals_ref.append(_forced_move_value_cp(engine, board, move, ref_depth, cache=cache))

    ref_non_null = [v for v in vals_ref if v is not None]
    scale = _std_of(ref_non_null) + eps

    weights: List[float] = []
    for vw, vr in zip(vals_work, vals_ref):
        if vw is None or vr is None:
            u = 0.0
        else:
            u = abs(vr - vw) / scale
        weights.append(math.sqrt(u + eps))
    return weights


def allocate_child_budgets_from_weights(
    weights: List[float],
    total_budget: int,
) -> List[int]:
    """Allocate `total_budget` units across candidates proportional to weights.

    Children with very low weight can receive 0 — they terminate at the current
    node and add their path as a leaf trajectory.  Rounding residuals are
    distributed to the highest-weight candidates first.
    """
    n = len(weights)
    if n == 0 or total_budget <= 0:
        return [0] * n

    s = sum(weights)
    if s <= 0:
        weights = [1.0] * n
        s = float(n)

    raw = [w / s * total_budget for w in weights]
    budgets = [int(math.floor(x)) for x in raw]   # allow 0; no forced minimum
    used = sum(budgets)

    # Distribute leftover units to highest-weight candidates first
    if used < total_budget:
        order = sorted(range(n), key=lambda i: weights[i], reverse=True)
        j = 0
        while used < total_budget:
            budgets[order[j % n]] += 1
            used += 1
            j += 1

    return budgets


def compute_difficulty_budget(
    rating: float,
    rating_min: float,
    rating_max: float,
    budget_min: int,
    budget_max: int,
) -> int:
    """Linear interpolation of total trajectory budget from puzzle Rating."""
    span = rating_max - rating_min
    normalized = max(0.0, min(1.0, (rating - rating_min) / span)) if span > 0 else 0.5
    return int(round(budget_min + (budget_max - budget_min) * normalized))


def _compute_total_budget(args, row_dict: dict, n_candidates: int) -> int:
    """Return the total trajectory budget for this puzzle."""
    if getattr(args, "budget_mode", "fixed") == "difficulty_adaptive":
        rating = row_dict.get("Rating", None)
        if rating is not None and not (isinstance(rating, float) and math.isnan(rating)):
            return compute_difficulty_budget(
                float(rating),
                getattr(args, "rating_min", 500.0),
                getattr(args, "rating_max", 3000.0),
                getattr(args, "budget_min", args.num_trajectories * n_candidates),
                getattr(args, "budget_max", 2 * args.num_trajectories * n_candidates),
            )
    return args.num_trajectories * n_candidates


def _sf_softmax(scores: List[float], temp: float) -> List[float]:
    if not scores:
        return []
    scaled = [s / temp for s in scores]
    m = max(scaled)
    exps = [math.exp(x - m) for x in scaled]
    z = sum(exps)
    return [e / z for e in exps]


def _sf_entropy(probs: List[float], eps: float = 1e-12) -> float:
    h = 0.0
    for p in probs:
        p = max(p, eps)
        h -= p * math.log(p)
    return h


def _sf_choose_width(h: float, max_width: int) -> int:
    """Linear mapping from entropy to candidate width in [1, max_width]."""
    if max_width <= 1:
        return 1
    h_max = math.log(max_width)
    frac = 0.0 if h_max <= 0.0 else h / h_max
    return max(1, min(max_width, int(round(1 + frac * (max_width - 1)))))


# ── Puzzle solution trajectory ───────────────────────────────────────────────── #

def build_puzzle_solution_trajectory(
    board: chess.Board,
    moves_list: List[str],
    max_depth: Optional[int] = None,
) -> Optional[List[chess.Move]]:
    """
    Parse the puzzle solution into chess.Move objects for tree injection.

    `board` must be the position AFTER moves_list[0] has been applied.
    The returned trajectory starts with moves_list[1] (the model's first move)
    and continues for the full puzzle solution.

    `max_depth` limits the trajectory length when set; pass None (the default)
    to include the complete solution regardless of tree depth budget.

    Returns None if parsing fails entirely.
    """
    remaining = moves_list[1:] if max_depth is None else moves_list[1 : 1 + max_depth]
    if not remaining:
        return None

    trajectory: List[chess.Move] = []
    cur = board.copy()
    for san in remaining:
        if cur.is_game_over():
            break
        try:
            mv = cur.parse_san(san)
        except ValueError:
            try:
                mv = chess.Move.from_uci(san)
                if mv not in cur.legal_moves:
                    break
            except ValueError:
                break
        trajectory.append(mv)
        cur.push(mv)

    return trajectory or None


# ── Result label reformatter ─────────────────────────────────────────────────── #

def _reformat_result_labels(s: str) -> str:
    """
    Convert subtree_minimax_generator's result tags to the puzzle format.
    e.g.  <result>+1</result>  →  <verify> <+1>
          <result>0</result>   →  <verify> <0>
          <result>-1</result>  →  <verify> <-1>
    """
    return re.sub(r"<result>([^<]+)</result>", r"<verify> <\1>", s)


def _minimax_label_str(v: int) -> str:
    """Convert minimax value (+1, 0, -1) to verify label string."""
    if v > 0:
        return "+1"
    if v < 0:
        return "-1"
    return "0"


def _strip_result_labels(s: str) -> str:
    """Remove all <verify> <value> tokens from a CoT string.

    Covers minimax labels (+1, 0, -1) and stratified CP-score labels
    (+3, +2, +0.5, -0.5, -2, -3, etc.).  The pattern matches:
      <verify>  (optional whitespace)  <anything-that-is-not->
    so it cannot accidentally match other structural tokens like <T>, <sep>.
    """
    return re.sub(r"\s*<verify>\s*<[^>]+>", "", s)


def build_candidate_cot_traversal(
    board: chess.Board,
    tree_root: "TrajectoryTreeNode",
    minimax_values: Dict[chess.Move, int],
    method: str,
) -> str:
    """
    Build a CoT traversal where each top-level candidate's full subtree is
    explored and then summarised with a single minimax-based <verify> label.

    Format:
      cand1 [subtree1] <verify> <+1> <sep> cand2 [subtree2] <verify> <-1> ...

    Candidate ordering:
      policy  methods  → generation_order (order sampled by policy)
      verifier methods → descending stockfish_value (best candidate first)

    Internal subtree traversal uses the matching DFS/BFS function with no
    leaf-level result labels (the candidate-level <verify> summarises them).
    """
    # Choose candidate ordering
    if "verifier" in method:
        ordered_candidates = sorted(
            tree_root.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )
    else:
        ordered_candidates = sorted(
            tree_root.children.items(),
            key=lambda x: x[1].generation_order,
        )

    parts: List[str] = []
    for cand_move, cand_node in ordered_candidates:
        cand_lan = move_to_lan(board, cand_move)
        cand_board = board.copy()
        cand_board.push(cand_move)

        # Build subtree traversal (no leaf labels — candidate <verify> summarises)
        if cand_node.children:
            if method == "dfs_policy":
                sub_trav = dfs_policy_order(
                    cand_board, cand_node,
                    include_directions=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            elif method == "dfs_verifier":
                sub_trav = dfs_verifier_order(
                    cand_board, cand_node,
                    include_directions=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            elif method == "bfs_policy":
                sub_trav = bfs_policy_order(
                    cand_board, cand_node,
                    include_level_markers=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            elif method == "bfs_verifier":
                sub_trav = bfs_verifier_order(
                    cand_board, cand_node,
                    include_level_markers=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            else:
                sub_trav = ""
        else:
            sub_trav = ""

        minimax_val = minimax_values.get(cand_move, 0)
        verify = f"<verify> <{_minimax_label_str(minimax_val)}>"

        token_parts = [cand_lan]
        if sub_trav:
            token_parts.append(sub_trav)
        token_parts.append(verify)
        parts.append(" ".join(token_parts))

    return " <sep> ".join(parts)


# ── Puzzle sequence (the part after </T>) ────────────────────────────────────── #

def generate_puzzle_sequence(
    board: chess.Board,
    moves_list: List[str],
    call_env_token: str = "<call_env>",
    max_puzzle_moves: int = 5,
) -> str:
    """
    Build the interleaved model/env sequence that appears after </T>.

    Indexing convention (1-based):
        moves_list[0]  already applied to `board` (opponent's trigger move)
        moves_list[1]  model move  → emitted, then <call_env>
        moves_list[2]  env move    → emitted
        moves_list[3]  model move  → emitted, then <call_env>
        …              (up to max_puzzle_moves moves from moves_list[1:])

    Output example (max_puzzle_moves=5, 5 moves available):
        move2 <call_env> move3 move4 <call_env> move5 move6 <call_env> move7

    <call_env> is always emitted after a model move, even if it is the last
    move in the sequence (signalling the model expects an env response).
    All moves are rendered in LAN format.
    """
    remaining = moves_list[1 : 1 + max_puzzle_moves]
    if not remaining:
        return ""

    tokens: List[str] = []
    cur = board.copy()

    for i, san in enumerate(remaining):
        if cur.is_game_over():
            break
        try:
            mv = cur.parse_san(san)
        except ValueError:
            try:
                mv = chess.Move.from_uci(san)
                if mv not in cur.legal_moves:
                    break
            except ValueError:
                break

        try:
            lan = move_to_lan(cur, mv)
        except Exception:
            lan = mv.uci()

        tokens.append(lan)
        cur.push(mv)

        is_model_move = (i % 2 == 0)  # i=0 → moves_list[1], always a model move
        if is_model_move:
            tokens.append(call_env_token)

    return " ".join(tokens)


# ── Full CoT builder ─────────────────────────────────────────────────────────── #

def generate_puzzle_cot_with_sequence(
    traversal_str: str,
    board: chess.Board,
    moves_list: List[str],
    target_move: Optional[chess.Move] = None,
    pgn_context: str = "",
    call_env_token: str = "<call_env>",
    max_puzzle_moves: int = 5,
) -> Dict[str, str]:
    """
    Combine the thinking trace and the puzzle sequence into two output formats:

    Multi-step:   {pgn_context} <T> {traversal_str} </T> move2 <call_env> move3 …
    First-move:   {pgn_context} <T> {traversal_str} </T> move2

    The multi-step sequence is driven by moves_list (the puzzle solution).
    The first-move format uses target_move, which should be the ground-truth
    move (gt_move) passed from the caller.

    Returns a dict with:
        cot_format                             – multi-step, no pgn prefix
        cot_format_with_context                – multi-step, pgn prepended
        cot_format_no_labels                   – multi-step, no labels, no pgn prefix
        cot_format_with_context_no_labels      – multi-step, no labels, pgn prepended
        cot_format_first_move                  – first-move only, no pgn prefix
        cot_format_first_move_with_context     – first-move only, pgn prepended
        cot_format_first_move_no_labels        – first-move only, no labels, no pgn prefix
        cot_format_first_move_with_context_no_labels – first-move only, no labels, pgn prepended
        puzzle_sequence                        – raw move2 <call_env> … string
        first_move_lan                         – LAN of the target move
    """
    cot_inner = f"<T> {traversal_str} </T>" if traversal_str else "</T>"
    cot_inner_nl = f"<T> {_strip_result_labels(traversal_str)} </T>" if traversal_str else "</T>"

    # ── Multi-step sequence ────────────────────────────────────────────────── #
    puzzle_seq = generate_puzzle_sequence(
        board=board,
        moves_list=moves_list,
        call_env_token=call_env_token,
        max_puzzle_moves=max_puzzle_moves,
    )
    cot_format = f"{cot_inner} {puzzle_seq}".strip() if puzzle_seq else cot_inner
    cot_format_with_context = (
        f"{pgn_context} {cot_format}".strip() if pgn_context else cot_format
    )
    cot_format_nl = f"{cot_inner_nl} {puzzle_seq}".strip() if puzzle_seq else cot_inner_nl
    cot_format_with_context_nl = (
        f"{pgn_context} {cot_format_nl}".strip() if pgn_context else cot_format_nl
    )

    # ── First-move-only format ─────────────────────────────────────────────── #
    first_move_lan = ""
    if target_move is not None:
        try:
            first_move_lan = move_to_lan(board, target_move)
        except Exception:
            first_move_lan = target_move.uci()

    cot_first_move = (
        f"{cot_inner} {first_move_lan}".strip() if first_move_lan else cot_inner
    )
    cot_first_move_with_context = (
        f"{pgn_context} {cot_first_move}".strip() if pgn_context else cot_first_move
    )
    cot_first_move_nl = (
        f"{cot_inner_nl} {first_move_lan}".strip() if first_move_lan else cot_inner_nl
    )
    cot_first_move_with_context_nl = (
        f"{pgn_context} {cot_first_move_nl}".strip() if pgn_context else cot_first_move_nl
    )

    return {
        "cot_format": cot_format,
        "cot_format_with_context": cot_format_with_context,
        "cot_format_no_labels": cot_format_nl,
        "cot_format_with_context_no_labels": cot_format_with_context_nl,
        "cot_format_first_move": cot_first_move,
        "cot_format_first_move_with_context": cot_first_move_with_context,
        "cot_format_first_move_no_labels": cot_first_move_nl,
        "cot_format_first_move_with_context_no_labels": cot_first_move_with_context_nl,
        "puzzle_sequence": puzzle_seq,
        "first_move_lan": first_move_lan,
    }


# ── Extra output format helpers ──────────────────────────────────────────────── #

def cp_score_to_stratified_label(cp: float) -> str:
    """Map a Stockfish centipawn score to a stratified label string.

    Thresholds (from the root player's perspective):
        cp > 300  → "+3"
        cp > 200  → "+2"
        cp > 100  → "+1"
        cp > 50   → "+0.5"
        cp > -50  → "0"
        cp > -100 → "-0.5"
        cp > -200 → "-1"
        cp > -300 → "-2"
        otherwise → "-3"
    """
    if cp > 300:
        return "+3"
    if cp > 200:
        return "+2"
    if cp > 100:
        return "+1"
    if cp > 50:
        return "+0.5"
    if cp > -50:
        return "0"
    if cp > -100:
        return "-0.5"
    if cp > -200:
        return "-1"
    if cp > -300:
        return "-2"
    return "-3"


def _trajectory_to_string_stratified(
    root_board: chess.Board,
    trajectory: List[chess.Move],
    cp: float,
) -> str:
    """Like trajectory_to_string_with_label but uses stratified CP-score <result> label."""
    tokens: List[str] = []
    board = root_board.copy()
    for move in trajectory:
        try:
            tokens.append(move_to_lan(board, move))
        except Exception:
            tokens.append(move.uci())
        board.push(move)
    tokens.append(f"<result>{cp_score_to_stratified_label(cp)}</result>")
    return " ".join(tokens)


def build_candidate_cot_traversal_stratified(
    board: chess.Board,
    tree_root: "TrajectoryTreeNode",
    method: str,
) -> str:
    """Like build_candidate_cot_traversal but uses stratified CP-score <verify> labels.

    Instead of the minimax <verify> <+1>/<0>/<-1>, each candidate gets
    <verify> <+3>/<+2>/…/<-3> derived from its direct Stockfish CP value via
    cp_score_to_stratified_label.

    Candidate ordering follows the same rules as build_candidate_cot_traversal.
    """
    if "verifier" in method:
        ordered_candidates = sorted(
            tree_root.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )
    else:
        ordered_candidates = sorted(
            tree_root.children.items(),
            key=lambda x: x[1].generation_order,
        )

    parts: List[str] = []
    for cand_move, cand_node in ordered_candidates:
        cand_lan = move_to_lan(board, cand_move)
        cand_board = board.copy()
        cand_board.push(cand_move)

        if cand_node.children:
            if method == "dfs_policy":
                sub_trav = dfs_policy_order(
                    cand_board, cand_node,
                    include_directions=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            elif method == "dfs_verifier":
                sub_trav = dfs_verifier_order(
                    cand_board, cand_node,
                    include_directions=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            elif method == "bfs_policy":
                sub_trav = bfs_policy_order(
                    cand_board, cand_node,
                    include_level_markers=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            elif method == "bfs_verifier":
                sub_trav = bfs_verifier_order(
                    cand_board, cand_node,
                    include_level_markers=False,
                    include_subtree_sep=False,
                    include_result_labels=False,
                )
            else:
                sub_trav = ""
        else:
            sub_trav = ""

        cp = cand_node.stockfish_value if cand_node.stockfish_value is not None else 0.0
        verify = f"<verify> <{cp_score_to_stratified_label(cp)}>"

        token_parts = [cand_lan]
        if sub_trav:
            token_parts.append(sub_trav)
        token_parts.append(verify)
        parts.append(" ".join(token_parts))

    return " <sep> ".join(parts)


def build_successful_only_cot_inner(
    board: chess.Board,
    moves_list: List[str],
    max_depth: int,
) -> str:
    """Render the puzzle solution continuation as a LAN sequence for the CoT.

    Parses moves_list[1:1+max_depth] on board and renders them as LAN,
    producing the ground-truth line used inside <T>…</T>.
    Example output: "Qh7+ Kxh7 Rh1+ Kg8 Rh8+"
    """
    remaining = moves_list[1 : 1 + max_depth]
    if not remaining:
        return ""

    cur = board.copy()
    move_strs: List[str] = []
    for san in remaining:
        if cur.is_game_over():
            break
        try:
            mv = cur.parse_san(san)
        except ValueError:
            try:
                mv = chess.Move.from_uci(san)
                if mv not in cur.legal_moves:
                    break
            except ValueError:
                break
        try:
            move_strs.append(move_to_lan(cur, mv))
        except Exception:
            move_strs.append(mv.uci())
        cur.push(mv)

    return " ".join(move_strs)


def build_monotonic_improvement_cot_inner(
    board: chess.Board,
    all_trajectories: List[List[chess.Move]],
    tree_root: "TrajectoryTreeNode",
) -> str:
    """Serialize all trajectories in ascending endpoint Stockfish value order (worst→best).

    Each trajectory is rendered as space-separated LAN moves.
    Trajectories are joined by <sep>, producing a monotonically improving sequence.
    Example: "Bxf7+ Nf7 <sep> Ng5 Nf6 Bd3 <sep> Qh7+ Kxh7 Rh1+"
    """
    if not all_trajectories:
        return ""

    def _endpoint_score(traj: List[chess.Move]) -> float:
        cur = tree_root
        for mv in traj:
            if mv in cur.children:
                cur = cur.children[mv]
        return cur.stockfish_value if cur.stockfish_value is not None else 0.0

    sorted_trajs = sorted(all_trajectories, key=_endpoint_score)

    parts: List[str] = []
    for traj in sorted_trajs:
        cur = board.copy()
        move_strs: List[str] = []
        for mv in traj:
            try:
                move_strs.append(move_to_lan(cur, mv))
            except Exception:
                move_strs.append(mv.uci())
            cur.push(mv)
        if move_strs:
            parts.append(" ".join(move_strs))

    return " <sep> ".join(parts)


# ── Sampling policy factory ──────────────────────────────────────────────────── #

def create_sampling_policy(args, device):
    """Build and return the callable sampling policy."""
    if args.sampling_policy == "random":
        seed = getattr(args, "seed", 42)

        def _rand(board, legal_moves):
            return random_sampling_policy(board, legal_moves, seed=seed)

        return _rand

    if args.sampling_policy == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)
        _sf_depth = getattr(args, "stockfish_depth_work", None)
        _sf_top_k = getattr(args, "stockfish_top_k", 6)

        def _sf(board, legal_moves):
            return stockfish_sampling_policy(
                board,
                legal_moves,
                engine,
                time_limit=args.stockfish_time,
                temperature=args.temperature,
                seed=getattr(args, "seed", 42),
                depth=_sf_depth,
                top_k=_sf_top_k,
            )

        # Expose the engine so the caller can quit() it on exit.
        _sf._engine = engine
        return _sf

    if args.sampling_policy == "hf_model":
        model, tokenizer = load_hf_model_and_tokenizer(
            args.model_path, str(device), config_path=args.config_path
        )
        return ModelSamplingPolicy(
            model, tokenizer, device, temperature=args.temperature
        )

    raise ValueError(f"Unknown sampling policy: {args.sampling_policy}")


# ── Batched trajectory generation ────────────────────────────────────────────── #

def _parse_move_sequence(
    output: str,
    start_board: chess.Board,
    max_depth: int,
    parse_move_fn,
) -> List[chess.Move]:
    """Parse a sequence of chess moves from raw model output text.

    Iterates through whitespace-separated tokens, skipping PGN move numbers
    (e.g. "1.", "12.", "..."), and tries to parse each remaining token as a
    legal move on the evolving board.  Stops at the first unrecognised token,
    game-over, or max_depth.
    """
    moves: List[chess.Move] = []
    board = start_board.copy()
    for token in output.split():
        if len(moves) >= max_depth or board.is_game_over():
            break
        if re.match(r'^\d+\.+$', token) or token == '...':
            continue
        legal = list(board.legal_moves)
        if not legal:
            break
        mv = parse_move_fn(token, board, legal)
        if mv is not None:
            moves.append(mv)
            board.push(mv)
        else:
            break
    return moves


def generate_trajectories_batched(
    root_board: chess.Board,
    candidate_moves: List[chess.Move],
    sampling_policy,
    n_traj_per_cand: List[int],
    depths_per_slot: List[List[int]],
    batch_size: int = 32,
    tokens_per_move: int = 6,
) -> Tuple[List[List[chess.Move]], Dict[chess.Move, List[List[chess.Move]]]]:
    """Generate all trajectory rollouts in a single batched model call.

    For each (candidate, trajectory) slot the board state after the candidate
    move is prepared and passed to the model.  A single call to
    generate_moves_batch is made with max_new_tokens set to cover the deepest
    slot in the batch (max_cont_depth * tokens_per_move).  Moves are then
    parsed greedily from each output string; parsing stops at the first
    unrecognisable token, game-over, or the slot's individual depth limit.

    Batch slots are padded to a multiple of 16 for GPU efficiency.  Only
    ModelSamplingPolicy instances support this path.

    Args:
        root_board:       Position before any candidate move.
        candidate_moves:  Ordered list of candidates.
        sampling_policy:  ModelSamplingPolicy (must have .model, .tokenizer,
                          .device, .temperature, .seed, _board_to_state,
                          _parse_move_from_output).
        n_traj_per_cand:  n_traj_per_cand[i] = number of rollouts for
                          candidate_moves[i].
        depths_per_slot:  depths_per_slot[i][j] = total trajectory depth
                          (including the candidate move) for candidate i,
                          rollout j.  Continuation depth = depth - 1.
        batch_size:       Sub-batch size forwarded to generate_moves_batch.
        tokens_per_move:  Token budget per move for max_new_tokens calculation.
    """
    from evaluation.inference import generate_moves_batch

    # Build flat slot list: (cand_idx, start_board_after_cand, cont_depth)
    slots: List[Tuple[int, chess.Board, int]] = []
    for ci, cand_move in enumerate(candidate_moves):
        cand_board = root_board.copy()
        cand_board.push(cand_move)
        for ti in range(n_traj_per_cand[ci]):
            cont_depth = max(0, depths_per_slot[ci][ti] - 1)
            slots.append((ci, cand_board.copy(), cont_depth))

    if not slots:
        return [], {mv: [] for mv in candidate_moves}

    n_slots = len(slots)
    max_cont_depth = max(d for _, _, d in slots)
    max_new_tokens = max(1, max_cont_depth * tokens_per_move)

    states = [sampling_policy._board_to_state(b) for _, b, _ in slots]

    # Pad to nearest multiple of 16
    padded_n = ((n_slots + 15) // 16) * 16
    states_padded = states + [states[0]] * (padded_n - n_slots)

    try:
        full_outputs, _ = generate_moves_batch(
            model=sampling_policy.model,
            tokenizer=sampling_policy.tokenizer,
            states=states_padded,
            device=sampling_policy.device,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=sampling_policy.temperature,
            top_k=None,
            seed=sampling_policy.seed,
        )
    except Exception as e:
        print(f"Warning: batched trajectory generation failed ({e}); returning empty.")
        traceback.print_exc()
        return [], {mv: [] for mv in candidate_moves}

    all_trajectories: List[List[chess.Move]] = []
    trajectories_per_candidate: Dict[chess.Move, List[List[chess.Move]]] = {
        mv: [] for mv in candidate_moves
    }

    for si, (ci, start_board, cont_depth) in enumerate(slots):
        cand_move = candidate_moves[ci]
        output = full_outputs[si] if si < len(full_outputs) else ""
        continuation = _parse_move_sequence(
            output, start_board, cont_depth,
            sampling_policy._parse_move_from_output,
        )
        full_traj = [cand_move] + continuation
        all_trajectories.append(full_traj)
        trajectories_per_candidate[cand_move].append(full_traj)

    return all_trajectories, trajectories_per_candidate


# ── Multi-position batched generation ────────────────────────────────────────── #

def prepare_position_slots(
    idx: int,
    row_dict: dict,
    args,
    sampling_policy,
    stockfish_engine: chess.engine.SimpleEngine,
    has_ground_truth: bool,
    moves_col: str,
    rng: Optional[np.random.Generator] = None,
) -> Optional[dict]:
    """Phase 1: parse PGN, generate candidates, compute slot specs.

    Returns a spec dict with all data needed for batched generation and
    finalization, or None if the position should be skipped.  Only used for
    ModelSamplingPolicy; non-model policies call process_one_position_puzzle.
    """
    pgn_str = row_dict.get("ctx")
    if pgn_str is np.nan or pgn_str is None:
        print(f"Skipping position {idx}: no PGN string")
        return None

    actual_pgn = pgn_str
    moves_list: List[str] = []
    ground_truth_move_san: Optional[str] = None

    if has_ground_truth and pd.notna(row_dict.get(moves_col)):
        moves_str = str(row_dict[moves_col]).strip()
        moves_list = moves_str.split()
        if len(moves_list) < 2:
            print(f"Skipping position {idx}: Moves column has fewer than 2 moves")
            return None

        ground_truth_move_san = moves_list[1]

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_str))
            board_tmp = game.end().board() if game is not None else chess.Board()
            opp_mv = board_tmp.parse_san(moves_list[0])
            board_tmp.push(opp_mv)
            new_game = chess.pgn.Game.from_board(board_tmp)
            exporter = chess.pgn.StringExporter(
                headers=False, variations=False, comments=False
            )
            new_pgn = re.sub(
                r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", new_game.accept(exporter)
            ).strip()
            actual_pgn = new_pgn
        except Exception as e:
            print(f"Position {idx}: failed to apply first move – {e}")
            return None

    try:
        game = chess.pgn.read_game(io.StringIO(actual_pgn))
        board = game.end().board() if game is not None else chess.Board()
        root_color = board.turn
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            print(f"Position {idx}: no legal moves (game over)")
            return None

        gt_move: Optional[chess.Move] = None
        if ground_truth_move_san:
            try:
                gt_move = board.parse_san(ground_truth_move_san)
            except ValueError:
                try:
                    mv = chess.Move.from_uci(ground_truth_move_san)
                    if mv in legal_moves:
                        gt_move = mv
                except ValueError:
                    pass

        if args.candidate_selection == "policy":
            candidate_moves = sample_candidates_with_policy(
                board=board,
                sampling_policy=sampling_policy,
                num_candidates=args.num_candidates,
            )
        else:
            scored = order_moves_by_stockfish(
                board=board,
                legal_moves=legal_moves,
                engine=stockfish_engine,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
            )
            candidate_moves = [mv for mv, _ in scored[: args.num_candidates]]

        if not candidate_moves:
            print(f"Position {idx}: no candidates generated")
            return None

        gt_injected = False
        if args.inject_answer and gt_move is not None:
            if gt_move not in candidate_moves:
                replace_idx = (
                    int(rng.integers(0, len(candidate_moves)))
                    if rng is not None
                    else random.randrange(len(candidate_moves))
                )
                candidate_moves[replace_idx] = gt_move
            gt_injected = True

        # GT solution trajectory (excluded from model generation slots)
        gt_solution_traj: Optional[List[chess.Move]] = None
        if (
            gt_injected
            and gt_move is not None
            and len(moves_list) >= 2
            and args.trajectory_depth > 0
        ):
            gt_solution_traj = build_puzzle_solution_trajectory(
                board=board,
                moves_list=moves_list,
                # max_depth omitted: always include the full puzzle solution.
            )

        # For stockfish policy: shuffle candidate order so the tree's generation_order
        # is not biased toward Stockfish score ranking.
        if args.sampling_policy == "stockfish" and len(candidate_moves) > 1:
            if rng is not None:
                rng.shuffle(candidate_moves)
            else:
                random.shuffle(candidate_moves)

        # Compute uncertainty-based budget allocation across candidates
        _prep_eval_cache: dict = {}
        _unc_weights = compute_uncertainty_weights(
            stockfish_engine, board, candidate_moves,
            getattr(args, "stockfish_depth_work", 10),
            getattr(args, "stockfish_depth_ref", 18),
            cache=_prep_eval_cache,
        )
        _total_budget = _compute_total_budget(args, row_dict, len(candidate_moves))
        _n_traj_base = allocate_child_budgets_from_weights(_unc_weights, _total_budget)

        # Pre-sample n_traj and depths for every (candidate, rollout) slot
        # GT solution trajectory is injected on top in finalize (no budget deduction).
        n_traj_per_cand: List[int] = []
        depths_per_slot: List[List[int]] = []
        for i in range(len(candidate_moves)):
            if args.trajectory_depth <= 0:
                n_traj_per_cand.append(0)
                depths_per_slot.append([])
                continue
            n_traj = _n_traj_base[i]
            n_traj_per_cand.append(n_traj)
            depths_per_slot.append(
                [_sample_depth(args, rng) for _ in range(n_traj)]
            )

        return {
            "idx": idx,
            "board": board,
            "root_color": root_color,
            "candidate_moves": candidate_moves,
            "n_traj_per_cand": n_traj_per_cand,
            "depths_per_slot": depths_per_slot,
            "gt_move": gt_move,
            "gt_injected": gt_injected,
            "gt_solution_traj": gt_solution_traj,
            "ground_truth_move_san": ground_truth_move_san,
            "moves_list": moves_list,
            "actual_pgn": actual_pgn,
            "pgn_str": pgn_str,
        }

    except Exception as e:
        print(f"Error preparing position {idx}: {e}")
        traceback.print_exc()
        return None


def generate_trajectories_multi_position_batched(
    position_specs: List[dict],
    sampling_policy,
    batch_size: int = 32,
    tokens_per_move: int = 6,
) -> List[Dict[chess.Move, List[List[chess.Move]]]]:
    """Generate trajectories for multiple positions in one padded model call.

    Pools all (candidate × trajectory) slots from every position into a single
    flat list, pads to a multiple of 16 for GPU efficiency, executes one
    generate_moves_batch call, then distributes parsed move sequences back to
    each position.

    Args:
        position_specs:  List of dicts from prepare_position_slots.
        sampling_policy: ModelSamplingPolicy instance.
        batch_size:      Sub-batch size forwarded to generate_moves_batch.
        tokens_per_move: Token budget per move for max_new_tokens.

    Returns:
        List of gen_per_cand dicts (one per input spec), each mapping
        chess.Move → List[List[chess.Move]] of model-generated rollouts.
        Positions with no generation slots get empty dicts.
    """
    from evaluation.inference import generate_moves_batch

    # Build flat slot list: (pos_idx, cand_idx, start_board_after_cand, cont_depth)
    flat_slots: List[Tuple[int, int, chess.Board, int]] = []
    pos_slot_offsets: List[int] = []  # flat_slots start index per position

    for pi, spec in enumerate(position_specs):
        pos_slot_offsets.append(len(flat_slots))
        board = spec["board"]
        candidate_moves = spec["candidate_moves"]
        n_traj_per_cand = spec["n_traj_per_cand"]
        depths_per_slot = spec["depths_per_slot"]
        for ci, cand_move in enumerate(candidate_moves):
            cand_board = board.copy()
            cand_board.push(cand_move)
            for ti in range(n_traj_per_cand[ci]):
                cont_depth = max(0, depths_per_slot[ci][ti] - 1)
                flat_slots.append((pi, ci, cand_board.copy(), cont_depth))

    # Initialise empty result dicts
    results: List[Dict[chess.Move, List[List[chess.Move]]]] = [
        {mv: [] for mv in spec["candidate_moves"]} for spec in position_specs
    ]

    if not flat_slots:
        return results

    n_slots = len(flat_slots)

    # Sort slots by cont_depth so each chunk shares a similar token budget,
    # avoiding over-generation for shallow slots forced to match a deep one.
    sorted_order = sorted(range(n_slots), key=lambda i: flat_slots[i][3])
    sorted_slots = [flat_slots[i] for i in sorted_order]
    all_states = [sampling_policy._board_to_state(b) for _, _, b, _ in sorted_slots]

    # Process in depth-sorted chunks, each with its own max_new_tokens.
    outputs_sorted: List[str] = [""] * n_slots
    try:
        for chunk_start in range(0, n_slots, batch_size):
            chunk_end = min(chunk_start + batch_size, n_slots)
            chunk_slots = sorted_slots[chunk_start:chunk_end]
            chunk_states = all_states[chunk_start:chunk_end]

            chunk_max_depth = max(d for _, _, _, d in chunk_slots)
            chunk_max_new_tokens = max(1, chunk_max_depth * tokens_per_move)

            # Pad chunk to nearest multiple of 16
            chunk_n = len(chunk_states)
            padded_n = ((chunk_n + 15) // 16) * 16
            chunk_states_padded = chunk_states + [chunk_states[0]] * (padded_n - chunk_n)

            chunk_outputs, _ = generate_moves_batch(
                model=sampling_policy.model,
                tokenizer=sampling_policy.tokenizer,
                states=chunk_states_padded,
                device=sampling_policy.device,
                batch_size=batch_size,
                max_new_tokens=chunk_max_new_tokens,
                temperature=sampling_policy.temperature,
                top_k=None,
                seed=sampling_policy.seed,
            )
            for i in range(chunk_n):
                outputs_sorted[chunk_start + i] = (
                    chunk_outputs[i] if i < len(chunk_outputs) else ""
                )
    except Exception as e:
        print(f"Warning: multi-position batched generation failed ({e}); returning empty.")
        traceback.print_exc()
        return results

    # Map outputs back to original slot order, then distribute to positions.
    full_outputs: List[str] = [""] * n_slots
    for sorted_i, orig_i in enumerate(sorted_order):
        full_outputs[orig_i] = outputs_sorted[sorted_i]

    for si, (pi, ci, start_board, cont_depth) in enumerate(flat_slots):
        spec = position_specs[pi]
        cand_move = spec["candidate_moves"][ci]
        output = full_outputs[si]
        continuation = _parse_move_sequence(
            output, start_board, cont_depth,
            sampling_policy._parse_move_from_output,
        )
        results[pi][cand_move].append([cand_move] + continuation)

    return results


def finalize_position_puzzle(
    spec: dict,
    gen_per_cand: Dict[chess.Move, List[List[chess.Move]]],
    args,
    stockfish_engine: chess.engine.SimpleEngine,
    sf_annotation_pool=None,
) -> Optional[Tuple[dict, Optional[dict]]]:
    """Phase 2: assemble trajectories, build tree, annotate, compute CoT strings.

    Args:
        spec:          Output of prepare_position_slots.
        gen_per_cand:  Model-generated rollouts per candidate from
                       generate_trajectories_multi_position_batched (or an
                       empty dict for depth-0 positions).

    Returns (result_dict, cot_entry) on success, None on error.
    """
    try:
        idx = spec["idx"]
        board = spec["board"]
        root_color = spec["root_color"]
        candidate_moves = spec["candidate_moves"]
        gt_move = spec["gt_move"]
        gt_injected = spec["gt_injected"]
        gt_solution_traj = spec["gt_solution_traj"]
        ground_truth_move_san = spec["ground_truth_move_san"]
        moves_list = spec["moves_list"]
        actual_pgn = spec["actual_pgn"]
        pgn_str = spec["pgn_str"]

        # ── Assemble trajectories ──────────────────────────────────────────── #
        all_trajectories: List[List[chess.Move]] = []
        trajectories_per_candidate: Dict[chess.Move, List[List[chess.Move]]] = {}

        if args.trajectory_depth <= 0:
            for cand_move in candidate_moves:
                traj = [cand_move]
                trajectories_per_candidate[cand_move] = [traj]
                all_trajectories.append(traj)
        else:
            for cand_move in candidate_moves:
                is_gt = gt_injected and gt_move is not None and cand_move == gt_move
                cand_trajs: List[List[chess.Move]] = list(gen_per_cand.get(cand_move, []))
                # Inject GT at a random position; dedup handles exact duplicates.
                if is_gt and gt_solution_traj is not None:
                    insert_at = random.randint(0, len(cand_trajs))
                    cand_trajs.insert(insert_at, gt_solution_traj)

                seen_keys: set = set()
                deduped: List[List[chess.Move]] = []
                for traj in cand_trajs:
                    key = tuple(mv.uci() for mv in traj)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        deduped.append(traj)

                all_trajectories.extend(deduped)
                trajectories_per_candidate[cand_move] = deduped

        # ── Step 4: Build tree ─────────────────────────────────────────────── #
        tree_root = build_tree_from_trajectories(
            root_board=board, all_trajectories=all_trajectories
        )

        # ── Step 5: Annotate with Stockfish ───────────────────────────────── #
        if sf_annotation_pool is not None:
            annotate_tree_with_labels_parallel(
                node=tree_root,
                pool=sf_annotation_pool,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
                win_threshold=args.win_threshold,
            )
        else:
            annotate_tree_with_labels(
                node=tree_root,
                engine=stockfish_engine,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
                win_threshold=args.win_threshold,
            )

        # ── Step 6: Minimax evaluation ────────────────────────────────────── #
        tree_stats = compute_trajectory_tree_stats(tree_root)
        leaf_labels = collect_leaf_labels_from_trajectory_tree(tree_root)
        minimax_values = compute_candidate_minimax_values(tree_root, root_color)

        trajectory_labels: List[int] = []
        trajectory_cp_values: List[float] = []
        for traj in all_trajectories:
            cur = tree_root
            for mv in traj:
                if mv in cur.children:
                    cur = cur.children[mv]
            cp_val = cur.stockfish_value if cur.stockfish_value is not None else 0.0
            if cur.leaf_label is not None:
                lbl = cur.leaf_label
            elif cur.stockfish_value is not None:
                sf = cur.stockfish_value
                lbl = 1 if sf > args.win_threshold else (-1 if sf < -args.win_threshold else 0)
            else:
                lbl = 0
            trajectory_labels.append(lbl)
            trajectory_cp_values.append(cp_val)

        direct_move_values: Dict[chess.Move, float] = {}
        move_leaf_max_values: Dict[chess.Move, float] = {}
        candidate_details: List[dict] = []

        for cand_move in candidate_moves:
            if cand_move not in tree_root.children:
                continue
            cand_node = tree_root.children[cand_move]
            cand_leaf_vals = collect_leaf_values_from_trajectory_tree(cand_node)
            cand_leaf_labs = collect_leaf_labels_from_trajectory_tree(cand_node)
            direct_val = cand_node.stockfish_value or 0.0
            best_leaf_val = max(cand_leaf_vals) if cand_leaf_vals else 0.0
            minimax_val = minimax_values.get(cand_move, 0)

            direct_move_values[cand_move] = direct_val
            move_leaf_max_values[cand_move] = best_leaf_val

            cand_stats = compute_trajectory_tree_stats(cand_node)
            candidate_details.append(
                {
                    "move": cand_move.uci(),
                    "move_san": board.san(cand_move),
                    "is_gt_injected": gt_injected and cand_move == gt_move,
                    "direct_value": float(direct_val),
                    "best_leaf_value": float(best_leaf_val),
                    "minimax_value": int(minimax_val),
                    "leaf_labels": cand_leaf_labs,
                    "leaf_count": cand_stats["leaf_count"],
                    "max_depth": cand_stats["max_depth"],
                    "node_count": cand_stats["node_count"],
                    "num_trajectories": len(
                        trajectories_per_candidate.get(cand_move, [])
                    ),
                }
            )

        # ── Step 7: Select target move ─────────────────────────────────────── #
        if args.evaluation_policy == "minimax":
            target_move = select_best_move_by_minimax(minimax_values, direct_move_values)
            best_val = minimax_values.get(target_move, 0) if target_move else None
        elif args.evaluation_policy == "stockfish_leaf_max":
            target_move = (
                max(move_leaf_max_values, key=move_leaf_max_values.get)
                if move_leaf_max_values
                else None
            )
            best_val = move_leaf_max_values.get(target_move) if target_move else None
        elif args.evaluation_policy == "stockfish_direct_max":
            target_move = (
                max(direct_move_values, key=direct_move_values.get)
                if direct_move_values
                else None
            )
            best_val = direct_move_values.get(target_move) if target_move else None
        else:
            raise ValueError(f"Unknown evaluation policy: {args.evaluation_policy}")

        target_move_san = board.san(target_move) if target_move else None

        # ── Step 8: Build traversal strings ───────────────────────────────── #
        traversals: Dict[str, str] = {}
        for method in args.traversal_methods:
            if method not in ("dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"):
                print(f"Warning: unknown traversal method '{method}', skipping")
                continue
            traversals[method] = build_candidate_cot_traversal(
                board, tree_root, minimax_values, method
            )

        # ── Step 9: Build full puzzle CoT strings ─────────────────────────── #
        call_env_token = getattr(args, "call_env_token", "<call_env>")
        max_puzzle_moves = getattr(args, "max_puzzle_moves", 5)

        cot_results: Dict[str, dict] = {}
        for method, trav_str in traversals.items():
            cot_results[method] = generate_puzzle_cot_with_sequence(
                traversal_str=trav_str,
                board=board,
                moves_list=moves_list,
                target_move=gt_move,
                pgn_context=actual_pgn,
                call_env_token=call_env_token,
                max_puzzle_moves=max_puzzle_moves,
            )

        # ── trajectory_sep: shuffle candidate and within-candidate order ──────── #
        # all_trajectories reflects construction order (e.g. EXPAND explores best
        # move first), which makes the GT answer systematically first.  Regroup via
        # trajectories_per_candidate, shuffle both levels, then flatten.
        _traj_cp_map = {
            tuple(mv.uci() for mv in t): cp
            for t, cp in zip(all_trajectories, trajectory_cp_values)
        }
        _cand_order = list(trajectories_per_candidate.keys())
        random.shuffle(_cand_order)
        _ordered_trajs: List[List[chess.Move]] = []
        for _cand in _cand_order:
            _cand_trajs = list(trajectories_per_candidate[_cand])
            random.shuffle(_cand_trajs)
            _ordered_trajs.extend(_cand_trajs)

        traj_strs = [
            _trajectory_to_string_stratified(
                board, traj,
                _traj_cp_map.get(tuple(mv.uci() for mv in traj), 0.0),
            )
            for traj in _ordered_trajs
        ]
        traj_sep_inner = _reformat_result_labels(" <sep> ".join(traj_strs))
        cot_results["trajectory_sep"] = generate_puzzle_cot_with_sequence(
            traversal_str=traj_sep_inner,
            board=board,
            moves_list=moves_list,
            target_move=gt_move,
            pgn_context=actual_pgn,
            call_env_token=call_env_token,
            max_puzzle_moves=max_puzzle_moves,
        )

        # ── Step 10: Extra CoT formats (IDs 1–3, 5–6) ────────────────────── #
        _puzzle_seq_extra = generate_puzzle_sequence(
            board=board, moves_list=moves_list,
            call_env_token=call_env_token, max_puzzle_moves=max_puzzle_moves,
        )
        _first_move_lan_extra = ""
        if gt_move is not None:
            try:
                _first_move_lan_extra = move_to_lan(board, gt_move)
            except Exception:
                _first_move_lan_extra = gt_move.uci()

        # ID 1: best_move_only – <state> Qh7+
        _bmo = _first_move_lan_extra
        _bmo_ctx = f"{actual_pgn} {_bmo}".strip() if _bmo else actual_pgn
        cot_results["best_move_only"] = {
            "cot_format": _bmo,
            "cot_format_with_context": _bmo_ctx,
            "cot_format_no_labels": _bmo,
            "cot_format_with_context_no_labels": _bmo_ctx,
            "cot_format_first_move": _bmo,
            "cot_format_first_move_with_context": _bmo_ctx,
            "cot_format_first_move_no_labels": _bmo,
            "cot_format_first_move_with_context_no_labels": _bmo_ctx,
            "puzzle_sequence": _bmo,
            "first_move_lan": _first_move_lan_extra,
        }

        # ID 2: solution_continuation – <state> Qh7+ Kxh7 Rh1+ ...
        _sc_ctx = f"{actual_pgn} {_puzzle_seq_extra}".strip() if _puzzle_seq_extra else actual_pgn
        _sc_fm_ctx = (
            f"{actual_pgn} {_first_move_lan_extra}".strip() if _first_move_lan_extra else actual_pgn
        )
        cot_results["solution_continuation"] = {
            "cot_format": _puzzle_seq_extra,
            "cot_format_with_context": _sc_ctx,
            "cot_format_no_labels": _puzzle_seq_extra,
            "cot_format_with_context_no_labels": _sc_ctx,
            "cot_format_first_move": _first_move_lan_extra,
            "cot_format_first_move_with_context": _sc_fm_ctx,
            "cot_format_first_move_no_labels": _first_move_lan_extra,
            "cot_format_first_move_with_context_no_labels": _sc_fm_ctx,
            "puzzle_sequence": _puzzle_seq_extra,
            "first_move_lan": _first_move_lan_extra,
        }

        # ID 3: successful_only_cot – <state> <T> Qh7+ Kxh7 … </T> Qh7+
        _succ_inner = build_successful_only_cot_inner(board, moves_list, max_puzzle_moves)
        cot_results["successful_only_cot"] = generate_puzzle_cot_with_sequence(
            traversal_str=_succ_inner,
            board=board,
            moves_list=moves_list,
            target_move=gt_move,
            pgn_context=actual_pgn,
            call_env_token=call_env_token,
            max_puzzle_moves=max_puzzle_moves,
        )

        # ID 5: stratified_verify – <verify> <+3>/<+2>/…/<-3> for each traversal method
        for _method in args.traversal_methods:
            if _method not in ("dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"):
                continue
            _strat_trav = build_candidate_cot_traversal_stratified(board, tree_root, _method)
            cot_results[f"stratified_{_method}"] = generate_puzzle_cot_with_sequence(
                traversal_str=_strat_trav,
                board=board,
                moves_list=moves_list,
                target_move=gt_move,
                pgn_context=actual_pgn,
                call_env_token=call_env_token,
                max_puzzle_moves=max_puzzle_moves,
            )

        # ID 6: monotonic_improvement – trajectories ordered worst→best by endpoint CP
        _mono_inner = build_monotonic_improvement_cot_inner(board, all_trajectories, tree_root)
        cot_results["monotonic_improvement"] = generate_puzzle_cot_with_sequence(
            traversal_str=_mono_inner,
            board=board,
            moves_list=moves_list,
            target_move=gt_move,
            pgn_context=actual_pgn,
            call_env_token=call_env_token,
            max_puzzle_moves=max_puzzle_moves,
        )

        # ── GT comparison ──────────────────────────────────────────────────── #
        is_match: Optional[bool] = None
        if ground_truth_move_san and target_move:
            try:
                gt_parsed = board.parse_san(ground_truth_move_san)
                is_match = target_move == gt_parsed
            except ValueError:
                is_match = target_move_san == ground_truth_move_san

        # ── Assemble result dict ───────────────────────────────────────────── #
        result = {
            "index": idx,
            "original_pgn": pgn_str,
            "pgn": actual_pgn,
            "moves_list": moves_list,
            "opponent_move": moves_list[0] if moves_list else None,
            "ground_truth_move": ground_truth_move_san,
            "gt_injected": gt_injected,
            "target_move": target_move.uci() if target_move else None,
            "target_move_san": target_move_san,
            "is_match": is_match,
            "best_val": (
                int(best_val)
                if isinstance(best_val, int)
                else float(best_val)
                if best_val is not None
                else None
            ),
            "num_candidates": len(candidate_moves),
            "num_trajectories_total": len(all_trajectories),
            "minimax_values": {mv.uci(): int(v) for mv, v in minimax_values.items()},
            "direct_move_values": {
                mv.uci(): float(v) for mv, v in direct_move_values.items()
            },
            "cot_by_method": cot_results,
            "candidate_details": candidate_details,
            "tree_stats": tree_stats,
            "leaf_label_distribution": {
                "wins": leaf_labels.count(1),
                "draws": leaf_labels.count(0),
                "losses": leaf_labels.count(-1),
            },
        }

        cot_entry = None
        if getattr(args, "cot_output_path", None):
            cot_entry = {
                "index": idx,
                "pgn": actual_pgn,
                "moves_list": moves_list,
                "target_move_san": target_move_san,
            }
            for method, cot_data in cot_results.items():
                cot_entry[f"cot_{method}"] = cot_data["cot_format_with_context"]
                cot_entry[f"cot_{method}_first_move"] = cot_data["cot_format_first_move_with_context"]
                cot_entry[f"puzzle_seq_{method}"] = cot_data["puzzle_sequence"]

        return result, cot_entry

    except Exception as e:
        print(f"Error finalizing position {spec.get('idx', '?')}: {e}")
        traceback.print_exc()
        return None


# ── Stockfish recursive tree expansion (Adaptive Tree Synthesis) ─────────────── #

def _principal_rollout_stockfish(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    remaining_budget: int,
    tree_depth: int,
    current_path: List[chess.Move],
    all_trajectories: List[List[chess.Move]],
    work_depth: int,
    max_tree_depth: int,
    node_analysis_cache: Optional[dict] = None,
) -> None:
    """Follow the principal (best-scoring) line until budget or max depth is exhausted.

    Each move consumes 1 budget unit.  The completed path is appended to
    all_trajectories only if at least one move was played (i.e. current_path
    is non-empty after the rollout).
    """
    path = list(current_path)
    board = board.copy()

    while remaining_budget > 0 and tree_depth < max_tree_depth:
        if board.is_game_over():
            break
        fen = board.fen()
        cache_key = (fen, work_depth, 1)
        if node_analysis_cache is not None and cache_key in node_analysis_cache:
            info_list = node_analysis_cache[cache_key]
        else:
            info_list = engine.analyse(board, chess.engine.Limit(depth=work_depth), multipv=1)
            if node_analysis_cache is not None:
                node_analysis_cache[cache_key] = info_list
        if not info_list:
            break
        pv = info_list[0].get("pv", [])
        if not pv:
            break
        best_move = pv[0]
        path.append(best_move)
        board.push(best_move)
        remaining_budget -= 1
        tree_depth += 1

    if path:
        all_trajectories.append(path)


def _expand_node_stockfish(
    engine: chess.engine.SimpleEngine,
    board: chess.Board,
    remaining_budget: int,
    tree_depth: int,
    current_path: List[chess.Move],
    all_trajectories: List[List[chess.Move]],
    work_depth: int,
    ref_depth: int,
    max_candidates: int,
    max_tree_depth: int,
    stop_entropy: float,
    temp: float,
    eps: float,
    forced_eval_cache: Optional[dict] = None,
    node_analysis_cache: Optional[dict] = None,
) -> None:
    """Recursive Adaptive Tree Synthesis (EXPAND from the pseudocode).

    At every node:
      1. Fetch up to max_candidates lines from Stockfish at work_depth.
      2. Compute softmax → entropy H → effective width W.
      3. If H < stop_entropy: delegate to _principal_rollout_stockfish.
      4. Otherwise: compute V_work / V_ref for each of the W candidates,
         derive uncertainty weights, allocate the remaining budget among
         children proportionally, and recurse.

    Leaf paths (non-empty current_path) are appended to all_trajectories.
    """
    # ── Base cases ──────────────────────────────────────────────────────────── #
    if remaining_budget <= 0 or tree_depth >= max_tree_depth:
        if current_path:
            all_trajectories.append(list(current_path))
        return

    if board.is_game_over():
        if current_path:
            all_trajectories.append(list(current_path))
        return

    k = min(max_candidates, len(list(board.legal_moves)))
    if k == 0:
        if current_path:
            all_trajectories.append(list(current_path))
        return

    # ── Fetch candidates at working depth ────────────────────────────────────── #
    _analyse_key = (board.fen(), work_depth, k)
    if node_analysis_cache is not None and _analyse_key in node_analysis_cache:
        info_list = node_analysis_cache[_analyse_key]
    else:
        info_list = engine.analyse(board, chess.engine.Limit(depth=work_depth), multipv=k)
        if node_analysis_cache is not None:
            node_analysis_cache[_analyse_key] = info_list
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
        cp_white = sc.white().score(mate_score=100000)
        cp_side = float(cp_white) if board.turn == chess.WHITE else float(-cp_white)
        candidates.append(mv)
        scores.append(cp_side)

    if not candidates:
        if current_path:
            all_trajectories.append(list(current_path))
        return

    # ── Entropy → width → low-entropy shortcut ──────────────────────────────── #
    probs = _sf_softmax(scores, temp)
    h = _sf_entropy(probs)

    # At tree_depth==0 (root) we must always branch regardless of entropy:
    # puzzles almost always have one clearly best move, so entropy is near 0
    # there, and applying the shortcut would collapse the entire tree into a
    # single principal-line trajectory (the puzzle answer itself).
    at_root = (tree_depth == 0)
    width = len(candidates) if at_root else _sf_choose_width(h, len(candidates))

    logger.debug(
        "[EXPAND] depth=%d budget=%d n_cands=%d entropy=%.4f width=%d at_root=%s  path=[%s]",
        tree_depth, remaining_budget, len(candidates), h, width, at_root,
        " ".join(mv.uci() for mv in current_path) or "<root>",
    )

    if h < stop_entropy and not at_root:
        logger.debug(
            "[EXPAND] depth=%d  H=%.4f < stop=%.4f → principal rollout from [%s]",
            tree_depth, h, stop_entropy,
            " ".join(mv.uci() for mv in current_path),
        )
        _principal_rollout_stockfish(
            engine=engine,
            board=board,
            remaining_budget=remaining_budget,
            tree_depth=tree_depth,
            current_path=current_path,
            all_trajectories=all_trajectories,
            work_depth=work_depth,
            max_tree_depth=max_tree_depth,
            node_analysis_cache=node_analysis_cache,
        )
        return

    # ── Uncertainty-based budget allocation to children ──────────────────────── #
    selected = candidates[:width]

    vals_work: List[Optional[float]] = [
        _forced_move_value_cp(engine, board, mv, work_depth, cache=forced_eval_cache)
        for mv in selected
    ]
    vals_ref: List[Optional[float]] = [
        _forced_move_value_cp(engine, board, mv, ref_depth, cache=forced_eval_cache)
        for mv in selected
    ]

    ref_non_null = [v for v in vals_ref if v is not None]
    scale = _std_of(ref_non_null) + eps

    weights: List[float] = []
    for vw, vr in zip(vals_work, vals_ref):
        u = abs(vr - vw) / scale if (vw is not None and vr is not None) else 0.0
        weights.append(math.sqrt(u + eps))

    child_budget_total = max(0, remaining_budget - 1)
    child_budgets = allocate_child_budgets_from_weights(weights, child_budget_total)

    logger.debug(
        "[EXPAND] depth=%d  child_budget_total=%d  allocations=%s",
        tree_depth, child_budget_total,
        " ".join(f"{mv.uci()}:{b}" for mv, b in zip(selected, child_budgets)),
    )

    for mv, child_budget in zip(selected, child_budgets):
        child_board = board.copy()
        child_board.push(mv)
        _expand_node_stockfish(
            engine=engine,
            board=child_board,
            remaining_budget=child_budget,
            tree_depth=tree_depth + 1,
            current_path=current_path + [mv],
            all_trajectories=all_trajectories,
            work_depth=work_depth,
            ref_depth=ref_depth,
            max_candidates=max_candidates,
            max_tree_depth=max_tree_depth,
            stop_entropy=stop_entropy,
            temp=temp,
            eps=eps,
            forced_eval_cache=forced_eval_cache,
            node_analysis_cache=node_analysis_cache,
        )


# ── Core position processor ──────────────────────────────────────────────────── #

def process_one_position_puzzle(
    idx: int,
    row_dict: dict,
    args,
    stockfish_engine: chess.engine.SimpleEngine,
    sampling_policy,
    has_ground_truth: bool,
    moves_col: str,
    sf_annotation_pool=None,
    rng: Optional[np.random.Generator] = None,
) -> Optional[Tuple[dict, Optional[dict]]]:
    """
    Process one puzzle position.
    Returns (result_dict, cot_entry) on success, None if skipped or errored.
    """
    pgn_str = row_dict.get("ctx")
    if pgn_str is np.nan or pgn_str is None:
        print(f"Skipping position {idx}: no PGN string")
        return None

    actual_pgn = pgn_str
    moves_list: List[str] = []
    ground_truth_move_san: Optional[str] = None

    # ── Parse Moves column and apply opponent trigger move ─────────────────── #
    if has_ground_truth and pd.notna(row_dict.get(moves_col)):
        moves_str = str(row_dict[moves_col]).strip()
        moves_list = moves_str.split()
        if len(moves_list) < 2:
            print(f"Skipping position {idx}: Moves column has fewer than 2 moves")
            return None

        ground_truth_move_san = moves_list[1]

        try:
            game = chess.pgn.read_game(io.StringIO(pgn_str))
            board_tmp = game.end().board() if game is not None else chess.Board()
            opp_mv = board_tmp.parse_san(moves_list[0])
            board_tmp.push(opp_mv)
            new_game = chess.pgn.Game.from_board(board_tmp)
            exporter = chess.pgn.StringExporter(
                headers=False, variations=False, comments=False
            )
            new_pgn = re.sub(
                r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", new_game.accept(exporter)
            ).strip()
            actual_pgn = new_pgn
        except Exception as e:
            print(f"Position {idx}: failed to apply first move – {e}")
            return None

    try:
        game = chess.pgn.read_game(io.StringIO(actual_pgn))
        board = game.end().board() if game is not None else chess.Board()
        root_color = board.turn
        legal_moves = list(board.legal_moves)

        if not legal_moves:
            print(f"Position {idx}: no legal moves (game over)")
            return None

        # ── Parse ground-truth move ─────────────────────────────────────────── #
        gt_move: Optional[chess.Move] = None
        if ground_truth_move_san:
            try:
                gt_move = board.parse_san(ground_truth_move_san)
            except ValueError:
                try:
                    mv = chess.Move.from_uci(ground_truth_move_san)
                    if mv in legal_moves:
                        gt_move = mv
                except ValueError:
                    pass

        # ── Step 1: Generate candidate moves ───────────────────────────────── #
        if args.candidate_selection == "policy":
            candidate_moves = sample_candidates_with_policy(
                board=board,
                sampling_policy=sampling_policy,
                num_candidates=args.num_candidates,
            )
        else:
            scored = order_moves_by_stockfish(
                board=board,
                legal_moves=legal_moves,
                engine=stockfish_engine,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
            )
            candidate_moves = [mv for mv, _ in scored[: args.num_candidates]]

        if not candidate_moves:
            print(f"Position {idx}: no candidates generated")
            return None

        # ── Step 2: Inject ground-truth answer ──────────────────────────────── #
        gt_injected = False
        if args.inject_answer and gt_move is not None:
            if gt_move not in candidate_moves:
                replace_idx = int(rng.integers(0, len(candidate_moves))) if rng is not None else random.randrange(len(candidate_moves))
                candidate_moves[replace_idx] = gt_move
            gt_injected = True  # already present or just inserted

        # Shuffle candidate order so the GT candidate is not systematically first
        # in all_trajectories (which determines generation_order / policy traversal
        # order).  Stockfish uses EXPAND which builds its own candidates, so no
        # shuffle is needed there.
        if args.sampling_policy != "stockfish" and len(candidate_moves) > 1:
            if rng is not None:
                rng.shuffle(candidate_moves)
            else:
                random.shuffle(candidate_moves)

        # ── Step 3: Generate trajectories per candidate ──────────────────────── #
        all_trajectories: List[List[chess.Move]] = []
        trajectories_per_candidate: Dict[chess.Move, List[List[chess.Move]]] = {}

        if args.trajectory_depth <= 0:
            # Depth-0: single-move trajectories only, no rollouts needed
            for cand_move in candidate_moves:
                traj = [cand_move]
                trajectories_per_candidate[cand_move] = [traj]
                all_trajectories.append(traj)

        elif isinstance(sampling_policy, ModelSamplingPolicy):
            # ── Batched generation for model policies ─────────────────────── #
            # GT solution trajectory is injected separately (not model-generated).
            gt_solution_traj: Optional[List[chess.Move]] = None
            if gt_injected and gt_move is not None and len(moves_list) >= 2:
                gt_solution_traj = build_puzzle_solution_trajectory(
                    board=board,
                    moves_list=moves_list,
                    max_depth=args.trajectory_depth,
                )

            # For stockfish policy: shuffle candidate order before tree building.
            if args.sampling_policy == "stockfish" and len(candidate_moves) > 1:
                if rng is not None:
                    rng.shuffle(candidate_moves)
                else:
                    random.shuffle(candidate_moves)

            # Uncertainty-based budget allocation; GT solution injected on top.
            _model_eval_cache: dict = {}
            _unc_weights = compute_uncertainty_weights(
                stockfish_engine, board, candidate_moves,
                getattr(args, "stockfish_depth_work", 10),
                getattr(args, "stockfish_depth_ref", 18),
                cache=_model_eval_cache,
            )
            _total_budget = _compute_total_budget(args, row_dict, len(candidate_moves))
            _n_traj_base = allocate_child_budgets_from_weights(_unc_weights, _total_budget)

            n_traj_per_cand: List[int] = []
            depths_per_slot: List[List[int]] = []
            for i, cand_move in enumerate(candidate_moves):
                n_traj = _n_traj_base[i]
                n_traj_per_cand.append(n_traj)
                depths_per_slot.append(
                    [_sample_depth(args, rng) for _ in range(n_traj)]
                )

            traj_batch_size = getattr(args, "trajectory_batch_size", 32)
            tokens_per_move = getattr(args, "tokens_per_move", 6)
            _, gen_per_cand = generate_trajectories_batched(
                root_board=board,
                candidate_moves=candidate_moves,
                sampling_policy=sampling_policy,
                n_traj_per_cand=n_traj_per_cand,
                depths_per_slot=depths_per_slot,
                batch_size=traj_batch_size,
                tokens_per_move=tokens_per_move,
            )

            # Assemble: model-generated rollouts + GT solution at a random position.
            # Random insertion mirrors finalize_position_puzzle so the GT answer is
            # not always the first trajectory in generation_order / policy traversal.
            for cand_move in candidate_moves:
                is_gt = gt_injected and gt_move is not None and cand_move == gt_move
                cand_trajs: List[List[chess.Move]] = list(gen_per_cand.get(cand_move, []))
                if is_gt and gt_solution_traj is not None:
                    insert_at = (
                        int(rng.integers(0, len(cand_trajs) + 1))
                        if rng is not None
                        else random.randint(0, len(cand_trajs))
                    )
                    cand_trajs.insert(insert_at, gt_solution_traj)

                seen_keys: set = set()
                deduped: List[List[chess.Move]] = []
                for traj in cand_trajs:
                    key = tuple(mv.uci() for mv in traj)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        deduped.append(traj)

                all_trajectories.extend(deduped)
                trajectories_per_candidate[cand_move] = deduped

        else:
            # ── Sequential fallback for non-model policies ────────────────── #

            if args.sampling_policy == "stockfish":
                # ── Adaptive Tree Synthesis: recursive EXPAND ─────────────── #
                # Budget allocation and branching happen at every node, not just
                # at the root.  The root budget comes from _compute_total_budget
                # (fixed = num_trajectories × num_candidates, or
                # difficulty_adaptive scaling with puzzle Rating).
                _root_budget = _compute_total_budget(args, row_dict, args.num_candidates)
                logger.debug(
                    "[EXPAND] puzzle=%s  root_budget=%d  work_depth=%d  ref_depth=%d  "
                    "stop_entropy=%.3f  max_candidates=%d  max_tree_depth=%d",
                    row_dict.get("PuzzleId", idx), _root_budget,
                    getattr(args, "stockfish_depth_work", 10),
                    getattr(args, "stockfish_depth_ref", 18),
                    getattr(args, "stop_entropy", 0.45),
                    args.num_candidates, args.trajectory_depth,
                )
                _forced_eval_cache: dict = {}
                _node_analysis_cache: dict = {}
                _expand_node_stockfish(
                    engine=stockfish_engine,
                    board=board,
                    remaining_budget=_root_budget,
                    tree_depth=0,
                    current_path=[],
                    all_trajectories=all_trajectories,
                    work_depth=getattr(args, "stockfish_depth_work", 10),
                    ref_depth=getattr(args, "stockfish_depth_ref", 18),
                    max_candidates=args.num_candidates,
                    max_tree_depth=args.trajectory_depth,
                    stop_entropy=getattr(args, "stop_entropy", 0.45),
                    temp=args.temperature,
                    eps=1e-6,
                    forced_eval_cache=_forced_eval_cache,
                    node_analysis_cache=_node_analysis_cache,
                )
                logger.debug(
                    "[EXPAND] puzzle=%s  EXPAND produced %d raw trajectories",
                    row_dict.get("PuzzleId", idx), len(all_trajectories),
                )

                # Inject GT solution only if gt_move was not reached by EXPAND.
                if gt_injected and gt_move is not None and len(moves_list) >= 2:
                    gt_explored = any(
                        traj and traj[0] == gt_move for traj in all_trajectories
                    )
                    if not gt_explored:
                        sol_traj = build_puzzle_solution_trajectory(
                            board=board,
                            moves_list=moves_list,
                            max_depth=args.trajectory_depth,
                        )
                        if sol_traj:
                            all_trajectories.append(sol_traj)

                # Global dedup (EXPAND can produce duplicate paths in rare cases)
                _seen_keys: set = set()
                _deduped: List[List[chess.Move]] = []
                for traj in all_trajectories:
                    key = tuple(mv.uci() for mv in traj)
                    if key not in _seen_keys:
                        _seen_keys.add(key)
                        _deduped.append(traj)
                all_trajectories = _deduped

                # Derive candidate_moves and trajectories_per_candidate from
                # the paths produced by EXPAND (first move = root candidate).
                _cand_seen: set = set()
                candidate_moves = []
                for traj in all_trajectories:
                    if not traj:
                        continue
                    first = traj[0]
                    if first.uci() not in _cand_seen:
                        _cand_seen.add(first.uci())
                        candidate_moves.append(first)
                    if first not in trajectories_per_candidate:
                        trajectories_per_candidate[first] = []
                    trajectories_per_candidate[first].append(traj)

            else:
                # ── Uncertainty-based flat rollout (random policy, etc.) ───── #
                _unc_weights = compute_uncertainty_weights(
                    stockfish_engine, board, candidate_moves,
                    getattr(args, "stockfish_depth_work", 10),
                    getattr(args, "stockfish_depth_ref", 18),
                )
                _total_budget = _compute_total_budget(args, row_dict, len(candidate_moves))
                _n_traj_per_cand = allocate_child_budgets_from_weights(_unc_weights, _total_budget)

                for cand_move, n_traj in zip(candidate_moves, _n_traj_per_cand):
                    is_gt = gt_injected and gt_move is not None and cand_move == gt_move

                    cand_trajs: List[List[chess.Move]] = []

                    cand_board = board.copy()
                    cand_board.push(cand_move)
                    for _ in range(n_traj):
                        depth = _sample_depth(args, rng)
                        continuation = generate_single_trajectory(
                            board=cand_board,
                            sampling_policy=sampling_policy,
                            max_depth=max(0, depth - 1),
                        )
                        cand_trajs.append([cand_move] + continuation)

                    # Inject GT at a random position; dedup handles exact duplicates.
                    if is_gt and len(moves_list) >= 2:
                        sol_traj = build_puzzle_solution_trajectory(
                            board=board,
                            moves_list=moves_list,
                            max_depth=args.trajectory_depth,
                        )
                        if sol_traj:
                            insert_at = (
                                int(rng.integers(0, len(cand_trajs) + 1))
                                if rng is not None
                                else random.randint(0, len(cand_trajs))
                            )
                            cand_trajs.insert(insert_at, sol_traj)

                    seen_keys: set = set()
                    deduped: List[List[chess.Move]] = []
                    for traj in cand_trajs:
                        key = tuple(mv.uci() for mv in traj)
                        if key not in seen_keys:
                            seen_keys.add(key)
                            deduped.append(traj)

                    all_trajectories.extend(deduped)
                    trajectories_per_candidate[cand_move] = deduped

        # ── Step 4: Build tree from trajectories ────────────────────────────── #
        tree_root = build_tree_from_trajectories(
            root_board=board, all_trajectories=all_trajectories
        )

        # ── Step 5: Annotate tree with Stockfish labels ─────────────────────── #
        if sf_annotation_pool is not None:
            annotate_tree_with_labels_parallel(
                node=tree_root,
                pool=sf_annotation_pool,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
                win_threshold=args.win_threshold,
            )
        else:
            annotate_tree_with_labels(
                node=tree_root,
                engine=stockfish_engine,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
                win_threshold=args.win_threshold,
            )

        # ── Step 6: Minimax evaluation ──────────────────────────────────────── #
        tree_stats = compute_trajectory_tree_stats(tree_root)
        leaf_labels = collect_leaf_labels_from_trajectory_tree(tree_root)
        minimax_values = compute_candidate_minimax_values(tree_root, root_color)

        # Per-trajectory leaf labels: walk tree along each trajectory path.
        # If the trajectory endpoint is an internal node (shared prefix with a
        # longer trajectory), leaf_label is None — fall back to stockfish_value.
        trajectory_labels: List[int] = []
        trajectory_cp_values: List[float] = []
        for traj in all_trajectories:
            cur = tree_root
            for mv in traj:
                if mv in cur.children:
                    cur = cur.children[mv]
            cp_val = cur.stockfish_value if cur.stockfish_value is not None else 0.0
            if cur.leaf_label is not None:
                lbl = cur.leaf_label
            elif cur.stockfish_value is not None:
                sf = cur.stockfish_value
                lbl = 1 if sf > args.win_threshold else (-1 if sf < -args.win_threshold else 0)
            else:
                lbl = 0
            trajectory_labels.append(lbl)
            trajectory_cp_values.append(cp_val)

        direct_move_values: Dict[chess.Move, float] = {}
        move_leaf_max_values: Dict[chess.Move, float] = {}
        candidate_details: List[dict] = []

        for cand_move in candidate_moves:
            if cand_move not in tree_root.children:
                continue
            cand_node = tree_root.children[cand_move]
            cand_leaf_vals = collect_leaf_values_from_trajectory_tree(cand_node)
            cand_leaf_labs = collect_leaf_labels_from_trajectory_tree(cand_node)
            direct_val = cand_node.stockfish_value or 0.0
            best_leaf_val = max(cand_leaf_vals) if cand_leaf_vals else 0.0
            minimax_val = minimax_values.get(cand_move, 0)

            direct_move_values[cand_move] = direct_val
            move_leaf_max_values[cand_move] = best_leaf_val

            cand_stats = compute_trajectory_tree_stats(cand_node)
            candidate_details.append(
                {
                    "move": cand_move.uci(),
                    "move_san": board.san(cand_move),
                    "is_gt_injected": gt_injected and cand_move == gt_move,
                    "direct_value": float(direct_val),
                    "best_leaf_value": float(best_leaf_val),
                    "minimax_value": int(minimax_val),
                    "leaf_labels": cand_leaf_labs,
                    "leaf_count": cand_stats["leaf_count"],
                    "max_depth": cand_stats["max_depth"],
                    "node_count": cand_stats["node_count"],
                    "num_trajectories": len(
                        trajectories_per_candidate.get(cand_move, [])
                    ),
                }
            )

        # ── Step 7: Select target move ──────────────────────────────────────── #
        if args.evaluation_policy == "minimax":
            target_move = select_best_move_by_minimax(minimax_values, direct_move_values)
            best_val = minimax_values.get(target_move, 0) if target_move else None
        elif args.evaluation_policy == "stockfish_leaf_max":
            target_move = (
                max(move_leaf_max_values, key=move_leaf_max_values.get)
                if move_leaf_max_values
                else None
            )
            best_val = move_leaf_max_values.get(target_move) if target_move else None
        elif args.evaluation_policy == "stockfish_direct_max":
            target_move = (
                max(direct_move_values, key=direct_move_values.get)
                if direct_move_values
                else None
            )
            best_val = direct_move_values.get(target_move) if target_move else None
        else:
            raise ValueError(f"Unknown evaluation policy: {args.evaluation_policy}")

        target_move_san = board.san(target_move) if target_move else None

        # ── Step 8: Build traversal strings ────────────────────────────────── #
        # Each candidate's full subtree is explored, then summarised with a
        # single minimax-based <verify> label (not per-leaf Stockfish labels).
        # Format: cand1 [subtree] <verify> <+1> <sep> cand2 [subtree] <verify> <-1>
        traversals: Dict[str, str] = {}
        for method in args.traversal_methods:
            if method not in ("dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"):
                print(f"Warning: unknown traversal method '{method}', skipping")
                continue
            traversals[method] = build_candidate_cot_traversal(
                board, tree_root, minimax_values, method
            )

        # ── Step 9: Build full puzzle CoT strings ───────────────────────────── #
        call_env_token = getattr(args, "call_env_token", "<call_env>")
        max_puzzle_moves = getattr(args, "max_puzzle_moves", 5)

        cot_results: Dict[str, dict] = {}
        for method, trav_str in traversals.items():
            cot_results[method] = generate_puzzle_cot_with_sequence(
                traversal_str=trav_str,
                board=board,
                moves_list=moves_list,
                target_move=gt_move,
                pgn_context=actual_pgn,
                call_env_token=call_env_token,
                max_puzzle_moves=max_puzzle_moves,
            )

        # ── Trajectory-sep CoT: each trajectory ends with its own <verify> ─── #
        # Inner format: "traj1_moves <verify> <+1> <sep> traj2_moves <verify> <-1> ..."
        # ── trajectory_sep: shuffle candidate and within-candidate order ──────── #
        # all_trajectories reflects construction order (e.g. EXPAND explores best
        # move first), which makes the GT answer systematically first.  Regroup via
        # trajectories_per_candidate, shuffle both levels, then flatten.
        _traj_cp_map = {
            tuple(mv.uci() for mv in t): cp
            for t, cp in zip(all_trajectories, trajectory_cp_values)
        }
        _cand_order = list(trajectories_per_candidate.keys())
        random.shuffle(_cand_order)
        _ordered_trajs: List[List[chess.Move]] = []
        for _cand in _cand_order:
            _cand_trajs = list(trajectories_per_candidate[_cand])
            random.shuffle(_cand_trajs)
            _ordered_trajs.extend(_cand_trajs)

        traj_strs = [
            _trajectory_to_string_stratified(
                board, traj,
                _traj_cp_map.get(tuple(mv.uci() for mv in traj), 0.0),
            )
            for traj in _ordered_trajs
        ]
        traj_sep_inner = _reformat_result_labels(" <sep> ".join(traj_strs))
        cot_results["trajectory_sep"] = generate_puzzle_cot_with_sequence(
            traversal_str=traj_sep_inner,
            board=board,
            moves_list=moves_list,
            target_move=gt_move,
            pgn_context=actual_pgn,
            call_env_token=call_env_token,
            max_puzzle_moves=max_puzzle_moves,
        )

        # ── Step 10: Extra CoT formats (IDs 1–3, 5–6) ───────────────────────── #
        _puzzle_seq_extra = generate_puzzle_sequence(
            board=board, moves_list=moves_list,
            call_env_token=call_env_token, max_puzzle_moves=max_puzzle_moves,
        )
        _first_move_lan_extra = ""
        if gt_move is not None:
            try:
                _first_move_lan_extra = move_to_lan(board, gt_move)
            except Exception:
                _first_move_lan_extra = gt_move.uci()

        # ID 1: best_move_only – <state> Qh7+
        _bmo = _first_move_lan_extra
        _bmo_ctx = f"{actual_pgn} {_bmo}".strip() if _bmo else actual_pgn
        cot_results["best_move_only"] = {
            "cot_format": _bmo,
            "cot_format_with_context": _bmo_ctx,
            "cot_format_no_labels": _bmo,
            "cot_format_with_context_no_labels": _bmo_ctx,
            "cot_format_first_move": _bmo,
            "cot_format_first_move_with_context": _bmo_ctx,
            "cot_format_first_move_no_labels": _bmo,
            "cot_format_first_move_with_context_no_labels": _bmo_ctx,
            "puzzle_sequence": _bmo,
            "first_move_lan": _first_move_lan_extra,
        }

        # ID 2: solution_continuation – <state> Qh7+ Kxh7 Rh1+ ...
        _sc_ctx = f"{actual_pgn} {_puzzle_seq_extra}".strip() if _puzzle_seq_extra else actual_pgn
        _sc_fm_ctx = (
            f"{actual_pgn} {_first_move_lan_extra}".strip() if _first_move_lan_extra else actual_pgn
        )
        cot_results["solution_continuation"] = {
            "cot_format": _puzzle_seq_extra,
            "cot_format_with_context": _sc_ctx,
            "cot_format_no_labels": _puzzle_seq_extra,
            "cot_format_with_context_no_labels": _sc_ctx,
            "cot_format_first_move": _first_move_lan_extra,
            "cot_format_first_move_with_context": _sc_fm_ctx,
            "cot_format_first_move_no_labels": _first_move_lan_extra,
            "cot_format_first_move_with_context_no_labels": _sc_fm_ctx,
            "puzzle_sequence": _puzzle_seq_extra,
            "first_move_lan": _first_move_lan_extra,
        }

        # ID 3: successful_only_cot – <state> <T> Qh7+ Kxh7 … </T> Qh7+
        _succ_inner = build_successful_only_cot_inner(board, moves_list, max_puzzle_moves)
        cot_results["successful_only_cot"] = generate_puzzle_cot_with_sequence(
            traversal_str=_succ_inner,
            board=board,
            moves_list=moves_list,
            target_move=gt_move,
            pgn_context=actual_pgn,
            call_env_token=call_env_token,
            max_puzzle_moves=max_puzzle_moves,
        )

        # ID 5: stratified_verify – <verify> <+3>/<+2>/…/<-3> for each traversal method
        for _method in args.traversal_methods:
            if _method not in ("dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"):
                continue
            _strat_trav = build_candidate_cot_traversal_stratified(board, tree_root, _method)
            cot_results[f"stratified_{_method}"] = generate_puzzle_cot_with_sequence(
                traversal_str=_strat_trav,
                board=board,
                moves_list=moves_list,
                target_move=gt_move,
                pgn_context=actual_pgn,
                call_env_token=call_env_token,
                max_puzzle_moves=max_puzzle_moves,
            )

        # ID 6: monotonic_improvement – trajectories ordered worst→best by endpoint CP
        _mono_inner = build_monotonic_improvement_cot_inner(board, all_trajectories, tree_root)
        cot_results["monotonic_improvement"] = generate_puzzle_cot_with_sequence(
            traversal_str=_mono_inner,
            board=board,
            moves_list=moves_list,
            target_move=gt_move,
            pgn_context=actual_pgn,
            call_env_token=call_env_token,
            max_puzzle_moves=max_puzzle_moves,
        )

        # ── Ground-truth comparison ─────────────────────────────────────────── #
        is_match: Optional[bool] = None
        if ground_truth_move_san and target_move:
            try:
                gt_parsed = board.parse_san(ground_truth_move_san)
                is_match = target_move == gt_parsed
            except ValueError:
                is_match = target_move_san == ground_truth_move_san

        # ── Assemble result dict ────────────────────────────────────────────── #
        result = {
            "index": idx,
            "original_pgn": pgn_str,
            "pgn": actual_pgn,
            "moves_list": moves_list,
            "opponent_move": moves_list[0] if moves_list else None,
            "ground_truth_move": ground_truth_move_san,
            "gt_injected": gt_injected,
            "target_move": target_move.uci() if target_move else None,
            "target_move_san": target_move_san,
            "is_match": is_match,
            "best_val": (
                int(best_val)
                if isinstance(best_val, int)
                else float(best_val)
                if best_val is not None
                else None
            ),
            "num_candidates": len(candidate_moves),
            "num_trajectories_total": len(all_trajectories),
            "minimax_values": {mv.uci(): int(v) for mv, v in minimax_values.items()},
            "direct_move_values": {
                mv.uci(): float(v) for mv, v in direct_move_values.items()
            },
            "cot_by_method": cot_results,
            "candidate_details": candidate_details,
            "tree_stats": tree_stats,
            "leaf_label_distribution": {
                "wins": leaf_labels.count(1),
                "draws": leaf_labels.count(0),
                "losses": leaf_labels.count(-1),
            },
        }

        # ── CoT entry for separate text output file ─────────────────────────── #
        cot_entry = None
        if getattr(args, "cot_output_path", None):
            cot_entry = {
                "index": idx,
                "pgn": actual_pgn,
                "moves_list": moves_list,
                "target_move_san": target_move_san,
            }
            for method, cot_data in cot_results.items():
                cot_entry[f"cot_{method}"] = cot_data["cot_format_with_context"]
                cot_entry[f"cot_{method}_first_move"] = cot_data["cot_format_first_move_with_context"]
                cot_entry[f"puzzle_seq_{method}"] = cot_data["puzzle_sequence"]

        return result, cot_entry

    except Exception as e:
        print(f"Error processing position {idx}: {e}")
        traceback.print_exc()
        return None


# ── Main ─────────────────────────────────────────────────────────────────────── #

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Puzzle sequence generator.\n"
            "Output format: {pgn} <T> [tree CoT] </T> move2 <call_env> move3 move4 <call_env> …\n\n"
            "Moves column convention:\n"
            "  moves_list[0]  opponent trigger (already applied)\n"
            "  moves_list[1]  model move  (even 1-indexed)\n"
            "  moves_list[2]  env response (odd 1-indexed)\n"
            "  …"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ──────────────────────────────────────────────────────────────────── #
    parser.add_argument(
        "--data_path", required=True, help="CSV with 'ctx' and 'Moves' columns"
    )
    parser.add_argument(
        "--output_path", required=True, help="Output JSON path or directory"
    )
    parser.add_argument(
        "--cot_output_path", default=None, help="Optional readable CoT text file"
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_skip_samples", type=int, default=None)
    parser.add_argument("--data_name", default="puzzle")

    # ── Sampling policy ───────────────────────────────────────────────────────── #
    parser.add_argument(
        "--sampling_policy",
        choices=["random", "stockfish", "hf_model"],
        default="random",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    # ── Candidates and trajectories ──────────────────────────────────────────── #
    parser.add_argument(
        "--candidate_selection", choices=["policy", "stockfish"], default="policy"
    )
    parser.add_argument(
        "--num_candidates", type=int, default=5,
        help="Number of candidate moves to explore in the thinking trace"
    )
    parser.add_argument(
        "--num_trajectories", type=int, default=3,
        help="Rollouts per non-answer candidate"
    )
    parser.add_argument(
        "--num_trajectories_answer", type=int, default=None,
        help="Max rollouts for the injected answer candidate "
             "(default: 2 × num_trajectories). "
             "One slot is always used by the puzzle solution line.",
    )
    parser.add_argument(
        "--num_trajectories_min", type=int, default=1,
        help="Minimum rollouts per candidate when depth_mode=random"
    )
    parser.add_argument(
        "--num_trajectories_mean", type=float, default=None,
        help="Target mean of lognormal trajectory-count distribution (default: num_trajectories)"
    )
    parser.add_argument(
        "--num_trajectories_sigma", type=float, default=0.8,
        help="Sigma of lognormal trajectory-count distribution"
    )
    parser.add_argument("--trajectory_depth", type=int, default=4)

    # ── Depth sampling ────────────────────────────────────────────────────────── #
    parser.add_argument(
        "--depth_mode",
        choices=["fixed", "random"],
        default="fixed",
        help=(
            "'fixed': all rollouts use --trajectory_depth; "
            "'random': sample each rollout depth from Lognormal(mean, sigma) "
            "clipped to [depth_min, trajectory_depth]"
        ),
    )
    parser.add_argument(
        "--depth_min", type=int, default=1,
        help="Minimum depth when depth_mode=random"
    )
    parser.add_argument(
        "--depth_mean", type=float, default=8.0,
        help="Target mean of lognormal depth distribution (E[depth] ≈ depth_mean)"
    )
    parser.add_argument(
        "--depth_sigma", type=float, default=0.8,
        help="Sigma of lognormal depth distribution"
    )

    # ── Answer injection ──────────────────────────────────────────────────────── #
    parser.add_argument(
        "--inject_answer",
        action="store_true",
        help=(
            "Force ground-truth move into candidates. "
            "Injects the full puzzle solution line as one trajectory for that "
            "candidate plus num_trajectories_answer additional rollouts."
        ),
    )

    # ── Puzzle sequence ───────────────────────────────────────────────────────── #
    parser.add_argument(
        "--max_puzzle_moves", type=int, default=5,
        help="Max moves from Moves column to include after </T> (cap at 5)"
    )
    parser.add_argument(
        "--call_env_token", default="<call_env>",
        help="Token emitted by the model after each of its moves"
    )

    # ── Evaluation ───────────────────────────────────────────────────────────── #
    parser.add_argument(
        "--evaluation_policy",
        choices=["minimax", "stockfish_leaf_max", "stockfish_direct_max"],
        default="minimax",
    )
    parser.add_argument("--win_threshold", type=int, default=300)
    parser.add_argument(
        "--traversal_methods",
        nargs="+",
        default=["dfs_policy"],
        choices=["dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"],
    )

    # ── Stockfish ────────────────────────────────────────────────────────────── #
    parser.add_argument("--stockfish_path", default="/usr/games/stockfish")
    parser.add_argument("--stockfish_time", type=float, default=0.01)
    parser.add_argument(
        "--stockfish_depth_work", type=int, default=10,
        help="Working depth for Stockfish candidate generation and uncertainty V_work"
    )
    parser.add_argument(
        "--stockfish_depth_ref", type=int, default=18,
        help="Reference depth for uncertainty estimation V_ref"
    )
    parser.add_argument(
        "--stockfish_top_k", type=int, default=6,
        help="Max candidates (K_max) for Stockfish depth-based multipv sampling"
    )
    parser.add_argument(
        "--stop_entropy", type=float, default=0.45,
        help=(
            "Entropy threshold (tau_H) for the EXPAND procedure: positions with "
            "entropy below this switch to principal-line rollout instead of branching"
        ),
    )

    # ── Budget allocation ─────────────────────────────────────────────────────── #
    parser.add_argument(
        "--budget_mode",
        choices=["fixed", "difficulty_adaptive"],
        default="fixed",
        help=(
            "'fixed': total trajectory budget = num_trajectories × num_candidates; "
            "'difficulty_adaptive': budget scales linearly with puzzle Rating "
            "between budget_min and budget_max"
        ),
    )
    parser.add_argument(
        "--budget_min", type=int, default=None,
        help="Min total trajectory budget for easiest puzzles (default: num_trajectories × num_candidates)"
    )
    parser.add_argument(
        "--budget_max", type=int, default=None,
        help="Max total trajectory budget for hardest puzzles (default: 2 × num_trajectories × num_candidates)"
    )
    parser.add_argument(
        "--rating_min", type=float, default=500.0,
        help="Puzzle Rating treated as minimum difficulty for budget scaling"
    )
    parser.add_argument(
        "--rating_max", type=float, default=3000.0,
        help="Puzzle Rating treated as maximum difficulty for budget scaling"
    )

    # ── Hardware ─────────────────────────────────────────────────────────────── #
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--num_workers", type=int, default=1,
        help="Stockfish annotation worker processes (recommended >1 for model policies)"
    )
    parser.add_argument(
        "--trajectory_batch_size", type=int, default=128,
        help="Sub-batch size for batched trajectory generation (model policies only). "
             "For efficiency, set num_candidates * num_trajectories to a multiple of 16."
    )
    parser.add_argument(
        "--position_batch_size", type=int, default=2,
        help="Number of positions to process together in one batched model call "
             "(model/hf_model policies only). Effective GPU batch = "
             "position_batch_size × num_candidates × num_trajectories. "
             "Set to a value >1 to improve GPU utilisation when individual "
             "positions have few candidates/trajectories."
    )
    parser.add_argument(
        "--tokens_per_move", type=int, default=6,
        help="Token budget per move used to compute max_new_tokens for batched "
             "trajectory generation (max_new_tokens = max_depth * tokens_per_move)."
    )
    parser.add_argument(
        "--debug", action="store_true", default=False,
        help="Enable DEBUG-level logging (shows EXPAND node decisions, entropy, "
             "budget splits, trajectory counts) – stockfish policy only.",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )

    # Fill defaults
    if args.num_trajectories_answer is None:
        args.num_trajectories_answer = 2 * args.num_trajectories
    if args.num_trajectories_mean is None:
        args.num_trajectories_mean = float(args.num_trajectories)
    if args.budget_min is None:
        args.budget_min = args.num_trajectories * args.num_candidates
    if args.budget_max is None:
        args.budget_max = 2 * args.num_trajectories * args.num_candidates

    # ── Load data ─────────────────────────────────────────────────────────────── #
    print(f"Reading data from {args.data_path} …")
    df = pd.read_csv(args.data_path)
    if "ctx" not in df.columns:
        raise ValueError("CSV must have a 'ctx' column with PGN positions")

    has_ground_truth = "Moves" in df.columns or "moves" in df.columns
    moves_col = ("Moves" if "Moves" in df.columns else "moves") if has_ground_truth else ""
    if has_ground_truth:
        print(f"Found ground-truth moves in '{moves_col}' column")

    if args.num_samples is not None:
        start = args.num_skip_samples or 0
        df = df.iloc[start : start + args.num_samples]

    print(f"Processing {len(df)} positions …")
    print(f"  sampling_policy     : {args.sampling_policy}")
    print(
        f"  depth_mode          : {args.depth_mode}"
        + (
            f"  (depth lognormal mean={args.depth_mean}, sigma={args.depth_sigma}, min={args.depth_min};"
            f" traj lognormal mean={args.num_trajectories_mean}, sigma={args.num_trajectories_sigma},"
            f" clip=[{args.num_trajectories_min}, {args.num_trajectories}])"
            if args.depth_mode == "random"
            else f"  (fixed depth={args.trajectory_depth})"
        )
    )
    print(f"  inject_answer       : {args.inject_answer}")
    print(f"  num_candidates      : {args.num_candidates}")
    print(f"  num_trajectories    : {args.num_trajectories}")
    print(f"  num_traj_answer     : {args.num_trajectories_answer}")
    print(f"  max_puzzle_moves    : {args.max_puzzle_moves}")
    print(f"  call_env_token      : {args.call_env_token!r}")
    print(f"  traversal_methods   : {args.traversal_methods}")
    print(f"  evaluation_policy   : {args.evaluation_policy}")

    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    sampling_policy = create_sampling_policy(args, device)
    stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

    sf_annotation_pool = None
    if args.num_workers > 1:
        print(f"  Using {args.num_workers} Stockfish annotation workers")
        sf_annotation_pool = multiprocessing.Pool(
            processes=args.num_workers,
            initializer=_init_sf_worker,
            initargs=(args.stockfish_path,),
        )

    traj_batch_size = getattr(args, "trajectory_batch_size", 64)
    tokens_per_move = getattr(args, "tokens_per_move", 6)
    position_batch_size = getattr(args, "position_batch_size", 1)
    use_position_batching = (
        isinstance(sampling_policy, ModelSamplingPolicy) and position_batch_size > 1
    )

    results: List[dict] = []
    cot_format_outputs: List[dict] = []

    try:
        if use_position_batching:
            rows_list = list(df.iterrows())
            n_batches = (len(rows_list) + position_batch_size - 1) // position_batch_size
            for batch_start in tqdm(range(0, len(rows_list), position_batch_size),
                                    total=n_batches, desc="Generating"):
                batch_rows = rows_list[batch_start : batch_start + position_batch_size]

                # Phase 1: parse positions and build slot specs
                specs = []
                for idx, row in batch_rows:
                    spec = prepare_position_slots(
                        idx=idx,
                        row_dict=row.to_dict(),
                        args=args,
                        sampling_policy=sampling_policy,
                        stockfish_engine=stockfish_engine,
                        has_ground_truth=has_ground_truth,
                        moves_col=moves_col,
                        rng=rng,
                    )
                    specs.append(spec)

                valid_specs = [s for s in specs if s is not None]
                if not valid_specs:
                    continue

                # Phase 2: single batched model call across all positions
                gen_per_cand_list = generate_trajectories_multi_position_batched(
                    position_specs=valid_specs,
                    sampling_policy=sampling_policy,
                    batch_size=traj_batch_size,
                    tokens_per_move=tokens_per_move,
                )

                # Phase 3: finalize each position
                for spec, gen_per_cand in zip(valid_specs, gen_per_cand_list):
                    out = finalize_position_puzzle(
                        spec=spec,
                        gen_per_cand=gen_per_cand,
                        args=args,
                        stockfish_engine=stockfish_engine,
                        sf_annotation_pool=sf_annotation_pool,
                    )
                    if out is not None:
                        result, cot_entry = out
                        results.append(result)
                        if cot_entry is not None:
                            cot_format_outputs.append(cot_entry)
        else:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating"):
                out = process_one_position_puzzle(
                    idx=idx,
                    row_dict=row.to_dict(),
                    args=args,
                    stockfish_engine=stockfish_engine,
                    sampling_policy=sampling_policy,
                    has_ground_truth=has_ground_truth,
                    moves_col=moves_col,
                    sf_annotation_pool=sf_annotation_pool,
                    rng=rng,
                )
                if out is not None:
                    result, cot_entry = out
                    results.append(result)
                    if cot_entry is not None:
                        cot_format_outputs.append(cot_entry)
    finally:
        if sf_annotation_pool is not None:
            sf_annotation_pool.terminate()
            sf_annotation_pool.join()
        stockfish_engine.quit()
        # Close the sampling-policy engine if it is a stockfish closure
        # (it opens its own subprocess that must be explicitly terminated).
        _policy_engine = getattr(sampling_policy, "_engine", None)
        if _policy_engine is not None:
            try:
                _policy_engine.quit()
            except Exception:
                pass

    results.sort(key=lambda r: r["index"])
    cot_format_outputs.sort(key=lambda e: e["index"])

    matches = sum(1 for r in results if r.get("is_match") is True)
    total_gt = sum(1 for r in results if r.get("is_match") is not None)

    # ── Build output path with hyperparams encoded in directory name ─────────── #
    base = Path(args.output_path)
    hyperparam_dir = (
        f"{args.sampling_policy}"
        f"_seed{args.seed}"
        f"_eval{args.evaluation_policy}"
        f"_cand{args.candidate_selection}"
        f"_nc{args.num_candidates}"
        f"_nt{args.num_trajectories}"
        f"_nta{args.num_trajectories_answer}"
        f"_d{args.trajectory_depth}"
        f"_dm{args.depth_mode}"
        f"_wt{args.win_threshold}"
        f"_pm{args.max_puzzle_moves}"
        + ("_inj" if args.inject_answer else "")
    )
    if args.num_samples is not None and args.num_skip_samples is not None:
        shard = f"data{args.data_name}_shard{args.num_skip_samples}-{args.num_skip_samples + args.num_samples}.json"
    elif args.num_samples is not None:
        shard = f"data{args.data_name}_n{args.num_samples}.json"
    else:
        shard = f"data{args.data_name}_results.json"

    base_dir = (base.parent / base.stem) if base.suffix else base
    output_path = base_dir / hyperparam_dir / shard
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Summary statistics ────────────────────────────────────────────────────── #
    total_traj = sum(r["num_trajectories_total"] for r in results)
    total_wins = sum(r["leaf_label_distribution"]["wins"] for r in results)
    total_draws = sum(r["leaf_label_distribution"]["draws"] for r in results)
    total_losses = sum(r["leaf_label_distribution"]["losses"] for r in results)
    total_labels = total_wins + total_draws + total_losses

    summary: dict = {
        "total_positions": len(results),
        "total_trajectories": total_traj,
        "avg_trajectories_per_position": total_traj / len(results) if results else 0,
        "traversal_methods": args.traversal_methods,
        "evaluation_policy": args.evaluation_policy,
        "win_threshold": args.win_threshold,
        "inject_answer": args.inject_answer,
        "depth_mode": args.depth_mode,
        "max_puzzle_moves": args.max_puzzle_moves,
        "leaf_label_distribution": {
            "total_wins": total_wins,
            "total_draws": total_draws,
            "total_losses": total_losses,
            "win_rate": total_wins / total_labels if total_labels else 0,
            "draw_rate": total_draws / total_labels if total_labels else 0,
            "loss_rate": total_losses / total_labels if total_labels else 0,
        },
    }

    if has_ground_truth and total_gt > 0:
        acc = matches / total_gt
        summary["ground_truth_comparison"] = {
            "total_compared": total_gt,
            "matches": matches,
            "accuracy": acc,
            "accuracy_percent": f"{acc * 100:.2f}%",
        }

    # ── Save JSON results ─────────────────────────────────────────────────────── #
    with open(output_path, "w") as f:
        json.dump(
            {"config": vars(args), "summary": summary, "results": results},
            f,
            indent=2,
        )
    print(f"Results saved to {output_path}")

    # ── Save readable CoT text file ──────────────────────────────────────────── #
    if args.cot_output_path and cot_format_outputs:
        cot_path = Path(args.cot_output_path)
        cot_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cot_path, "w") as f:
            for entry in cot_format_outputs:
                f.write(f"# Position {entry['index']}\n")
                f.write(f"# PGN    : {entry['pgn']}\n")
                f.write(f"# Moves  : {entry['moves_list']}\n")
                f.write(f"# Target : {entry['target_move_san']}\n")
                _extra_method_names = (
                    ["best_move_only", "solution_continuation", "successful_only_cot"]
                    + [f"stratified_{m}" for m in args.traversal_methods
                       if m in ("dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier")]
                    + ["monotonic_improvement"]
                )
                for method in list(args.traversal_methods) + ["trajectory_sep"] + _extra_method_names:
                    f.write(
                        f"# [{method}] multi-step CoT:\n"
                        f"{entry.get(f'cot_{method}', '')}\n"
                    )
                    f.write(
                        f"# [{method}] first-move-only CoT:\n"
                        f"{entry.get(f'cot_{method}_first_move', '')}\n"
                    )
                    f.write(
                        f"# [{method}] puzzle sequence only:\n"
                        f"{entry.get(f'puzzle_seq_{method}', '')}\n"
                    )
                f.write("\n")
        print(f"CoT format saved to {cot_path}")

    # ── Print summary ─────────────────────────────────────────────────────────── #
    print(f"\n{'='*60}")
    print(f"Processed    : {len(results)} positions")
    print(f"Trajectories : {total_traj} total")
    if total_labels:
        print(
            f"Leaf labels  : +1={total_wins}  0={total_draws}  -1={total_losses}"
            f"  win_rate={100 * total_wins / total_labels:.1f}%"
        )
    if has_ground_truth and total_gt > 0:
        print(
            f"GT accuracy  : {summary['ground_truth_comparison']['accuracy_percent']}"
            f"  ({matches}/{total_gt})"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
