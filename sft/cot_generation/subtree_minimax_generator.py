import sys
import argparse
import multiprocessing
import pandas as pd
import torch
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import chess
import chess.engine
import chess.pgn
import json
from tqdm import tqdm
import numpy as np
import io
import random
import re
import traceback
from collections import deque

# Add parent directory to path so we can import cot_generation
parent_dir = Path(__file__).parent.parent
if str(parent_dir) not in sys.path:
    sys.path.insert(0, str(parent_dir))

from cot_generation.policy import (
    random_sampling_policy,
    stockfish_sampling_policy,
    stockfish_eval,
)
from cot_generation.generator import TreeNode, board_from_pgn
from evaluation.inference import generate_move


def move_to_lan(board: chess.Board, move: chess.Move) -> str:
    """
    Convert a chess.Move to LAN string (combined format).
    """
    if board.is_castling(move):
        if chess.square_file(move.to_square) == 6:
            castle = "O-O"
        else:
            castle = "O-O-O"
        board.push(move)
        suffix = "#" if board.is_checkmate() else "+" if board.is_check() else ""
        board.pop()
        return castle + suffix

    parts = []
    piece = board.piece_at(move.from_square)
    parts.append(piece.symbol().upper() if piece else "P")
    parts.append(chess.square_name(move.from_square))
    if board.is_capture(move):
        parts.append("x")
    parts.append(chess.square_name(move.to_square))
    if move.promotion:
        parts.append("=")
        parts.append(chess.piece_symbol(move.promotion).upper())
    board.push(move)
    if board.is_checkmate():
        parts.append("#")
    elif board.is_check():
        parts.append("+")
    board.pop()
    return "".join(parts)


class ModelSamplingPolicy:
    """Wrapper for model-based sampling policy."""

    def __init__(self, model, tokenizer, device, temperature=1.0, seed=None):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.temperature = temperature
        self.model.eval()
        self.seed = seed

    def _board_to_state(self, board: chess.Board) -> str:
        """Convert a chess board to PGN state string."""
        game = chess.pgn.Game()
        node = game
        move_stack = []
        for move in board.move_stack:
            move_stack.append(move)
        for move in move_stack:
            node = node.add_variation(move)
        exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
        pgn_str = game.accept(exporter).strip()
        return pgn_str if pgn_str else ""

    def _parse_move_from_output(self, first_move: str, board: chess.Board, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        """Try to parse a move string into a chess.Move from legal_moves."""
        if not first_move:
            return None
        try:
            parsed_move = chess.Move.from_uci(first_move)
            if parsed_move in legal_moves:
                return parsed_move
        except Exception:
            pass
        try:
            parsed_move = board.parse_san(first_move)
            if parsed_move in legal_moves:
                return parsed_move
        except Exception:
            pass
        for move in legal_moves:
            if move.uci() == first_move or board.san(move) == first_move:
                return move
        return None

    def __call__(self, board: chess.Board, legal_moves: List[chess.Move]) -> Optional[chess.Move]:
        if not legal_moves:
            return None

        state = self._board_to_state(board)

        try:
            full_output, first_move = generate_move(
                model=self.model,
                tokenizer=self.tokenizer,
                state=state,
                device=self.device,
                max_new_tokens=20,
                temperature=self.temperature,
                top_k=None,
                seed=self.seed,
            )
            parsed_move = self._parse_move_from_output(first_move, board, legal_moves)
            if parsed_move:
                return parsed_move
            return random.choice(legal_moves)
        except Exception as e:
            print(f"Error in model sampling: {e}")
            traceback.print_exc()
            return random.choice(legal_moves)

    def batch_sample(self, boards: List[chess.Board], legal_moves_list: List[List[chess.Move]], batch_size: int = 32) -> List[Optional[chess.Move]]:
        """
        Batch sample moves from multiple boards.
        """
        if len(boards) != len(legal_moves_list):
            raise ValueError("boards and legal_moves_list must have the same length")

        if not boards:
            return []

        states = [self._board_to_state(board) for board in boards]

        try:
            from evaluation.inference import generate_moves_batch
            moves, first_moves = generate_moves_batch(
                model=self.model,
                tokenizer=self.tokenizer,
                states=states,
                device=self.device,
                batch_size=batch_size,
                max_new_tokens=20,
                temperature=self.temperature,
                top_k=None,
                seed=self.seed,
            )

            results = []
            for i, (board, legal_moves, first_move) in enumerate(zip(boards, legal_moves_list, first_moves)):
                if not legal_moves:
                    results.append(None)
                    continue
                parsed_move = self._parse_move_from_output(first_move, board, legal_moves)
                if parsed_move:
                    results.append(parsed_move)
                else:
                    results.append(random.choice(legal_moves))
            return results
        except Exception as e:
            print(f"Error in batch model sampling: {e}")
            traceback.print_exc()
            return [random.choice(legal_moves) if legal_moves else None for legal_moves in legal_moves_list]


def load_hf_model_and_tokenizer(
    hf_model_path: str,
    device: str,
    config_path: str,
    torch_dtype: Optional[torch.dtype] = None,
):
    """Load a HF CausalLM via from_pretrained."""
    from transformers import AutoModelForCausalLM, AutoConfig
    from llm_tokens.chess.tokenizer_factory import init_tokenizer

    if not config_path:
        raise ValueError("--config_path is required to build the SAME tokenizer")

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    tokenizer = init_tokenizer(name=cfg["tokenizer"]["name"], config=cfg["tokenizer"])

    hf_cfg = AutoConfig.from_pretrained(hf_model_path)
    hf_cfg.vocab_size = tokenizer.get_vocab_size()
    hf_cfg.bos_token_id = tokenizer.bos_id()
    hf_cfg.eos_token_id = tokenizer.eos_id()
    hf_cfg.pad_token_id = tokenizer.pad_id()

    kwargs = {"config": hf_cfg}
    if torch_dtype is not None:
        kwargs["torch_dtype"] = torch_dtype

    model = AutoModelForCausalLM.from_pretrained(hf_model_path, **kwargs)
    model = model.to(device)
    model.eval()

    return model, tokenizer


def create_sampling_policy(args, device):
    if args.sampling_policy == "random":
        return random_sampling_policy(seed=args.seed)
    if args.sampling_policy == "stockfish":
        engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

        def policy(board, legal_moves):
            return stockfish_sampling_policy(
                board,
                legal_moves,
                engine,
                time_limit=args.stockfish_time,
                temperature=args.temperature,
                seed=args.seed,
            )

        return policy
    if args.sampling_policy == "hf_model":
        if not args.model_path:
            raise ValueError("--model_path is required when using HF sampling policy")
        if not args.config_path:
            raise ValueError("--config_path is required when using HF sampling policy")
        model, tokenizer = load_hf_model_and_tokenizer(args.model_path, device, config_path=args.config_path)
        return ModelSamplingPolicy(model, tokenizer, device, temperature=args.temperature, seed=None)
    raise ValueError(f"Unknown sampling policy: {args.sampling_policy}")


# --- Trajectory Tree Node -------------------------------------------------- #

class TrajectoryTreeNode:
    """Tree node built from trajectories with generation order tracking and minimax labels."""

    def __init__(
        self,
        board: chess.Board,
        move: Optional[chess.Move] = None,
        parent: Optional["TrajectoryTreeNode"] = None,
        generation_order: int = 0,
    ):
        self.board = board
        self.move = move
        self.parent = parent
        self.children: Dict[chess.Move, "TrajectoryTreeNode"] = {}
        self.generation_order = generation_order
        self.stockfish_value: Optional[float] = None
        # New fields for minimax with labels
        self.leaf_label: Optional[int] = None  # +1, 0, or -1 for leaves
        self.minimax_value: Optional[int] = None  # Propagated minimax value


# --- Leaf Label and Minimax Functions -------------------------------------- #

def get_leaf_label(
    board: chess.Board,
    engine: chess.engine.SimpleEngine,
    root_color: chess.Color,
    time_limit: float,
    win_threshold: int = 300,
) -> int:
    """
    Returns +1 (win), 0 (draw), or -1 (loss) from root_color's perspective.

    Args:
        board: Chess board at the leaf position
        engine: Stockfish engine
        root_color: The color of the root player (for perspective)
        time_limit: Time limit for Stockfish evaluation
        win_threshold: Centipawn threshold for win/loss classification

    Returns:
        +1 if winning, 0 if draw, -1 if losing (from root_color's perspective)
    """
    # Check for terminal game state first
    if board.is_game_over():
        outcome = board.outcome()
        if outcome is None or outcome.winner is None:
            return 0  # Draw
        return 1 if outcome.winner == root_color else -1

    # Use Stockfish evaluation
    info = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score = info["score"].pov(root_color)

    # Handle mate scores
    if score.is_mate():
        mate_in = score.mate()
        return 1 if mate_in > 0 else -1

    # Use centipawn thresholds
    cp = score.score()
    if cp is None:
        return 0
    if cp > win_threshold:
        return 1
    elif cp < -win_threshold:
        return -1
    else:
        return 0


def annotate_tree_with_labels(
    node: TrajectoryTreeNode,
    engine: chess.engine.SimpleEngine,
    root_color: chess.Color,
    stockfish_time: float,
    win_threshold: int = 300,
):
    """
    Recursively annotate all leaf nodes in the tree with win/draw/loss labels.
    Also stores stockfish_value for backward compatibility and tie-breaking.
    """
    # Always compute stockfish value for tie-breaking purposes
    try:
        node.stockfish_value = stockfish_eval(node.board, engine, stockfish_time, root_color)
    except Exception as e:
        print(f"Warning: failed to eval node: {e}")
        node.stockfish_value = 0.0

    if not node.children:
        # Leaf node - get label
        try:
            node.leaf_label = get_leaf_label(
                node.board, engine, root_color, stockfish_time, win_threshold
            )
        except Exception as e:
            print(f"Warning: failed to get leaf label: {e}")
            node.leaf_label = 0
    else:
        for child in node.children.values():
            annotate_tree_with_labels(child, engine, root_color, stockfish_time, win_threshold)


def minimax_on_tree(
    node: TrajectoryTreeNode,
    root_color: chess.Color,
) -> int:
    """
    Computes minimax value over the existing tree using leaf labels.
    Returns +1, 0, or -1.

    At nodes where it's root_color's turn: maximize
    At nodes where it's opponent's turn: minimize
    """
    # Leaf node: return stored label
    if not node.children:
        return node.leaf_label if node.leaf_label is not None else 0

    child_values = [minimax_on_tree(child, root_color) for child in node.children.values()]

    # Max at root player's turn, min at opponent's turn
    if node.board.turn == root_color:
        return max(child_values)
    else:
        return min(child_values)


def compute_candidate_minimax_values(
    tree_root: TrajectoryTreeNode,
    root_color: chess.Color,
) -> Dict[chess.Move, int]:
    """
    Compute minimax value for each candidate move (direct children of root).
    """
    candidate_values = {}
    for move, child in tree_root.children.items():
        child.minimax_value = minimax_on_tree(child, root_color)
        candidate_values[move] = child.minimax_value
    return candidate_values


def select_best_move_by_minimax(
    candidate_values: Dict[chess.Move, int],
    tie_break_values: Dict[chess.Move, float],
) -> Optional[chess.Move]:
    """
    Select best move based on minimax values.
    Break ties using stockfish evaluation.
    """
    if not candidate_values:
        return None

    max_minimax = max(candidate_values.values())
    best_candidates = [mv for mv, val in candidate_values.items() if val == max_minimax]

    if len(best_candidates) == 1:
        return best_candidates[0]

    # Tie-break using stockfish evaluation
    return max(best_candidates, key=lambda mv: tie_break_values.get(mv, 0))


def label_to_string(label: int) -> str:
    """Convert label (+1, 0, -1) to string format."""
    if label == 1:
        return "+1"
    elif label == -1:
        return "-1"
    else:
        return "0"


# --- Candidate and Trajectory Generation ----------------------------------- #

def sample_candidates_with_policy(
    board: chess.Board,
    sampling_policy,
    num_candidates: int,
    batch_size: int = 32,
) -> List[chess.Move]:
    """
    Sample candidate moves using the policy (without replacement).
    Returns moves in the order they were sampled by the policy.
    """
    legal_moves = list(board.legal_moves)
    if not legal_moves:
        return []

    is_model_policy = isinstance(sampling_policy, ModelSamplingPolicy)

    if is_model_policy and num_candidates > 1:
        candidates: List[chess.Move] = []
        remaining = set(legal_moves)

        num_to_sample = min(num_candidates * 2, len(legal_moves))

        batch_boards = [board] * num_to_sample
        batch_legal_moves = [list(remaining)] * num_to_sample

        sampled_moves = sampling_policy.batch_sample(
            boards=batch_boards,
            legal_moves_list=batch_legal_moves,
            batch_size=batch_size,
        )

        seen = set()
        for mv in sampled_moves:
            if mv is not None and mv in remaining and mv not in seen:
                candidates.append(mv)
                seen.add(mv)
                remaining.remove(mv)
                if len(candidates) >= num_candidates:
                    break

        if len(candidates) < num_candidates and remaining:
            additional_needed = num_candidates - len(candidates)
            additional_to_sample = min(additional_needed * 2, len(remaining))

            batch_boards = [board] * additional_to_sample
            batch_legal_moves = [list(remaining)] * additional_to_sample

            sampled_moves = sampling_policy.batch_sample(
                boards=batch_boards,
                legal_moves_list=batch_legal_moves,
                batch_size=batch_size,
            )

            for mv in sampled_moves:
                if mv is not None and mv in remaining and mv not in seen:
                    candidates.append(mv)
                    seen.add(mv)
                    remaining.remove(mv)
                    if len(candidates) >= num_candidates:
                        break

        return candidates[:num_candidates]
    else:
        candidates: List[chess.Move] = []
        remaining = set(legal_moves)

        while len(candidates) < num_candidates and remaining:
            mv = sampling_policy(board, list(remaining))
            if mv is None or mv not in remaining:
                break
            candidates.append(mv)
            remaining.remove(mv)

        return candidates


def generate_single_trajectory(
    board: chess.Board,
    sampling_policy,
    max_depth: int,
) -> List[chess.Move]:
    """Generate a single trajectory (rollout) using the policy."""
    trajectory: List[chess.Move] = []
    current_board = board.copy()

    for _ in range(max_depth):
        if current_board.is_game_over():
            break
        legal_moves = list(current_board.legal_moves)
        if not legal_moves:
            break
        mv = sampling_policy(current_board, legal_moves)
        if mv is None:
            break
        trajectory.append(mv)
        current_board.push(mv)

    return trajectory


def generate_trajectories_for_candidate(
    root_board: chess.Board,
    candidate_move: chess.Move,
    sampling_policy,
    num_trajectories: int,
    trajectory_depth: int,
) -> List[List[chess.Move]]:
    """
    Generate multiple trajectories starting with the candidate move.
    """
    trajectories: List[List[chess.Move]] = []
    cand_board = root_board.copy()
    cand_board.push(candidate_move)

    for _ in range(num_trajectories):
        continuation = generate_single_trajectory(
            board=cand_board,
            sampling_policy=sampling_policy,
            max_depth=trajectory_depth - 1,
        )
        full_trajectory = [candidate_move] + continuation
        trajectories.append(full_trajectory)

    return trajectories


# --- Tree Building --------------------------------------------------------- #

def build_tree_from_trajectories(
    root_board: chess.Board,
    all_trajectories: List[List[chess.Move]],
) -> TrajectoryTreeNode:
    """
    Build a tree by merging trajectories with common prefixes.
    """
    root = TrajectoryTreeNode(board=root_board.copy(), generation_order=0)
    order_counter = [1]

    for trajectory in all_trajectories:
        current_node = root
        current_board = root_board.copy()

        for move in trajectory:
            if move in current_node.children:
                current_node = current_node.children[move]
                current_board.push(move)
            else:
                current_board.push(move)
                new_node = TrajectoryTreeNode(
                    board=current_board.copy(),
                    move=move,
                    parent=current_node,
                    generation_order=order_counter[0],
                )
                order_counter[0] += 1
                current_node.children[move] = new_node
                current_node = new_node

    return root


def compute_trajectory_tree_stats(node: TrajectoryTreeNode) -> Dict:
    """Compute statistics for the trajectory tree."""
    node_count = 0
    leaf_count = 0
    max_depth = 0
    branching_factors: List[int] = []

    def _dfs(n: TrajectoryTreeNode, depth: int):
        nonlocal node_count, leaf_count, max_depth
        node_count += 1

        if not n.children:
            leaf_count += 1
            if depth > max_depth:
                max_depth = depth
        else:
            branching_factors.append(len(n.children))
            for child in n.children.values():
                _dfs(child, depth + 1)

    _dfs(node, 0)

    avg_branching = sum(branching_factors) / len(branching_factors) if branching_factors else 0.0

    return {
        "node_count": node_count,
        "leaf_count": leaf_count,
        "max_depth": max_depth,
        "avg_branching_factor": avg_branching,
        "max_branching_factor": max(branching_factors) if branching_factors else 0,
    }


def collect_leaf_values_from_trajectory_tree(
    node: TrajectoryTreeNode,
) -> List[float]:
    """Collect all leaf Stockfish values from the tree."""
    values: List[float] = []

    def _collect(n: TrajectoryTreeNode):
        if not n.children:
            if n.stockfish_value is not None:
                values.append(n.stockfish_value)
        else:
            for child in n.children.values():
                _collect(child)

    _collect(node)
    return values


def collect_leaf_labels_from_trajectory_tree(
    node: TrajectoryTreeNode,
) -> List[int]:
    """Collect all leaf labels from the tree."""
    labels: List[int] = []

    def _collect(n: TrajectoryTreeNode):
        if not n.children:
            if n.leaf_label is not None:
                labels.append(n.leaf_label)
        else:
            for child in n.children.values():
                _collect(child)

    _collect(node)
    return labels


# --- Traversal Methods with Result Labels ---------------------------------- #

def dfs_policy_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_directions: bool = False,
    include_subtree_sep: bool = False,
    include_result_labels: bool = True,
) -> str:
    """
    DFS traversal ordered by generation order (policy order).
    Appends <result>+1</result>, <result>0</result>, or <result>-1</result> at leaf nodes.
    """
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode):
        ordered_children = sorted(n.children.items(), key=lambda x: x[1].generation_order)

        for mv, child in ordered_children:
            if include_directions:
                tokens.append("<down>")
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            board.push(mv)

            # Check if this is a leaf node - append result label
            if not child.children and include_result_labels:
                label = child.leaf_label if child.leaf_label is not None else 0
                tokens.append(f"<result>{label_to_string(label)}</result>")

            _dfs(child)
            board.pop()
            if include_directions:
                tokens.append("<up>")

    # Process root's children (candidates) with optional <sep>
    ordered_root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)

    for i, (mv, child) in enumerate(ordered_root_children):
        if include_subtree_sep and i > 0:
            tokens.append("<sep>")
        if include_directions:
            tokens.append("<down>")
        try:
            tokens.append(move_to_lan(board, mv))
        except Exception:
            tokens.append(mv.uci())
        board.push(mv)

        # Check if this is a leaf node - append result label
        if not child.children and include_result_labels:
            label = child.leaf_label if child.leaf_label is not None else 0
            tokens.append(f"<result>{label_to_string(label)}</result>")

        _dfs(child)
        board.pop()
        if include_directions:
            tokens.append("<up>")

    return " ".join(tokens)


def dfs_verifier_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_directions: bool = False,
    include_subtree_sep: bool = False,
    include_result_labels: bool = True,
) -> str:
    """
    DFS traversal ordered by Stockfish value (verifier order).
    Appends <result> tags at leaf nodes.
    """
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode):
        ordered_children = sorted(
            n.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )

        for mv, child in ordered_children:
            if include_directions:
                tokens.append("<down>")
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            board.push(mv)

            if not child.children and include_result_labels:
                label = child.leaf_label if child.leaf_label is not None else 0
                tokens.append(f"<result>{label_to_string(label)}</result>")

            _dfs(child)
            board.pop()
            if include_directions:
                tokens.append("<up>")

    ordered_root_children = sorted(
        node.children.items(),
        key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
        reverse=True,
    )

    for i, (mv, child) in enumerate(ordered_root_children):
        if include_subtree_sep and i > 0:
            tokens.append("<sep>")
        if include_directions:
            tokens.append("<down>")
        try:
            tokens.append(move_to_lan(board, mv))
        except Exception:
            tokens.append(mv.uci())
        board.push(mv)

        if not child.children and include_result_labels:
            label = child.leaf_label if child.leaf_label is not None else 0
            tokens.append(f"<result>{label_to_string(label)}</result>")

        _dfs(child)
        board.pop()
        if include_directions:
            tokens.append("<up>")

    return " ".join(tokens)


def bfs_policy_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_level_markers: bool = False,
    include_subtree_sep: bool = False,
    include_result_labels: bool = True,
) -> str:
    """
    BFS traversal ordered by generation order within each level.
    Appends <result> tags at leaf nodes.
    """
    if include_subtree_sep:
        subtree_strs: List[str] = []
        root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)

        for mv, child in root_children:
            temp_root = TrajectoryTreeNode(board=root_board.copy())
            temp_root.children[mv] = child
            subtree_str = bfs_policy_order(root_board, temp_root, include_level_markers, include_subtree_sep=False, include_result_labels=include_result_labels)
            subtree_strs.append(subtree_str)

        return " <sep> ".join(subtree_strs)

    tokens: List[str] = []
    queue: deque = deque()

    root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)
    current_level_size = len(root_children)
    next_level_size = 0

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy()))

    if include_level_markers and current_level_size > 0:
        tokens.append("<level>")

    nodes_in_level = 0
    while queue:
        current, move, parent_board = queue.popleft()
        nodes_in_level += 1

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())

        # Append result label for leaf nodes
        if not current.children and include_result_labels:
            label = current.leaf_label if current.leaf_label is not None else 0
            tokens.append(f"<result>{label_to_string(label)}</result>")

        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(current.children.items(), key=lambda x: x[1].generation_order)
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy()))
            next_level_size += 1

        if nodes_in_level == current_level_size:
            if include_level_markers and next_level_size > 0:
                tokens.append("<level>")
            current_level_size = next_level_size
            next_level_size = 0
            nodes_in_level = 0

    return " ".join(tokens)


def bfs_verifier_order(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_level_markers: bool = False,
    include_subtree_sep: bool = False,
    include_result_labels: bool = True,
) -> str:
    """
    BFS traversal ordered by Stockfish value within each level.
    Appends <result> tags at leaf nodes.
    """
    if include_subtree_sep:
        subtree_strs: List[str] = []
        root_children = sorted(
            node.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )

        for mv, child in root_children:
            temp_root = TrajectoryTreeNode(board=root_board.copy())
            temp_root.children[mv] = child
            subtree_str = bfs_verifier_order(root_board, temp_root, include_level_markers, include_subtree_sep=False, include_result_labels=include_result_labels)
            subtree_strs.append(subtree_str)

        return " <sep> ".join(subtree_strs)

    tokens: List[str] = []
    queue: deque = deque()

    root_children = sorted(
        node.children.items(),
        key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
        reverse=True,
    )
    current_level_size = len(root_children)
    next_level_size = 0

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy()))

    if include_level_markers and current_level_size > 0:
        tokens.append("<level>")

    nodes_in_level = 0
    while queue:
        current, move, parent_board = queue.popleft()
        nodes_in_level += 1

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())

        if not current.children and include_result_labels:
            label = current.leaf_label if current.leaf_label is not None else 0
            tokens.append(f"<result>{label_to_string(label)}</result>")

        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(
            current.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy()))
            next_level_size += 1

        if nodes_in_level == current_level_size:
            if include_level_markers and next_level_size > 0:
                tokens.append("<level>")
            current_level_size = next_level_size
            next_level_size = 0
            nodes_in_level = 0

    return " ".join(tokens)


def dfs_policy_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_result_labels: bool = True,
) -> str:
    """DFS traversal with layer markers [l{depth}] and result labels at leaves."""
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode, depth: int):
        ordered_children = sorted(n.children.items(), key=lambda x: x[1].generation_order)

        for mv, child in ordered_children:
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            tokens.append(f"[l{depth}]")
            board.push(mv)

            if not child.children and include_result_labels:
                label = child.leaf_label if child.leaf_label is not None else 0
                tokens.append(f"<result>{label_to_string(label)}</result>")

            _dfs(child, depth + 1)
            board.pop()

    _dfs(node, 1)
    return " ".join(tokens)


def dfs_verifier_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_result_labels: bool = True,
) -> str:
    """DFS verifier order with layer markers and result labels."""
    tokens: List[str] = []
    board = root_board.copy()

    def _dfs(n: TrajectoryTreeNode, depth: int):
        ordered_children = sorted(
            n.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )

        for mv, child in ordered_children:
            try:
                tokens.append(move_to_lan(board, mv))
            except Exception:
                tokens.append(mv.uci())
            tokens.append(f"[l{depth}]")
            board.push(mv)

            if not child.children and include_result_labels:
                label = child.leaf_label if child.leaf_label is not None else 0
                tokens.append(f"<result>{label_to_string(label)}</result>")

            _dfs(child, depth + 1)
            board.pop()

    _dfs(node, 1)
    return " ".join(tokens)


def bfs_policy_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_result_labels: bool = True,
) -> str:
    """BFS policy order with layer markers and result labels."""
    tokens: List[str] = []
    queue: deque = deque()

    root_children = sorted(node.children.items(), key=lambda x: x[1].generation_order)

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy(), 1))

    while queue:
        current, move, parent_board, depth = queue.popleft()

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())
        tokens.append(f"[l{depth}]")

        if not current.children and include_result_labels:
            label = current.leaf_label if current.leaf_label is not None else 0
            tokens.append(f"<result>{label_to_string(label)}</result>")

        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(current.children.items(), key=lambda x: x[1].generation_order)
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy(), depth + 1))

    return " ".join(tokens)


def bfs_verifier_order_with_layers(
    root_board: chess.Board,
    node: TrajectoryTreeNode,
    include_result_labels: bool = True,
) -> str:
    """BFS verifier order with layer markers and result labels."""
    tokens: List[str] = []
    queue: deque = deque()

    root_children = sorted(
        node.children.items(),
        key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
        reverse=True,
    )

    for mv, child in root_children:
        queue.append((child, mv, root_board.copy(), 1))

    while queue:
        current, move, parent_board, depth = queue.popleft()

        try:
            tokens.append(move_to_lan(parent_board, move))
        except Exception:
            tokens.append(move.uci())
        tokens.append(f"[l{depth}]")

        if not current.children and include_result_labels:
            label = current.leaf_label if current.leaf_label is not None else 0
            tokens.append(f"<result>{label_to_string(label)}</result>")

        current_board = parent_board.copy()
        current_board.push(move)
        ordered_children = sorted(
            current.children.items(),
            key=lambda x: x[1].stockfish_value if x[1].stockfish_value is not None else 0.0,
            reverse=True,
        )
        for mv, child in ordered_children:
            queue.append((child, mv, current_board.copy(), depth + 1))

    return " ".join(tokens)


# --- CoT Generation -------------------------------------------------------- #

def generate_traversal_cot(
    traversal_str: str,
    target_move: chess.Move,
    root_board: chess.Board,
    pgn_context: str = "",
) -> dict:
    """Build CoT string from a single traversal string."""
    if not traversal_str or not target_move:
        return {
            "cot_format": "",
            "cot_format_with_context": "",
            "target_move_lan": "",
        }

    target_lan = move_to_lan(root_board, target_move)
    cot_format = f"<T> {traversal_str} </T> {target_lan}"
    cot_format_with_context = f"{pgn_context} {cot_format}".strip() if pgn_context else cot_format

    return {
        "cot_format": cot_format,
        "cot_format_with_context": cot_format_with_context,
        "target_move_lan": target_lan,
    }


def trajectory_to_string_with_label(
    root_board: chess.Board,
    trajectory: List[chess.Move],
    leaf_label: int,
    include_result_labels: bool = True,
) -> str:
    """Convert a trajectory to string, optionally with result label at the end."""
    tokens: List[str] = []
    board = root_board.copy()
    for move in trajectory:
        try:
            tokens.append(move_to_lan(board, move))
        except Exception:
            tokens.append(move.uci())
        board.push(move)
    if include_result_labels:
        tokens.append(f"<result>{label_to_string(leaf_label)}</result>")
    return " ".join(tokens)


def generate_trajectory_sep_cot(
    root_board: chess.Board,
    all_trajectories: List[List[chess.Move]],
    trajectory_labels: List[int],
    target_move: chess.Move,
    pgn_context: str = "",
    include_result_labels: bool = True,
) -> dict:
    """
    Build CoT string where each trajectory is separated by <sep> and ends with <result>.
    """
    if not all_trajectories or not target_move:
        return {
            "cot_format": "",
            "cot_format_with_context": "",
            "target_move_lan": "",
        }

    trajectory_strs: List[str] = []
    for trajectory, label in zip(all_trajectories, trajectory_labels):
        traj_str = trajectory_to_string_with_label(root_board, trajectory, label, include_result_labels=include_result_labels)
        trajectory_strs.append(traj_str)

    cot_parts = ["<T>"]
    for traj_str in trajectory_strs:
        cot_parts.append(traj_str)
        cot_parts.append("<sep>")
    cot_parts.pop()  # remove the last <sep>
    cot_parts.append("</T>")
    target_lan = move_to_lan(root_board, target_move)
    cot_parts.append(target_lan)
    cot_format = " ".join(cot_parts)
    cot_format_with_context = f"{pgn_context} {cot_format}".strip() if pgn_context else cot_format

    return {
        "cot_format": cot_format,
        "cot_format_with_context": cot_format_with_context,
        "target_move_lan": target_lan,
        "num_trajectories": len(trajectory_strs),
    }


# --- Legacy utilities ------------------------------------------------------ #

def order_moves_by_stockfish(
    board: chess.Board,
    legal_moves: List[chess.Move],
    engine: chess.engine.SimpleEngine,
    root_color: chess.Color,
    stockfish_time: float,
) -> List[Tuple[chess.Move, float]]:
    scored_moves: List[Tuple[chess.Move, float]] = []
    for mv in legal_moves:
        b = board.copy()
        b.push(mv)
        val = stockfish_eval(b, engine, stockfish_time, root_color)
        scored_moves.append((mv, val))
    scored_moves.sort(key=lambda x: x[1], reverse=True)
    return scored_moves


# --- Parallel worker support ----------------------------------------------- #

# --- Node-level Stockfish annotation pool (used with any sampling policy) ---

_sf_worker_engine: Optional[chess.engine.SimpleEngine] = None


def _init_sf_worker(stockfish_path: str):
    """Initialize a Stockfish engine for node annotation workers."""
    import atexit
    global _sf_worker_engine
    _sf_worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    atexit.register(_sf_worker_engine.quit)


def _sf_eval_node_worker(args_tuple):
    """Evaluate a single board position (stockfish_value + optional leaf_label)."""
    fen, is_leaf, root_color, stockfish_time, win_threshold = args_tuple
    board = chess.Board(fen)
    try:
        sf_val = stockfish_eval(board, _sf_worker_engine, stockfish_time, root_color)
    except Exception:
        sf_val = 0.0
    leaf_label = None
    if is_leaf:
        try:
            leaf_label = get_leaf_label(board, _sf_worker_engine, root_color, stockfish_time, win_threshold)
        except Exception:
            leaf_label = 0
    return sf_val, leaf_label


def annotate_tree_with_labels_parallel(
    node: TrajectoryTreeNode,
    pool: "multiprocessing.Pool",
    root_color: chess.Color,
    stockfish_time: float,
    win_threshold: int = 300,
):
    """
    Annotate all nodes using a pre-created pool of Stockfish workers.
    Collects all nodes, evaluates in parallel, writes results back.
    """
    all_nodes: List[TrajectoryTreeNode] = []

    def _collect(n: TrajectoryTreeNode):
        all_nodes.append(n)
        for child in n.children.values():
            _collect(child)

    _collect(node)

    work_items = [
        (n.board.fen(), not bool(n.children), root_color, stockfish_time, win_threshold)
        for n in all_nodes
    ]

    results = pool.map(_sf_eval_node_worker, work_items)

    for n, (sf_val, leaf_label) in zip(all_nodes, results):
        n.stockfish_value = sf_val
        if not n.children:
            n.leaf_label = leaf_label


# --- Position-level pool (random/stockfish sampling policy only) -------------

_worker_engine: Optional[chess.engine.SimpleEngine] = None
_worker_sampling_policy = None


def _init_worker(stockfish_path: str, policy_type: str, policy_kwargs: dict):
    """Initialize per-worker Stockfish engine and sampling policy (called once per subprocess)."""
    import atexit
    global _worker_engine, _worker_sampling_policy
    _worker_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    atexit.register(_worker_engine.quit)

    if policy_type == "random":
        _worker_sampling_policy = random_sampling_policy(seed=policy_kwargs.get("seed", 42))
    elif policy_type == "stockfish":
        sf_engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        atexit.register(sf_engine.quit)
        stockfish_time = policy_kwargs.get("stockfish_time", 0.01)
        temperature = policy_kwargs.get("temperature", 1.0)
        seed = policy_kwargs.get("seed", 42)

        def policy(board, legal_moves, _e=sf_engine, _t=stockfish_time, _temp=temperature, _s=seed):
            return stockfish_sampling_policy(board, legal_moves, _e, time_limit=_t, temperature=_temp, seed=_s)

        _worker_sampling_policy = policy
    else:
        raise ValueError(f"Policy '{policy_type}' does not support --num_workers > 1")


def _worker_fn(work_item):
    """Top-level worker function for position-level multiprocessing pool."""
    import types
    idx, row_dict, args_dict, has_ground_truth, moves_col = work_item
    args = types.SimpleNamespace(**args_dict)
    return process_one_position(
        idx, row_dict, args, _worker_engine, _worker_sampling_policy, has_ground_truth, moves_col,
        sf_annotation_pool=None,  # each worker uses its own engine sequentially
    )


def process_one_position(
    idx: int,
    row_dict: dict,
    args,
    stockfish_engine: chess.engine.SimpleEngine,
    sampling_policy,
    has_ground_truth: bool,
    moves_col: str,
    sf_annotation_pool=None,  # if provided, uses parallel Stockfish annotation
):
    """
    Process a single chess position.
    Returns (result_dict, cot_entry_or_None) on success, or None if the position is skipped/errors.
    """
    pgn_str = row_dict.get("ctx")
    if pgn_str is np.nan or pgn_str is None:
        print(f"Skipping position {idx} because it has no PGN string")
        return None

    ground_truth_move = None
    is_match = None
    actual_pgn = pgn_str
    moves_list: List[str] = []

    if has_ground_truth and pd.notna(row_dict.get(moves_col)):
        moves_str = str(row_dict[moves_col]).strip()
        moves_list = moves_str.split()
        if len(moves_list) >= 2:
            first_move_san = moves_list[0]
            ground_truth_san = moves_list[1]
            ground_truth_move = ground_truth_san
            try:
                game = chess.pgn.read_game(io.StringIO(pgn_str))
                board = game.end().board() if game is not None else chess.Board()
                opp_move = board.parse_san(first_move_san)
                board.push(opp_move)
                new_game = chess.pgn.Game.from_board(board)
                exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
                new_pgn = new_game.accept(exporter)
                new_pgn = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", new_pgn).strip()
                actual_pgn = new_pgn
            except Exception as e:
                print(f"Error applying first move for position {idx}: {e}")
                return None
        else:
            print(f"Skipping position {idx}: not enough moves in Moves column")
            return None

    try:
        game = chess.pgn.read_game(io.StringIO(actual_pgn))
        board = game.end().board() if game is not None else chess.Board()
        root_color = board.turn

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            print(f"Position {idx} has no legal moves (game over)")
            return None

        # Step 1: Generate candidates
        if args.candidate_selection == "policy":
            candidate_moves = sample_candidates_with_policy(
                board=board,
                sampling_policy=sampling_policy,
                num_candidates=args.num_candidates,
            )
        else:
            scored_moves = order_moves_by_stockfish(
                board=board,
                legal_moves=legal_moves,
                engine=stockfish_engine,
                root_color=root_color,
                stockfish_time=args.stockfish_time,
            )
            candidate_moves = [mv for mv, _ in scored_moves[: min(args.num_candidates, len(scored_moves))]]

        if not candidate_moves:
            print(f"Position {idx}: no candidates generated")
            return None

        # Step 2: Generate trajectories for each candidate
        all_trajectories: List[List[chess.Move]] = []
        trajectories_per_candidate: Dict[chess.Move, List[List[chess.Move]]] = {}

        if args.num_trajectories <= 0 or args.trajectory_depth <= 1:
            for cand_move in candidate_moves:
                trajectories_per_candidate[cand_move] = [[cand_move]]
                all_trajectories.append([cand_move])
        else:
            for cand_move in candidate_moves:
                cand_trajectories = generate_trajectories_for_candidate(
                    root_board=board,
                    candidate_move=cand_move,
                    sampling_policy=sampling_policy,
                    num_trajectories=args.num_trajectories,
                    trajectory_depth=args.trajectory_depth,
                )
                trajectories_per_candidate[cand_move] = cand_trajectories
                all_trajectories.extend(cand_trajectories)

        # Step 3: Build tree from all trajectories
        tree_root = build_tree_from_trajectories(
            root_board=board,
            all_trajectories=all_trajectories,
        )

        # Step 4: Annotate tree with labels (and stockfish values for tie-breaking)
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

        # Compute tree statistics
        tree_stats = compute_trajectory_tree_stats(tree_root)

        # Collect leaf labels and values
        leaf_labels = collect_leaf_labels_from_trajectory_tree(tree_root)
        leaf_values = collect_leaf_values_from_trajectory_tree(tree_root)

        # Compute minimax values for candidates
        minimax_values = compute_candidate_minimax_values(tree_root, root_color)

        # Compute stockfish direct values for tie-breaking and backward compatibility
        direct_move_values: Dict[chess.Move, float] = {}
        move_leaf_max_values: Dict[chess.Move, float] = {}
        candidate_details: List[Dict] = []

        for cand_move in candidate_moves:
            if cand_move in tree_root.children:
                cand_node = tree_root.children[cand_move]
                cand_leaf_values = collect_leaf_values_from_trajectory_tree(cand_node)
                cand_leaf_labels = collect_leaf_labels_from_trajectory_tree(cand_node)
                best_leaf_val = max(cand_leaf_values) if cand_leaf_values else 0.0
                direct_val = cand_node.stockfish_value if cand_node.stockfish_value is not None else 0.0
                minimax_val = minimax_values.get(cand_move, 0)

                move_leaf_max_values[cand_move] = best_leaf_val
                direct_move_values[cand_move] = direct_val

                cand_tree_stats = compute_trajectory_tree_stats(cand_node)

                candidate_details.append({
                    "move": cand_move.uci(),
                    "move_san": board.san(cand_move),
                    "direct_value": float(direct_val),
                    "best_leaf_value": float(best_leaf_val),
                    "minimax_value": int(minimax_val),
                    "leaf_labels": cand_leaf_labels,
                    "leaf_values": [float(v) for v in cand_leaf_values],
                    "leaf_count": cand_tree_stats["leaf_count"],
                    "max_depth": cand_tree_stats["max_depth"],
                    "node_count": cand_tree_stats["node_count"],
                    "num_trajectories": len(trajectories_per_candidate.get(cand_move, [])),
                })

        # Determine target move based on evaluation policy
        if args.evaluation_policy == "minimax":
            target_move = select_best_move_by_minimax(minimax_values, direct_move_values)
            best_val = minimax_values.get(target_move, 0) if target_move else None
        elif args.evaluation_policy == "stockfish_leaf_max":
            target_move = max(move_leaf_max_values.keys(), key=lambda m: move_leaf_max_values[m]) if move_leaf_max_values else None
            best_val = move_leaf_max_values.get(target_move, 0) if target_move else None
        elif args.evaluation_policy == "stockfish_direct_max":
            target_move = max(direct_move_values.keys(), key=lambda m: direct_move_values[m]) if direct_move_values else None
            best_val = direct_move_values.get(target_move, 0) if target_move else None
        else:
            raise ValueError(f"Unknown evaluation policy: {args.evaluation_policy}")

        target_move_san = board.san(target_move) if target_move else None

        # Step 5: Generate traversals with different methods
        traversals: Dict[str, Dict[str, str]] = {}

        for method in args.traversal_methods:
            if method == "dfs_policy":
                trav = dfs_policy_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_policy_order(board, tree_root, include_directions=True)
                trav_sep = dfs_policy_order(board, tree_root, include_directions=False, include_subtree_sep=True)
                trav_nl = dfs_policy_order(board, tree_root, include_directions=False, include_result_labels=False)
                trav_dirs_nl = dfs_policy_order(board, tree_root, include_directions=True, include_result_labels=False)
                trav_sep_nl = dfs_policy_order(board, tree_root, include_directions=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "dfs_verifier":
                trav = dfs_verifier_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_verifier_order(board, tree_root, include_directions=True)
                trav_sep = dfs_verifier_order(board, tree_root, include_directions=False, include_subtree_sep=True)
                trav_nl = dfs_verifier_order(board, tree_root, include_directions=False, include_result_labels=False)
                trav_dirs_nl = dfs_verifier_order(board, tree_root, include_directions=True, include_result_labels=False)
                trav_sep_nl = dfs_verifier_order(board, tree_root, include_directions=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "bfs_policy":
                trav = bfs_policy_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_policy_order(board, tree_root, include_level_markers=True)
                trav_sep = bfs_policy_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
                trav_nl = bfs_policy_order(board, tree_root, include_level_markers=False, include_result_labels=False)
                trav_dirs_nl = bfs_policy_order(board, tree_root, include_level_markers=True, include_result_labels=False)
                trav_sep_nl = bfs_policy_order(board, tree_root, include_level_markers=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "bfs_verifier":
                trav = bfs_verifier_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_verifier_order(board, tree_root, include_level_markers=True)
                trav_sep = bfs_verifier_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
                trav_nl = bfs_verifier_order(board, tree_root, include_level_markers=False, include_result_labels=False)
                trav_dirs_nl = bfs_verifier_order(board, tree_root, include_level_markers=True, include_result_labels=False)
                trav_sep_nl = bfs_verifier_order(board, tree_root, include_level_markers=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "dfs_policy_layers":
                trav = dfs_policy_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_policy_order_with_layers(board, tree_root)
                trav_sep = dfs_policy_order(board, tree_root, include_directions=False, include_subtree_sep=True)
                trav_nl = dfs_policy_order(board, tree_root, include_directions=False, include_result_labels=False)
                trav_dirs_nl = dfs_policy_order_with_layers(board, tree_root, include_result_labels=False)
                trav_sep_nl = dfs_policy_order(board, tree_root, include_directions=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "dfs_verifier_layers":
                trav = dfs_verifier_order(board, tree_root, include_directions=False)
                trav_dirs = dfs_verifier_order_with_layers(board, tree_root)
                trav_sep = dfs_verifier_order(board, tree_root, include_directions=False, include_subtree_sep=True)
                trav_nl = dfs_verifier_order(board, tree_root, include_directions=False, include_result_labels=False)
                trav_dirs_nl = dfs_verifier_order_with_layers(board, tree_root, include_result_labels=False)
                trav_sep_nl = dfs_verifier_order(board, tree_root, include_directions=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "bfs_policy_layers":
                trav = bfs_policy_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_policy_order_with_layers(board, tree_root)
                trav_sep = bfs_policy_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
                trav_nl = bfs_policy_order(board, tree_root, include_level_markers=False, include_result_labels=False)
                trav_dirs_nl = bfs_policy_order_with_layers(board, tree_root, include_result_labels=False)
                trav_sep_nl = bfs_policy_order(board, tree_root, include_level_markers=False, include_subtree_sep=True, include_result_labels=False)
            elif method == "bfs_verifier_layers":
                trav = bfs_verifier_order(board, tree_root, include_level_markers=False)
                trav_dirs = bfs_verifier_order_with_layers(board, tree_root)
                trav_sep = bfs_verifier_order(board, tree_root, include_level_markers=False, include_subtree_sep=True)
                trav_nl = bfs_verifier_order(board, tree_root, include_level_markers=False, include_result_labels=False)
                trav_dirs_nl = bfs_verifier_order_with_layers(board, tree_root, include_result_labels=False)
                trav_sep_nl = bfs_verifier_order(board, tree_root, include_level_markers=False, include_subtree_sep=True, include_result_labels=False)
            else:
                print(f"Warning: Unknown traversal method '{method}', skipping")
                continue

            traversals[method] = {
                "traversal": trav,
                "traversal_with_markers": trav_dirs,
                "traversal_with_subtree_sep": trav_sep,
                "traversal_no_labels": trav_nl,
                "traversal_with_markers_no_labels": trav_dirs_nl,
                "traversal_with_subtree_sep_no_labels": trav_sep_nl,
            }

        # Generate CoT for each traversal method
        cot_results: Dict[str, Dict] = {}
        for method, trav_data in traversals.items():
            cot = generate_traversal_cot(
                traversal_str=trav_data["traversal_with_subtree_sep"],
                target_move=target_move,
                root_board=board,
                pgn_context=actual_pgn,
            )
            cot_with_markers = generate_traversal_cot(
                traversal_str=trav_data["traversal_with_markers"],
                target_move=target_move,
                root_board=board,
                pgn_context=actual_pgn,
            )
            cot_nl = generate_traversal_cot(
                traversal_str=trav_data["traversal_with_subtree_sep_no_labels"],
                target_move=target_move,
                root_board=board,
                pgn_context=actual_pgn,
            )
            cot_with_markers_nl = generate_traversal_cot(
                traversal_str=trav_data["traversal_with_markers_no_labels"],
                target_move=target_move,
                root_board=board,
                pgn_context=actual_pgn,
            )
            cot_results[method] = {
                "cot_format": cot["cot_format"],
                "cot_format_with_context": cot["cot_format_with_context"],
                "cot_format_with_markers": cot_with_markers["cot_format"],
                "cot_format_with_markers_and_context": cot_with_markers["cot_format_with_context"],
                "cot_format_no_labels": cot_nl["cot_format"],
                "cot_format_with_context_no_labels": cot_nl["cot_format_with_context"],
                "cot_format_with_markers_no_labels": cot_with_markers_nl["cot_format"],
                "cot_format_with_markers_and_context_no_labels": cot_with_markers_nl["cot_format_with_context"],
                "target_move_lan": cot["target_move_lan"],
            }

        # Per-trajectory CoT with labels
        trajectory_labels: List[int] = []
        for trajectory in all_trajectories:
            current_node = tree_root
            for move in trajectory:
                if move in current_node.children:
                    current_node = current_node.children[move]
            trajectory_labels.append(current_node.leaf_label if current_node.leaf_label is not None else 0)

        trajectory_sep_cot = generate_trajectory_sep_cot(
            root_board=board,
            all_trajectories=all_trajectories,
            trajectory_labels=trajectory_labels,
            target_move=target_move,
            pgn_context=actual_pgn,
        )
        trajectory_sep_cot_no_labels = generate_trajectory_sep_cot(
            root_board=board,
            all_trajectories=all_trajectories,
            trajectory_labels=trajectory_labels,
            target_move=target_move,
            pgn_context=actual_pgn,
            include_result_labels=False,
        )

        # Check against ground truth
        if has_ground_truth and ground_truth_move:
            if target_move:
                try:
                    gt_parsed = board.parse_san(ground_truth_move)
                    is_match = target_move == gt_parsed
                except ValueError:
                    if target_move_san:
                        is_match = target_move_san == ground_truth_move
                    else:
                        is_match = False
            else:
                is_match = False

        # Build result dict
        minimax_values_uci = {mv.uci(): int(val) for mv, val in minimax_values.items()} if minimax_values else {}
        direct_move_values_uci = {mv.uci(): float(val) for mv, val in direct_move_values.items()} if direct_move_values else {}

        result = {
            "index": idx,
            "original_pgn": pgn_str,
            "pgn": actual_pgn,
            "opponent_move": moves_list[0] if (has_ground_truth and pd.notna(row_dict.get(moves_col, None)) and len(moves_list) >= 2) else None,
            "target_move": target_move.uci() if target_move else None,
            "target_move_san": target_move_san,
            "target_move_lan": cot_results.get(args.traversal_methods[0], {}).get("target_move_lan", ""),
            "best_val": int(best_val) if isinstance(best_val, int) else float(best_val) if best_val is not None else None,
            "minimax_values": minimax_values_uci,
            "direct_move_values": direct_move_values_uci,
            "ground_truth_move": ground_truth_move,
            "is_match": is_match,
            "num_candidates": len(candidate_moves),
            "num_trajectories_total": len(all_trajectories),
            "cot_by_method": cot_results,
            "trajectory_sep_cot": trajectory_sep_cot,
            "trajectory_sep_cot_no_labels": trajectory_sep_cot_no_labels,
            "traversals": traversals,
            "candidate_details": candidate_details,
            "tree_stats": tree_stats,
            "leaf_label_distribution": {
                "wins": leaf_labels.count(1),
                "draws": leaf_labels.count(0),
                "losses": leaf_labels.count(-1),
            },
        }

        # Backward compatibility
        primary_method = args.traversal_methods[0] if args.traversal_methods else "dfs_policy"
        if primary_method in cot_results:
            result["cot_format"] = cot_results[primary_method]["cot_format"]
            result["cot_format_with_context"] = cot_results[primary_method]["cot_format_with_context"]
            result["cot_format_with_directions"] = cot_results[primary_method].get("cot_format_with_markers", "")
            result["cot_format_with_directions_and_context"] = cot_results[primary_method].get("cot_format_with_markers_and_context", "")

        cot_entry = None
        if args.cot_output_path:
            cot_entry = {
                "index": idx,
                "pgn": actual_pgn,
                "target_move_san": target_move_san,
                "target_move_lan": result.get("target_move_lan", ""),
            }
            for method, cot_data in cot_results.items():
                cot_entry[f"cot_{method}"] = cot_data["cot_format_with_context"]
                cot_entry[f"cot_{method}_with_markers"] = cot_data["cot_format_with_markers_and_context"]
                cot_entry[f"cot_{method}_no_labels"] = cot_data["cot_format_with_context_no_labels"]
                cot_entry[f"cot_{method}_with_markers_no_labels"] = cot_data["cot_format_with_markers_and_context_no_labels"]
            cot_entry["cot_trajectory_sep"] = trajectory_sep_cot["cot_format_with_context"]
            cot_entry["cot_trajectory_sep_no_labels"] = trajectory_sep_cot_no_labels["cot_format_with_context"]

        return result, cot_entry

    except Exception as e:
        print(f"Error processing position {idx}: {e}")
        traceback.print_exc()
        return None


# --- Main pipeline -------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Generate CoT data with minimax labels and result annotations")

    parser.add_argument("--data_path", type=str, required=True, help="Path to CSV file containing positions")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save generated CoT data (JSON)")
    parser.add_argument("--cot_output_path", type=str, default=None, help="Path to save CoT format file (optional)")
    parser.add_argument("--num_samples", type=int, default=None, help="Number of samples to process (default: all)")
    parser.add_argument("--num_skip_samples", type=int, default=None, help="Number of samples to skip (default: none)")

    parser.add_argument("--sampling_policy", type=str, choices=["random", "stockfish", "hf_model"], default="random")
    parser.add_argument("--evaluation_policy", type=str, choices=["minimax", "stockfish_leaf_max", "stockfish_direct_max"], default="minimax",
                        help="How to select best move: minimax uses win/draw/loss propagation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--config_path", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=1.0)

    parser.add_argument("--num_candidates", type=int, default=5, help="Number of candidate moves to explore")
    parser.add_argument("--num_trajectories", type=int, default=3, help="Number of trajectories per candidate")
    parser.add_argument("--trajectory_depth", type=int, default=4, help="Maximum depth for each trajectory")
    parser.add_argument("--data_name", type=str, default="puzzle", help="Name of the data")
    parser.add_argument("--win_threshold", type=int, default=300,
                        help="Centipawn threshold for win/loss classification (default: 300)")
    parser.add_argument(
        "--candidate_selection",
        type=str,
        choices=["policy", "stockfish"],
        default="policy",
        help="How to select candidate moves: 'policy' samples from policy, 'stockfish' orders by stockfish",
    )
    parser.add_argument(
        "--traversal_methods",
        type=str,
        nargs="+",
        default=["dfs_policy", "dfs_verifier", "bfs_policy", "bfs_verifier"],
        help="Traversal methods to use",
    )

    parser.add_argument("--stockfish_path", type=str, default="/usr/games/stockfish")
    parser.add_argument("--stockfish_time", type=float, default=0.01)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of parallel worker processes. >1 only supported for 'random' and 'stockfish' sampling policies.")

    args = parser.parse_args()

    print(f"Reading data from {args.data_path}...")
    df = pd.read_csv(args.data_path)
    if "ctx" not in df.columns:
        raise ValueError("CSV file must contain 'ctx' column with PGN positions")

    has_ground_truth = "Moves" in df.columns or "moves" in df.columns
    if has_ground_truth:
        moves_col = "Moves" if "Moves" in df.columns else "moves"
        print(f"Found ground truth moves in '{moves_col}' column")

    if args.num_samples is not None:
        if args.num_skip_samples is None:
            df = df.head(args.num_samples)
        else:
            df = df.iloc[args.num_skip_samples : args.num_skip_samples + args.num_samples]

    print(f"Processing {len(df)} positions...")

    print(f"Using trajectory-based tree generator with minimax labels:")
    print(f"  Candidate selection: {args.candidate_selection}")
    print(f"  Num candidates: {args.num_candidates}")
    print(f"  Trajectories per candidate: {args.num_trajectories}")
    print(f"  Trajectory depth: {args.trajectory_depth}")
    print(f"  Win threshold: {args.win_threshold} cp")
    print(f"  Evaluation policy: {args.evaluation_policy}")
    print(f"  Traversal methods: {args.traversal_methods}")
    print(f"  Num workers: {args.num_workers}")

    results = []
    cot_format_outputs = []
    moves_col_safe = moves_col if has_ground_truth else ""

    parallel_policies = ("random", "stockfish")
    use_parallel = args.num_workers > 1 and args.sampling_policy in parallel_policies

    if args.num_workers > 1 and not use_parallel:
        print(f"Warning: --num_workers > 1 is only supported for {parallel_policies} policies. Running sequentially.")

    if use_parallel:
        policy_kwargs = {
            "seed": args.seed,
            "stockfish_time": args.stockfish_time,
            "temperature": args.temperature,
        }
        work_items = [
            (idx, row.to_dict(), vars(args), has_ground_truth, moves_col_safe)
            for idx, row in df.iterrows()
        ]
        with multiprocessing.Pool(
            processes=args.num_workers,
            initializer=_init_worker,
            initargs=(args.stockfish_path, args.sampling_policy, policy_kwargs),
        ) as pool:
            for output in tqdm(
                pool.imap_unordered(_worker_fn, work_items),
                total=len(df),
                desc="Generating CoT data",
            ):
                if output is not None:
                    result, cot_entry = output
                    results.append(result)
                    if cot_entry is not None:
                        cot_format_outputs.append(cot_entry)
    else:
        device = torch.device(args.device)
        sampling_policy = create_sampling_policy(args, device)
        stockfish_engine = chess.engine.SimpleEngine.popen_uci(args.stockfish_path)

        # For model-based policies, parallelize Stockfish node annotation
        sf_annotation_pool = None
        if args.num_workers > 1:
            print(f"  Using {args.num_workers} Stockfish workers for parallel tree annotation")
            sf_annotation_pool = multiprocessing.Pool(
                processes=args.num_workers,
                initializer=_init_sf_worker,
                initargs=(args.stockfish_path,),
            )

        try:
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="Generating CoT data"):
                output = process_one_position(
                    idx, row.to_dict(), args, stockfish_engine, sampling_policy,
                    has_ground_truth, moves_col_safe,
                    sf_annotation_pool=sf_annotation_pool,
                )
                if output is not None:
                    result, cot_entry = output
                    results.append(result)
                    if cot_entry is not None:
                        cot_format_outputs.append(cot_entry)
        finally:
            if sf_annotation_pool is not None:
                sf_annotation_pool.terminate()
                sf_annotation_pool.join()
            stockfish_engine.quit()

    # Sort by original index for deterministic output order
    results.sort(key=lambda r: r["index"])
    cot_format_outputs.sort(key=lambda e: e["index"])

    # Compute accuracy from collected results
    matches = sum(1 for r in results if r.get("is_match") is True)
    total_with_ground_truth = sum(1 for r in results if r.get("is_match") is not None)

    from pathlib import Path

    base_output_path = Path(args.output_path)

    hyperparam_dir = (
        f"{args.sampling_policy}"
        f"_seed{args.seed}"
        f"_eval{args.evaluation_policy}"
        f"_cand{args.candidate_selection}"
        f"_nc{args.num_candidates}"
        f"_nt{args.num_trajectories}"
        f"_d{args.trajectory_depth}"
        f"_wt{args.win_threshold}"
        f"_temp{args.temperature}"
        f"_sf{args.stockfish_time}"
    )

    if args.num_samples is not None and args.num_skip_samples is not None:
        shard_name = f"data{args.data_name}_shard{args.num_skip_samples}-{args.num_skip_samples + args.num_samples}.json"
    elif args.num_samples is not None:
        shard_name = f"data{args.data_name}_n{args.num_samples}.json"
    else:
        shard_name = f"data{args.data_name}_results.json"

    if base_output_path.suffix:
        base_dir = base_output_path.parent / base_output_path.stem
    else:
        base_dir = base_output_path

    output_path = base_dir / hyperparam_dir / shard_name

    print(f"Saving results to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Compute summary statistics
    total_trajectories = sum(r["num_trajectories_total"] for r in results)
    summary = {
        "total_positions": len(results),
        "total_trajectories": total_trajectories,
        "avg_trajectories_per_position": total_trajectories / len(results) if results else 0,
        "traversal_methods": args.traversal_methods,
        "win_threshold": args.win_threshold,
        "evaluation_policy": args.evaluation_policy,
    }

    # Aggregate tree stats
    node_counts = [r["tree_stats"]["node_count"] for r in results if "tree_stats" in r]
    leaf_counts = [r["tree_stats"]["leaf_count"] for r in results if "tree_stats" in r]
    depths = [r["tree_stats"]["max_depth"] for r in results if "tree_stats" in r]
    branching_factors = [r["tree_stats"]["avg_branching_factor"] for r in results if "tree_stats" in r]

    if node_counts:
        summary["avg_node_count"] = sum(node_counts) / len(node_counts)
    if leaf_counts:
        summary["avg_leaf_count"] = sum(leaf_counts) / len(leaf_counts)
    if depths:
        summary["avg_max_depth"] = sum(depths) / len(depths)
    if branching_factors:
        summary["avg_branching_factor"] = sum(branching_factors) / len(branching_factors)

    # Aggregate label distribution
    total_wins = sum(r.get("leaf_label_distribution", {}).get("wins", 0) for r in results)
    total_draws = sum(r.get("leaf_label_distribution", {}).get("draws", 0) for r in results)
    total_losses = sum(r.get("leaf_label_distribution", {}).get("losses", 0) for r in results)
    total_labels = total_wins + total_draws + total_losses

    summary["leaf_label_distribution"] = {
        "total_wins": total_wins,
        "total_draws": total_draws,
        "total_losses": total_losses,
        "win_rate": total_wins / total_labels if total_labels > 0 else 0,
        "draw_rate": total_draws / total_labels if total_labels > 0 else 0,
        "loss_rate": total_losses / total_labels if total_labels > 0 else 0,
    }

    if has_ground_truth and total_with_ground_truth > 0:
        accuracy = matches / total_with_ground_truth
        summary["ground_truth_comparison"] = {
            "total_compared": total_with_ground_truth,
            "matches": matches,
            "accuracy": accuracy,
            "accuracy_percent": f"{accuracy * 100:.2f}%",
        }

    output_data = {
        "config": {
            "sampling_policy": args.sampling_policy,
            "candidate_selection": args.candidate_selection,
            "evaluation_policy": args.evaluation_policy,
            "num_candidates": args.num_candidates,
            "num_trajectories": args.num_trajectories,
            "trajectory_depth": args.trajectory_depth,
            "win_threshold": args.win_threshold,
            "traversal_methods": args.traversal_methods,
            "temperature": args.temperature,
            "stockfish_time": args.stockfish_time,
            "note": "Trajectory-based tree with minimax labels and result annotations",
        },
        "summary": summary,
        "results": results,
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    if args.cot_output_path and cot_format_outputs:
        base_path = Path(args.cot_output_path)
        hyperparam_str = f"_{hyperparam_dir}" if hyperparam_dir else ""
        shard_str = f"_{shard_name.replace('.json', '')}" if shard_name else ""
        if base_path.suffix:
            cot_output_path = base_path.parent / f"{base_path.stem}{hyperparam_str}{shard_str}{base_path.suffix}"
        else:
            cot_output_path = base_path / f"cot_output{hyperparam_str}{shard_str}.txt"
        print(f"Saving CoT format to {cot_output_path}...")
        cot_output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cot_output_path, "w") as f:
            for entry in cot_format_outputs:
                f.write(f"# Position {entry['index']}\n")
                f.write(f"# PGN: {entry['pgn']}\n")
                f.write(f"# Target (SAN): {entry['target_move_san']}\n")
                f.write(f"# Target (LAN): {entry['target_move_lan']}\n")
                for method in args.traversal_methods:
                    f.write(f"# CoT ({method} with subtree <sep>):\n")
                    f.write(f"{entry.get(f'cot_{method}', '')}\n")
                    f.write(f"# CoT ({method} with markers):\n")
                    f.write(f"{entry.get(f'cot_{method}_with_markers', '')}\n")
                f.write(f"# CoT (trajectory_sep - per trajectory <sep>):\n")
                f.write(f"{entry.get('cot_trajectory_sep', '')}\n")
                f.write("\n")

    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Successfully generated CoT data for {len(results)} positions")
    print(f"Total trajectories generated: {summary['total_trajectories']}")
    print(f"Average trajectories per position: {summary['avg_trajectories_per_position']:.2f}")
    print(f"\nTree Statistics:")
    if "avg_node_count" in summary:
        print(f"  Average node count: {summary['avg_node_count']:.2f}")
    if "avg_leaf_count" in summary:
        print(f"  Average leaf count: {summary['avg_leaf_count']:.2f}")
    if "avg_max_depth" in summary:
        print(f"  Average max depth: {summary['avg_max_depth']:.2f}")
    if "avg_branching_factor" in summary:
        print(f"  Average branching factor: {summary['avg_branching_factor']:.2f}")
    print(f"\nLeaf Label Distribution:")
    print(f"  Wins (+1): {summary['leaf_label_distribution']['total_wins']} ({summary['leaf_label_distribution']['win_rate']*100:.1f}%)")
    print(f"  Draws (0): {summary['leaf_label_distribution']['total_draws']} ({summary['leaf_label_distribution']['draw_rate']*100:.1f}%)")
    print(f"  Losses (-1): {summary['leaf_label_distribution']['total_losses']} ({summary['leaf_label_distribution']['loss_rate']*100:.1f}%)")
    if has_ground_truth and total_with_ground_truth > 0:
        print("\nGround Truth Comparison:")
        print(f"  Positions compared: {total_with_ground_truth}")
        print(f"  Matches: {matches}")
        print(f"  Accuracy: {summary['ground_truth_comparison']['accuracy_percent']}")
    print(f"\nEvaluation policy: {args.evaluation_policy}")
    print(f"Win threshold: {args.win_threshold} cp")
    print(f"Traversal methods: {', '.join(args.traversal_methods)}")
    print(f"\nResults saved to {output_path}")
    if args.cot_output_path:
        print(f"CoT format saved to {cot_output_path if 'cot_output_path' in locals() else args.cot_output_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
