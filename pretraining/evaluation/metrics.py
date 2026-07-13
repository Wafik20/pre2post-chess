"""Metric calculation for chess move prediction evaluation."""

import chess
import chess.pgn
import io
from typing import List, Tuple, Dict, Optional
import contextlib
import os


def is_move_legal(state: str, predicted_move: str, move_format: str = "uci") -> bool:
    """
    Check if a predicted move is legal given the current state.
    
    Args:
        state: Chess state as string (e.g., PGN sequence or FEN)
        predicted_move: The predicted move string
        move_format: Format of the move ("san", "uci", "lan")
    
    Returns:
        True if move is legal, False otherwise
    """
    try:
        # Get board from state
        board = _state_to_board(state)
        if board is None:
            return False
        
        # Try to parse and validate the move
        if move_format.lower() == "san":
            move = board.parse_san(predicted_move)
        elif move_format.lower() == "uci":
            move = chess.Move.from_uci(predicted_move)
        elif move_format.lower() == "lan":
            move = board.parse_san(predicted_move)  # LAN is similar to SAN
        else:
            return False
        
        # Check if move is legal
        return move in board.legal_moves
    
    except (ValueError, chess.InvalidMoveError, chess.IllegalMoveError):
        return False
    except Exception:
        return False


def _state_to_board(state: str) -> Optional[chess.Board]:
    """
    Convert a state string to a chess.Board object.
    Handles PGN sequences and FEN strings.
    
    Args:
        state: State as PGN game or FEN string
    
    Returns:
        chess.Board object or None if parsing fails
    """
    state = state.strip()
    
    # Try FEN first (simpler)
    if ' ' in state and len(state.split()) >= 4:
        try:
            return chess.Board(state)
        except Exception:
            pass
    
    # Try PGN
    try:
        # Suppress stderr warnings from chess.pgn
        with open(os.devnull, "w") as devnull, contextlib.redirect_stderr(devnull):
            game = chess.pgn.read_game(io.StringIO(state))
        
        if game is None:
            return None
        
        # Play through the game to get final position
        board = game.board()
        for move in game.mainline_moves():
            board.push(move)
        
        return board
    
    except Exception:
        return None


def calculate_legal_move_accuracy(
    states: List[str],
    predicted_moves: List[str],
    move_format: str = "san"
) -> Dict[str, float]:
    """
    Calculate legal move accuracy.
    
    Args:
        states: List of chess states
        predicted_moves: List of predicted moves
        move_format: Format of moves ("san", "uci", "lan")
    
    Returns:
        Dictionary with:
            - "legal_accuracy": Fraction of legal moves
            - "num_legal": Number of legal moves
            - "num_total": Total number of predictions
    """
    assert len(states) == len(predicted_moves), "States and predictions must have same length"
    
    num_legal = 0
    num_total = len(states)
    
    for state, pred_move in zip(states, predicted_moves):
        if is_move_legal(state, pred_move, move_format):
            num_legal += 1
    
    accuracy = num_legal / num_total if num_total > 0 else 0.0
    
    return {
        "legal_accuracy": accuracy,
        "num_legal": num_legal,
        "num_total": num_total,
    }


def calculate_move_matching_accuracy(
    predicted_moves: List[str],
    target_moves: List[str],
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate move matching accuracy (exact match between prediction and target).
    
    Args:
        predicted_moves: List of predicted moves
        target_moves: List of target/ground truth moves
        normalize: If True, normalize moves before comparison (remove +, #, spaces)
    
    Returns:
        Dictionary with:
            - "match_accuracy": Fraction of exact matches
            - "num_matches": Number of matches
            - "num_total": Total number of predictions
    """
    assert len(predicted_moves) == len(target_moves), "Predictions and targets must have same length"
    
    num_matches = 0
    num_total = len(predicted_moves)
    
    for pred, target in zip(predicted_moves, target_moves):
        if normalize:
            pred = _normalize_move(pred)
            target = _normalize_move(target)
        
        if pred == target:
            num_matches += 1
    
    accuracy = num_matches / num_total if num_total > 0 else 0.0
    
    return {
        "match_accuracy": accuracy,
        "num_matches": num_matches,
        "num_total": num_total,
    }


def _normalize_move(move: str) -> str:
    """
    Normalize a move string for comparison.
    Removes check/checkmate symbols, whitespace, etc.
    """
    move = move.strip()
    # Remove check and checkmate symbols
    move = move.rstrip('+#')
    # Remove whitespace
    move = move.replace(' ', '')
    return move.lower()


def calculate_pass_at_k(
    predicted_moves_list: List[List[str]],
    target_moves: List[str],
    k: int,
    normalize: bool = True
) -> Dict[str, float]:
    """
    Calculate pass@k metric: fraction of problems where at least one of k predictions matches the target.
    
    Args:
        predicted_moves_list: List of lists, where each inner list contains k predicted moves for that state
        target_moves: List of target/ground truth moves (one per state)
        k: Number of samples per state (k in pass@k)
        normalize: If True, normalize moves before comparison
    
    Returns:
        Dictionary with:
            - "pass_at_k": Fraction of states where at least one prediction matches
            - "num_passed": Number of states that passed
            - "num_total": Total number of states
    """
    assert len(predicted_moves_list) == len(target_moves), "Predictions and targets must have same length"
    
    num_passed = 0
    num_total = len(target_moves)
    
    for pred_list, target in zip(predicted_moves_list, target_moves):
        # Check if any of the k predictions matches the target
        passed = False
        for pred in pred_list:
            if pred is None:
                continue
            if normalize:
                pred_norm = _normalize_move(pred)
                target_norm = _normalize_move(target)
            else:
                pred_norm = pred.strip()
                target_norm = target.strip()
            
            if pred_norm == target_norm:
                passed = True
                break
        
        if passed:
            num_passed += 1
    
    pass_at_k = num_passed / num_total if num_total > 0 else 0.0
    
    return {
        "pass_at_k": pass_at_k,
        "num_passed": num_passed,
        "num_total": num_total,
    }


def evaluate_predictions(
    states: List[str],
    predicted_moves: List[str],
    target_moves: List[str],
    move_format: str = "san",
    normalize: bool = True
) -> Dict[str, float]:
    """
    Comprehensive evaluation of predictions.
    Calculates both legal move accuracy and move matching accuracy.

    Args:
        states: List of chess states
        predicted_moves: List of predicted moves
        target_moves: List of target/ground truth moves
        move_format: Format of moves ("san", "uci", "lan")
        normalize: If True, normalize moves before comparison

    Returns:
        Dictionary containing all metrics
    """
    legal_metrics = calculate_legal_move_accuracy(states, predicted_moves, move_format)
    match_metrics = calculate_move_matching_accuracy(predicted_moves, target_moves, normalize)
    # cp_gap_metrics = calculate_cp_gap(states, predicted_moves, move_format)
    # Combine metrics
    return {
        **legal_metrics,
        **match_metrics,
        # **cp_gap_metrics,
    }


def calculate_cp_gap(
    states: List[str],
    predicted_moves: List[str],
    move_format: str = "uci",
    stockfish_path: str = "${REPO_ROOT}/Stockfish/src/stockfish",
    search_time: float = 0.1,
) -> Dict[str, float]:
    """
    Calculate centipawn gap between the model's move and Stockfish's best move.

    For each position the engine evaluates the position, yielding the best-move
    score.  Then the model's predicted move is applied and the resulting
    position is evaluated (negated, since the side to move flipped).  The gap
    is ``best_score - model_score`` in centipawns from the perspective of the
    side to move.  Mate scores are capped at +/- 10000 cp.

    Args:
        states: List of chess state strings (PGN or FEN).
        predicted_moves: List of predicted move strings.
        move_format: Move format ("uci" or "san").
        stockfish_path: Path to the Stockfish binary.
        search_time: Per-position search time in seconds.

    Returns:
        Dictionary with:
            - "cp_gap_mean": Mean centipawn gap (lower is better).
            - "cp_gap_median": Median centipawn gap.
            - "num_evaluated": Number of positions successfully evaluated.
    """
    import chess.engine
    import statistics

    MATE_CP = 10_000  # cap for mate scores

    def score_to_cp(pov_score: chess.engine.PovScore) -> float:
        """Convert a PovScore (white-relative) to centipawns, capping mates."""
        sc = pov_score.white()
        if sc.is_mate():
            mate_in = sc.mate()
            return MATE_CP if mate_in > 0 else -MATE_CP
        return float(sc.score())

    gaps: List[float] = []

    try:
        engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    except Exception as e:
        print(f"[cp_gap] Failed to start Stockfish: {e}")
        return {"cp_gap_mean": 0.0, "cp_gap_median": 0.0, "num_evaluated": 0}

    try:
        for state, pred_move in zip(states, predicted_moves):
            if not pred_move or pred_move.strip() == "":
                continue
            try:
                board = _state_to_board(state)
                if board is None:
                    continue

                # Parse the predicted move
                if move_format.lower() == "uci":
                    move = chess.Move.from_uci(pred_move.strip())
                else:
                    move = board.parse_san(pred_move.strip())

                if move not in board.legal_moves:
                    continue

                # Evaluate the position before the move (best line)
                stm = board.turn  # side to move
                info_best = engine.analyse(board, chess.engine.Limit(time=search_time))
                best_cp = score_to_cp(info_best["score"])
                # Make it relative to side-to-move
                if stm == chess.BLACK:
                    best_cp = -best_cp

                # Apply the model's move, evaluate the resulting position
                board.push(move)
                info_after = engine.analyse(board, chess.engine.Limit(time=search_time))
                after_cp = score_to_cp(info_after["score"])
                # Negate because side to move flipped
                if stm == chess.WHITE:
                    after_cp = -after_cp

                gap = best_cp - after_cp
                gaps.append(max(gap, 0.0))  # gap should be >= 0; clamp rounding noise
            except Exception:
                continue
    finally:
        engine.quit()

    if not gaps:
        return {"cp_gap_mean": 0.0, "cp_gap_median": 0.0, "num_evaluated": 0}

    return {
        "cp_gap_mean": statistics.mean(gaps),
        "cp_gap_median": statistics.median(gaps),
        "num_evaluated": len(gaps),
    }

