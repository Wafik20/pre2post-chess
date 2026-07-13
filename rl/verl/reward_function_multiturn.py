import os
import re
import random

DEBUG = os.environ.get('DEBUG', 'False').lower() == 'true'


def _check_move_legality(fen: str, uci_move: str) -> float:
    """Returns 1.0 if uci_move is legal on the board described by fen, 0.0 otherwise."""
    if not fen or not uci_move:
        return 0.0
    try:
        import chess
        board = chess.Board(fen)
        move = chess.Move.from_uci(uci_move)
        return 1.0 if move in board.legal_moves else 0.0
    except Exception:
        return 0.0


if DEBUG:
    print(f"REWARD_MODEL_TYPE: {os.environ.get('REWARD_MODEL_TYPE', 'NOT_SET')}")

REWARD_MODEL_TYPE = os.environ.get('REWARD_MODEL_TYPE', 'RULE_BASED').upper()


def lan_to_uci(lan: str, side_to_move: str = 'white') -> str:
    """
    Convert custom LAN move to UCI format.
    
    Args:
        lan: Custom LAN string, e.g., "Pd2d4", "Pd4xe5", "Pe7e8=Q", "O-O"
        side_to_move: 'white' or 'black' for castling conversion
        
    Returns:
        UCI string, e.g., "d2d4", "d4e5", "e7e8q", "e1g1"
        
    Raises:
        ValueError if invalid format
    """
    import re
    
    # Strip check/checkmate symbols
    lan = lan.rstrip('+#').strip()
    
    # Handle castling
    if lan == 'O-O':
        if side_to_move == 'white':
            return 'e1g1'
        elif side_to_move == 'black':
            return 'e8g8'
        else:
            raise ValueError("Invalid side_to_move for castling")
    elif lan == 'O-O-O':
        if side_to_move == 'white':
            return 'e1c1'
        elif side_to_move == 'black':
            return 'e8c8'
        else:
            raise ValueError("Invalid side_to_move for castling")
    
    # Pattern for regular moves: Piece from (x)? to (=Promotion)?
    match = re.match(r'^([PNBRQK])([a-h][1-8])(x)?([a-h][1-8])(=([QRBN]))?$', lan)
    if not match:
        raise ValueError(f"Invalid LAN format: {lan}")
    
    piece, from_sq, capture, to_sq, promo_group, promo = match.groups()
    
    uci = from_sq + to_sq
    if promo:
        uci += promo.lower()  # UCI uses lowercase for promotion (q/r/b/n)
    
    return uci

def _is_complete_move(text: str) -> bool:
    """
    Check if text represents a complete chess move in custom format.
    Handles: Pd2d4, Pd4xe5, O-O, Pe7e8=Q, etc., with optional + or #.
    """
    if not text:
        return False
    # Remove trailing check/checkmate symbols
    move = text.rstrip('+#')
    
    # Castling (O-O or O-O-O, possibly with piece K?)
    if move in ['O-O', 'O-O-O']:
        return True
    
    # Pattern: Piece [a-h][1-8] (x)? [a-h][1-8] (= [QRBN])?
    pattern = r'^[PNBRQK][a-h][1-8](x)?[a-h][1-8](=[QRBN])?$'
    if re.match(pattern, move):
        return True
    
    return False


def _extract_first_move(text: str) -> str:
    """
    Extract the first complete move from generated text.
    Handles cases where model generates multiple tokens.
    """
    text = text.strip()
    
    # Split on whitespace
    moves = text.split()
    if not moves:
        return text
    
    for move in moves:
        # Skip move numbers (e.g., "1.", "2.", "1...")
        if re.match(r'^\d+\.{1,3}$', move):
            continue
        if _is_complete_move(move):
            return move
    # If no complete move found, return what we have
    return None


def _extract_last_move(text: str) -> str:
    """
    Extract the last complete move from text (i.e., the move closest to <call_env>).
    """
    text = text.strip()
    for move in reversed(text.split()):
        if re.match(r'^\d+\.{1,3}$', move):
            continue
        if _is_complete_move(move):
            return move
    return None


def _extract_move_after_thinking(text: str) -> tuple:
    """
    Extract the move that appears after the </T> token in SFT-generated text.
    
    Strict mode:
    - If </T> exists: parse the first move after </T>
    - If </T> doesn't exist: return None (count as wrong)
    - follows_format is True only if both <T> and </T> are present
    
    Args:
        text: Generated text that may contain <T>...</T> format
        
    Returns:
        tuple: (move_after_T, follows_format)
            - move_after_T: The first complete move found after </T>, or None
            - follows_format: Boolean indicating if text follows <T>...</T> format
    """
    text = text.strip()
    
    # Check if the text follows the <T>...</T> format
    has_closing_T = '</T>' in text
    follows_format = has_closing_T
    
    if not follows_format:
        # Strict mode: No </T> token found, return None (count as wrong)
        return None, False
    
    # Find the position after </T>
    closing_tag_end = text.find('</T>') + len('</T>')
    text_after_T = text[closing_tag_end:].strip()
    
    if not text_after_T:
        return None, follows_format
    
    # Extract the first complete move after </T>
    first_move = _extract_first_move(text_after_T)
    return first_move, follows_format


def _extract_all_my_moves(text: str) -> list:
    """
    Extract all model moves from multiturn text.
    Each my_move appears before a <call_env> tag.
    Format: [my_move] <call_env> [env_move] [my_move] <call_env> ...
    Thinking blocks <T>...</T> may precede each move.

    Returns:
        List of extracted moves (None for any turn where extraction failed).
    """
    segments = text.split('<call_env>')
    my_moves = []

    # All segments except the last contain a my_move at the end
    for segment in segments[:-1]:
        move, _ = _extract_move_after_thinking(segment)
        if move is None:
            move = _extract_last_move(segment)
        my_moves.append(move)

    return my_moves


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    """
    Compute chess move scores for multiturn trajectories.
    Extracts all my_moves (moves before <call_env>) and compares to ground truth list.
    Reward is 1.0 only if every extracted move matches the corresponding ground truth move.
    """
    import json

    if DEBUG:
        print(f"BATCH INPUT DEBUG: data_sources={len(data_sources)}, solution_strs={len(solution_strs)}, ground_truths={len(ground_truths)}, extra_infos={len(extra_infos)}")
        print(f"REWARD_MODEL_TYPE: {REWARD_MODEL_TYPE}")

    results = []
    no_move_extracted_count = 0

    for data_source, solution_str, ground_truth, extra_info in zip(
        data_sources, solution_strs, ground_truths, extra_infos, strict=True
    ):
        # Parse ground truth into a list of UCI target moves
        if isinstance(ground_truth, str):
            try:
                ground_truth = json.loads(ground_truth)
            except json.JSONDecodeError:
                try:
                    import ast
                    ground_truth = ast.literal_eval(ground_truth)
                except (ValueError, SyntaxError):
                    pass

        if isinstance(ground_truth, list):
            target_moves = [str(m).strip() for m in ground_truth]
        else:
            # Space-separated string
            target_moves = str(ground_truth).strip().split()

        # Extract all my_moves from the multiturn trajectory
        extracted_moves = _extract_all_my_moves(solution_str)

        if not extracted_moves:
            no_move_extracted_count += 1
            score = 0.0
            extracted_ucis = []
        else:
            # Convert each extracted move to UCI and compare
            extracted_ucis = []
            all_match = True

            if len(extracted_moves) != len(target_moves):
                all_match = False

            for i, move in enumerate(extracted_moves):
                if move is None:
                    extracted_ucis.append('')
                    all_match = False
                    no_move_extracted_count += 1
                else:
                    try:
                        uci = lan_to_uci(move)
                    except ValueError:
                        uci = move  # keep as-is if conversion fails
                    extracted_ucis.append(uci)
                    if i >= len(target_moves) or uci != target_moves[i]:
                        all_match = False

            score = 1.0 if all_match else 0.0

        first_pred_uci = extracted_ucis[0] if extracted_ucis else ''
        first_gt_uci = target_moves[0] if target_moves else ''
        first_move_score = 1.0 if (first_pred_uci and first_pred_uci == first_gt_uci) else 0.0

        _fen = extra_info.get('FEN') or extra_info.get('fen', '') if isinstance(extra_info, dict) else ''
        first_move_legality_score = _check_move_legality(_fen, first_pred_uci)

        results.append({
            "score": float(score),
            "ground_truth": str(ground_truth),
            "reward_method": "CHESS_MULTITURN_PARSING",
            "extracted_moves": ",".join(str(m) for m in extracted_ucis),
            "target_moves": ",".join(str(m) for m in target_moves),
            "data_source": data_source,
            "first_move_score": first_move_score,
            "first_move_legality_score": first_move_legality_score,
        })

    if DEBUG and results:
        correct_count = sum(1 for r in results if r["score"] > 0.5)
        total_count = len(results)
        accuracy = correct_count / total_count if total_count > 0 else 0.0
        print(f"\nCHESS_MULTITURN_PARSING BATCH SUMMARY:")
        print(f"   Overall accuracy: {correct_count}/{total_count} = {accuracy:.3f}")
        print(f"   No-move-extracted count: {no_move_extracted_count}")
        print()

    return results