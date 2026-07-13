"""
Multi-turn puzzle evaluator for SFT models using <T>…</T> + <call_env> format.

Generation flow per puzzle:
  1. CoT phase   : tokenize `{pgn} <T>`, generate until </T> is produced.
  2. Move loop   : repeatedly generate until <call_env> is produced,
                   decode the preceding tokens as the model's move,
                   inject the ground-truth env move, and continue.

Metrics reported:
  - format_rate      : fraction of outputs that contain both <T> and </T>
                       AND have at least one "move <call_env>" pattern
  - first_move_acc   : accuracy on the first model move (moves_list[1])
  - full_puzzle_acc  : fraction of puzzles where ALL model moves are correct
  - per_depth_acc    : dict depth → accuracy (depth = number of model moves)
"""

import sys
import io
import re
import pathlib
from collections import defaultdict
from typing import List, Optional, Dict, Tuple

import torch
import chess
import chess.pgn

repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from evaluation.utils import lan_to_uci


# ── Move pattern (LAN) ────────────────────────────────────────────────────────── #
_MOVE_RE = re.compile(
    r'^([PNBRQK])([a-h][1-8])(x)?([a-h][1-8])(=[QRBN])?[+#]?$'
)
_CASTLING = {"O-O", "O-O-O"}


def _is_lan_move(token: str) -> bool:
    """Return True if *token* looks like a LAN move."""
    t = token.rstrip('+#')
    return t in _CASTLING or bool(_MOVE_RE.match(token))


def _extract_lan_move(tokens: List[str]) -> Optional[str]:
    """
    Return the first token in *tokens* that is a complete LAN move,
    or None if none found.
    """
    for tok in tokens:
        if _is_lan_move(tok):
            return tok
    return None


# ── Board helpers ─────────────────────────────────────────────────────────────── #

def _pgn_to_board(pgn: str) -> Optional[chess.Board]:
    """Parse a PGN string and return the resulting board, or None on failure."""
    try:
        game = chess.pgn.read_game(io.StringIO(pgn))
        if game is None:
            return None
        return game.end().board()
    except Exception:
        return None


def _uci_side(board: chess.Board) -> str:
    """Return 'white' or 'black' for the side to move."""
    return "white" if board.turn == chess.WHITE else "black"


def _apply_uci(board: chess.Board, uci: str) -> bool:
    """Push a UCI move onto the board. Return True on success."""
    try:
        mv = chess.Move.from_uci(uci)
        if mv in board.legal_moves:
            board.push(mv)
            return True
    except Exception:
        pass
    return False


# ── Generation helpers ────────────────────────────────────────────────────────── #

def _vocab(tokenizer) -> Dict[str, int]:
    return tokenizer.get_vocab()


def generate_cot_phase(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    t_end_id: int,
    eos_id: int,
    max_new_tokens: int = 1024,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate the CoT section, stopping when </T> or EOS is produced.

    Args:
        input_ids : (1, seq_len) tensor already containing the prompt + <T> token.

    Returns:
        (1, seq_len + new_tokens) tensor including the stop token.
    """
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            use_cache=True,
            pad_token_id=eos_id,
            eos_token_id=[t_end_id, eos_id],
        )
    return out  # shape (1, extended_len)


def generate_until_call_env(
    model,
    tokenizer,
    context_ids: torch.Tensor,
    call_env_id: int,
    eos_id: int,
    max_new_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, List[int], bool]:
    """
    Generate a model move, stopping when <call_env> or EOS is produced.

    Args:
        context_ids : (1, seq_len) tensor of the current context.

    Returns:
        (extended_context, new_token_ids, hit_call_env)
        - extended_context : context tensor after appending new tokens
        - new_token_ids    : list of new token IDs (including stop token)
        - hit_call_env     : True if generation stopped at <call_env>
    """
    with torch.no_grad():
        out = model.generate(
            input_ids=context_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            do_sample=True,
            use_cache=True,
            pad_token_id=eos_id,
            eos_token_id=[call_env_id, eos_id],
        )
    new_ids = out[0, context_ids.shape[1]:].tolist()
    hit_call_env = bool(new_ids) and new_ids[-1] == call_env_id
    return out, new_ids, hit_call_env


# ── Format checker ────────────────────────────────────────────────────────────── #

def check_puzzle_cot_format(text: str) -> bool:
    """
    Return True if the generated text follows the expected format:
      1. Contains <T> … </T>
      2. After </T>, there is at least one occurrence of "move <call_env>"
         (i.e., a LAN move token followed by <call_env>).
    """
    if "<T>" not in text or "</T>" not in text:
        return False

    return True


# ── LAN → UCI conversion (board-aware for castling) ──────────────────────────── #

def _lan_to_uci_board(lan: str, board: chess.Board) -> Optional[str]:
    """Convert LAN token to UCI, using the board for castling side detection."""
    if not lan:
        return None
    lan_clean = lan.rstrip("+#")
    if lan_clean in _CASTLING:
        side = _uci_side(board)
        try:
            return lan_to_uci(lan_clean, side_to_move=side)
        except Exception:
            return None
    try:
        return lan_to_uci(lan)
    except Exception:
        return None


# ── Main evaluator ────────────────────────────────────────────────────────────── #

def evaluate_multiturn_puzzle(
    model,
    tokenizer,
    puzzle_file: str,
    device: torch.device,
    max_samples: Optional[int] = None,
    max_puzzle_moves: int = 5,
    max_cot_tokens: int = 1024,
    max_move_tokens: int = 32,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    use_ctx: bool = True,
    verbose: bool = True,
    save_path: Optional[str] = None,
    use_thinking: bool = True,
) -> dict:
    """
    Run multi-turn puzzle evaluation.

    Args:
        model          : Trained GPT/HF model.
        tokenizer      : LanTokenizerSFT instance.
        puzzle_file    : Path to CSV file with 'FEN', 'Moves', 'ctx' columns.
        device         : Torch device.
        max_samples    : Cap the number of puzzles evaluated.
        max_puzzle_moves: Max model moves to attempt per puzzle (default 5).
        max_cot_tokens : Max tokens for the CoT phase.
        max_move_tokens: Max tokens for each move generation phase.
        temperature    : Sampling temperature.
        top_k          : Top-k sampling (None = disabled).
        use_ctx        : If True use PGN (ctx); if False use FEN.
        verbose        : Print progress.
        save_path      : If provided, save per-puzzle results to this parquet path.

    Returns:
        dict with keys:
            format_rate, first_move_acc, full_puzzle_acc,
            per_depth_acc, per_puzzle_results, total
    """
    import pandas as pd

    # ── Load data ──────────────────────────────────────────────────────────── #
    if puzzle_file.endswith(".csv"):
        df = pd.read_csv(puzzle_file)
    else:
        df = pd.read_parquet(puzzle_file)

    if max_samples is not None:
        df = df.head(max_samples)

    state_col = "ctx" if use_ctx else "FEN"
    if state_col not in df.columns:
        alt = "FEN" if use_ctx else "ctx"
        if alt in df.columns:
            if verbose:
                print(f"Warning: '{state_col}' not found, falling back to '{alt}'")
            state_col = alt
        else:
            raise ValueError(f"Neither 'ctx' nor 'FEN' column found in {puzzle_file}")

    if "Moves" not in df.columns:
        raise ValueError("'Moves' column not found in puzzle file")

    df = df.dropna(subset=[state_col, "Moves"]).reset_index(drop=True)

    # ── Tokenizer special IDs ─────────────────────────────────────────────── #
    vocab = _vocab(tokenizer)
    t_id = vocab.get("<T>")
    t_end_id = vocab.get("</T>")
    call_env_id = vocab.get("<call_env>")
    eos_id = tokenizer.eos_id()

    if t_id is None or t_end_id is None or call_env_id is None:
        raise ValueError(
            "Tokenizer vocab is missing one of: <T>, </T>, <call_env>. "
            "Ensure LanTokenizerSFT was built with include_env_tokens=True."
        )

    model.eval()

    # ── Per-puzzle tracking ───────────────────────────────────────────────── #
    per_puzzle_results = []
    format_ok_count = 0
    first_move_correct = 0
    full_puzzle_correct = 0
    per_depth_correct: Dict[int, int] = defaultdict(int)
    per_depth_total: Dict[int, int] = defaultdict(int)
    # Failure reason counters (mutually exclusive, first failure per puzzle)
    cot_format_fail_count = 0    # CoT did not end with </T>
    move_format_fail_count = 0   # a move turn hit EOS instead of <call_env>
    illegal_move_fail_count = 0  # model produced <call_env> but move is illegal
    wrong_move_fail_count = 0    # all moves legal, but at least one is wrong

    total = len(df)
    if verbose:
        print(f"Evaluating {total} puzzles …")

    for idx, row in df.iterrows():
        state = str(row[state_col]).strip()
        moves_str = str(row["Moves"]).strip()
        moves_list = moves_str.split()  # list of UCI moves

        # Need at least opponent trigger + one model move
        if len(moves_list) < 2:
            continue

        # Build board after opponent's trigger move
        board = _pgn_to_board(state)
        if board is None:
            # Fallback: try FEN
            try:
                board = chess.Board(state)
            except Exception:
                if verbose:
                    print(f"  [skip] Could not parse board for puzzle {idx}")
                continue

        # Apply opponent trigger (moves_list[0])
        if not _apply_uci(board, moves_list[0]):
            if verbose:
                print(f"  [skip] Could not apply trigger move {moves_list[0]} for puzzle {idx}")
            continue

        # Update state to include the trigger move (model must see the resulting position)
        exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
        new_game = chess.pgn.Game.from_board(board)
        new_pgn = new_game.accept(exporter)
        movetext = re.sub(r"\[[^\]]*\]\s*", "", new_pgn)
        movetext = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", movetext).strip()
        state = movetext

        # ── Encode prompt ────────────────────────────────────────────────── #
        state_ids = tokenizer.encode(state)
        # Strip trailing EOS if present
        if state_ids and state_ids[-1] == eos_id:
            state_ids = state_ids[:-1]

        if use_thinking:
            # Append <T> to trigger CoT generation
            state_ids = state_ids + [t_id]

        context_ids = torch.tensor([state_ids], dtype=torch.long, device=device)

        if use_thinking:
            # ── CoT phase ──────────────────────────────────────────────── #
            context_ids = generate_cot_phase(
                model=model,
                tokenizer=tokenizer,
                input_ids=context_ids,
                t_end_id=t_end_id,
                eos_id=eos_id,
                max_new_tokens=max_cot_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            # If CoT did not end with </T>, skip move generation entirely
            cot_new_ids = context_ids[0, len(state_ids):].tolist()
            if t_end_id not in cot_new_ids:
                cot_format_fail_count += 1
                raw_output = tokenizer.decode(cot_new_ids).strip()
                per_puzzle_results.append({
                    "index": idx,
                    "moves_list": moves_list,
                    "predicted_moves_uci": [],
                    "target_model_moves": moves_list[1::2][:max_puzzle_moves],
                    "first_move_correct": False,
                    "all_correct": False,
                    "depth": min(len(moves_list[1::2]), max_puzzle_moves),
                    "failure_reason": "cot_format",
                    "raw_output": raw_output,
                })
                continue

        # ── Move loop ───────────────────────────────────────────────────── #
        # moves_list[1], [3], [5], … are model moves (to evaluate)
        # moves_list[2], [4], [6], … are env moves (to inject)
        model_move_targets = moves_list[1::2]   # UCI
        env_moves = moves_list[2::2]            # UCI

        num_model_moves = min(len(model_move_targets), max_puzzle_moves)
        depth = num_model_moves
        per_depth_total[depth] += 1

        predicted_moves_uci: List[Optional[str]] = []
        all_correct = True
        failure_reason: Optional[str] = None
        board_copy = board.copy()  # track board state through the sequence

        for move_i in range(num_model_moves):
            context_ids, new_ids, hit_call_env = generate_until_call_env(
                model=model,
                tokenizer=tokenizer,
                context_ids=context_ids,
                call_env_id=call_env_id,
                eos_id=eos_id,
                max_new_tokens=max_move_tokens,
                temperature=temperature,
                top_k=top_k,
            )

            # Decode new tokens (exclude the trailing <call_env> / eos)
            move_ids = new_ids[:-1] if new_ids else []
            move_text = tokenizer.decode(move_ids).strip() if move_ids else ""
            move_tokens = move_text.split()

            # If the model didn't produce <call_env>, it violated the move format
            if not hit_call_env:
                predicted_moves_uci.append(None)
                all_correct = False
                failure_reason = "move_format"
                break

            predicted_lan = _extract_lan_move(move_tokens)
            predicted_uci = _lan_to_uci_board(predicted_lan, board_copy) if predicted_lan else None
            predicted_moves_uci.append(predicted_uci)

            # If the predicted move is illegal, end the loop early
            try:
                is_legal = (
                    predicted_uci is not None
                    and chess.Move.from_uci(predicted_uci) in board_copy.legal_moves
                )
            except Exception:
                is_legal = False

            if not is_legal:
                all_correct = False
                failure_reason = "illegal_move"
                break

            target_uci = model_move_targets[move_i].strip().lower()
            pred_uci_norm = predicted_uci.strip().lower()
            correct = pred_uci_norm == target_uci

            if move_i == 0:
                if correct:
                    first_move_correct += 1

            if not correct:
                all_correct = False
                failure_reason = "wrong_move"
                break

            # Advance board with the correct model move (not prediction)
            # so subsequent env moves are applied to the right position
            _apply_uci(board_copy, target_uci)

            # Inject env move if available
            if move_i < len(env_moves):
                env_uci = env_moves[move_i].strip()
                env_lan = _uci_to_lan(env_uci, board_copy)

                if env_lan is not None:
                    env_ids = _tokenize_lan_move(tokenizer, env_lan)
                    if env_ids:
                        env_tensor = torch.tensor(
                            [env_ids], dtype=torch.long, device=device
                        )
                        context_ids = torch.cat([context_ids, env_tensor], dim=1)

                # Advance board with env move
                _apply_uci(board_copy, env_uci)

        if all_correct and num_model_moves > 0:
            full_puzzle_correct += 1
            per_depth_correct[depth] += 1
        elif not all_correct and failure_reason is None:
            failure_reason = "wrong_move"

        # Update failure counters
        if failure_reason == "move_format":
            move_format_fail_count += 1
        elif failure_reason == "illegal_move":
            illegal_move_fail_count += 1
        elif failure_reason == "wrong_move":
            wrong_move_fail_count += 1

        raw_output = tokenizer.decode(context_ids[0, len(state_ids):].tolist()).strip()
        per_puzzle_results.append({
            "index": idx,
            "moves_list": moves_list,
            "predicted_moves_uci": predicted_moves_uci,
            "target_model_moves": model_move_targets[:num_model_moves],
            "first_move_correct": (
                len(predicted_moves_uci) > 0
                and (predicted_moves_uci[0] or "").strip().lower()
                    == (model_move_targets[0] if model_move_targets else "").strip().lower()
            ),
            "all_correct": all_correct,
            "depth": depth,
            "failure_reason": failure_reason,
            "raw_output": raw_output,
        })

        if verbose and (idx + 1) % 50 == 0:
            print(
                f"  [{idx + 1}/{total}] "
                f"fmt={format_ok_count}/{idx+1} "
                f"1st={first_move_correct}/{idx+1} "
                f"full={full_puzzle_correct}/{idx+1}"
            )

    n = len(per_puzzle_results)
    if n == 0:
        return {"error": "No valid puzzles evaluated", "total": 0}

    per_depth_acc = {
        d: per_depth_correct[d] / per_depth_total[d]
        for d in sorted(per_depth_total)
    }

    results = {
        "format_rate": format_ok_count / n,
        "first_move_acc": first_move_correct / n,
        "full_puzzle_acc": full_puzzle_correct / n,
        "per_depth_acc": per_depth_acc,
        "per_depth_total": dict(per_depth_total),
        "per_puzzle_results": per_puzzle_results,
        "total": n,
        # Failure breakdown (mutually exclusive, counts first failure per puzzle)
        "failure_counts": {
            "cot_format": cot_format_fail_count,
            "move_format": move_format_fail_count,
            "illegal_move": illegal_move_fail_count,
            "wrong_move": wrong_move_fail_count,
        },
    }

    if save_path is not None:
        results_df = pd.DataFrame(per_puzzle_results)
        # Convert list columns to strings so parquet writers with limited type
        # support (e.g. fastparquet) don't choke; pyarrow handles native lists fine.
        for col in ("moves_list", "predicted_moves_uci", "target_model_moves"):
            if col in results_df.columns:
                results_df[col] = results_df[col].apply(
                    lambda x: " ".join(str(v) if v is not None else "" for v in x) if isinstance(x, list) else x
                )
        results_df.to_parquet(save_path, index=False)
        if verbose:
            print(f"Results saved to {save_path}")

    if verbose:
        print("\n" + "=" * 55)
        print("MULTI-TURN PUZZLE EVALUATION RESULTS")
        print("=" * 55)
        print(f"Total puzzles   : {n}")
        print(f"Format rate     : {results['format_rate']:.2%} ({format_ok_count}/{n})")
        print(f"First-move acc  : {results['first_move_acc']:.2%} ({first_move_correct}/{n})")
        print(f"Full-puzzle acc : {results['full_puzzle_acc']:.2%} ({full_puzzle_correct}/{n})")
        if per_depth_acc:
            print("Per-depth acc   :")
            for d, acc in per_depth_acc.items():
                cnt = per_depth_total[d]
                print(f"  depth {d}: {acc:.2%}  ({per_depth_correct[d]}/{cnt})")
        print("Failure breakdown:")
        print(f"  cot_format  : {cot_format_fail_count}/{n}")
        print(f"  move_format : {move_format_fail_count}/{n}")
        print(f"  illegal_move: {illegal_move_fail_count}/{n}")
        print(f"  wrong_move  : {wrong_move_fail_count}/{n}")
        print("=" * 55)

    return results


# ── LAN move tokenization (bypass PGN path in encode()) ─────────────────────── #

def _tokenize_lan_move(tokenizer, lan: str) -> List[int]:
    """
    Tokenize a single LAN move (e.g., "Pe7e5", "O-O") into token IDs
    without going through the PGN parser.

    Uses _lan_move_to_tokens() to split the move into individual tokens
    (piece, squares, capture, promotion), then maps each token to its ID
    using the vocabulary.  Unknown tokens are mapped to <unk>.
    """
    tok2id = tokenizer._tok2id
    unk_id = tok2id.get(tokenizer._unk, 0)
    raw_tokens = tokenizer._lan_move_to_tokens(lan)
    return [tok2id.get(t, unk_id) for t in raw_tokens]


# ── UCI → LAN conversion (needed for env move injection) ─────────────────────── #

def _uci_to_lan(uci: str, board: chess.Board) -> Optional[str]:
    """
    Convert a UCI move to the custom LAN format the tokenizer uses,
    e.g. "d2d4" → "Pd2d4", "e1g1" → "O-O".

    Returns None if the move is illegal or parsing fails.
    """
    try:
        mv = chess.Move.from_uci(uci)
        if mv not in board.legal_moves:
            return None
        piece = board.piece_at(mv.from_square)
        if piece is None:
            return None

        # Castling
        if board.is_castling(mv):
            return "O-O" if chess.square_file(mv.to_square) > 4 else "O-O-O"

        piece_sym = piece.symbol().upper()
        from_sq = chess.square_name(mv.from_square)
        to_sq = chess.square_name(mv.to_square)
        capture = "x" if board.is_capture(mv) else ""
        promo = f"={mv.promotion.symbol().upper()}" if mv.promotion else ""

        return f"{piece_sym}{from_sq}{capture}{to_sq}{promo}"
    except Exception:
        return None
