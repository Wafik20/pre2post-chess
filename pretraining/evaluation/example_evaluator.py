"""Example evaluator implementation showing how to customize BaseEvaluator."""

import pandas as pd
from .base_evaluator import BaseEvaluator, SFTEvaluator
import re
RESULT_RE = re.compile(r"\s+(1-0|0-1|1/2-1/2|\*)\s*$")
class ExampleEvaluator(BaseEvaluator):
    """
    Example evaluator for a parquet file with 'state' and 'move' columns.
    
    Usage:
        evaluator = ExampleEvaluator(
            model=model,
            tokenizer=tokenizer,
            device=device,
            move_format="san"
        )
        metrics = evaluator.evaluate("path/to/eval_data.parquet")
    """
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load parquet file."""
        return pd.read_parquet(file_path)
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "move"


class CustomEvaluator(BaseEvaluator):
    """
    Example of a more customized evaluator with preprocessing.
    
    This shows how to:
    - Handle custom column names
    - Preprocess states (e.g., add special tokens)
    - Preprocess moves (e.g., normalize format)
    """
    
    def __init__(self, state_col: str, move_col: str, **kwargs):
        """
        Args:
            state_col: Name of the state column in your dataset
            move_col: Name of the move column in your dataset
            **kwargs: Other arguments passed to BaseEvaluator
        """
        super().__init__(**kwargs)
        self.state_col_name = state_col
        self.move_col_name = move_col
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from parquet or CSV."""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    def get_state_column(self) -> str:
        return self.state_col_name
    
    def get_move_column(self) -> str:
        return self.move_col_name
    
    def preprocess_state(self, state: str) -> str:
        """
        Example preprocessing: ensure state ends with a space
        so the model knows to generate the next move.
        """
        state = state.strip()
        if not state.endswith(' '):
            state += ' '
        return state
    
    def preprocess_move(self, move: str) -> str:
        """
        Example preprocessing: strip whitespace and remove check symbols.
        """
        return move.strip()

class HumanGamesEvaluator(BaseEvaluator):
    """
    Evaluator for human games test dataset.
    """
    
    def load_data(self, file_path="${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_opening.parquet") -> pd.DataFrame:
        """Load parquet file."""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "next_move"
    
class PuzzlesEvaluator(BaseEvaluator):
    """
    Evaluator for human games test dataset.
    """
    
    def load_data(self, file_path="${REPO_ROOT}/LLM-Pretraining/data/chess/test/puzzles_test.csv") -> pd.DataFrame:
        """Load parquet file."""
        import chess.pgn
        import io

        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        df.dropna(subset=["ctx", "Moves"],inplace=True)
        # Assume 'ctx' column is PGN string, and 'moves' column has moves in space-separated format
        # We apply the first move in 'moves' to 'ctx', get the new PGN, then get the second move and set as a new column

        def apply_first_move_and_get_new_pgn(row):
            ctx = row["ctx"]
            moves = row["Moves"]
            moves_split = moves.strip().split()
            if len(moves_split) < 2:
                # not enough moves, set to None
                return pd.Series({"current_state": None, "best_next_move": None})
            first_move, second_move = moves_split[0], moves_split[1]

            # Parse PGN
            game = chess.pgn.read_game(io.StringIO(ctx))
            board = game.end().board() if game is not None else chess.Board()
            try:
                move_obj = board.parse_san(first_move)
            except Exception:
                # Invalid move
                return pd.Series({"current_state": None, "best_next_move": None})

            board.push(move_obj)

            # Build a new PGN after applying the move
            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            new_game = chess.pgn.Game.from_board(board)
            new_pgn = new_game.accept(exporter)
            movetext = re.sub(r"\[[^\]]*\]\s*", "", new_pgn)
            movetext = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", movetext).strip()
            return pd.Series({"current_state": movetext, "best_next_move": second_move})

        new_cols = df.apply(apply_first_move_and_get_new_pgn, axis=1)
        df["current_state"] = new_cols["current_state"]
        df["best_next_move"] = new_cols["best_next_move"]
        return df
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "current_state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "best_next_move"
    
class RandomGamesEvaluator(BaseEvaluator):
    """
    Evaluator for practice games test dataset.
    """
    
    def load_data(self, file_path="${REPO_ROOT}/LLM-Pretraining/data/chess/test/random_games_100.parquet") -> pd.DataFrame:
        """Load parquet file."""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "next_move_san"


# ============ SFT Evaluators (with <T>...</T> format support) ============

class HumanGamesEvaluatorSFT(SFTEvaluator):
    """
    SFT Evaluator for human games test dataset.
    Parses </T> token and extracts move after it, tracks format compliance.
    """
    
    def load_data(self, file_path="${REPO_ROOT}/LLM-Pretraining/data/chess/test/human_games_test_opening.parquet") -> pd.DataFrame:
        """Load parquet file."""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "next_move"


class PuzzlesEvaluatorSFT(SFTEvaluator):
    """
    SFT Evaluator for puzzles test dataset.
    Parses </T> token and extracts move after it, tracks format compliance.
    """
    
    def load_data(self, file_path="${REPO_ROOT}/LLM-Pretraining/data/chess/test/puzzles_test.csv") -> pd.DataFrame:
        """Load parquet file."""
        import chess.pgn
        import io

        if file_path.endswith('.parquet'):
            df = pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        df.dropna(subset=["ctx", "Moves"], inplace=True)

        def apply_first_move_and_get_new_pgn(row):
            ctx = row["ctx"]
            moves = row["Moves"]
            moves_split = moves.strip().split()
            if len(moves_split) < 2:
                return pd.Series({"current_state": None, "best_next_move": None})
            first_move, second_move = moves_split[0], moves_split[1]

            game = chess.pgn.read_game(io.StringIO(ctx))
            board = game.end().board() if game is not None else chess.Board()
            try:
                move_obj = board.parse_san(first_move)
            except Exception:
                return pd.Series({"current_state": None, "best_next_move": None})

            board.push(move_obj)

            exporter = chess.pgn.StringExporter(headers=True, variations=False, comments=False)
            new_game = chess.pgn.Game.from_board(board)
            new_pgn = new_game.accept(exporter)
            movetext = re.sub(r"\[[^\]]*\]\s*", "", new_pgn)
            movetext = re.sub(r"\s*(1-0|0-1|1/2-1/2|\*)\s*$", "", movetext).strip()
            return pd.Series({"current_state": movetext, "best_next_move": second_move})

        new_cols = df.apply(apply_first_move_and_get_new_pgn, axis=1)
        df["current_state"] = new_cols["current_state"]
        df["best_next_move"] = new_cols["best_next_move"]
        return df
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "current_state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "best_next_move"


class RandomGamesEvaluatorSFT(SFTEvaluator):
    """
    SFT Evaluator for random games test dataset.
    Parses </T> token and extracts move after it, tracks format compliance.
    """
    
    def load_data(self, file_path="${REPO_ROOT}/LLM-Pretraining/data/chess/test/random_games_100.parquet") -> pd.DataFrame:
        """Load parquet file."""
        if file_path.endswith('.parquet'):
            return pd.read_parquet(file_path)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path)
    
    def get_state_column(self) -> str:
        """Return the column name for chess states."""
        return "state"
    
    def get_move_column(self) -> str:
        """Return the column name for target moves."""
        return "next_move_san"
    