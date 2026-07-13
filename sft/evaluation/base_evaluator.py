"""Base evaluator class for custom evaluation tasks."""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import pandas as pd
import torch
from pathlib import Path
import chess

from .inference import generate_moves_batch, generate_moves_batch_sft, generate_moves_batch_k, generate_moves_batch_sft_k
from .metrics import evaluate_predictions, calculate_pass_at_k
from .utils import lan_to_uci, san_to_uci


class BaseEvaluator(ABC):
    """
    Base class for evaluation tasks.
    
    Users should subclass this and implement the abstract methods
    to customize data loading and column mapping for their specific datasets.
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        move_format: str = "san",
        batch_size: int = 32,
        max_new_tokens: int = 20,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Initialize the evaluator.
        
        Args:
            model: The GPT model to evaluate
            tokenizer: Tokenizer instance
            device: Device to run inference on
            move_format: Format of moves ("san", "uci", "lan")
            batch_size: Batch size for inference
            max_new_tokens: Max tokens to generate per move
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.move_format = move_format
        self.batch_size = batch_size
        self.generation_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_k": top_k,
        }
    
    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load evaluation data from file.
        
        Args:
            file_path: Path to the evaluation file (e.g., parquet file)
        
        Returns:
            pandas DataFrame containing the evaluation data
        
        Example implementation:
            return pd.read_parquet(file_path)
        """
        raise NotImplementedError("Subclasses must implement load_data()")
    
    @abstractmethod
    def get_state_column(self) -> str:
        """
        Return the name of the column containing chess states.
        
        Returns:
            Column name as string (e.g., "state", "position", "pgn")
        
        Example implementation:
            return "state"
        """
        raise NotImplementedError("Subclasses must implement get_state_column()")
    
    @abstractmethod
    def get_move_column(self) -> str:
        """
        Return the name of the column containing target moves.
        
        Returns:
            Column name as string (e.g., "move", "target", "best_move")
        
        Example implementation:
            return "move"
        """
        raise NotImplementedError("Subclasses must implement get_move_column()")
    
    def preprocess_state(self, state: str) -> str:
        """
        Optional preprocessing of state strings before inference.
        Override this if you need custom preprocessing.
        
        Args:
            state: Raw state string from dataset
        
        Returns:
            Preprocessed state string
        """
        return state
    
    def preprocess_move(self, move: str) -> str:
        """
        Optional preprocessing of target move strings.
        Override this if you need custom preprocessing.
        
        Args:
            move: Raw move string from dataset
        
        Returns:
            Preprocessed move string
        """
        return move
    
    def evaluate(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run evaluation on a dataset.
        
        Args:
            file_path: Path to evaluation file
            max_samples: If set, only evaluate first N samples
            verbose: If True, print progress information
        
        Returns:
            Dictionary of evaluation metrics
        """
        if verbose:
            print(f"Loading data from {file_path}...")
        
        # Load data using custom loader
        df = self.load_data(file_path)
        
        # Get column names
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        # Validate columns exist
        if state_col not in df.columns:
            raise ValueError(f"State column '{state_col}' not found in dataset. Available: {df.columns.tolist()}")
        if move_col not in df.columns:
            raise ValueError(f"Move column '{move_col}' not found in dataset. Available: {df.columns.tolist()}")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.head(max_samples)
        
        if verbose:
            print(f"Evaluating on {len(df)} samples...")
        
        # Extract and preprocess states and moves
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        # Generate predictions
        if verbose:
            print("Generating predictions...")
        
        _, predicted_moves = generate_moves_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        # Calculate metrics
        if verbose:
            print("Calculating metrics...")
        
        predicted_uci = []
        for pred, state in zip(predicted_moves, states):
            try:
                predicted_uci.append(lan_to_uci(pred))  # Assumes lan_to_uci handles side_to_move based on state
            except ValueError:
                predicted_uci.append("")  # Invalid LAN -> empty

        target_uci = []
        for target, state in zip(target_moves, states):
            try:
                target_uci.append(san_to_uci(target, state))
            except ValueError:
                target_uci.append("")  # Invalid SAN -> empty

        metrics = evaluate_predictions(
            states=states,
            predicted_moves=predicted_uci,
            target_moves=target_uci,
            move_format="uci",
            normalize=True
        )
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS")
            print("="*50)
            print(f"Legal Move Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
            print(f"Move Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
            print("="*50 + "\n")
        
        return metrics
    
    def evaluate_pass_at_k(
        self,
        file_path: str,
        k: int = 5,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        output_path: Optional[str] = None,
    ) -> Dict[str, float]:
        """
        Run pass@k evaluation on a dataset.
        Generates k samples per state and checks if any match the target.
        
        Args:
            file_path: Path to evaluation file
            k: Number of samples to generate per state
            max_samples: If set, only evaluate first N samples
            verbose: If True, print progress information
        
        Returns:
            Dictionary of evaluation metrics including pass@k
        """
        if verbose:
            print(f"Loading data from {file_path}...")
        
        # Load data using custom loader
        df = self.load_data(file_path)
        
        # Get column names
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        # Validate columns exist
        if state_col not in df.columns:
            raise ValueError(f"State column '{state_col}' not found in dataset. Available: {df.columns.tolist()}")
        if move_col not in df.columns:
            raise ValueError(f"Move column '{move_col}' not found in dataset. Available: {df.columns.tolist()}")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.head(max_samples)
        
        if verbose:
            print(f"Evaluating pass@{k} on {len(df)} samples (generating {k} samples per state)...")
        
        # Extract and preprocess states and moves
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        # Generate k predictions per state
        if verbose:
            print(f"Generating {k} predictions per state...")
        
        all_moves, all_first_moves = generate_moves_batch_k(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            k=k,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        # Convert all predictions to UCI
        if verbose:
            print("Converting predictions to UCI format...")
        
        all_predicted_uci = []
        for pred_list in all_first_moves:
            state_uci_list = []
            for pred in pred_list:
                try:
                    if pred is not None:
                        state_uci_list.append(lan_to_uci(pred))
                    else:
                        state_uci_list.append("")
                except ValueError:
                    state_uci_list.append("")
            all_predicted_uci.append(state_uci_list)
        
        # Convert targets to UCI
        target_uci = []
        for target, state in zip(target_moves, states):
            try:
                target_uci.append(san_to_uci(target, state))
            except ValueError:
                target_uci.append("")
        
        # Calculate pass@k
        if verbose:
            print("Calculating pass@k metrics...")
        
        pass_at_k_metrics = calculate_pass_at_k(
            predicted_moves_list=all_predicted_uci,
            target_moves=target_uci,
            k=k,
            normalize=True
        )
        
        # Also calculate individual metrics for the first prediction (for comparison)
        first_predicted_uci = [pred_list[0] if pred_list else "" for pred_list in all_predicted_uci]
        single_metrics = evaluate_predictions(
            states=states,
            predicted_moves=first_predicted_uci,
            target_moves=target_uci,
            move_format="uci",
            normalize=True
        )
        
        # Combine metrics
        metrics = {
            **single_metrics,
            **pass_at_k_metrics,
        }
        
        if verbose:
            print("\n" + "="*50)
            print(f"PASS@{k} EVALUATION RESULTS")
            print("="*50)
            print(f"Pass@{k}: {metrics['pass_at_k']:.2%} ({metrics['num_passed']}/{metrics['num_total']})")
            print(f"Single Sample Legal Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
            print(f"Single Sample Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
            print("="*50 + "\n")
        
        if output_path is not None:
            import json
            from pathlib import Path

            df_results = df.copy()
            df_results["state"] = states
            df_results["target_moves_uci"] = target_uci

            df_results[f"predicted_moves@{k}"] = all_predicted_uci
            df_results[f"pass@{k}_hit"] = [
                any((p is not None) and (t != "") and (str(p).strip().lower() == str(t).strip().lower())
                    for p in preds[:k])
                for preds, t in zip(df_results[f"predicted_moves@{k}"], target_uci)
            ]

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".parquet":
                df_results.to_parquet(output_path, index=False)
            elif output_path.suffix == ".csv":
                df_csv = df_results.copy()
                for col in [f"predicted_moves@{k}"]:
                    df_csv[col] = df_csv[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
                df_csv.to_csv(output_path, index=False)
            else:
                df_results.to_parquet(str(output_path) + ".parquet", index=False)

            if verbose:
                print(f"\nPass@{k} results saved to {output_path}")
                
        return metrics
    
    def evaluate_and_save_predictions(
        self,
        file_path: str,
        output_path: str,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run evaluation and save predictions to file.
        
        Args:
            file_path: Path to evaluation file
            output_path: Path to save predictions
            max_samples: If set, only evaluate first N samples
            verbose: If True, print progress information
        
        Returns:
            Dictionary of evaluation metrics
        """
        # Load data
        df = self.load_data(file_path)
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        if max_samples is not None:
            df = df.head(max_samples)
        
        # Generate predictions
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        predicted_moves_seq, predicted_moves = generate_moves_batch(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )

        target_moves_uci = []
        for target, state in zip(target_moves, states):
            try:
                target_moves_uci.append(san_to_uci(target, state))
            except:
                target_moves_uci.append("")  # Invalid SAN -> empty
        
        predicted_moves_uci = []
        for pred, state in zip(predicted_moves, states):
            try:
                predicted_moves_uci.append(lan_to_uci(pred))  # Assumes lan_to_uci handles side_to_move based on state
            except:
                predicted_moves_uci.append("")  # Invalid LAN -> empty
        # Calculate metrics
        metrics = evaluate_predictions(
            states=states,
            predicted_moves=predicted_moves_uci,
            target_moves=target_moves_uci,
            move_format=self.move_format,
            normalize=True
        )
        
        # Add predictions to dataframe
        df_results = df.copy()
        df_results['state'] = states
        df_results['target_moves_uci'] = target_moves_uci
        df_results['predicted_moves_uci'] = predicted_moves_uci
        df_results['predicted_moves_seq'] = predicted_moves_seq
        df_results['is_legal'] = [
            self._check_legal(s, p) for s, p in zip(states, predicted_moves)
        ]
        df_results['is_match'] = [
            p.strip().lower() == t.strip().lower() if p is not None and t is not None else False
            for p, t in zip(predicted_moves_uci, target_moves_uci)
        ]
        
        # print the first 10 predicted and target moves
        print(df_results[['predicted_moves_uci', 'target_moves_uci']].head(10))
        print(df_results[['is_legal', 'is_match']].head(10))
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df_results.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df_results.to_csv(output_path, index=False)
        else:
            # Default to parquet
            df_results.to_parquet(str(output_path) + '.parquet', index=False)
        
        if verbose:
            print(f"Predictions saved to {output_path}")
        
        return metrics
    
    def _check_legal(self, state: str, move: str) -> bool:
        """Helper to check if a move is legal."""
        from .metrics import is_move_legal
        from .utils import lan_to_uci
        if move == "" or move is None:
            return False
        if not move.startswith("O-O"):
            move = lan_to_uci(move)
            return is_move_legal(state, move, "uci")
        else:
            return is_move_legal(state, move, "san")


class SFTEvaluator(BaseEvaluator):
    """
    Evaluator for SFT models that generate <T>...</T> thinking format.
    
    Extends BaseEvaluator to use generate_moves_batch_sft which:
    - Parses the </T> token and extracts the move after it
    - Tracks format compliance (how many outputs follow <T>...</T> format)
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        device: torch.device,
        move_format: str = "san",
        batch_size: int = 32,
        max_new_tokens: int = 2048,  # Longer for thinking
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ):
        """
        Initialize the SFT evaluator.
        
        Args:
            model: The GPT model to evaluate
            tokenizer: Tokenizer instance
            device: Device to run inference on
            move_format: Format of moves ("san", "uci", "lan")
            batch_size: Batch size for inference
            max_new_tokens: Max tokens to generate (default 512 for thinking)
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
        """
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            device=device,
            move_format=move_format,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
    
    def evaluate(
        self,
        file_path: str,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run evaluation on a dataset using SFT inference.
        
        Args:
            file_path: Path to evaluation file
            max_samples: If set, only evaluate first N samples
            verbose: If True, print progress information
        
        Returns:
            Dictionary of evaluation metrics including format compliance
        """
        if verbose:
            print(f"Loading data from {file_path}...")
        
        # Load data using custom loader
        df = self.load_data(file_path)
        
        # Get column names
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        # Validate columns exist
        if state_col not in df.columns:
            raise ValueError(f"State column '{state_col}' not found in dataset. Available: {df.columns.tolist()}")
        if move_col not in df.columns:
            raise ValueError(f"Move column '{move_col}' not found in dataset. Available: {df.columns.tolist()}")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.head(max_samples)
        
        if verbose:
            print(f"Evaluating on {len(df)} samples...")
        
        # Extract and preprocess states and moves
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        # Generate predictions using SFT inference
        if verbose:
            print("Generating predictions (SFT mode)...")
        
        sft_results = generate_moves_batch_sft(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        predicted_moves = sft_results['moves']
        
        if verbose:
            print(f"Format compliance: {sft_results['num_following_format']}/{sft_results['total']} ({sft_results['format_compliance_rate']:.1%})")
        
        # Calculate metrics
        if verbose:
            print("Calculating metrics...")
        
        predicted_uci = []
        for pred, state in zip(predicted_moves, states):
            try:
                if pred is not None:
                    predicted_uci.append(lan_to_uci(pred))
                else:
                    predicted_uci.append("")
            except ValueError:
                predicted_uci.append("")  # Invalid LAN -> empty

        target_uci = []
        for target, state in zip(target_moves, states):
            try:
                target_uci.append(san_to_uci(target, state))
            except ValueError:
                target_uci.append("")  # Invalid SAN -> empty

        metrics = evaluate_predictions(
            states=states,
            predicted_moves=predicted_uci,
            target_moves=target_uci,
            move_format="uci",
            normalize=True
        )
        
        # Add format compliance metrics
        metrics['format_compliance_rate'] = sft_results['format_compliance_rate']
        metrics['num_following_format'] = sft_results['num_following_format']
        
        if verbose:
            print("\n" + "="*50)
            print("EVALUATION RESULTS (SFT)")
            print("="*50)
            print(f"Format Compliance:   {metrics['format_compliance_rate']:.2%} ({metrics['num_following_format']}/{metrics['num_total']})")
            print(f"Legal Move Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
            print(f"Move Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
            print("="*50 + "\n")
        
        return metrics
    
    def evaluate_pass_at_k(
    self,
        file_path: str,
        k: int = 5,
        max_samples: Optional[int] = None,
        verbose: bool = True,
        output_path: Optional[str] = None,  
    ) -> Dict[str, float]:
        """
        Run pass@k evaluation on a dataset using SFT inference.
        Generates k samples per state and checks if any match the target.
        
        Args:
            file_path: Path to evaluation file
            k: Number of samples to generate per state
            max_samples: If set, only evaluate first N samples
            verbose: If True, print progress information
        
        Returns:
            Dictionary of evaluation metrics including pass@k and format compliance
        """
        if verbose:
            print(f"Loading data from {file_path}...")
        
        # Load data using custom loader
        df = self.load_data(file_path)
        
        # Get column names
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        # Validate columns exist
        if state_col not in df.columns:
            raise ValueError(f"State column '{state_col}' not found in dataset. Available: {df.columns.tolist()}")
        if move_col not in df.columns:
            raise ValueError(f"Move column '{move_col}' not found in dataset. Available: {df.columns.tolist()}")
        
        # Limit samples if requested
        if max_samples is not None:
            df = df.head(max_samples)
        
        if verbose:
            print(f"Evaluating pass@{k} on {len(df)} samples (generating {k} samples per state)...")
        
        # Extract and preprocess states and moves
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        # Generate k predictions per state using SFT inference
        if verbose:
            print(f"Generating {k} predictions per state (SFT mode)...")
        
        sft_results = generate_moves_batch_sft_k(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            k=k,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        all_moves = sft_results['moves']
        
        if verbose:
            print(f"Format compliance: {sft_results['format_compliance_rate']:.1%} ({sft_results['num_following_format']}/{sft_results['total']})")
        
        # Convert all predictions to UCI
        if verbose:
            print("Converting predictions to UCI format...")
        
        all_predicted_uci = []
        for pred_list in all_moves:
            state_uci_list = []
            for pred in pred_list:
                try:
                    if pred is not None:
                        state_uci_list.append(lan_to_uci(pred))
                    else:
                        state_uci_list.append("")
                except ValueError:
                    state_uci_list.append("")
            all_predicted_uci.append(state_uci_list)
        
        # Convert targets to UCI
        target_uci = []
        for target, state in zip(target_moves, states):
            try:
                target_uci.append(san_to_uci(target, state))
            except ValueError:
                target_uci.append("")
        
        # Calculate pass@k
        if verbose:
            print("Calculating pass@k metrics...")
        
        pass_at_k_metrics = calculate_pass_at_k(
            predicted_moves_list=all_predicted_uci,
            target_moves=target_uci,
            k=k,
            normalize=True
        )
        
        # Also calculate individual metrics for the first prediction (for comparison)
        first_predicted_uci = [pred_list[0] if pred_list else "" for pred_list in all_predicted_uci]
        single_metrics = evaluate_predictions(
            states=states,
            predicted_moves=first_predicted_uci,
            target_moves=target_uci,
            move_format="uci",
            normalize=True
        )
        
        # Combine metrics
        metrics = {
            **single_metrics,
            **pass_at_k_metrics,
            'format_compliance_rate': sft_results['format_compliance_rate'],
            'num_following_format': sft_results['num_following_format'],
        }
        
        if verbose:
            print("\n" + "="*50)
            print(f"PASS@{k} EVALUATION RESULTS (SFT)")
            print("="*50)
            print(f"Pass@{k}: {metrics['pass_at_k']:.2%} ({metrics['num_passed']}/{metrics['num_total']})")
            print(f"Format Compliance:   {metrics['format_compliance_rate']:.2%} ({metrics['num_following_format']}/{sft_results['total']})")
            print(f"Single Sample Legal Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
            print(f"Single Sample Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
            print("="*50 + "\n")
        
        if output_path is not None:
            import json
            from pathlib import Path

            df_results = df.copy()
            df_results["state"] = states
            df_results["target_moves_uci"] = target_moves_uci


            df_results[f"predicted_moves@{k}"] = sft_results["moves"]
            df_results[f"raw_outputs@{k}"] = sft_results["raw_outputs"]
            df_results[f"format_compliance@{k}"] = sft_results["format_compliance"]

            df_results[f"pass@{k}_hit"] = [
                any((p is not None) and (t != "") and (str(p).strip().lower() == str(t).strip().lower())
                    for p in preds[:k])
                for preds, t in zip(df_results[f"predicted_moves@{k}"], target_moves_uci)
            ]

            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            if output_path.suffix == ".parquet":
                df_results.to_parquet(output_path, index=False)
            elif output_path.suffix == ".csv":
                df_csv = df_results.copy()
                for col in [f"predicted_moves@{k}", f"raw_outputs@{k}", f"format_compliance@{k}"]:
                    df_csv[col] = df_csv[col].apply(lambda x: json.dumps(x, ensure_ascii=False))
                df_csv.to_csv(output_path, index=False)
            else:
                df_results.to_parquet(str(output_path) + ".parquet", index=False)

            if verbose:
                print(f"\nPass@{k} results saved to {output_path}")
        return metrics
    
    def evaluate_and_save_predictions(
        self,
        file_path: str,
        output_path: str,
        max_samples: Optional[int] = None,
        verbose: bool = True
    ) -> Dict[str, float]:
        """
        Run SFT evaluation and save predictions to file.
        
        Args:
            file_path: Path to evaluation file
            output_path: Path to save predictions
            max_samples: If set, only evaluate first N samples
            verbose: If True, print progress information
        
        Returns:
            Dictionary of evaluation metrics including format compliance
        """
        # Load data
        df = self.load_data(file_path)
        state_col = self.get_state_column()
        move_col = self.get_move_column()
        
        if max_samples is not None:
            df = df.head(max_samples)
        
        # Generate predictions using SFT inference
        states = [self.preprocess_state(str(s)) for s in df[state_col].tolist()]
        target_moves = [self.preprocess_move(str(m)) for m in df[move_col].tolist()]
        
        if verbose:
            print("Generating predictions (SFT mode)...")
        
        sft_results = generate_moves_batch_sft(
            model=self.model,
            tokenizer=self.tokenizer,
            states=states,
            device=self.device,
            batch_size=self.batch_size,
            **self.generation_kwargs
        )
        
        predicted_moves = sft_results['moves']
        raw_outputs = sft_results['raw_outputs']
        format_compliance = sft_results['format_compliance']
        
        if verbose:
            print(f"Format compliance: {sft_results['num_following_format']}/{sft_results['total']} ({sft_results['format_compliance_rate']:.1%})")

        target_moves_uci = []
        for target, state in zip(target_moves, states):
            try:
                target_moves_uci.append(san_to_uci(target, state))
            except:
                target_moves_uci.append("")  # Invalid SAN -> empty
        
        predicted_moves_uci = []
        for pred, state in zip(predicted_moves, states):
            try:
                if pred is not None:
                    predicted_moves_uci.append(lan_to_uci(pred))
                else:
                    predicted_moves_uci.append("")
            except:
                predicted_moves_uci.append("")  # Invalid LAN -> empty
        
        # Calculate metrics
        metrics = evaluate_predictions(
            states=states,
            predicted_moves=predicted_moves_uci,
            target_moves=target_moves_uci,
            move_format=self.move_format,
            normalize=True
        )
        
        # Add format compliance metrics
        metrics['format_compliance_rate'] = sft_results['format_compliance_rate']
        metrics['num_following_format'] = sft_results['num_following_format']
        
        # Add predictions to dataframe
        df_results = df.copy()
        df_results['state'] = states
        df_results['target_moves_uci'] = target_moves_uci
        df_results['predicted_moves_uci'] = predicted_moves_uci
        df_results['raw_output'] = raw_outputs  # Full generated text with <T>...</T>
        df_results['follows_format'] = format_compliance
        df_results['is_legal'] = [
            self._check_legal(s, p) if p is not None else False 
            for s, p in zip(states, predicted_moves)
        ]
        df_results['is_match'] = [
            p.strip().lower() == t.strip().lower() if p is not None and t is not None and p != "" else False
            for p, t in zip(predicted_moves_uci, target_moves_uci)
        ]
        
        # Print the first 10 examples
        print("\n" + "="*50)
        print("Sample predictions:")
        print("="*50)
        print(df_results[['predicted_moves_uci', 'target_moves_uci', 'follows_format']].head(10))
        print(df_results[['is_legal', 'is_match']].head(10))
        
        # Save to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if output_path.suffix == '.parquet':
            df_results.to_parquet(output_path, index=False)
        elif output_path.suffix == '.csv':
            df_results.to_csv(output_path, index=False)
        else:
            # Default to parquet
            df_results.to_parquet(str(output_path) + '.parquet', index=False)
        
        if verbose:
            print(f"\nPredictions saved to {output_path}")
            print("\n" + "="*50)
            print("EVALUATION RESULTS (SFT)")
            print("="*50)
            print(f"Format Compliance:   {metrics['format_compliance_rate']:.2%} ({metrics['num_following_format']}/{metrics['num_total']})")
            print(f"Legal Move Accuracy: {metrics['legal_accuracy']:.2%} ({metrics['num_legal']}/{metrics['num_total']})")
            print(f"Move Match Accuracy: {metrics['match_accuracy']:.2%} ({metrics['num_matches']}/{metrics['num_total']})")
            print("="*50 + "\n")
        
        return metrics

