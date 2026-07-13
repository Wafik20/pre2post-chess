"""Model inference functions for generating chess moves."""

import torch
from typing import List, Optional, Union
import re

def generate_move(
    model,
    tokenizer,
    state: str,
    device: torch.device,
    max_new_tokens: int = 20,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[List[str]] = None,
    seed: Optional[int] = None,
) -> str:
    """
    Generate a single move prediction using Hugging Face's generate() with KV cache.
    """
    model.eval()
    
    # Encode input state using your custom tokenizer
    input_ids = tokenizer.encode(state)
    if tokenizer.eos_id() is not None and input_ids[-1] == tokenizer.eos_id():
        input_ids = input_ids[:-1]
    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    
    if seed is not None:
        from transformers import set_seed
        set_seed(seed)
    
    # Use model's generate() method with caching
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=True,  # Enables KV cache
            do_sample=True,  # For temperature/top_k sampling
            pad_token_id=tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') else tokenizer.eos_id(),
            eos_token_id=tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else None,
        )
    
    # Decode only the newly generated tokens
    generated_ids = generated_ids[0, input_ids.shape[1]:]  # Skip input
    move = tokenizer.decode(generated_ids.tolist()).strip()
    
    # Apply your custom cleanup (e.g., extract first move, handle stop tokens)
    if stop_tokens:
        for stop in stop_tokens:
            if move.endswith(stop):
                move = move[:-len(stop)].strip()
    
    first_move = _extract_first_move(move)
    
    return move, first_move

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
    import re
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


def generate_moves_batch(
    model,
    tokenizer,
    states: List[str],
    device: torch.device,
    batch_size: int = 32,
    seed: Optional[int] = None,
    **generation_kwargs
) -> tuple:
    """
    Generate moves for a batch of states using true batched generation.
    
    Args:
        model: The GPT model
        tokenizer: Tokenizer instance
        states: List of input state strings
        device: Device to run inference on
        batch_size: Batch size for processing
        seed: Random seed for generation
        **generation_kwargs: Additional arguments passed to generate
    
    Returns:
        tuple: (moves, first_moves) where moves is list of full outputs and first_moves is list of extracted first moves
    """
    model.eval()
    moves = []
    first_moves = []
    
    max_new_tokens = generation_kwargs.get('max_new_tokens', 10)
    temperature = generation_kwargs.get('temperature', 1.0)
    top_k = generation_kwargs.get('top_k', None)
    
    pad_id = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id() is not None else 0
    eos_id = tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else None
    
    max_ctx = model.config.max_position_embeddings
    # Process in batches
    for batch_start in range(0, len(states), batch_size):
        batch_states = states[batch_start:batch_start + batch_size]
        
        # Encode all states in batch
        encoded_batch = []
        valid_indices = []  
        skipped = 0

        for i, state in enumerate(batch_states):
            ids = tokenizer.encode(state)

            if eos_id is not None and ids and ids[-1] == eos_id:
                ids = ids[:-1]

            if len(ids) + max_new_tokens > max_ctx: # check if the input is too long
                skipped += 1
                continue 

            encoded_batch.append(ids)
            valid_indices.append(i)
        
        if len(encoded_batch) == 0:
            continue 

        # Pad to max length (left padding for generation)
        print("Skipped %d states due to length" % skipped)
        max_len = max(len(ids) for ids in encoded_batch)
        input_ids = torch.full((len(encoded_batch), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(encoded_batch), max_len), dtype=torch.long, device=device)
        
        for i, ids in enumerate(encoded_batch):
            # Left padding
            start_pos = max_len - len(ids)
            input_ids[i, start_pos:] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, start_pos:] = 1
        
        # Generate in batch
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                do_sample=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        
        # Decode each generated sequence
        batch_moves = [None] * len(batch_states)
        batch_first_moves = [None] * len(batch_states)

        for out_idx, orig_idx in enumerate(valid_indices):
            gen_ids = generated_ids[out_idx]
            new_tokens = gen_ids[max_len:].tolist()
            new_tokens = [t for t in new_tokens if t != pad_id and t != eos_id]
            move = tokenizer.decode(new_tokens).strip()

            first_move = _extract_first_move(move)

            batch_first_moves[orig_idx] = first_move
            batch_moves[orig_idx] = move
        
        moves.extend(batch_moves)
        first_moves.extend(batch_first_moves)
    return moves, first_moves


def generate_moves_batch_k(
    model,
    tokenizer,
    states: List[str],
    device: torch.device,
    k: int = 5,
    batch_size: int = 32,
    seed: Optional[int] = None,
    **generation_kwargs
) -> tuple:
    """
    Generate k samples per state for pass@k evaluation.
    
    Args:
        model: The GPT model
        tokenizer: Tokenizer instance
        states: List of input state strings
        device: Device to run inference on
        k: Number of samples to generate per state
        batch_size: Batch size for processing (will process k*batch_size total)
        seed: Random seed for generation
        **generation_kwargs: Additional arguments passed to generate
    
    Returns:
        tuple: (all_moves, all_first_moves) where:
            - all_moves: List of lists, each inner list contains k full outputs for that state
            - all_first_moves: List of lists, each inner list contains k extracted first moves for that state
    """
    model.eval()
    all_moves = []
    all_first_moves = []
    
    max_new_tokens = generation_kwargs.get('max_new_tokens', 20)
    temperature = generation_kwargs.get('temperature', 1.0)
    top_k = generation_kwargs.get('top_k', None)
    stop_tokens = generation_kwargs.get('stop_tokens', None)
    
    pad_id = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id() is not None else 0
    eos_id = tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else None
    
    from transformers import set_seed
    if seed is not None:
        set_seed(seed)
    
    # Process each state k times
    for state_idx, state in enumerate(states):
        state_moves = []
        state_first_moves = []
        
        # Generate k samples for this state
        # We'll process in batches of k to be efficient
        for sample_idx in range(0, k, batch_size):
            batch_k = min(batch_size, k - sample_idx)
            batch_states = [state] * batch_k
            
            # Encode all states in batch
            encoded_batch = []
            for _ in batch_states:
                ids = tokenizer.encode(state)
                # Remove trailing EOS if present
                if eos_id is not None and ids and ids[-1] == eos_id:
                    ids = ids[:-1]
                encoded_batch.append(ids)
            
            if not encoded_batch:
                continue
            
            # Pad to max length (right padding for generation)
            max_len = max(len(ids) for ids in encoded_batch)
            input_ids = torch.full((len(batch_states), max_len), pad_id, dtype=torch.long, device=device)
            attention_mask = torch.zeros((len(batch_states), max_len), dtype=torch.long, device=device)
            
            for i, ids in enumerate(encoded_batch):
                input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                attention_mask[i, :len(ids)] = 1
            
            # Generate in batch
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    use_cache=True,
                    do_sample=True,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                    num_return_sequences=1,  # Generate one sequence per input
                )
            
            # Decode each generated sequence (skip input tokens)
            for i, (gen_ids, orig_ids) in enumerate(zip(generated_ids, encoded_batch)):
                # Skip input tokens
                new_tokens = gen_ids[len(orig_ids):].tolist()
                move = tokenizer.decode(new_tokens).strip()
                
                # Apply stop token cleanup
                if stop_tokens:
                    for stop in stop_tokens:
                        if move.endswith(stop):
                            move = move[:-len(stop)].strip()
                
                first_move = _extract_first_move(move)
                state_moves.append(move)
                state_first_moves.append(first_move)
        
        all_moves.append(state_moves)
        all_first_moves.append(state_first_moves)
    
    return all_moves, all_first_moves


# ============ SFT Inference Functions ============

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


def generate_move_sft(
    model,
    tokenizer,
    state: str,
    device: torch.device,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_k: Optional[int] = None,
    stop_tokens: Optional[List[str]] = None,
) -> tuple:
    """
    Generate a move prediction for SFT models that use <T></T> thinking format.

    Returns:
        tuple: (raw_output, move_after_T, follows_format)
            - raw_output: The full generated text
            - move_after_T: The extracted move after </T>
            - follows_format: Boolean indicating if output follows <T>...</T> format
    """
    model.eval()

    # Encode input state
    input_ids = tokenizer.encode(state)
    if tokenizer.eos_id() is not None and input_ids[-1] == tokenizer.eos_id():
        input_ids = input_ids[:-1]
    input_ids.append(tokenizer.get_vocab()["<T>"])

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Generate with model
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            use_cache=True,
            do_sample=True,
            pad_token_id=tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') else None,
            eos_token_id=tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else None,
        )

    # Decode the newly generated tokens
    generated_ids = generated_ids[0, input_ids.shape[1]:]
    raw_output = tokenizer.decode(generated_ids.tolist()).strip()

    # Apply stop token cleanup
    if stop_tokens:
        for stop in stop_tokens:
            if raw_output.endswith(stop):
                raw_output = raw_output[:-len(stop)].strip()

    # Extract move after </T> and check format
    move_after_T, follows_format = _extract_move_after_thinking(raw_output)

    return raw_output, move_after_T, follows_format


def generate_moves_batch_sft(
    model,
    tokenizer,
    states: List[str],
    device: torch.device,
    batch_size: int = 32,
    **generation_kwargs
) -> dict:
    """
    Generate moves for a batch of states using SFT models with <T></T> format.
    Uses true batched generation for speed.
    
    Args:
        model: The GPT model
        tokenizer: Tokenizer instance
        states: List of input state strings
        device: Device to run inference on
        batch_size: Batch size for processing
        **generation_kwargs: Additional arguments passed to generate
    
    Returns:
        dict with keys:
            - raw_outputs: List of full generated texts
            - moves: List of extracted moves after </T>
            - format_compliance: List of booleans indicating format compliance
            - format_compliance_rate: Float, percentage of outputs following format
    """
    model.eval()
    raw_outputs = []
    moves = []
    format_compliance = []
    
    max_new_tokens = generation_kwargs.get('max_new_tokens', 512)
    temperature = generation_kwargs.get('temperature', 1.0)
    top_k = generation_kwargs.get('top_k', None)
    
    pad_id = tokenizer.pad_id() if hasattr(tokenizer, 'pad_id') and tokenizer.pad_id() is not None else 0
    eos_id = tokenizer.eos_id() if hasattr(tokenizer, 'eos_id') else None
    
    max_ctx = model.config.max_position_embeddings
    # Process in batches
    for batch_start in range(0, len(states), batch_size):
        batch_states = states[batch_start:batch_start + batch_size]
        
        # Encode all states in batch
        encoded_batch = []
        valid_indices = []  
        skipped = 0

        for i, state in enumerate(batch_states):
            ids = tokenizer.encode(state)

            if eos_id is not None and ids and ids[-1] == eos_id:
                ids = ids[:-1]

            ids.append(tokenizer.get_vocab()["<T>"])

            if len(ids) + max_new_tokens > max_ctx: # check if the input is too long
                skipped += 1
                continue 

            encoded_batch.append(ids)
            valid_indices.append(i)
        
        if len(encoded_batch) == 0:
            continue 

        # Pad to max length (left padding for generation)
        max_len = max(len(ids) for ids in encoded_batch)
        input_ids = torch.full((len(encoded_batch), max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((len(encoded_batch), max_len), dtype=torch.long, device=device)
        
        for i, ids in enumerate(encoded_batch):
            # Left padding
            start_pos = max_len - len(ids)
            input_ids[i, start_pos:] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[i, start_pos:] = 1
        
        # Generate in batch
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                do_sample=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )
        
        # Decode each generated sequence
        batch_raw = [None] * len(batch_states)
        batch_moves = [None] * len(batch_states)
        batch_format = [False] * len(batch_states)

        for out_idx, orig_idx in enumerate(valid_indices):
            gen_ids = generated_ids[out_idx]
            new_tokens = gen_ids[max_len:].tolist()
            raw_output = tokenizer.decode(new_tokens).strip()

            move, follows_format = _extract_move_after_thinking(raw_output)

            batch_raw[orig_idx] = raw_output
            batch_moves[orig_idx] = move
            batch_format[orig_idx] = follows_format

        raw_outputs.extend(batch_raw)
        moves.extend(batch_moves)
        format_compliance.extend(batch_format)

    # Calculate format compliance rate
    num_following_format = sum(format_compliance)
    format_compliance_rate = num_following_format / len(states) if states else 0.0
    
    return {
        'raw_outputs': raw_outputs,
        'moves': moves,
        'format_compliance': format_compliance,
        'format_compliance_rate': format_compliance_rate,
        'num_following_format': num_following_format,
        'total': len(states)
    }

def generate_moves_batch_sft_k(
    model,
    tokenizer,
    states: List[str],
    device: torch.device,
    k: int = 5,
    batch_size: int = 32,
    **generation_kwargs
) -> dict:
    """
    Vectorized pass@k generation:
    1) repeat each state k times
    2) do normal batched generation
    3) reshape back to (len(states), k)
    """
    model.eval()

    max_new_tokens = generation_kwargs.get("max_new_tokens", 512)
    temperature = generation_kwargs.get("temperature", 1.0)
    top_k = generation_kwargs.get("top_k", None)

    pad_id = tokenizer.pad_id() if hasattr(tokenizer, "pad_id") and tokenizer.pad_id() is not None else 0
    eos_id = tokenizer.eos_id() if hasattr(tokenizer, "eos_id") else None

    max_ctx = model.config.max_position_embeddings
    T_id = tokenizer.get_vocab().get("<T>", None)
    if T_id is None:
        raise ValueError("Tokenizer vocab does not contain '<T>' token.")

    n = len(states)
    if n == 0:
        return {
            "raw_outputs": [],
            "moves": [],
            "format_compliance": [],
            "format_compliance_rate": 0.0,
            "num_following_format": 0,
            "total": 0,
        }

    # 1) Repeat each state k times, stable order:
    # index mapping: flat_idx = i*k + j
    flat_states = [s for s in states for _ in range(k)]
    flat_state_ids = []
    for i in range(n):
        for _ in range(k):
            flat_state_ids.append(i)
    total = len(flat_states)

    flat_raw = [None] * total
    flat_moves = [None] * total
    flat_format = [False] * total

    # 2) Batched generation over repeated list
    for start in range(0, total, batch_size):
        end = min(total, start + batch_size)
        batch_states = flat_states[start:end]

        encoded = []
        keep_flat_indices = []

        for bi, s in enumerate(batch_states):
            ids = tokenizer.encode(s)

            if eos_id is not None and ids and ids[-1] == eos_id:
                ids = ids[:-1]

            ids = ids + [T_id]

            if len(ids) + max_new_tokens > max_ctx:
                # too long -> mark as invalid directly
                flat_idx = start + bi
                flat_raw[flat_idx] = None
                flat_moves[flat_idx] = None
                flat_format[flat_idx] = False
                continue

            encoded.append(ids)
            keep_flat_indices.append(start + bi)

        if not encoded:
            continue

        max_len = max(len(ids) for ids in encoded)
        b = len(encoded)

        input_ids = torch.full((b, max_len), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((b, max_len), dtype=torch.long, device=device)

        for row, ids in enumerate(encoded):
            start_pos = max_len - len(ids)  # left pad
            input_ids[row, start_pos:] = torch.tensor(ids, dtype=torch.long, device=device)
            attention_mask[row, start_pos:] = 1

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                use_cache=True,
                do_sample=True,
                pad_token_id=pad_id,
                eos_token_id=eos_id,
            )

        # 3) Decode and write back to the original flat positions
        for row, flat_idx in enumerate(keep_flat_indices):
            gen_ids = generated_ids[row]
            new_tokens = gen_ids[max_len:].tolist()
            raw_output = tokenizer.decode(new_tokens).strip()

            move, follows_format = _extract_move_after_thinking(raw_output)

            flat_raw[flat_idx] = raw_output
            flat_moves[flat_idx] = move
            flat_format[flat_idx] = follows_format

    # 4) Reshape back to per-state lists of length k
    all_raw_outputs = []
    all_moves = []
    all_format_compliance = []
    
    assert len(flat_state_ids) == len(flat_raw)

    for i in range(n):
        lo = i * k
        hi = (i + 1) * k
        all_raw_outputs.append(flat_raw[lo:hi])
        all_moves.append(flat_moves[lo:hi])
        all_format_compliance.append(flat_format[lo:hi])
        block_ids = flat_state_ids[lo:hi]
        assert all(x == i for x in block_ids), \
            f"State index mismatch in block {i}: {block_ids}"

    total_samples = n * k
    num_following_format = sum(sum(x) for x in all_format_compliance)
    format_compliance_rate = num_following_format / total_samples if total_samples > 0 else 0.0

    return {
        "raw_outputs": all_raw_outputs,
        "moves": all_moves,
        "format_compliance": all_format_compliance,
        "format_compliance_rate": format_compliance_rate,
        "num_following_format": num_following_format,
        "total": total_samples,
    }
