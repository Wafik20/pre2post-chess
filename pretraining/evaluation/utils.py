
import chess 

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

def san_to_uci(san: str, state: str) -> str:
    """
    Convert SAN move to UCI given the state.
    
    Args:
        san: SAN string, e.g., "d4", "Nxd5"
        state: PGN or FEN string for board context
        
    Returns:
        UCI string, e.g., "d2d4", "b1c3"
        
    Raises:
        ValueError if invalid
    """
    import chess
    import chess.pgn
    import io
    
    # Get board from state (reuse or implement _state_to_board if needed)
    def _state_to_board(state: str) -> chess.Board:
        state = state.strip()
        try:
            return chess.Board(state)  # Try FEN
        except:
            pass
        try:
            game = chess.pgn.read_game(io.StringIO(state))
            board = game.end().board() if game else chess.Board()
            return board
        except:
            raise ValueError("Invalid state format")
    
    board = _state_to_board(state)
    try:
        move = board.parse_san(san)
        return move.uci()
    except Exception as e:
        raise ValueError(f"Invalid SAN: {san} in state: {state}") from e

#test
# if __name__ == "__main__":
#     print(lan_to_uci("Pd2d4"))
#     print(lan_to_uci("Pd4xe5"))
#     print(lan_to_uci("Pe7e8=Q"))
#     print(lan_to_uci("O-O"))
#     print(lan_to_uci("O-O-O"))
#     print(lan_to_uci("Pg7g8#"))