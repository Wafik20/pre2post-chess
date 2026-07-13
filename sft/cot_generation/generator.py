import io
from typing import Callable, List, Optional, Dict, Tuple

import chess
import chess.pgn


class TreeNode:
    def __init__(self, board: chess.Board, move: Optional[chess.Move] = None, parent: Optional["TreeNode"] = None):
        self.board = board
        self.move = move              # move taken from parent to reach this node (None for root)
        self.parent = parent
        self.children: Dict[chess.Move, "TreeNode"] = {}
        self.value = None             # optional: for evaluation_policy to store scores

    def is_root(self) -> bool:
        return self.parent is None


def board_from_pgn(pgn_str: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn_str))
    if game is None:
        raise ValueError("Invalid PGN input")
    board = game.board()
    for move in game.mainline_moves():
        board.push(move)
    return board


class CoTDataGenerator:
    def __init__(
        self,
        sampling_policy: Callable[[chess.Board, List[chess.Move]], Optional[chess.Move]],
        evaluation_policy: Callable[[TreeNode], chess.Move],
        depth_limit: int,
        move_budget: int,
    ):
        self.sampling_policy = sampling_policy
        self.evaluation_policy = evaluation_policy
        self.depth_limit = depth_limit
        self.move_budget = move_budget

    def compute_max_traces(self) -> int:
        if self.depth_limit <= 0:
            return 0
        return self.move_budget // self.depth_limit

    def generate(
        self,
        root_pgn: str,
    ) -> Tuple[TreeNode, List[List[TreeNode]], chess.Move, Optional[float], Optional[Dict]]:
        """
        Returns:
            root: tree root node
            trajectories: list of trajectories, each is a list of TreeNode from root→leaf
            target_move: chosen best move at root from evaluation_policy
            best_val: evaluation value of the best move (if available)
            move_values: dict of move -> value for all evaluated moves (if available)
        """
        root_board = board_from_pgn(root_pgn)
        root = TreeNode(board=root_board)
        trajectories: List[List[TreeNode]] = []

        max_traces = self.compute_max_traces()
        moves_used = 0

        for _ in range(max_traces):
            if moves_used >= self.move_budget:
                break

            traj, moves_used = self._rollout_trajectory(root, moves_used)
            if len(traj) <= 1:  # only root, nothing expanded
                break
            trajectories.append(traj)

        # Evaluation policy can return either just a move, or (move, val, move_values)
        result = self.evaluation_policy(root)
        if isinstance(result, tuple):
            target_move, best_val, move_values = result
        else:
            target_move = result
            best_val = None
            move_values = None
        
        return root, trajectories, target_move, best_val, move_values

    def _rollout_trajectory(
        self,
        root: TreeNode,
        moves_used: int,
    ) -> Tuple[List[TreeNode], int]:
        """
        From root, sample moves with sampling_policy, build/update tree,
        and return the list of nodes along this trajectory.
        """
        board = root.board.copy()
        node = root
        trajectory_nodes = [node]

        for _ in range(self.depth_limit):
            if moves_used >= self.move_budget:
                break
            if board.is_game_over():
                break

            legal_moves = list(board.legal_moves)
            if not legal_moves:
                break

            move = self.sampling_policy(board, legal_moves)
            if move is None:
                break
            if move not in legal_moves:
                break

            board.push(move)
            moves_used += 1

            if move in node.children:
                node = node.children[move]
            else:
                child = TreeNode(board=board.copy(), move=move, parent=node)
                node.children[move] = child
                node = child

            trajectory_nodes.append(node)

        return trajectory_nodes, moves_used
