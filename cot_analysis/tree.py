"""Parse a model's reasoning text into a prefix tree of chess moves.

The model's reasoning region is delimited by <T>...</T>. Within it, individual
candidate lines are separated by <sep>. Each line is a sequence of move
tokens written in a custom long-algebraic form (e.g. "Pe2e4", "Qh5xf7+").

We build a *prefix tree*: lines are walked from the root, creating new nodes
for unseen prefixes and re-entering existing nodes for shared prefixes. The
order in which nodes are written is preserved in `ordered_visits`, which
later powers the search-behavior metrics (revisit, DFS-consistency).

Example
-------
Output text:
    <T> Pe2e4 Pe7e5 Ng1f3 <sep> Pe2e4 Pc7c5 <sep> Pd2d4 </T>
Parses to three lines, then to a prefix tree with five distinct nodes:

           root
          /    \\
        Pe2e4   Pd2d4
       /     \\
     Pe7e5    Pc7c5
       |
     Ng1f3

ordered_visits = [Pe2e4, Pe7e5, Ng1f3, Pe2e4, Pc7c5, Pd2d4]
(the second Pe2e4 is a revisit of the existing depth-1 node.)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TreeNode:
    node_id: int
    move: str | None              # None for the root
    parent: int | None
    depth: int
    children: dict[str, int] = field(default_factory=dict)  # move -> child node_id

    def is_root(self) -> bool:
        return self.parent is None


class PrefixSearchTree:
    """A trie of move-prefixes plus the temporal write-order."""

    def __init__(self) -> None:
        self.nodes: dict[int, TreeNode] = {
            0: TreeNode(node_id=0, move="<root>", parent=None, depth=0),
        }
        self.root_id: int = 0
        self._next_id: int = 1
        self.ordered_visits: list[int] = []

    def add_line(self, moves: list[str]) -> None:
        cur = self.nodes[self.root_id]
        for move in moves:
            if move not in cur.children:
                node_id = self._next_id
                self._next_id += 1
                self.nodes[node_id] = TreeNode(
                    node_id=node_id, move=move,
                    parent=cur.node_id, depth=cur.depth + 1,
                )
                cur.children[move] = node_id
            child_id = cur.children[move]
            self.ordered_visits.append(child_id)
            cur = self.nodes[child_id]

    def is_ancestor(self, ancestor_id: int, node_id: int) -> bool:
        cur = self.nodes[node_id]
        while cur.parent is not None:
            if cur.parent == ancestor_id:
                return True
            cur = self.nodes[cur.parent]
        return False

    def leaves(self) -> list[int]:
        return [n.node_id for n in self.nodes.values()
                if n.node_id != self.root_id and not n.children]


# -----------------------------------------------------------------------------
# Text extraction
# -----------------------------------------------------------------------------

def extract_reasoning_text(input_text: str, output_text: str) -> str:
    """Return the content between <T>...</T> in the model output.

    If the output already includes </T>, we take everything before it.
    Otherwise, if <T> appears, we take everything after it.
    """
    if "</T>" in output_text:
        return output_text.split("</T>")[0].strip()
    if "<T>" in output_text:
        return output_text.split("<T>", 1)[1].strip()
    return ""


def parse_candidate_lines(reasoning_text: str) -> list[list[str]]:
    """Split reasoning into candidate lines on '<sep>' and tokenize each line.

    Special markup tokens are filtered out.
    """
    if not reasoning_text:
        return []
    lines: list[list[str]] = []
    for raw in reasoning_text.split("<sep>"):
        moves = [m for m in raw.strip().split()
                 if m not in {"<T>", "</T>", "<sep>", "<call_env>"}]
        if moves:
            lines.append(moves)
    return lines


def build_tree_from_row(row: dict) -> tuple[PrefixSearchTree, list[list[str]], str]:
    """Convenience: input/output dict → (tree, parsed lines, raw reasoning)."""
    reasoning = extract_reasoning_text(row.get("input", ""), row.get("output", ""))
    lines = parse_candidate_lines(reasoning)
    tree = PrefixSearchTree()
    for line in lines:
        tree.add_line(line)
    return tree, lines, reasoning
