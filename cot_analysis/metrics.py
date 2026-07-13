"""Search-shape, search-behavior, and reasoning-length metrics.

Each function takes a parsed PrefixSearchTree (or a raw rollout row, for
the convenience wrapper at the bottom) and returns plain Python numbers.
"""
from __future__ import annotations

from tree import PrefixSearchTree, build_tree_from_row, extract_reasoning_text


# =============================================================================
# Search shape
# =============================================================================

def compute_shape_metrics(tree: PrefixSearchTree) -> dict[str, float]:
    """Structural descriptors of the prefix tree.

    Returns:
      candidate_count     # distinct root candidates (depth-1 children)
      num_nodes           # distinct prefixes excluding the root
      max_depth           # depth of the deepest node
      effective_branching # geometric-mean fanout: num_nodes ^ (1/max_depth)
      width_depth_index   # #leaves / max_depth  (proxy for wide-vs-deep)
    """
    root = tree.nodes[tree.root_id]
    num_nodes = len(tree.nodes) - 1
    candidate_count = len(root.children)

    if num_nodes == 0:
        return dict(candidate_count=0, num_nodes=0, max_depth=0,
                    effective_branching=0.0, width_depth_index=0.0)

    depths = [n.depth for n in tree.nodes.values() if not n.is_root()]
    max_depth = max(depths)
    leaf_ids = tree.leaves()
    effective_branching = num_nodes ** (1.0 / max_depth) if max_depth > 0 else 0.0
    width_depth_index = len(leaf_ids) / max(max_depth, 1)

    return dict(
        candidate_count=candidate_count,
        num_nodes=num_nodes,
        max_depth=max_depth,
        effective_branching=effective_branching,
        width_depth_index=width_depth_index,
    )


# =============================================================================
# Search behavior (over the temporal write-order)
# =============================================================================

def compute_revisit_rate(tree: PrefixSearchTree) -> float:
    """Fraction of node-writes that re-enter an already-visited node.

    Each move written by the model corresponds to one entry in
    `ordered_visits`. A revisit fires when the model writes its way back
    to a prefix it has already traversed (e.g., two candidate lines that
    share a prefix re-walk that prefix's nodes).

    Identity of a "node" is its full move-prefix from the root, so the
    same move-string at two different depths counts as different nodes
    (no revisit), while the same prefix written twice does count.
    """
    visits = tree.ordered_visits
    if not visits:
        return 0.0
    seen: set[int] = set()
    revisits = 0
    for nid in visits:
        if nid in seen:
            revisits += 1
        else:
            seen.add(nid)
    return revisits / len(visits)


def compute_dfs_consistency(tree: PrefixSearchTree) -> float:
    """Kendall's tau between the model's first-visit order and the canonical
    depth-first preorder of the same tree.

    Canonical DFS preorder sorts each node's children by their *first-visit
    time* in the model's writes. Then we ask: ignoring revisits, does the
    model's first-visit ordering match this preorder?

      tau = 1.0   model finishes each subtree before opening the next  (pure DFS)
      tau = 0.0   maximally interleaved across subtrees                (mixed)
      tau < 0     anti-DFS (rare in practice)

    Returns 1.0 when there are <2 distinct visits (no signal).
    """
    seen: set[int] = set()
    first_visit_order: list[int] = []
    for nid in tree.ordered_visits:
        if nid not in seen:
            first_visit_order.append(nid)
            seen.add(nid)
    if len(first_visit_order) < 2:
        return 1.0

    first_visit_time = {nid: i for i, nid in enumerate(first_visit_order)}

    # canonical DFS preorder: at each node, recurse into children sorted by
    # their own first-visit time. Records each child as it's entered.
    canonical: list[int] = []
    def dfs(nid: int) -> None:
        kids = sorted(tree.nodes[nid].children.values(),
                      key=lambda c: first_visit_time.get(c, float("inf")))
        for c in kids:
            canonical.append(c)
            dfs(c)
    dfs(tree.root_id)

    canonical_rank = {nid: i for i, nid in enumerate(canonical)}

    # Kendall tau: count concordant vs discordant pairs in the model order
    # relative to canonical order.
    n = len(first_visit_order)
    concordant = discordant = 0
    for i in range(n):
        ri = canonical_rank[first_visit_order[i]]
        for j in range(i + 1, n):
            rj = canonical_rank[first_visit_order[j]]
            if ri < rj:
                concordant += 1
            elif ri > rj:
                discordant += 1
    total = concordant + discordant
    return 1.0 if total == 0 else (concordant - discordant) / total


# =============================================================================
# Reasoning length
# =============================================================================

def compute_length_metrics(tree: PrefixSearchTree, reasoning_text: str) -> dict:
    """Two simple length signals of the reasoning region:
        reasoning_char_count: total characters between <T>...</T>
        n_ordered_visits    : total move tokens written into the tree
    """
    return dict(
        reasoning_char_count=len(reasoning_text),
        n_ordered_visits=len(tree.ordered_visits),
    )


# =============================================================================
# All-in-one convenience wrapper
# =============================================================================

def compute_all_metrics_for_row(row: dict) -> dict:
    """End-to-end: rollout row → all non-Stockfish metrics."""
    tree, _, reasoning = build_tree_from_row(row)
    out: dict = {}
    out.update(compute_shape_metrics(tree))
    out["revisit_rate"] = compute_revisit_rate(tree)
    out["dfs_consistency"] = compute_dfs_consistency(tree)
    out.update(compute_length_metrics(tree, reasoning))
    return out
