"""AI-B: standalone generalized Gomoku, Monte Carlo Tree Search with UCT.

Only Python's standard library is used. All tree values use the ROOT player's
view; UCT negates exploitation only when the selecting player is the opponent.
Rollouts use runtime K-window tactics and random choices, never a response book.
"""

import math
import heapq
import random
import time

NAME = "Monte Carlo"
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class _SearchTimeout(Exception):
    pass


class _Node:
    __slots__ = ("move", "player", "children", "untried", "visits",
                 "value_sum", "winner")

    def __init__(self, move, player):
        self.move = move
        self.player = player
        self.children = []
        self.untried = None
        self.visits = 0
        self.value_sum = 0.0
        self.winner = None


class _Position:
    """One simulation's mutable state; immutable geometry belongs to its AI.

    Window occupancy, threat reference counts, move gains and neighboring-cell
    counts update through windows/cells touched by a move. A simulation clones
    these caches, so even an interrupted rollout cannot change the root board.
    """

    def __init__(self, ai, cells):
        self.ai = ai
        self.cells = list(cells)
        size = len(cells)
        self.empty = self.cells.count(0)
        self.neighbor_counts = [0] * size
        for cell, stone in enumerate(cells):
            if cell % 32 == 0:
                ai._check_time()
            if stone:
                for neighbor in ai.neighbors[cell]:
                    self.neighbor_counts[neighbor] += 1
        self.frontier = {i for i, v in enumerate(cells)
                         if not v and self.neighbor_counts[i]}
        self.black = []
        self.white = []
        self.gaps = []
        self.gains = [[0] * size, [0] * size]
        self.ranks = [ai.tiebreak.copy(), ai.tiebreak.copy()]
        self.threat_refs = [[0] * size, [0] * size]
        self.threats = [set(), set()]
        # Sparse links: placing x creates a next-turn winning cell y. Distinct
        # targets, not window counts, determine whether x creates a fork.
        self.fork_links = [{}, {}]
        self.forks = [set(), set()]
        self.score = 0
        self.live_windows = 0
        for index, window in enumerate(ai.windows):
            if index % 16 == 0:
                ai._check_time()
            a = sum(cells[i] == 1 for i in window)
            b = sum(cells[i] == 2 for i in window)
            mask = sum(1 << i for i in window if not cells[i])
            self.black.append(a)
            self.white.append(b)
            self.live_windows += int(not a or not b)
            self.gaps.append(mask)
            self.score += self._window_score(a, b)
            attack, defense = self._window_gains(a, b)
            for cell in window:
                if not cells[cell]:
                    self.gains[0][cell] += attack
                    self.gains[1][cell] += defense
                    self.ranks[0][cell] += (16*attack + 10*defense) * ai.rank_scale
                    self.ranks[1][cell] += (16*defense + 10*attack) * ai.rank_scale
            self._adjust_threat(index, 1)
            self._adjust_fork(index, 1)

    def clone(self):
        other = _Position.__new__(_Position)
        other.ai = self.ai
        other.cells = self.cells.copy()
        other.empty = self.empty
        other.neighbor_counts = self.neighbor_counts.copy()
        other.frontier = self.frontier.copy()
        other.black = self.black.copy()
        other.white = self.white.copy()
        other.gaps = self.gaps.copy()
        other.gains = [side.copy() for side in self.gains]
        other.ranks = [side.copy() for side in self.ranks]
        other.threat_refs = [side.copy() for side in self.threat_refs]
        other.threats = [side.copy() for side in self.threats]
        other.fork_links = [{cell: links.copy() for cell, links in side.items()}
                           for side in self.fork_links]
        other.forks = [side.copy() for side in self.forks]
        other.score = self.score
        other.live_windows = self.live_windows
        return other

    def _window_score(self, a, b):
        return self.ai.window_values[a][b][0]

    def _window_gains(self, a, b):
        return self.ai.window_values[a][b][1:]

    def _adjust_fork(self, index, change):
        a, b = self.black[index], self.white[index]
        if not b and a == self.ai.win_length - 2:
            side = 0
        elif not a and b == self.ai.win_length - 2:
            side = 1
        else:
            return
        gaps = self.gaps[index]
        low = gaps & -gaps
        x, y = low.bit_length() - 1, (gaps ^ low).bit_length() - 1
        mapping, forks = self.fork_links[side], self.forks[side]
        for cell, target in ((x, y), (y, x)):
            links = mapping.setdefault(cell, {})
            value = links.get(target, 0) + change
            if value:
                links[target] = value
            else:
                del links[target]
            if len(links) >= 2:
                forks.add(cell)
            else:
                forks.discard(cell)
            if not links:
                del mapping[cell]

    def _adjust_threat(self, index, change):
        a, b = self.black[index], self.white[index]
        k = self.ai.win_length
        if b == 0 and a == k - 1:
            side = 0
        elif a == 0 and b == k - 1:
            side = 1
        else:
            return
        cell = self.gaps[index].bit_length() - 1
        self.threat_refs[side][cell] += change
        if self.threat_refs[side][cell]:
            self.threats[side].add(cell)
        else:
            self.threats[side].discard(cell)

    def play(self, cell, player):
        """Apply to this local simulation; never checks time halfway through."""
        self.cells[cell] = player
        self.empty -= 1
        self.frontier.discard(cell)
        for neighbor in self.ai.neighbors[cell]:
            self.neighbor_counts[neighbor] += 1
            if not self.cells[neighbor]:
                self.frontier.add(neighbor)
        bit = 1 << cell
        gains1, gains2 = self.gains
        ranks1, ranks2 = self.ranks
        scale = self.ai.rank_scale
        values = self.ai.window_values
        near = self.ai.win_length - 2
        for index in self.ai.cell_windows[cell]:
            a, b = self.black[index], self.white[index]
            was_live = not a or not b
            old_score, old1, old2 = values[a][b]
            if (not b and a >= near) or (not a and b >= near):
                self._adjust_threat(index, -1)
                self._adjust_fork(index, -1)
            if player == 1:
                self.black[index] += 1
            else:
                self.white[index] += 1
            self.gaps[index] ^= bit
            a, b = self.black[index], self.white[index]
            if was_live and a and b:
                self.live_windows -= 1
            new_score, new1, new2 = values[a][b]
            self.score += new_score - old_score
            if (not b and a >= near) or (not a and b >= near):
                self._adjust_threat(index, 1)
                self._adjust_fork(index, 1)
            d1, d2 = new1 - old1, new2 - old2
            mask = self.gaps[index]
            if d1 or d2:
                dr1, dr2 = (16*d1+10*d2)*scale, (16*d2+10*d1)*scale
                while mask:
                    low = mask & -mask
                    target = low.bit_length() - 1
                    gains1[target] += d1
                    gains2[target] += d2
                    ranks1[target] += dr1
                    ranks2[target] += dr2
                    mask ^= low
            gains1[cell] -= old1
            gains2[cell] -= old2
        ranks1[cell] = ranks2[cell] = self.ai.tiebreak[cell]

    def won(self, cell, player):
        counts = self.black if player == 1 else self.white
        return any(counts[index] == self.ai.win_length
                   for index in self.ai.cell_windows[cell])

    def priority(self, cell, player):
        # Prefer both making a line and preventing one. These are computed from
        # the current board and K, rather than stored responses for an opponent.
        return (self.ranks[player - 1][cell]
                + (self.ai.fork_bonus if cell in self.forks[2 - player] else 0))

    def candidates(self, player, width):
        self.ai._check_time()
        own, enemy = self.threats[player - 1], self.threats[2 - player]
        if own:
            return sorted(own)
        if enemy:
            return sorted(enemy)
        if self.forks[player - 1]:
            return sorted(self.forks[player - 1])
        available = self.frontier
        if not available:
            available = [i for i, value in enumerate(self.cells) if not value]
        # Width adapts to remaining cells; urgent attacks/defenses survive the
        # cap. A single move may answer several threats, so do not drop all
        # other defenses just because the opponent has a possible fork.
        key = lambda cell: self.priority(cell, player)
        moves = heapq.nlargest(width, available, key=key)
        tactical = set(self.fork_links[player - 1]) | self.forks[2 - player]
        for cell in sorted(tactical, key=key, reverse=True):
            if not self.cells[cell] and cell not in moves:
                moves.append(cell)
        self.ai._check_time()
        return moves


class GomokuAI:
    def __init__(self, player_id: int, board_size: int, win_length: int):
        if type(player_id) is not int or player_id not in (1, 2):
            raise ValueError("player_id must be 1 or 2")
        if type(board_size) is not int or type(win_length) is not int:
            raise ValueError("N and K must be integers")
        if not 3 <= win_length <= board_size:
            raise ValueError("require 3 <= K <= N")
        self.player_id = player_id
        self.board_size = board_size
        self.win_length = win_length
        self.rng = random.Random()
        self.exploration = math.sqrt(2.0)
        self.root_width = min(18, max(10, 2 * win_length + 2))
        self.branch_width = min(12, max(6, win_length + 3))
        self.rollout_width = 6
        self.rollout_randomness = 0.15
        self.rollout_limit = board_size * board_size  # Complete games by default.
        self.max_iterations = None  # Optional reproducible simulation cap.
        self.windows = []
        self.cell_windows = [[] for _ in range(board_size * board_size)]
        self.neighbors = []
        self.centrality = []
        for r in range(board_size):
            for c in range(board_size):
                self.centrality.append(-abs(2 * r - board_size + 1)
                                       - abs(2 * c - board_size + 1))
                self.neighbors.append(tuple(
                    rr * board_size + cc
                    for rr in range(max(0, r - 2), min(board_size, r + 3))
                    for cc in range(max(0, c - 2), min(board_size, c + 3))
                    if (rr, cc) != (r, c)))
                for dr, dc in _DIRECTIONS:
                    er, ec = r + (win_length - 1) * dr, c + (win_length - 1) * dc
                    if 0 <= er < board_size and 0 <= ec < board_size:
                        window = tuple((r + j * dr) * board_size + c + j * dc
                                       for j in range(win_length))
                        index = len(self.windows)
                        self.windows.append(window)
                        for cell in window:
                            self.cell_windows[cell].append(index)
        # Exact integer ratios of runtime K-window weights; a line is most
        # valuable near completion for any legal K, including K > 5.
        self.weights = [0] + [8 ** (j - 1) for j in range(1, win_length + 1)]
        size = board_size * board_size
        self.rank_scale = 4 * board_size * size + 1
        self.tiebreak = [central * size - cell for cell, central in enumerate(self.centrality)]
        self.fork_bonus = self.weights[-1] * self.rank_scale * 64 * size
        # Arithmetic cache of the heuristic formula, built from K at runtime;
        # this contains no positions or precomputed move responses.
        w, k = self.weights, win_length
        self.window_values = [[(0 if a and b else w[a] - w[b],
                                w[a + 1] - w[a] if not b and a < k else 0,
                                w[b + 1] - w[b] if not a and b < k else 0)
                               for b in range(k + 1)] for a in range(k + 1)]
        self.last_stats = {}
        self.root = None

    def get_move(self, board, last_opponent_move, time_limit):
        """Protocol v1: read-only board, legal (int, int), wall-clock deadline."""
        started = time.perf_counter()
        n = self.board_size
        if len(board) != n or any(len(row) != n for row in board):
            raise ValueError("board shape does not match N")
        if any(type(value) is not int or value not in (0, 1, 2)
               for row in board for value in row):
            raise ValueError("board cells must be integers 0, 1, or 2")
        if (not isinstance(time_limit, (int, float))
                or not math.isfinite(time_limit) or time_limit <= 0):
            raise ValueError("time_limit must be finite and positive")
        cells = [value for row in board for value in row]
        empty = [i for i, value in enumerate(cells) if not value]
        if not empty:
            raise ValueError("no legal move on a full board")
        best = max(empty, key=lambda i: (self.centrality[i], -i))
        self.root = None
        self.last_stats = {"algorithm": "mcts_uct", "reason": "fallback",
                           "iterations": 0, "simulations": 0, "rollouts": 0,
                           "tree_nodes": 0, "nodes": 0, "tree_depth": 0,
                           "depth": 0, "cutoffs": 0, "rollout_moves": 0,
                           "rollout_truncations": 0, "rollout_terminal": 0}
        reserve = min(0.08, time_limit * 0.2)
        self.last_stats.update(reward_perspective="root_player", root_player=self.player_id,
                               tactical_rollouts=0, root_candidates=0,
                               proven_draw_rollouts=0, proven_draw_nodes=0,
                               time_reserve=reserve,
                               usable_time=time_limit - reserve)
        if len(empty) == n * n or time_limit <= 0.01:
            self.last_stats["reason"] = "opening" if len(empty) == n * n else "tiny_budget"
            return self._finish(best, started)
        self.deadline = started + time_limit - reserve
        try:
            position = _Position(self, cells)
            own, enemy = position.threats[self.player_id - 1], position.threats[2 - self.player_id]
            if own:
                self.last_stats["reason"] = "immediate_win"
                return self._finish(min(own), started)
            if len(enemy) == 1:
                self.last_stats["reason"] = "forced_block"
                return self._finish(min(enemy), started)
            if len(enemy) > 1:
                self.last_stats["reason"] = "multiple_threats"
                return self._finish(max(enemy, key=lambda c: position.priority(c, self.player_id)), started)
            if position.forks[self.player_id - 1]:
                self.last_stats["reason"] = "forced_win"
                return self._finish(max(position.forks[self.player_id - 1],
                                        key=lambda c: position.priority(c, self.player_id)), started)
            if not position.live_windows:
                self.last_stats.update(reason="proven_draw", root_value=0.0)
                return self._finish(best, started)
            moves = position.candidates(self.player_id, self.root_width)
            best = moves[0]
            root = self.root = _Node(None, self.player_id)
            root.untried = list(reversed(moves))
            self.last_stats.update(reason="mcts", tree_nodes=1, nodes=1)
            self.last_stats["root_candidates"] = len(moves)
            search_started = time.perf_counter()
            while self.max_iterations is None or root.visits < self.max_iterations:
                self._check_time()
                state = position.clone()
                node = root
                path = [root]
                # Selection follows UCT until an unexpanded action or terminal.
                while node.winner is None:
                    self._check_time()
                    if node.untried is None:
                        width = self.root_width if node is root else self.branch_width
                        node.untried = list(reversed(state.candidates(node.player, width)))
                    if node.untried:
                        # Expansion: one previously unvisited action per pass.
                        move = node.untried.pop()
                        moving_player = node.player
                        state.play(move, moving_player)
                        child = _Node(move, 3 - moving_player)
                        if state.won(move, moving_player):
                            child.winner = moving_player
                        elif not state.empty or not state.live_windows:
                            child.winner = 0
                            if state.empty:
                                self.last_stats["proven_draw_nodes"] += 1
                        node.children.append(child)
                        node = child
                        path.append(node)
                        self.last_stats["tree_nodes"] += 1
                        break
                    if not node.children:
                        node.winner = 0
                        break
                    node = self._select_child(node)
                    state.play(node.move, 3 - node.player)
                    path.append(node)
                depth = len(path) - 1
                self.last_stats["tree_depth"] = max(self.last_stats["tree_depth"], depth)
                self._check_time()
                if node.winner is None:
                    value = self._rollout(state, node.player)
                    self.last_stats["rollouts"] += 1
                else:
                    value = self._terminal_value(node.winner)
                # Every node stores the SAME root-relative result.
                self._backpropagate(path, value)
                self.last_stats["iterations"] = root.visits
                self.last_stats["simulations"] = root.visits
        except _SearchTimeout:
            self.last_stats["interrupted"] = True
        if self.root is not None:
            visited = [child for child in self.root.children if child.visits]
            if visited:
                choice = max(visited, key=lambda child: (child.visits, child.value_sum / child.visits))
                best = choice.move
                self.last_stats["root_value"] = choice.value_sum / choice.visits
                self.last_stats["root_children"] = [
                    {"move": [int(child.move // n), int(child.move % n)],
                     "visits": child.visits,
                     "root_value": child.value_sum / child.visits if child.visits else None}
                    for child in self.root.children]
        if self.root is not None:
            self.last_stats["search_elapsed"] = time.perf_counter() - search_started
        return self._finish(best, started)

    def _finish(self, move, started):
        self.last_stats["nodes"] = self.last_stats["tree_nodes"]
        self.last_stats["depth"] = self.last_stats["tree_depth"]
        self.last_stats["elapsed"] = time.perf_counter() - started
        elapsed = max(self.last_stats["elapsed"], 1e-12)
        self.last_stats["simulations_per_second"] = self.last_stats["simulations"] / elapsed
        self.last_stats["rollouts_per_second"] = self.last_stats["rollouts"] / elapsed
        return (int(move // self.board_size), int(move % self.board_size))

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise _SearchTimeout

    def _terminal_value(self, winner):
        return 0.0 if winner == 0 else 1.0 if winner == self.player_id else -1.0

    def _uct_score(self, parent, child):
        if not child.visits:
            return float("inf")
        sign = 1 if parent.player == self.player_id else -1
        exploitation = sign * child.value_sum / child.visits
        exploration = self.exploration * math.sqrt(math.log(max(1, parent.visits)) / child.visits)
        return exploitation + exploration

    def _select_child(self, parent):
        return max(parent.children, key=lambda child: (self._uct_score(parent, child), self.rng.random()))

    @staticmethod
    def _backpropagate(path, root_value):
        for node in reversed(path):
            node.visits += 1
            node.value_sum += root_value

    def _rollout(self, state, player):
        for _ in range(min(self.rollout_limit, state.empty)):
            self._check_time()
            # Exact draw proof, not a heuristic rollout horizon: every possible
            # K-line is permanently blocked by stones of both colors. The
            # counter is maintained only in windows touched by the last move.
            if not state.live_windows:
                self.last_stats["rollout_terminal"] += 1
                self.last_stats["proven_draw_rollouts"] += 1
                return 0.0
            own, enemy = state.threats[player - 1], state.threats[2 - player]
            if own:
                self.last_stats["rollout_terminal"] += 1
                return self._terminal_value(player)
            if len(enemy) > 1:
                # No single move can occupy both distinct immediate win cells.
                self.last_stats["rollout_terminal"] += 1
                return self._terminal_value(3 - player)
            if enemy:
                move = next(iter(enemy))
            elif state.forks[player - 1]:
                # Two distinct next-turn wins cannot both be blocked. This is
                # a local tactical proof, not a recursive minimax search.
                self.last_stats["rollout_terminal"] += 1
                self.last_stats["tactical_rollouts"] += 1
                return self._terminal_value(player)
            else:
                sample = self.rng.random()
                # Scan only the incremental frontier. Most rollouts need just
                # its maximum, not a complete sort of all empty cells.
                available = state.frontier
                if not available:
                    available = [c for c, v in enumerate(state.cells) if not v]
                # Defense cells share the same high fork bonus; the cached
                # numeric ranking avoids repeated heuristic arithmetic/calls.
                defensive = state.forks[2 - player]
                if defensive:
                    available = defensive
                key = state.ranks[player - 1].__getitem__
                if sample >= self.rollout_randomness:
                    move = max(available, key=key)
                else:
                    moves = heapq.nlargest(self.rollout_width, available, key=key)
                    move = self.rng.choices(moves, weights=range(len(moves), 0, -1), k=1)[0]
            state.play(move, player)
            self.last_stats["rollout_moves"] += 1
            if state.won(move, player):
                self.last_stats["rollout_terminal"] += 1
                return self._terminal_value(player)
            if not state.empty:
                self.last_stats["rollout_terminal"] += 1
                return 0.0
            player = 3 - player
        if not state.empty:
            self.last_stats["rollout_terminal"] += 1
            return 0.0
        # Used only if a caller explicitly lowers rollout_limit. Default rollout
        # length can fill the whole board. The cutoff estimate stays in [-.5,.5]
        # and is counted separately from terminal game outcomes.
        self.last_stats["rollout_truncations"] += 1
        scale = max(1, self.weights[self.win_length - 1] * 4)
        sign = 1 if self.player_id == 1 else -1
        return sign * 0.5 * math.tanh(state.score / scale)
