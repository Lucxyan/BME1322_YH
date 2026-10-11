"""Self-contained Gomoku AI Protocol v1 implementation (standard library only).

WindowSearch v2 uses incremental K-window evaluation and threat information,
iterative deepening Minimax/Alpha-Beta, and a collision-free transposition table.
The K-window model applies to every 3 <= K <= N, rather than only five-in-a-row.
"""

import math
import time

NAME = "Depth Search"
_DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
_MATE = 10 ** 12
_EXACT, _LOWER, _UPPER = 0, 1, 2


class _SearchTimeout(Exception):
    pass


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
        self.max_depth = 10
        self.root_width = 24
        self.branch_width = 12
        self.quiescence_depth = 6
        self.table_limit = 100000
        self.windows = []
        self.cell_windows = [[] for _ in range(board_size * board_size)]
        self.neighbors = []
        self.centrality = []
        for r in range(board_size):
            for c in range(board_size):
                self.neighbors.append(tuple(
                    rr * board_size + cc
                    for rr in range(max(0, r - 2), min(board_size, r + 3))
                    for cc in range(max(0, c - 2), min(board_size, c + 3))
                    if (rr, cc) != (r, c)))
                self.centrality.append(-abs(2 * r - board_size + 1)
                                       - abs(2 * c - board_size + 1))
                for dr, dc in _DIRECTIONS:
                    er, ec = r + (win_length - 1) * dr, c + (win_length - 1) * dc
                    if 0 <= er < board_size and 0 <= ec < board_size:
                        cells = tuple((r + j * dr) * board_size + c + j * dc
                                      for j in range(win_length))
                        w = len(self.windows)
                        self.windows.append(cells)
                        for cell in cells:
                            self.cell_windows[cell].append(w)
        self.weights = [0] + [100000 * 8 ** (j - win_length + 1)
                              for j in range(1, win_length)] + [_MATE]
        # All heuristic weights share this unit. Integer caches avoid losing
        # tiny long-K contributions when a temporary mate score is undone.
        self.unit_scale = 100000 / 8 ** (win_length - 2)
        self.unit_weights = [0] + [8 ** (j - 1) for j in range(1, win_length)] + [
            10 ** 7 * 8 ** (win_length - 2)]
        self.last_stats = {}

    def get_move(self, board, last_opponent_move, time_limit):
        """Return two Python ints, without modifying the supplied board.

        The board is authoritative; the advisory last move need not be supplied.
        Search state and its table are rebuilt between protocol calls. A full
        board has no legal move and lies outside the protocol's move domain.
        """
        started = time.perf_counter()
        n = self.board_size
        if len(board) != n or any(len(row) != n for row in board):
            raise ValueError("board shape does not match N")
        if any(type(x) is not int or x not in (0, 1, 2) for row in board for x in row):
            raise ValueError("board cells must be integers 0, 1, or 2")
        if not isinstance(time_limit, (int, float)) or not math.isfinite(time_limit) or time_limit <= 0:
            raise ValueError("time_limit must be finite and positive")
        self.cells = [x for row in board for x in row]
        empties = [i for i, x in enumerate(self.cells) if not x]
        if not empties:
            raise ValueError("no legal move on a full board")
        fallback = max(empties, key=lambda i: (self.centrality[i], -i))
        self.nodes = self.cutoffs = self.tt_hits = self.extensions = 0
        self.last_stats = {"depth": 0, "nodes": 0, "cutoffs": 0, "reason": "fallback"}
        if len(empties) == n * n or time_limit <= 0.02:
            self.last_stats["reason"] = "opening" if len(empties) == n * n else "tiny_budget"
            return self._finish(fallback, started)
        reserve = min(0.08, time_limit * 0.2)
        self.deadline = started + time_limit - reserve
        best = fallback
        try:
            self._initialize_position()
            own, enemy = self._threats(self.player_id)
            if own:
                self.last_stats["reason"] = "immediate_win"
                return self._finish(min(own), started)
            if enemy:
                best = min(enemy)
                if len(enemy) == 1:
                    self.last_stats["reason"] = "forced_block"
                    return self._finish(best, started)
            elif self.forks[self.player_id - 1]:
                # Two distinct next-turn winning cells cannot both be blocked.
                best = max(self.forks[self.player_id - 1],
                           key=lambda cell: self._priority(cell, self.player_id))
                self.last_stats.update(reason="forced_win", score=_MATE - 3)
                return self._finish(best, started)
            # Stones cannot be removed. If every K-window contains both
            # colors, no completion can win, regardless of remaining spaces.
            if all(a and b for a, b in self.counts):
                self.last_stats.update(reason="proven_draw", score=0)
                return self._finish(best, started)
            ordered = self._candidates(self.player_id, self.root_width)
            if ordered:
                best = ordered[0]
            self.last_stats["reason"] = "search"
            for depth in range(1, min(self.max_depth, self.empty_count) + 1):
                self._check_time()
                score, move = self._search_root(depth, best)
                best = move
                self.last_stats["depth"] = depth
                self.last_stats["score"] = score
                if abs(score) > _MATE / 2:
                    break
        except _SearchTimeout:
            self.last_stats["interrupted"] = True
        return self._finish(best, started)

    def _initialize_position(self):
        size = self.board_size * self.board_size
        self.empty_count = self.cells.count(0)
        self.black_mask = self.white_mask = self.empty_mask = 0
        self.neighbor_counts = [0] * size
        self.frontier = set()
        for cell, player in enumerate(self.cells):
            if cell % 32 == 0:
                self._check_time()
            bit = 1 << cell
            if player == 1:
                self.black_mask |= bit
            elif player == 2:
                self.white_mask |= bit
            else:
                self.empty_mask |= bit
            if player:
                for neighbor in self.neighbors[cell]:
                    self.neighbor_counts[neighbor] += 1
        self.frontier = {i for i in range(size) if not self.cells[i] and self.neighbor_counts[i]}
        self.counts = []
        self.window_empty_masks = []
        self.score = 0
        self.raw_score = 0
        self.move_values = [[0] * size, [0] * size]
        self.threat_refs = [[0] * size, [0] * size]
        self.threat_cells = [set(), set()]
        # Each edge a -> b means playing a makes b a winning move. Ref counts
        # deduplicate overlapping windows: a real fork needs distinct targets.
        self.fork_links = [[{} for _ in range(size)], [{} for _ in range(size)]]
        self.near_wins = [set(), set()]
        self.forks = [set(), set()]
        self.table = {}
        self.killers = {}
        self.history = [[0] * size, [0] * size]
        for w, window in enumerate(self.windows):
            if w % 16 == 0:
                self._check_time()
            a = b = mask = 0
            for cell in window:
                player = self.cells[cell]
                if player == 1:
                    a += 1
                elif player == 2:
                    b += 1
                else:
                    mask |= 1 << cell
            self.counts.append([a, b])
            self.window_empty_masks.append(mask)
            self.raw_score += self._window_raw_score(a, b)
            p1, p2 = self._window_move_values(a, b)
            for cell in window:
                if not self.cells[cell]:
                    self.move_values[0][cell] += p1
                    self.move_values[1][cell] += p2
            self._patterns(w, 1)
        self.score = self.raw_score * self.unit_scale

    def _finish(self, cell, started):
        self.last_stats.update(nodes=self.nodes, cutoffs=self.cutoffs,
                               tt_hits=self.tt_hits, extensions=self.extensions,
                               elapsed=time.perf_counter() - started)
        return (int(cell // self.board_size), int(cell % self.board_size))

    def _check_time(self):
        if time.perf_counter() >= self.deadline:
            raise _SearchTimeout

    def _window_score(self, a, b):
        if a and b:
            return 0
        return self.weights[a] - self.weights[b]

    def _window_move_values(self, a, b):
        return ((self.unit_weights[a + 1] - self.unit_weights[a])
                if not b and a < self.win_length else 0,
                (self.unit_weights[b + 1] - self.unit_weights[b])
                if not a and b < self.win_length else 0)

    def _window_raw_score(self, a, b):
        return 0 if a and b else self.unit_weights[a] - self.unit_weights[b]

    def _update_link(self, side, cell, target, delta):
        targets = self.fork_links[side][cell]
        refs = targets.get(target, 0) + delta
        if refs:
            targets[target] = refs
        else:
            del targets[target]
        if targets:
            self.near_wins[side].add(cell)
        else:
            self.near_wins[side].discard(cell)
        if len(targets) >= 2:
            self.forks[side].add(cell)
        else:
            self.forks[side].discard(cell)

    def _patterns(self, w, delta):
        a, b = self.counts[w]
        k = self.win_length
        if a and b:
            return
        side = 0 if a else 1
        stones = a if a else b
        mask = self.window_empty_masks[w]
        if stones == k - 1:
            cell = mask.bit_length() - 1
            self.threat_refs[side][cell] += delta
            if self.threat_refs[side][cell]:
                self.threat_cells[side].add(cell)
            else:
                self.threat_cells[side].discard(cell)
        elif stones == k - 2:
            first_bit = mask & -mask
            first = first_bit.bit_length() - 1
            second = (mask ^ first_bit).bit_length() - 1
            self._update_link(side, first, second, delta)
            self._update_link(side, second, first, delta)

    def _change(self, cell, player, delta):
        # Updates are atomic with respect to our time checks. Every search caller
        # undoes in finally, restoring all caches even during timeout unwinding.
        bit = 1 << cell
        self.cells[cell] = player if delta > 0 else 0
        self.empty_count -= delta
        self.empty_mask ^= bit
        if player == 1:
            self.black_mask ^= bit
        else:
            self.white_mask ^= bit
        self.frontier.discard(cell)
        for neighbor in self.neighbors[cell]:
            self.neighbor_counts[neighbor] += delta
            if not self.cells[neighbor] and self.neighbor_counts[neighbor]:
                self.frontier.add(neighbor)
            else:
                self.frontier.discard(neighbor)
        if delta < 0 and self.neighbor_counts[cell]:
            self.frontier.add(cell)
        for w in self.cell_windows[cell]:
            a, b = self.counts[w]
            old_score = self._window_raw_score(a, b)
            old1, old2 = self._window_move_values(a, b)
            old_mask = self.window_empty_masks[w]
            self._patterns(w, -1)
            self.counts[w][player - 1] += delta
            self.window_empty_masks[w] ^= bit
            a, b = self.counts[w]
            self.raw_score += self._window_raw_score(a, b) - old_score
            new1, new2 = self._window_move_values(a, b)
            self._patterns(w, 1)
            diff1, diff2 = new1 - old1, new2 - old2
            mask = old_mask & self.window_empty_masks[w]
            if diff1 or diff2:
                while mask:
                    target_bit = mask & -mask
                    target = target_bit.bit_length() - 1
                    self.move_values[0][target] += diff1
                    self.move_values[1][target] += diff2
                    mask ^= target_bit
            if delta > 0:
                self.move_values[0][cell] -= old1
                self.move_values[1][cell] -= old2
            else:
                self.move_values[0][cell] += new1
                self.move_values[1][cell] += new2
        self.score = self.raw_score * self.unit_scale

    def _threats(self, player):
        """All-board immediate wins, independent of distance or width limits."""
        self._check_time()
        return set(self.threat_cells[player - 1]), set(self.threat_cells[2 - player])

    def _priority(self, cell, player):
        attack = self.move_values[player - 1][cell] * self.unit_scale
        defense = self.move_values[2 - player][cell] * self.unit_scale
        forcing = len(self.fork_links[player - 1][cell]) * 200000
        if cell in self.forks[2 - player]:
            forcing += 10 ** 8
        if cell in self.forks[player - 1]:
            forcing += 2 * 10 ** 8
        return (attack + defense + forcing, attack, self.centrality[cell], -cell)

    def _tactical_defenses(self, player):
        """Include every way to disrupt an enemy fork or force a reply first."""
        enemy_side = 2 - player
        if not self.forks[enemy_side]:
            return set()
        responses = set(self.forks[enemy_side])
        for cell in self.forks[enemy_side]:
            responses.update(self.fork_links[enemy_side][cell])
        responses.update(self.near_wins[player - 1])
        return responses

    def _candidates(self, player, width):
        own, enemy = self._threats(player)
        if own:
            return sorted(own)
        if enemy:
            return sorted(enemy)
        nearby = self.frontier
        if not nearby:
            nearby = {i for i, value in enumerate(self.cells) if not value}
        ranked = []
        for i in nearby:
            self._check_time()
            ranked.append((self._priority(i, player), i))
        ranked.sort(reverse=True)
        moves = [i for _, i in ranked[:width]]
        # Quiet moves are selective; proven threats and their replies are not.
        tactical = self._tactical_defenses(player) | self.forks[player - 1]
        if tactical:
            selected = set(moves)
            moves.extend(sorted(tactical - selected,
                                key=lambda i: self._priority(i, player), reverse=True))
        return moves

    def _ordered_moves(self, player, width, ply, tt_move=None):
        moves = self._candidates(player, width)
        killers = self.killers.get(ply, ())
        history = self.history[player - 1]
        moves.sort(key=lambda i: (i == tt_move,
                                 (2 - killers.index(i)) if i in killers else 0,
                                 history[i]), reverse=True)
        return moves

    def _won(self, cell, player):
        return any(self.counts[w][player - 1] == self.win_length
                   for w in self.cell_windows[cell])

    @staticmethod
    def _pack_mate(value, ply):
        if value > _MATE / 2:
            return value + ply
        if value < -_MATE / 2:
            return value - ply
        return value

    @staticmethod
    def _unpack_mate(value, ply):
        if value > _MATE / 2:
            return value - ply
        if value < -_MATE / 2:
            return value + ply
        return value

    def _store(self, key, depth, value, flag, move, ply):
        previous = self.table.get(key)
        if previous is not None and previous[0] > depth:
            return
        if previous is not None or len(self.table) < self.table_limit:
            self.table[key] = (depth, self._pack_mate(value, ply), flag, move)

    def _search_root(self, depth, previous_best):
        self._check_time()
        moves = self._ordered_moves(self.player_id, self.root_width, 0, previous_best)
        alpha, beta = -2 * _MATE, 2 * _MATE
        best_move = moves[0]
        for cell in moves:
            self._check_time()
            self._change(cell, self.player_id, 1)
            try:
                value = (_MATE - 1 if self._won(cell, self.player_id) else
                         -self._negamax(depth - 1, -beta, -alpha, 3 - self.player_id, 1))
            finally:
                self._change(cell, self.player_id, -1)
            if value > alpha:
                alpha, best_move = value, cell
            if alpha > _MATE / 2:
                break
        return alpha, best_move

    def _negamax(self, depth, alpha, beta, player, ply, extra=0):
        """Symmetric zero-sum Minimax, with Alpha-Beta and forcing extensions."""
        self._check_time()
        self.nodes += 1
        if not self.empty_count:
            return 0
        own, enemy = self._threats(player)
        if own:
            return _MATE - ply - 1
        if len(enemy) > 1:
            return -_MATE + ply + 2
        if not enemy and self.forks[player - 1]:
            return _MATE - ply - 3
        quiescent = depth <= 0
        if quiescent:
            # A unique immediate block must be followed until it is quiet. The
            # finite board bounds that chain. Fork-defense branching is bounded.
            if not enemy and (not self.forks[2 - player] or extra >= self.quiescence_depth):
                return self.score if player == 1 else -self.score
            self.extensions += 1
        key = (self.black_mask, self.white_mask, player)
        tt_move = None
        entry = None if quiescent else self.table.get(key)
        if entry is not None:
            tt_move = entry[3]
            if entry[0] >= depth:
                self.tt_hits += 1
                value = self._unpack_mate(entry[1], ply)
                if entry[2] == _EXACT:
                    return value
                if entry[2] == _LOWER:
                    alpha = max(alpha, value)
                else:
                    beta = min(beta, value)
                if alpha >= beta:
                    return value
        window_alpha, window_beta = alpha, beta
        if enemy:
            moves = sorted(enemy)
        elif quiescent:
            moves = sorted(self._tactical_defenses(player),
                           key=lambda i: self._priority(i, player), reverse=True)
        else:
            moves = self._ordered_moves(player, self.branch_width, ply, tt_move)
        best, best_move = -2 * _MATE, moves[0]
        for cell in moves:
            self._check_time()
            self._change(cell, player, 1)
            try:
                value = (_MATE - ply - 1 if self._won(cell, player) else
                         -self._negamax(0 if quiescent else depth - 1,
                                       -beta, -alpha, 3 - player, ply + 1,
                                       extra + 1 if quiescent else extra))
            finally:
                self._change(cell, player, -1)
            if value > best:
                best, best_move = value, cell
            alpha = max(alpha, value)
            if alpha >= beta:
                self.cutoffs += 1
                if not quiescent and not enemy:
                    killers = self.killers.get(ply, [])
                    self.killers[ply] = [cell] + [i for i in killers if i != cell][:1]
                    self.history[player - 1][cell] += depth * depth
                break
        if not quiescent:
            flag = _UPPER if best <= window_alpha else _LOWER if best >= window_beta else _EXACT
            self._store(key, depth, best, flag, best_move, ply)
        return best
