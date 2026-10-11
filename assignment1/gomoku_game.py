"""Generalized Gomoku rules, independent of UI and search."""

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))


class GomokuGame:
    def __init__(self, board_size=15, win_length=5):
        if type(board_size) is not int or type(win_length) is not int or not 3 <= win_length <= board_size:
            raise ValueError("require integer N and K with 3 <= K <= N")
        self.board_size, self.win_length = board_size, win_length
        self.board = [[0] * board_size for _ in range(board_size)]
        self.current_player = 1
        self.winner = None  # None: ongoing; 0: draw; 1/2: winner
        self.history = []

    def legal_moves(self):
        n = self.board_size
        return [(r, c) for r in range(n) for c in range(n) if not self.board[r][c]]

    def play(self, move):
        if self.winner is not None:
            raise ValueError("game already finished")
        if not isinstance(move, tuple) or len(move) != 2 or any(type(x) is not int for x in move):
            raise ValueError("move must be an (int, int) tuple")
        r, c = move
        n = self.board_size
        if not (0 <= r < n and 0 <= c < n):
            raise ValueError("move is outside the board")
        if self.board[r][c]:
            raise ValueError("cell is occupied")
        player = self.current_player
        self.board[r][c] = player
        self.history.append((player, r, c))
        if self.is_winning_move(r, c, player):
            self.winner = player
        elif len(self.history) == n * n:
            self.winner = 0
        else:
            self.current_player = 3 - player
        return self.winner

    def is_winning_move(self, r, c, player):
        n = self.board_size
        for dr, dc in DIRECTIONS:
            total = 1
            for sign in (-1, 1):
                rr, cc = r + sign * dr, c + sign * dc
                while 0 <= rr < n and 0 <= cc < n and self.board[rr][cc] == player:
                    total += 1
                    rr, cc = rr + sign * dr, cc + sign * dc
            if total >= self.win_length:
                return True
        return False

    def render(self):
        symbols = (".", "X", "O")
        lines = ["    " + " ".join(f"{c:2d}" for c in range(self.board_size))]
        lines += [f"{r:3d} " + " ".join(f" {symbols[x]}" for x in row)
                  for r, row in enumerate(self.board)]
        return "\n".join(lines)
