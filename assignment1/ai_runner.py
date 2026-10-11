"""Load protocol AIs in persistent child processes with hard wall-clock limits."""
import importlib.util
import multiprocessing as mp
import time
import traceback
from pathlib import Path


class AIError(RuntimeError):
    pass


def _worker(conn, path, player, n, k, seed, settings):
    try:
        spec = importlib.util.spec_from_file_location("loaded_gomoku_ai", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        ai = module.GomokuAI(player, n, k)
        if seed is not None and hasattr(ai, "rng"):
            ai.rng.seed(seed)
        for key, value in (settings or {}).items():
            setattr(ai, key, value)
        conn.send(("ready", getattr(module, "NAME", Path(path).stem)))
        while True:
            request = conn.recv()
            if request is None:
                break
            board, last, limit = request
            start = time.perf_counter()
            try:
                move = ai.get_move(board, last, limit)
                elapsed = time.perf_counter() - start
                conn.send(("move", move, elapsed, getattr(ai, "last_stats", {})))
            except Exception:
                conn.send(("error", traceback.format_exc(limit=4)))
    except Exception:
        try:
            conn.send(("error", traceback.format_exc(limit=4)))
        except (EOFError, OSError):
            pass
    finally:
        conn.close()


class AIProcess:
    def __init__(self, path, player, n, k, seed=None, settings=None):
        path = str(Path(path).resolve())
        if not Path(path).is_file():
            raise AIError(f"AI file does not exist: {path}")
        ctx = mp.get_context("spawn")
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker, args=(child, path, player, n, k, seed, settings))
        self.process.daemon = True
        self.process.start()
        child.close()
        try:
            if not self.conn.poll(10):
                raise AIError("AI initialization timed out (10 seconds)")
            reply = self.conn.recv()
            if reply[0] != "ready":
                raise AIError(f"AI initialization failed: {reply[1]}")
            self.name = reply[1]
        except (AIError, EOFError, OSError) as exc:
            self.close()
            raise AIError(str(exc)) from exc

    def get_move(self, board, last, limit):
        try:
            start = time.perf_counter()
            self.conn.send(([row[:] for row in board], last, limit))
            remaining = max(0, limit - (time.perf_counter() - start))
            if not self.conn.poll(remaining):
                self.close()
                raise AIError(f"AI exceeded {limit:.3f}s wall-clock budget")
            reply = self.conn.recv()
            if reply[0] == "error":
                raise AIError(f"AI raised an exception: {reply[1]}")
            _, move, elapsed, stats = reply
            if elapsed > limit:
                raise AIError(f"AI exceeded budget: {elapsed:.6f}s > {limit:.6f}s")
            return move, elapsed, stats
        except (EOFError, OSError) as exc:
            raise AIError(f"AI process disconnected: {exc}") from exc

    def close(self):
        if hasattr(self, "process"):
            if self.process.is_alive():
                self.process.terminate()
            self.process.join(timeout=1)
        if hasattr(self, "conn"):
            self.conn.close()


def play_match(black_path, white_path, n=15, k=5, limit=5.0, seed=0,
               settings=None, on_move=None, opening=None):
    from gomoku_game import GomokuGame
    game = GomokuGame(n, k)
    players, names, moves = {}, {}, []
    reason = "normal"
    offender = None
    for raw_move in opening or []:
        side = game.current_player
        move = tuple(raw_move)
        game.play(move)
        moves.append({"player": side, "move": move, "seconds": 0.0,
                      "stats": {"reason": "opening", "preset": True}})
        if game.winner is not None:
            raise ValueError("preset opening must not end the game")
    try:
        for side, path in ((1, black_path), (2, white_path)):
            try:
                players[side] = AIProcess(path, side, n, k, seed + side,
                                          (settings or {}).get(side))
                names[side] = players[side].name
            except AIError as exc:
                game.winner, offender, reason = 3 - side, side, str(exc)
                break
        while game.winner is None:
            side = game.current_player
            last = tuple(game.history[-1][1:]) if game.history else None
            try:
                move, elapsed, stats = players[side].get_move(game.board, last, limit)
                game.play(move)
            except (AIError, ValueError) as exc:
                game.winner, offender, reason = 3 - side, side, str(exc)
                break
            moves.append({"player": side, "move": move, "seconds": elapsed, "stats": stats})
            if on_move:
                on_move(game, side, move, elapsed)
        return {"board_size": n, "win_length": k, "time_limit": limit, "seed": seed,
                "names": names, "winner": game.winner, "plies": len(game.history),
                "reason": reason, "offender": offender, "moves": moves,
                "opening": list(opening or [])}
    finally:
        for player in players.values():
            player.close()
