"""Interactive local-browser GUI, Python standard library only.

The browser handles drawing; models A, B and H run through the same isolated
protocol processes as the CLI. No internet is required.
"""
import argparse
import json
import math
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from multiprocessing import freeze_support
from pathlib import Path
from urllib.parse import urlsplit

from ai_runner import AIError, AIProcess
from gomoku_game import GomokuGame

ROOT = Path(__file__).resolve().parent
AI_FILES = {
    "model_a": ROOT / "gomoku_depth.py",
    "model_b": ROOT / "gomoku_mcts.py",
    "model_h": ROOT / "gomoku_ai.py",
    # Backward-compatible aliases for saved local configurations and tests.
    "optimized": ROOT / "gomoku_depth.py",
    "mcts": ROOT / "gomoku_mcts.py",
    "random": ROOT / "random_ai.py",
}
AI_LABELS = {
    "model_a": "Depth Search", "model_b": "Monte Carlo", "model_h": "Adaptive Search",
    "optimized": "Depth Search", "mcts": "Monte Carlo", "random": "随机 AI",
}


class GameController:
    def __init__(self):
        self.lock = threading.RLock()
        self.game_id = 0
        self.players = {}
        self.closed = False
        self.new_game({})

    @staticmethod
    def _dispose(players):
        for player in players.values():
            player.close()

    def _dispose_async(self, players):
        if players:
            threading.Thread(target=self._dispose, args=(players,), daemon=True).start()

    def new_game(self, config):
        n, k = config.get("board_size", 15), config.get("win_length", 5)
        if type(n) is not int or type(k) is not int or not 5 <= n <= 30 or not 3 <= k <= n:
            raise ValueError("界面要求 5 <= N <= 30，3 <= K <= N")
        mode, human = config.get("mode", "hva"), config.get("human", 1)
        if mode not in ("hvh", "hva", "ava") or type(human) is not int or human not in (1, 2):
            raise ValueError("请选择正确的对战模式和人类棋色")
        limit = config.get("time_limit", 1.0)
        if type(limit) not in (int, float) or not math.isfinite(limit) or not 0.03 <= limit <= 10:
            raise ValueError("每步时限须为 0.03 到 10 秒")
        paths, names = {}, {}
        for side, key in ((1, "black_ai"), (2, "white_ai")):
            choice = config.get(key, "model_a")
            if not isinstance(choice, str):
                raise ValueError("AI 路径须为字符串")
            path = AI_FILES.get(choice, Path(choice)).resolve()
            if not path.is_file() or path.suffix.lower() != ".py":
                raise ValueError(f"找不到 AI 文件：{choice}")
            paths[side], names[side] = path, AI_LABELS.get(choice, path.stem)
        with self.lock:
            old = self.players
            self.players = {}
            self.game_id += 1
            self.game = GomokuGame(n, k)
            self.mode, self.human, self.time_limit = mode, human, float(limit)
            self.paths, self.names = paths, names
            self.busy, self.paused = False, False
            self.error = self.last_ai = None
            self._schedule_locked()
            state = self._state_locked()
        self._dispose_async(old)
        return state

    def _state_locked(self):
        g = self.game
        return {"game_id": self.game_id, "board_size": g.board_size,
                "win_length": g.win_length, "mode": self.mode, "human": self.human,
                "time_limit": self.time_limit, "board": [r[:] for r in g.board],
                "current_player": g.current_player, "winner": g.winner,
                "history": [list(move) for move in g.history], "busy": self.busy,
                "paused": self.paused, "error": self.error, "last_ai": self.last_ai,
                "names": {str(side): name for side, name in self.names.items()}}

    def state(self):
        with self.lock:
            return self._state_locked()

    def _check_generation(self, data):
        if data.get("game_id") != self.game_id:
            raise ValueError("棋局已更新，请重试")

    def _is_human(self, side):
        return self.mode == "hvh" or (self.mode == "hva" and side == self.human)

    def move(self, data):
        with self.lock:
            self._check_generation(data)
            if self.busy or not self._is_human(self.game.current_player):
                raise ValueError("当前不是可操作的人类回合")
            self.game.play((data.get("row"), data.get("col")))
            self._schedule_locked()
            return self._state_locked()

    def undo(self, data):
        with self.lock:
            self._check_generation(data)
            history = self.game.history
            if not history:
                raise ValueError("还没有可撤销的落子")
            if self.mode == "hva":
                human_moves = [i for i, move in enumerate(history) if move[0] == self.human]
                if not human_moves:
                    raise ValueError("还没有人类落子可撤销")
                keep = human_moves[-1]
            else:
                keep = len(history) - 1
            retained = history[:keep]
            old = self.players
            self.players = {}
            self.game_id += 1
            self.game = GomokuGame(self.game.board_size, self.game.win_length)
            for _, r, c in retained:
                self.game.play((r, c))
            self.busy = False
            self.error = self.last_ai = None
            if self.mode == "ava":
                self.paused = True
            self._schedule_locked()
            state = self._state_locked()
        self._dispose_async(old)
        return state

    def pause(self, data):
        with self.lock:
            self._check_generation(data)
            if self.mode != "ava" or type(data.get("paused")) is not bool:
                raise ValueError("暂停仅适用于机机对战")
            self.paused = data["paused"]
            self._schedule_locked()
            return self._state_locked()

    def step(self, data):
        with self.lock:
            self._check_generation(data)
            if self.mode != "ava" or not self.paused or self.busy:
                raise ValueError("请先暂停机机对战并等待当前步结束")
            self._schedule_locked(force=True)
            return self._state_locked()

    def _schedule_locked(self, force=False):
        if (self.closed or self.busy or self.error or self.game.winner is not None
                or self._is_human(self.game.current_player) or (self.paused and not force)):
            return
        side = self.game.current_player
        token = self.game_id
        board = [row[:] for row in self.game.board]
        last = tuple(self.game.history[-1][1:]) if self.game.history else None
        self.busy = True
        thread = threading.Thread(target=self._think, args=(token, side, board, last,
                                  self.paths[side], self.time_limit), daemon=True)
        thread.start()

    def _think(self, token, side, board, last, path, limit):
        ai = None
        try:
            with self.lock:
                if token != self.game_id or self.closed:
                    return
                ai = self.players.get(side)
            if ai is None:
                created = AIProcess(path, side, len(board), self.game.win_length)
                with self.lock:
                    if token != self.game_id or self.closed:
                        created.close()
                        return
                    ai = self.players[side] = created
            move, elapsed, stats = ai.get_move(board, last, limit)
            with self.lock:
                if token != self.game_id or self.closed:
                    return
                self.game.play(move)
                self.last_ai = {"player": side, "move": list(move), "seconds": elapsed, "stats": stats}
                self.busy = False
                # Avoid a visually invisible chain of instant tactical moves.
                if self.mode == "ava" and not self.paused and self.game.winner is None:
                    threading.Timer(0.25, self._resume_generation, args=(token,)).start()
                else:
                    self._schedule_locked()
        except (AIError, ValueError, OSError, EOFError) as exc:
            with self.lock:
                if token == self.game_id and not self.closed:
                    self.error = f"{self.names[side]} 判负：{exc}"
                    self.game.winner = 3 - side
                    self.busy = False

    def _resume_generation(self, token):
        with self.lock:
            if token == self.game_id:
                self._schedule_locked()

    def close(self):
        with self.lock:
            self.closed = True
            self.game_id += 1
            old, self.players = self.players, {}
        self._dispose(old)


class LocalServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def _reply(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False, allow_nan=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _valid_local_request(self):
        port = self.server.server_address[1]
        allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
        if self.headers.get("Host") not in allowed:
            self._reply({"error": "Local host only"}, 403)
            return False
        origin = self.headers.get("Origin")
        if origin and origin not in {f"http://{host}" for host in allowed}:
            self._reply({"error": "Same-origin requests only"}, 403)
            return False
        return True

    def do_GET(self):
        if not self._valid_local_request():
            return
        path = urlsplit(self.path).path
        if path == "/api/state":
            self._reply(self.server.controller.state())
        elif path in ("/", "/index.html"):
            file = ROOT / "web" / "index.html"
            if not file.is_file():
                self._reply({"error": "GUI page missing"}, 404)
                return
            data = file.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            # Keep all browser resource requests on the local origin.
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; connect-src 'self'; "
                             "script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
                             "img-src 'self' data: blob:; font-src 'self'; "
                             "base-uri 'none'; form-action 'none'; frame-ancestors 'none'")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
        else:
            self._reply({"error": "Not found"}, 404)

    def do_POST(self):
        if not self._valid_local_request():
            return
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            self._reply({"error": "Expected application/json"}, 415)
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 8192:
                raise ValueError("请求长度不正确")
            data = json.loads(self.rfile.read(size))
            if not isinstance(data, dict):
                raise ValueError("请求必须是 JSON 对象")
            routes = {"/api/new": self.server.controller.new_game,
                      "/api/move": self.server.controller.move,
                      "/api/undo": self.server.controller.undo,
                      "/api/pause": self.server.controller.pause,
                      "/api/step": self.server.controller.step}
            method = routes.get(urlsplit(self.path).path)
            if method is None:
                self._reply({"error": "Not found"}, 404)
                return
            self._reply(method(data))
        except (ValueError, TypeError, KeyError) as exc:
            self._reply({"error": str(exc)}, 400)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Gomoku interactive local GUI")
    parser.add_argument("--port", type=int, default=0, help="0 chooses an available port")
    parser.add_argument("--no-browser", action="store_true", help="print URL without opening browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("port must be 0..65535")
    controller = GameController()
    try:
        server = LocalServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        controller.close()
        parser.error(f"could not bind local port: {exc}")
    server.controller = controller
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"Gomoku GUI: {url}", flush=True)
    print("Keep this terminal open. Ctrl+C stops the local server.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\nGUI stopped.", flush=True)
    finally:
        server.server_close()
        controller.close()
    return 0


if __name__ == "__main__":
    freeze_support()
    raise SystemExit(main())
