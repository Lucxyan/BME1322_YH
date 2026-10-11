"""All three required game modes, with zero-based row/column input."""
import argparse
from pathlib import Path

from ai_runner import AIError, AIProcess, play_match
from gomoku_game import GomokuGame

HERE = Path(__file__).resolve().parent


def _human_move(game):
    while True:
        raw = input(f"Player {game.current_player} (row col, or q): ").strip()
        if raw.lower() in ("q", "quit", "exit"):
            raise KeyboardInterrupt
        try:
            values = raw.replace(",", " ").split()
            if len(values) != 2:
                raise ValueError("enter two coordinates")
            move = tuple(int(x) for x in values)
            game.play(move)
            return move
        except ValueError as exc:
            print(f"Invalid move: {exc}; try again.")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generalized Gomoku (X=black, O=white)")
    parser.add_argument("--mode", choices=("hvh", "hva", "ava"), default="hva",
                        help="human-human / human-AI / AI-AI")
    parser.add_argument("--board", type=int, default=15)
    parser.add_argument("--win", type=int, default=5)
    parser.add_argument("--time", type=float, default=5.0)
    parser.add_argument("--human", type=int, choices=(1, 2), default=1)
    parser.add_argument("--black-ai", default=str(HERE / "gomoku_ai.py"))
    parser.add_argument("--white-ai", default=str(HERE / "gomoku_ai.py"))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)
    import math
    if not math.isfinite(args.time) or args.time <= 0:
        parser.error("--time must be finite and positive")
    try:
        game = GomokuGame(args.board, args.win)
    except ValueError as exc:
        parser.error(str(exc))

    def show(current, side, move, elapsed):
        if not args.quiet:
            print(f"\nPlayer {side}: {move}, {elapsed:.4f}s\n{current.render()}")

    if args.mode == "ava":
        result = play_match(args.black_ai, args.white_ai, args.board, args.win, args.time, on_move=show)
        print(f"Result: {'Draw' if result['winner'] == 0 else 'Player ' + str(result['winner']) + ' wins'}; "
              f"{result['plies']} plies; {result['reason']}")
        return 0 if result["reason"] == "normal" else 1
    ai = None
    try:
        if args.mode == "hva":
            side = 3 - args.human
            ai = AIProcess(args.black_ai if side == 1 else args.white_ai,
                           side, args.board, args.win)
        print(f"N={args.board}, K={args.win}; coordinates are zero-based. Enter q to stop.")
        print(game.render())
        while game.winner is None:
            side = game.current_player
            if args.mode == "hvh" or side == args.human:
                move = _human_move(game)
                show(game, side, move, 0)
            else:
                last = tuple(game.history[-1][1:]) if game.history else None
                try:
                    move, elapsed, _ = ai.get_move(game.board, last, args.time)
                    game.play(move)
                except (AIError, ValueError) as exc:
                    print(f"AI forfeits: {exc}; Player {3 - side} wins.")
                    return 1
                show(game, side, move, elapsed)
        print("Draw." if game.winner == 0 else f"Player {game.winner} wins.")
        return 0
    except (KeyboardInterrupt, EOFError):
        print("\nGame stopped.")
        return 0
    except AIError as exc:
        print(f"Could not start AI: {exc}")
        return 1
    finally:
        if ai:
            ai.close()
