from multiprocessing import freeze_support
from main_cli import main
import sys

if __name__ == "__main__":
    freeze_support()
    if "--gui" in sys.argv[1:]:
        from gui import main as gui_main
        raise SystemExit(gui_main([arg for arg in sys.argv[1:] if arg != "--gui"]))
    raise SystemExit(main())
