# 广义 N×K Gomoku

需要 Python 3.9 或更新版本。程序仅使用 Python 标准库，无需安装第三方包或执行 `pip install`。需要图形界面时，请使用本机浏览器；游戏可离线运行。

先在存放 `main.py` 的目录打开终端，执行 `python --version` 确认 Python 可用。Windows 若 `python` 不可用，可将下面命令的 `python` 替换为 `py -3`。若两者均不可用，请安装 Python，并将其加入 PATH；双击启动器同样要求系统能找到 `py` 或 `python`。

## 图形界面

```console
python main.py --gui
```

Windows 也可双击 `run_gui.cmd`。启动后会自动打开浏览器；若没有自动打开，请复制终端显示的 `http://127.0.0.1:端口/` 地址到本机浏览器。端口自动选择，每次可能不同。也可使用以下命令只显示地址：

```console
python main.py --gui --no-browser
```

保持启动终端运行，按 Ctrl+C 停止服务。页面由本机 Python 服务提供，直接双击 `web/index.html` 无法执行 AI 对战。

页面支持人人、人机、机机模式，可配置棋盘边长 N、获胜连子数 K、人类棋色和双方模型。界面范围为 `5 <= N <= 30`、`3 <= K <= N`。修改设置后点击“开始新对局”生效。

可选模型：

- **Depth Search**：Alpha-Beta、迭代加深、着法排序。
- **Monte Carlo**：MCTS、UCT、候选筛选及启发式模拟。
- **Adaptive Search**：战术预检查后，每步选择一个独立搜索器；对应参赛入口 `gomoku_ai.py`。

网页初始模型为 Depth Search，初始每步预算为1秒。若要按课程默认预算体验参赛模型，请选择 Adaptive Search，将“AI 每步思考”设为“挑战 · 5秒”，再开始新对局。

点击棋盘交叉点落子；支持悔棋、机机暂停/继续/单步和棋谱导出。AI 思考时可以悔棋或开始新局，旧搜索结果会被丢弃；机机暂停会等待当前一手结束。搜索观察区按模型展示实际搜索统计。

## 命令行游戏

### 人人对战

```console
python main.py --mode hvh --board 9 --win 4
```

### 人机对战

人执黑先行，AI 默认使用 Adaptive Search，每步最多5秒：

```console
python main.py --mode hva --human 1 --board 15 --win 5 --time 5
```

将 `--human 1` 改为 `--human 2`，可改为人执白后行。双击 `run_game.cmd` 会启动默认命令行人机模式：15×15、K=5、人执黑、Adaptive Search、每步5秒。

### 机机对战

先用小棋盘快速验证运行。下面使用本项目的随机基线 `random_ai.py`：

```console
python main.py --mode ava --white-ai random_ai.py --board 5 --win 3 --time 5
```

也可以指定双方搜索模型，例如 Depth Search 对 Monte Carlo：

```console
python main.py --mode ava --black-ai gomoku_depth.py --white-ai gomoku_mcts.py --board 9 --win 4 --time 5
```

`--black-ai` 和 `--white-ai` 均可指定本地协议 AI 文件，未指定时使用 Adaptive Search。相对路径按终端当前目录解析；路径含空格时加双引号。不同机器的搜索速度和墙钟截断可能产生不同棋谱。

## 坐标与参数

命令行坐标从0开始，输入 `行 列`，例如 `7 7`，也接受 `7,7`。`X` 为黑棋，`O` 为白棋，`.` 为空位；输入 `q` 结束，错误坐标会提示重新输入。

`--board N` 指定棋盘边长，`--win K` 指定连胜长度，要求整数且 `3 <= K <= N`。`--time` 为 AI 每步外部时限，单位秒，必须为正数，默认5秒。命令行默认 N=15、K=5、人机模式、人执黑；核心引擎没有 N 的固定上限，但极大棋盘性能未全面验证。

GUI 参数与命令行对战参数属于不同入口：`--gui` 后可用 `--no-browser` 和 `--port`，棋盘、模型及预算在页面中设置。
