#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connect-4 with AlphaZero-style self-play (PUCT MCTS + policy-value NN)
and full Tkinter GUI — April 2025 continuous-self-play edition.
"""

# ----------------------------------------------------------------------
# Imports & constants
# ----------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox
import numpy as np, random, math, threading, time, json, os, numba
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

ROW_COUNT, COLUMN_COUNT = 6, 7
SQUARESIZE, RADIUS      = 100, int(100/2 - 5)
WIDTH, HEIGHT           = COLUMN_COUNT * SQUARESIZE, (ROW_COUNT + 1) * SQUARESIZE

BLUE, BLACK = "#0000FF", "#000000"
RED,  GREEN = "#FF0000", "#00FF00"
EMPTY_COLOR = "#CCCCCC"

EMPTY, RED_PIECE, GREEN_PIECE = 0, 1, 2
PLAYER_MAP = {RED_PIECE: "Red", GREEN_PIECE: "Green"}
COLOR_MAP  = {RED_PIECE: RED,   GREEN_PIECE: GREEN}

DEFAULT_MCTS_ITERATIONS = 1000
DEFAULT_PUCT_C          = 1.25
NN_MODEL_FILE           = "C4_NN.pt"
MCTS_CONFIG_FILE        = "mcts_config.json"
MAX_TRAINING_EXAMPLES   = 20_000

# ----------------------------------------------------------------------
# Neural network (policy + value heads)
# ----------------------------------------------------------------------
class Connect4NN(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.in_size = ROW_COUNT * COLUMN_COUNT * 2 + 1
        self.fc1 = nn.Linear(self.in_size, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.policy_head = nn.Linear(hidden, COLUMN_COUNT)
        self.value_head  = nn.Linear(hidden, 1)
        self.relu, self.tanh = nn.ReLU(), nn.Tanh()
    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.policy_head(x), self.tanh(self.value_head(x))

class Connect4Dataset(Dataset):
    def __init__(self, s, a, o): 
        self.s, self.a, self.o = s, a, o
    def __len__(self): 
        return len(self.s)
    def __getitem__(self, i): 
        return self.s[i], self.a[i], self.o[i]

# ----------------------------------------------------------------------
# Numba helpers
# ----------------------------------------------------------------------
@numba.njit(cache=True)
def _is_valid(b, col, C, R): 
    return 0 <= col < C and b[R-1, col] == EMPTY

@numba.njit(cache=True)
def _next_row(b, col, R):
    for r in range(R):
        if b[r, col] == EMPTY: 
            return r
    return -1

@numba.njit(cache=True)
def _win(b, p, R, C):
    for c in range(C-3):
        for r in range(R):
            if (b[r, c:c+4] == p).all(): 
                return True
    for c in range(C):
        for r in range(R-3):
            if (b[r:r+4, c] == p).all(): 
                return True
    for c in range(C-3):
        for r in range(R-3):
            if b[r, c] == p and b[r+1, c+1] == p and b[r+2, c+2] == p and b[r+3, c+3] == p: 
                return True
    for c in range(C-3):
        for r in range(3, R):
            if b[r, c] == p and b[r-1, c+1] == p and b[r-2, c+2] == p and b[r-3, c+3] == p: 
                return True
    return False

@numba.njit(cache=True)
def _draw(b):
    for i in range(b.shape[0]):
        for j in range(b.shape[1]):
            if b[i, j] == EMPTY: 
                return False
    return True

@numba.njit(cache=True)
def _valid_cols(b, R, C):
    cols = []
    for col in range(C):
        if _is_valid(b, col, C, R): 
            cols.append(col)
    return np.array(cols, np.int64) if cols else np.empty(0, np.int64)

@numba.njit(cache=True)
def _random_playout(board, player, R, C, red, green):
    b = board.copy()
    cur = player
    while True:
        moves = _valid_cols(b, R, C)
        if len(moves) == 0: 
            return EMPTY
        col = moves[np.random.randint(len(moves))]
        row = _next_row(b, col, R); b[row, col] = cur
        if _win(b, cur, R, C): 
            return cur
        if _draw(b): 
            return EMPTY
        cur = green if cur == red else red

# ----------------------------------------------------------------------
# Core game state
# ----------------------------------------------------------------------
class Connect4Game:
    def __init__(self): 
        self.reset()
    def reset(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int64)
        self.current_player = RED_PIECE
        self.game_over = False
        self.winner = None
    def is_valid(self, c): 
        return _is_valid(self.board, c, COLUMN_COUNT, ROW_COUNT)
    def valid_moves(self): 
        return _valid_cols(self.board, ROW_COUNT, COLUMN_COUNT).tolist()
    def drop_piece(self, c):
        if not self.is_valid(c): 
            return False, -1
        r = _next_row(self.board, c, ROW_COUNT)
        self.board[r, c] = int(self.current_player)
        if _win(self.board, self.current_player, ROW_COUNT, COLUMN_COUNT):
            self.game_over = True; self.winner = self.current_player
        elif _draw(self.board):
            self.game_over = True; self.winner = 'Draw'
        return True, r
    def switch(self): 
        self.current_player = GREEN_PIECE if self.current_player == RED_PIECE else RED_PIECE
    def copy(self):
        g = Connect4Game()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g.game_over = self.game_over
        g.winner = self.winner
        return g
    get_state_copy = copy

# ----------------------------------------------------------------------
# Neural-network manager
# ----------------------------------------------------------------------
class NNManager:
    def __init__(self):
        self.net = Connect4NN()
        self.opt = optim.Adam(self.net.parameters(), lr=1e-3)
        self.pol_loss = nn.CrossEntropyLoss()
        self.val_loss = nn.MSELoss()
        self.data = {'states': [], 'actions': [], 'outcomes': []}
        self.pending = []
        if os.path.exists(NN_MODEL_FILE):
            ck = torch.load(NN_MODEL_FILE, map_location='cpu')
            self.net.load_state_dict(ck['model_state_dict'])
            self.opt.load_state_dict(ck['optimizer_state_dict'])
            print("Model loaded.")
        else:
            print("No existing model – new weights.")
    def _save(self):
        torch.save({
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.opt.state_dict()
        }, NN_MODEL_FILE)
    @staticmethod
    def _tensor(state: Connect4Game):
        red = (state.board == RED_PIECE).astype(np.float32)
        green = (state.board == GREEN_PIECE).astype(np.float32)
        turn = np.array([1.0 if state.current_player == RED_PIECE else 0.0], np.float32)
        return torch.tensor(np.concatenate([red.flatten(), green.flatten(), turn]), dtype=torch.float32)
    def add_example(self, state, action):
        self.pending.append({
            'state': self._tensor(state),
            'action': action,
            'player': state.current_player
        })
    add_training_example = add_example
    def finish_game(self, winner):
        for ex in self.pending:
            if winner == 'Draw':
                out = 0.0
            elif winner == ex['player']:
                out = 1.0
            else:
                out = -1.0
            self.data['states'].append(ex['state'])
            self.data['actions'].append(ex['action'])
            self.data['outcomes'].append(torch.tensor([out], dtype=torch.float32))
        self.pending = []
        if len(self.data['states']) > MAX_TRAINING_EXAMPLES:
            n = len(self.data['states']) - MAX_TRAINING_EXAMPLES
            for k in self.data:
                self.data[k] = self.data[k][n:]
    record_game_outcome = finish_game
    def train(self, batch_size=32, epochs=5):
        if not self.data['states']:
            return
        ds = Connect4Dataset(torch.stack(self.data['states']),
                             torch.tensor(self.data['actions'], dtype=torch.long),
                             torch.stack(self.data['outcomes']))
        dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        self.net.train()
        for ep in range(epochs):
            p = v = 0
            for s, a, o in dl:
                self.opt.zero_grad()
                logits, val = self.net(s)
                lp = self.pol_loss(logits, a); lv = self.val_loss(val, o)
                (lp + lv).backward()
                self.opt.step()
                p += lp.item(); v += lv.item()
            print(f"[Train] Epoch {ep+1}/{epochs}  policy={p/len(dl):.4f}  value={v/len(dl):.4f}")
        self._save()
    def policy_value(self, state: Connect4Game):
        self.net.eval()
        with torch.no_grad():
            t = self._tensor(state).unsqueeze(0)
            logits, val = self.net(t)
            logits = logits.squeeze(0)
            valid = state.valid_moves()
            for c in range(COLUMN_COUNT):
                if c not in valid:
                    logits[c] = -1e9
            return F.softmax(logits, dim=0).cpu().numpy(), float(val.item())

# ----------------------------------------------------------------------
# MCTS with PUCT
# ----------------------------------------------------------------------
class MCTSNode:
    def __init__(self, state, parent=None, move=None, prior=0.0, mcts=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.mcts = mcts
        self.children = []
        self.untried = state.valid_moves()
        self.visits = 0
        self.value_sum = 0.0
        self.player = state.current_player
        self._policy = None
    def puct(self, total, c):
        if self.visits == 0:
            return float('inf')
        return self.value_sum / self.visits + c * self.prior * math.sqrt(total) / (1 + self.visits)
    def _ensure_policy(self):
        if self._policy is not None:
            return
        if self.mcts and self.mcts.nn:
            self._policy, _ = self.mcts.nn.policy_value(self.state)
        else:
            self._policy = np.zeros(COLUMN_COUNT, np.float32)
            for col in self.untried:
                self._policy[col] = 1 / len(self.untried)
    def select(self, c):
        node = self
        while not node.untried and node.children:
            node = max(node.children, key=lambda ch: ch.puct(node.visits, c))
        return node
    def expand(self):
        self._ensure_policy()
        if not self.untried:
            return None
        m = self.untried.pop()
        ns = self.state.copy(); ns.drop_piece(m)
        if not ns.game_over:
            ns.switch()
        child = MCTSNode(ns, parent=self, move=m, prior=float(self._policy[m]), mcts=self.mcts)
        self.children.append(child)
        return child
    def evaluate(self):
        if self.state.game_over:
            if self.state.winner == 'Draw':
                return 0.0
            return 1.0 if self.state.winner == self.player else -1.0
        if self.mcts and self.mcts.nn:
            _, v = self.mcts.nn.policy_value(self.state)
            # If it's the opponent's turn from this node's perspective, invert the value
            return v if self.state.current_player == self.player else -v
        w = _random_playout(self.state.board.copy(),
                            int(self.state.current_player),
                            ROW_COUNT, COLUMN_COUNT, RED_PIECE, GREEN_PIECE)
        if w == EMPTY:
            return 0.0
        return 1.0 if w == self.player else -1.0
    def backprop(self, val):
        node = self
        while node:
            node.visits += 1
            node.value_sum += val
            val = -val
            node = node.parent

class MCTS:
    def __init__(self, iters, c, nn):
        self.I = iters; self.C = c; self.nn = nn
    def search(self, state):
        root = MCTSNode(state, mcts=self)
        if self.nn:
            root._ensure_policy()
            valid = state.valid_moves()
            noise = np.random.dirichlet([0.3] * len(valid))
            for i, m in enumerate(valid):
                root._policy[m] = 0.75 * root._policy[m] + 0.25 * noise[i]
        for _ in range(self.I):
            node = root.select(self.C)
            if not node.state.game_over:
                node = node.expand() or node
            v = node.evaluate()
            node.backprop(v)
        best = max(root.children, key=lambda ch: ch.visits)
        if self.nn:
            self.nn.add_example(state, best.move)
        return best.move

# ----------------------------------------------------------------------
# Players
# ----------------------------------------------------------------------
class Player: 
    pass

class HumanPlayer(Player): 
    pass

class RandomComputerPlayer(Player):
    def get_move(self, state, gui=None):
        time.sleep(0.1)
        return random.choice(state.valid_moves())

class MCTSComputerPlayer(Player):
    def __init__(self, iters, c, nn):
        self.mcts = MCTS(iters, c, nn)
    def get_move(self, state, gui=None):
        t = time.time()
        mv = self.mcts.search(state.copy())
        # Ensure a slight delay for UX (so AI move isn't instant)
        if time.time() - t < 0.1:
            time.sleep(0.1 - (time.time() - t))
        return mv

# ----------------------------------------------------------------------
# Dialogs
# ----------------------------------------------------------------------
class PlayerDialog(simpledialog.Dialog):
    def __init__(self, master, mcts, nn):
        self.mcts = mcts; self.nn = nn
        self.p1 = tk.StringVar(master, "Human")
        self.p2 = tk.StringVar(master, "Computer (MCTS)")
        super().__init__(master, "Select Players")
    def body(self, m):
        opts = ["Human", "Computer (Random)", "Computer (MCTS)"]
        ttk.Label(m, text="Red:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        for i, o in enumerate(opts, 1):
            ttk.Radiobutton(m, text=o, variable=self.p1, value=o).grid(row=0, column=i, sticky="w")
        ttk.Label(m, text="Green:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        for i, o in enumerate(opts, 1):
            ttk.Radiobutton(m, text=o, variable=self.p2, value=o).grid(row=1, column=i, sticky="w")
    def apply(self):
        def mk(sel):
            if sel == "Human": 
                return HumanPlayer()
            if sel == "Computer (Random)":
                return RandomComputerPlayer()
            return MCTSComputerPlayer(self.mcts['iterations'], self.mcts['C_param'], self.nn)
        self.result = {'red': mk(self.p1.get()), 'green': mk(self.p2.get())}

class MCTSDialog(simpledialog.Dialog):
    def __init__(self, master, it, c):
        self.it = tk.StringVar(master, str(it))
        self.c = tk.StringVar(master, f"{c:.2f}")
        super().__init__(master, "MCTS Config")
    def body(self, m):
        ttk.Label(m, text="Iterations:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(m, textvariable=self.it, width=10).grid(row=0, column=1, padx=5)
        ttk.Label(m, text="Exploration C:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(m, textvariable=self.c, width=10).grid(row=1, column=1, padx=5)
    def validate(self):
        try:
            return int(self.it.get()) > 0 and float(self.c.get()) >= 0
        except:
            messagebox.showwarning("Invalid", "Enter positive numbers.")
            return False
    def apply(self):
        self.result = {
            'iterations': int(self.it.get()),
            'C_param': float(self.c.get())
        }

class TrainDialog(simpledialog.Dialog):
    def __init__(self, master):
        self.e = tk.StringVar(master, "100")
        super().__init__(master, "Train NN")
    def body(self, m):
        ttk.Label(m, text="Self-play games:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(m, textvariable=self.e, width=10).grid(row=0, column=1, padx=5)
    def validate(self):
        try:
            return int(self.e.get()) > 0
        except:
            messagebox.showwarning("Invalid", "Enter positive integer.")
            return False
    def apply(self):
        self.result = int(self.e.get())

# ----------------------------------------------------------------------
# GUI
# ----------------------------------------------------------------------
class Connect4GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Connect 4 – AlphaZero Edition")
        self.resizable(False, False)

        # engine & state
        self.mcts_params = {'iterations': DEFAULT_MCTS_ITERATIONS, 'C_param': DEFAULT_PUCT_C}
        self.game = Connect4Game()
        self.nn = NNManager()
        self._load_cfg()
        self.players = {RED_PIECE: None, GREEN_PIECE: None}
        self.score = {'red': 0, 'green': 0, 'draws': 0, 'games': 0}
        self.turn_count = 0
        self.auto_job = None
        self.game_in_progress = False
        self.is_comp = False
        self.paused = False
        self.last_hover = None

        # layout (condensed)
        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0)
        self.canvas = tk.Canvas(main, width=WIDTH, height=HEIGHT, bg=BLUE, highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=3, padx=(0, 10))
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Motion>", self._hover)
        side = ttk.Frame(main)
        side.grid(row=0, column=1, rowspan=3, sticky="ns")
        side.grid_rowconfigure(5, weight=1)

        # Learn toggle (Yes/No)
        self.learn_var = tk.BooleanVar(self, True)
        self.learn_check = ttk.Checkbutton(side, text="Learn (Yes)", variable=self.learn_var,
                                          command=lambda: self.learn_check.config(text=f"Learn ({'Yes' if self.learn_var.get() else 'No'})"))
        self.learn_check.grid(row=2, column=0, columnspan=4, sticky="w", padx=5, pady=5)

        # Status and score labels
        self.status = ttk.Label(side, font=("Helvetica", 16, "bold"), width=25)
        self.status.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0, 5))
        self.score_lbl = ttk.Label(side, font=("Helvetica", 12))
        self.score_lbl.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0, 10))

        ttk.Button(side, text="MCTS Config", command=self._config).grid(row=0, column=3, padx=5)
        ttk.Button(side, text="Train NN", command=self._train).grid(row=1, column=3, padx=5)

        hist = ttk.Frame(side)
        hist.grid(row=3, column=0, columnspan=4, sticky="nsew")
        hist.grid_rowconfigure(1, weight=1)
        hist.grid_columnconfigure(0, weight=1)
        ttk.Label(hist, text="History:").grid(row=0, column=0, sticky="w")
        self.moves = tk.Text(hist, width=28, height=15, font=("Courier", 10), state="disabled")
        scr = ttk.Scrollbar(hist, command=self.moves.yview)
        self.moves['yscrollcommand'] = scr.set
        self.moves.grid(row=1, column=0, sticky="nsew")
        scr.grid(row=1, column=1, sticky="ns")

        ctl = ttk.Frame(side)
        ctl.grid(row=4, column=0, columnspan=4, pady=5)
        ttk.Button(ctl, text="Restart", command=lambda: self._new_game(False)).pack(side="left", padx=5)
        self.stop_btn = ttk.Button(ctl, text="Stop", state="disabled", command=self._pause)
        self.stop_btn.pack(side="left", padx=5)
        self.go_btn = ttk.Button(ctl, text="Continue", state="disabled", command=self._resume)
        self.go_btn.pack(side="left", padx=5)

        self._draw()
        self._set_status("Select players")
        self._update_score()
        self._choose_players()

    # --- drawing helpers ---
    def _mix(self, c1, c2, a):
        r1, g1, b1 = [int(c1[i:i+2], 16) for i in (1, 3, 5)]
        r2, g2, b2 = [int(c2[i:i+2], 16) for i in (1, 3, 5)]
        return f"#{int(r1*(1-a)+r2*a):02x}{int(g1*(1-a)+g2*a):02x}{int(b1*(1-a)+b2*a):02x}"

    def _draw(self):
        self.canvas.delete("all")
        # Draw board background
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                x1 = c * SQUARESIZE; y1 = HEIGHT - (r+1) * SQUARESIZE
                x2 = x1 + SQUARESIZE; y2 = y1 + SQUARESIZE
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=BLUE, outline=BLACK)
                cx = x1 + SQUARESIZE/2; cy = y1 + SQUARESIZE/2
                piece = self.game.board[r, c]
                fill = EMPTY_COLOR if piece == EMPTY else COLOR_MAP[piece]
                self.canvas.create_oval(cx-RADIUS, cy-RADIUS, cx+RADIUS, cy+RADIUS, fill=fill, outline=BLACK)
        # If hovering over a column (for human player), draw a translucent piece at top
        if (self.last_hover is not None and self.game_in_progress and
                isinstance(self.players[self.game.current_player], HumanPlayer)):
            col = self.last_hover
            cx = col * SQUARESIZE + SQUARESIZE/2
            cy = SQUARESIZE/2
            light = self._mix(COLOR_MAP[self.game.current_player], "#FFFFFF", 0.4)
            self.canvas.create_oval(cx-RADIUS, cy-RADIUS, cx+RADIUS, cy+RADIUS, fill=light, outline=BLACK, dash=(3, 3))

    def _set_status(self, msg, color=BLACK):
        self.status.config(text=msg, foreground=color)

    def _update_score(self):
        s = self.score
        self.score_lbl.config(text=f"Red {s['red']}  Green {s['green']}  Draw {s['draws']}  Games {s['games']}")

    # --- mouse events ---
    def _hover(self, e):
        col = e.x // SQUARESIZE
        # Highlight the column if valid move
        self.last_hover = col if 0 <= col < COLUMN_COUNT and self.game.is_valid(col) else None
        self._draw()

    def _click(self, e):
        if (not self.game_in_progress or self.game.game_over or self.paused or
                not isinstance(self.players[self.game.current_player], HumanPlayer)):
            return
        col = e.x // SQUARESIZE
        if self.game.is_valid(col):
            self._make_move(col)

    # --- moves ---
    def _log(self, t):
        self.moves.config(state="normal")
        self.moves.insert("end", t + "\n")
        self.moves.config(state="disabled")
        self.moves.see("end")

    def _make_move(self, col):
        # Drop piece into column, capturing state before move for training
        if not self.game.is_valid(col):
            return
        state_before = self.game.get_state_copy()
        ok, _ = self.game.drop_piece(col)
        if not ok:
            return
        # Record training example unless learning is disabled or already recorded by MCTS
        if self.nn and (self.learn_var.get() or self.is_comp):
            if not isinstance(self.players[self.game.current_player], MCTSComputerPlayer):
                self.nn.add_example(state_before, col)
        mv = (self.turn_count // 2) + 1
        if self.game.current_player == RED_PIECE:
            # Red's move
            if self.game.game_over:
                # Last move of the game (Red wins or draw on Red's move)
                self._log(f"{mv:>3}. {col+1}")
            else:
                # Log Red's move and wait for Green's move
                self.moves.config(state="normal")
                self.moves.insert("end", f"{mv:>3}. {col+1} -- ")
                self.moves.config(state="disabled")
                self.moves.see("end")
        else:
            # Green's move (completes the line started by Red)
            self._log(f"{col+1}")
        self.turn_count += 1
        self._draw()
        if self.game.game_over:
            self._finish()
        else:
            self.game.switch()
            self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", COLOR_MAP[self.game.current_player])
            self.after(30, self._next_turn)

    def _next_turn(self):
        if self.game.game_over or not self.game_in_progress or self.paused:
            return
        ply = self.players[self.game.current_player]
        if isinstance(ply, (RandomComputerPlayer, MCTSComputerPlayer)):
            self._set_status(f"{PLAYER_MAP[self.game.current_player]} (AI) thinking…",
                             COLOR_MAP[self.game.current_player])
            threading.Thread(target=lambda: self._ai_play(ply), daemon=True).start()

    def _ai_play(self, ply):
        mv = ply.get_move(self.game, self)
        # Schedule making the move back on the Tk main thread
        self.after(10, lambda: self._make_move(mv))

    # --- finish & restart ---
    def _finish(self):
        self.game_in_progress = False
        self.score['games'] += 1
        if self.game.winner == 'Draw':
            self.score['draws'] += 1
            self._set_status("Draw")
        else:
            if self.game.winner == RED_PIECE:
                self.score['red'] += 1; name = "Red"
            else:
                self.score['green'] += 1; name = "Green"
            self._set_status(f"{name} wins", COLOR_MAP[self.game.winner])
        self._update_score()
        # Record outcome and train the model (if enabled)
        if self.learn_var.get() or self.is_comp:
            self.nn.finish_game(self.game.winner)
            if self.nn.data['states']:
                self.nn.train(batch_size=32, epochs=3)
        else:
            self.nn.pending = []
        if self.is_comp and not self.paused:
            self.auto_job = self.after(1500, lambda: self._new_game(True))

    def _new_game(self, keep_players):
        if self.auto_job:
            self.after_cancel(self.auto_job)
            self.auto_job = None
        self.game.reset()
        self.turn_count = 0
        self.last_hover = None
        self.moves.config(state="normal")
        self.moves.delete("1.0", "end")
        self.moves.config(state="disabled")
        self.game_in_progress = True
        self.paused = False
        self._draw()
        if keep_players:
            self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", COLOR_MAP[self.game.current_player])
            self.after(30, self._next_turn)
        else:
            self._choose_players()

    # --- dialogs ---
    def _choose_players(self):
        dlg = PlayerDialog(self, self.mcts_params, self.nn)
        if not dlg.result:
            return
        self.players[RED_PIECE] = dlg.result['red']
        self.players[GREEN_PIECE] = dlg.result['green']
        self.is_comp = all(isinstance(p, (RandomComputerPlayer, MCTSComputerPlayer))
                           for p in self.players.values())
        self.game_in_progress = True            # ←← FIX
        self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", 
                         COLOR_MAP[self.game.current_player])
        self.after(30, self._next_turn)

    def _config(self):
        dlg = MCTSDialog(self, self.mcts_params['iterations'], self.mcts_params['C_param'])
        if not dlg.result:
            return
        self.mcts_params = dlg.result
        json.dump(self.mcts_params, open(MCTS_CONFIG_FILE, "w"), indent=4)
        messagebox.showinfo("Saved", "MCTS parameters updated.")

    def _train(self):
        dlg = TrainDialog(self); n = getattr(dlg, 'result', None)
        if not n:
            return
        # Training progress window
        win = tk.Toplevel(self)
        win.title("Training"); win.resizable(False, False)
        ttk.Label(win, text=f"Running {n} self-play games…").pack(pady=10)
        bar = ttk.Progressbar(win, length=300, mode="determinate", maximum=n)
        bar.pack(pady=5)
        prog = tk.StringVar(value=f"0 / {n}")
        ttk.Label(win, textvariable=prog).pack(pady=5)
        def worker():
            aiR = MCTSComputerPlayer(self.mcts_params['iterations'], self.mcts_params['C_param'], self.nn)
            aiG = MCTSComputerPlayer(self.mcts_params['iterations'], self.mcts_params['C_param'], self.nn)
            for i in range(n):
                g = Connect4Game(); cur = RED_PIECE
                while not g.game_over:
                    mv = (aiR if cur == RED_PIECE else aiG).get_move(g)
                    ok, _ = g.drop_piece(mv)
                    if ok and not g.game_over:
                        g.switch(); cur = g.current_player
                self.nn.finish_game(g.winner)
                # Update progress bar in UI thread
                def _upd(idx=i):
                    bar.configure(value=idx+1)
                    prog.set(f"{idx+1} / {n}")
                self.after(0, _upd)
            # Train on all collected self-play data (more epochs for a batch of games)
            self.nn.train(batch_size=32, epochs=10)
            self.after(0, win.destroy)
            self.after(0, lambda: messagebox.showinfo("Done", "Training finished."))
        threading.Thread(target=worker, daemon=True).start()

    # --- pause / resume ---
    def _pause(self):
        if not self.is_comp:
            return
        self.paused = True
        self.stop_btn['state'] = "disabled"
        self.go_btn['state'] = "normal"
        if self.auto_job:
            self.after_cancel(self.auto_job)
            self.auto_job = None
        self._set_status("Match paused")

    def _resume(self):
        if not self.paused:
            return
        self.paused = False
        self.stop_btn['state'] = "normal"
        self.go_btn['state'] = "disabled"
        if self.game.game_over:
            self._new_game(True)
        else:
            self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", 
                             COLOR_MAP[self.game.current_player])
            self._next_turn()

    # --- cfg loader ---
    def _load_cfg(self):
        if os.path.exists(MCTS_CONFIG_FILE):
            try:
                cfg = json.load(open(MCTS_CONFIG_FILE))
                if (isinstance(cfg.get('iterations'), int) and cfg['iterations'] > 0 and
                        isinstance(cfg.get('C_param'), (int, float)) and cfg['C_param'] >= 0):
                    self.mcts_params = cfg
            except:
                pass

# ----------------------------------------------------------------------
if __name__ == "__main__":
    Connect4GUI().mainloop()
