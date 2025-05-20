#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connect‑4 with AlphaZero‑style self‑play (PUCT MCTS + CNN policy‑value NN)
Corrected & enhanced – April 2025 overhaul.

Improvements implemented
------------------------
1. **CNN architecture** – spatial 3‑plane board encoding → deeper conv net.
2. **Single NN call per simulation** – evaluate leaf only once; value + priors returned together.
3. **Policy targets = root visit distribution** instead of chosen move.
4. **Training loss** – custom cross‑entropy for distribution + value MSE.
5. **UI safeguards** – Training disabled during active games; Learn toggle forced on for AI‑vs‑AI & grayed‑out.
6. **Clearer Learn toggle semantics** – user‑driven games respect switch fully.
7. **Minor clean‑ups** – removed redundant value inversion; small refactors.
8. **UI enhancements** – Settings cog, tooltips, training improvements

(C) 2025 – released under MIT License.
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

DEFAULT_MCTS_ITERATIONS = 800
DEFAULT_PUCT_C          = 1.25
DEFAULT_TRAIN_GAMES     = 200
NN_MODEL_FILE           = "C4_v4.pt"
MCTS_CONFIG_FILE        = "mcts_config.json"
MAX_TRAINING_EXAMPLES   = 30_000

# ----------------------------------------------------------------------
# Tooltip class for UI enhancements
# ----------------------------------------------------------------------
class ToolTip:
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)
    
    def show_tooltip(self, event=None):
        # Get widget position - works with all widget types
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
        # Create top level window
        self.tooltip = tk.Toplevel(self.widget)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x}+{y}")
        
        label = ttk.Label(self.tooltip, text=self.text, background="#ffffe0", relief="solid", borderwidth=1)
        label.pack(ipadx=5, ipady=5)
    
    def hide_tooltip(self, event=None):
        if self.tooltip:
            self.tooltip.destroy()
            self.tooltip = None

# ----------------------------------------------------------------------
# Neural network (CNN policy + value heads)
# ----------------------------------------------------------------------
class Connect4CNN(nn.Module):
    """3 input planes (own pieces, opp pieces, player plane) → conv stack → policy & value"""
    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)

        flat = channels * ROW_COUNT * COLUMN_COUNT
        # policy head
        self.fc_pol = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, COLUMN_COUNT)
        )
        # value head
        self.fc_val = nn.Sequential(
            nn.Linear(flat, 256), nn.ReLU(), nn.Linear(256, 1), nn.Tanh()
        )

    def forward(self, x):
        # x: [B, 3, 6, 7]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = x.view(x.size(0), -1)
        p = self.fc_pol(x)
        v = self.fc_val(x)
        return p, v.squeeze(1)

class Connect4Dataset(Dataset):
    def __init__(self, s, p, v):
        self.s, self.p, self.v = s, p, v
    def __len__(self):
        return len(self.s)
    def __getitem__(self, i):
        return self.s[i], self.p[i], self.v[i]

# ----------------------------------------------------------------------
# Numba helpers (same as before)
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
# Core game state (unchanged except small refactor)
# ----------------------------------------------------------------------
class Connect4Game:
    def __init__(self):
        self.reset()
    def reset(self):
        self.board = np.zeros((ROW_COUNT, COLUMN_COUNT), dtype=np.int8)
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
        self.board[r, c] = self.current_player
        if _win(self.board, self.current_player, ROW_COUNT, COLUMN_COUNT):
            self.game_over = True; self.winner = self.current_player
        elif _draw(self.board):
            self.game_over = True; self.winner = 'Draw'
        return True, r
    def switch(self):
        self.current_player = GREEN_PIECE if self.current_player == RED_PIECE else RED_PIECE
    def copy(self):
        g = Connect4Game()
        g.board = self.board.copy(); g.current_player = self.current_player
        g.game_over = self.game_over; g.winner = self.winner
        return g
    get_state_copy = copy

# ----------------------------------------------------------------------
# Neural‑network manager (policy distribution targets)
# ----------------------------------------------------------------------
class NNManager:
    def __init__(self, hyperparams=None):
        self.hyperparams = hyperparams or {'learning_rate': 1e-3, 'batch_size': 64, 'epochs': 5}
        self.net = Connect4CNN()
        self.opt = optim.Adam(self.net.parameters(), lr=self.hyperparams['learning_rate'])
        self.data = {'states': [], 'policies': [], 'values': []}
        self.pending = []  # moves from current game
        if os.path.exists(NN_MODEL_FILE):
            ck = torch.load(NN_MODEL_FILE, map_location='cpu', weights_only=True)
            self.net.load_state_dict(ck['model_state_dict'])
            self.opt.load_state_dict(ck['optimizer_state_dict'])
            print("Model loaded.")
        else:
            print("No existing model – new weights.")

    # 3‑plane tensor
    @staticmethod
    def _tensor(state: 'Connect4Game'):
        red_plane   = (state.board == RED_PIECE).astype(np.float32)
        green_plane = (state.board == GREEN_PIECE).astype(np.float32)
        turn_plane  = np.full_like(red_plane, 1.0 if state.current_player == RED_PIECE else 0.0)
        stacked = np.stack([red_plane, green_plane, turn_plane])  # [3,6,7]
        return torch.tensor(stacked, dtype=torch.float32)

    # inference
    def policy_value(self, state: 'Connect4Game'):
        self.net.eval()
        with torch.no_grad():
            t = self._tensor(state).unsqueeze(0)  # [1,3,6,7]
            logits, v = self.net(t)
            logits = logits.squeeze(0)  # [7]
            valid = state.valid_moves()
            mask = torch.full_like(logits, -1e9)
            mask[valid] = 0.0
            logits = logits + mask  # illegal moves neg‑inf
            probs = F.softmax(logits, dim=0).cpu().numpy()
            return probs, float(v.item())

    # called by MCTS: add example with state and full visit distribution
    def add_example(self, state, visit_probs):
        self.pending.append({'state': self._tensor(state), 'policy': torch.tensor(visit_probs, dtype=torch.float32),
                             'player': state.current_player})

    def finish_game(self, winner):
        for ex in self.pending:
            if winner == 'Draw':
                value = 0.0
            elif winner == ex['player']:
                value = 1.0
            else:
                value = -1.0
            self.data['states'].append(ex['state'])
            self.data['policies'].append(ex['policy'])
            self.data['values'].append(torch.tensor([value], dtype=torch.float32))
        self.pending.clear()
        # trim buffer
        if len(self.data['states']) > MAX_TRAINING_EXAMPLES:
            n = len(self.data['states']) - MAX_TRAINING_EXAMPLES
            for k in self.data:
                self.data[k] = self.data[k][n:]

    # custom loss with distribution targets
    def train(self, batch_size=None, epochs=None):
        if not self.data['states']:
            return
        batch_size = batch_size or self.hyperparams['batch_size']
        epochs = epochs or self.hyperparams['epochs']
        ds = Connect4Dataset(torch.stack(self.data['states']),
                             torch.stack(self.data['policies']),
                             torch.stack(self.data['values']).squeeze(1))
        dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        self.net.train()
        for ep in range(epochs):
            p_loss_sum = v_loss_sum = 0.0
            for s, p_target, v_target in dl:
                self.opt.zero_grad()
                logits, v_pred = self.net(s)
                log_probs = F.log_softmax(logits, dim=1)
                policy_loss = -(p_target * log_probs).sum(dim=1).mean()
                value_loss = F.mse_loss(v_pred, v_target)
                loss = policy_loss + value_loss
                loss.backward(); self.opt.step()
                p_loss_sum += policy_loss.item(); v_loss_sum += value_loss.item()
            print(f"[Train] Epoch {ep+1}/{epochs}  policy={p_loss_sum/len(dl):.4f}  value={v_loss_sum/len(dl):.4f}")
        # save
        torch.save({'model_state_dict': self.net.state_dict(), 'optimizer_state_dict': self.opt.state_dict()},
                   NN_MODEL_FILE)

# ----------------------------------------------------------------------
# MCTS with single NN call per simulation, storing visit counts
# ----------------------------------------------------------------------
class TreeNode:
    __slots__ = ('state', 'parent', 'move', 'prior', 'children', 'visits', 'value_sum', 'player')
    def __init__(self, state: 'Connect4Game', parent=None, move=None, prior=0.0):
        self.state = state; self.parent = parent; self.move = move; self.prior = prior
        self.children = {}  # move → child
        self.visits = 0; self.value_sum = 0.0
        self.player = state.current_player  # player to play at this node

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def u(self, c_puct, total_visits):
        return c_puct * self.prior * math.sqrt(total_visits) / (1 + self.visits)

    def best_child(self, c_puct):
        total = self.visits
        return max(self.children.values(), key=lambda n: n.q() + n.u(c_puct, total))

class MCTS:
    def __init__(self, iterations, c_puct, nn: NNManager, explore=True):
        self.I = iterations; self.c = c_puct; self.nn = nn; self.explore = explore

    def search(self, root_state: 'Connect4Game'):
        root = TreeNode(root_state.copy())
        # initial prior via NN + Dirichlet noise
        prior, _ = self.nn.policy_value(root.state)
        valid = root.state.valid_moves()
        if self.explore:
            dirichlet = np.random.dirichlet([0.3]*len(valid))
        for i, m in enumerate(valid):
            p = prior[m] if not self.explore else 0.75 * prior[m] + 0.25 * dirichlet[i]
            ns = root.state.copy(); ns.drop_piece(m)
            if not ns.game_over: ns.switch()
            root.children[m] = TreeNode(ns, parent=root, move=m, prior=p)

        for _ in range(self.I):
            node = root
            # SELECT
            while node.children:
                node = node.best_child(self.c)
            # EXPAND & EVALUATE (if game not over)
            if not node.state.game_over:
                probs, value = self.nn.policy_value(node.state)
                valid = node.state.valid_moves()
                for m in valid:
                    if m not in node.children:
                        ns = node.state.copy(); ns.drop_piece(m)
                        if not ns.game_over: ns.switch()
                        node.children[m] = TreeNode(ns, parent=node, move=m, prior=probs[m])
            else:
                # terminal node value
                if node.state.winner == 'Draw':
                    value = 0.0
                else:
                    value = 1.0 if node.state.winner == node.player else -1.0
            # BACKPROP
            cur = node
            while cur:
                cur.visits += 1
                cur.value_sum += value if cur.player == node.player else -value
                cur = cur.parent

        # Move selection by visit count
        visit_counts = np.zeros(COLUMN_COUNT, dtype=np.float32)
        for m, child in root.children.items():
            visit_counts[m] = child.visits
        visit_probs = visit_counts / visit_counts.sum()

        # record training example before making the move
        self.nn.add_example(root_state, visit_probs)
        # choose move (argmax visits)
        best_move = int(visit_counts.argmax())
        return best_move

# ----------------------------------------------------------------------
# Player classes (Human, Random, MCTS)
# ----------------------------------------------------------------------
class Player: pass
class HumanPlayer(Player): pass
class RandomComputerPlayer(Player):
    def get_move(self, state, gui=None):
        time.sleep(0.1)
        return random.choice(state.valid_moves())
class MCTSComputerPlayer(Player):
    def __init__(self, iters, c, nn: NNManager, explore=True):
        self.mcts = MCTS(iters, 0.0 if not explore else c, nn, explore=explore)
    def get_move(self, state, gui=None):
        start = time.time()
        mv = self.mcts.search(state)
        dt = time.time() - start
        if dt < 0.1:
            time.sleep(0.1 - dt)
        return mv

# ----------------------------------------------------------------------
# Dialogs (updated to include settings dialog)
# ----------------------------------------------------------------------
class PlayerDialog(simpledialog.Dialog):
    def __init__(self, master, mcts_params, nn):
        self.mcts = mcts_params; self.nn = nn
        self.p1 = tk.StringVar(master, "Human")
        self.p2 = tk.StringVar(master, "Computer (MCTS)")
        super().__init__(master, "Select Players")
    def body(self, m):
        opts = ["Human", "Computer (Random)", "Computer (MCTS)"]
        ttk.Label(m, text="Red:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        for i,o in enumerate(opts,1):
            rb = ttk.Radiobutton(m,text=o,variable=self.p1,value=o)
            rb.grid(row=0,column=i,sticky="w")
            ToolTip(rb, f"Select {o} for Red player")
            
        ttk.Label(m, text="Green:").grid(row=1,column=0,sticky="w", padx=5, pady=5)
        for i,o in enumerate(opts,1):
            rb = ttk.Radiobutton(m,text=o,variable=self.p2,value=o)
            rb.grid(row=1,column=i,sticky="w")
            ToolTip(rb, f"Select {o} for Green player")
    def apply(self):
        def mk(sel):
            if sel=="Human": return HumanPlayer()
            if sel=="Computer (Random)": return RandomComputerPlayer()
            return MCTSComputerPlayer(self.mcts['iterations'], self.mcts['C_param'], self.nn,
                                     explore=not self.master.fullplay_var.get())
        self.result={'red':mk(self.p1.get()),'green':mk(self.p2.get())}

class SettingsDialog(simpledialog.Dialog):
    def __init__(self, master, mcts_params, train_games=200, nn_params=None):
        self.it = tk.StringVar(master, str(mcts_params['iterations']))
        self.c = tk.StringVar(master, f"{mcts_params['C_param']:.2f}")
        self.games = tk.StringVar(master, str(train_games))
        self.nn_params = nn_params or {'learning_rate': 1e-3, 'batch_size': 64, 'epochs': 5}
        self.lr = tk.StringVar(master, str(self.nn_params.get('learning_rate', 1e-3)))
        self.batch = tk.StringVar(master, str(self.nn_params.get('batch_size', 64)))
        self.epochs = tk.StringVar(master, str(self.nn_params.get('epochs', 5)))
        super().__init__(master, "Settings")
        
    def body(self, m):
        # Create notebook with tabs
        nb = ttk.Notebook(m)
        nb.pack(fill='both', expand=True, padx=5, pady=5)
        
        # MCTS tab
        mcts_tab = ttk.Frame(nb)
        nb.add(mcts_tab, text="MCTS")
        
        ttk.Label(mcts_tab, text="Iterations:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        e1 = ttk.Entry(mcts_tab, textvariable=self.it, width=10)
        e1.grid(row=0, column=1, padx=5)
        ToolTip(e1, "Number of MCTS simulations per move\nHigher = stronger but slower")
        
        ttk.Label(mcts_tab, text="Exploration C:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        e2 = ttk.Entry(mcts_tab, textvariable=self.c, width=10)
        e2.grid(row=1, column=1, padx=5)
        ToolTip(e2, "PUCT exploration parameter\nHigher = more exploration")
        
        # Training tab
        train_tab = ttk.Frame(nb)
        nb.add(train_tab, text="Training")
        
        ttk.Label(train_tab, text="Self-play games:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        e3 = ttk.Entry(train_tab, textvariable=self.games, width=10)
        e3.grid(row=0, column=1, padx=5)
        ToolTip(e3, "Number of self-play games for training")
        
        # Advanced NN tab
        nn_tab = ttk.Frame(nb)
        nb.add(nn_tab, text="Advanced")
        
        ttk.Label(nn_tab, text="Learning rate:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        e4 = ttk.Entry(nn_tab, textvariable=self.lr, width=10)
        e4.grid(row=0, column=1, padx=5)
        ToolTip(e4, "Neural network learning rate\nTypical values: 0.001-0.0001")
        
        ttk.Label(nn_tab, text="Batch size:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        e5 = ttk.Entry(nn_tab, textvariable=self.batch, width=10)
        e5.grid(row=1, column=1, padx=5)
        ToolTip(e5, "Training batch size\nHigher = faster but uses more memory")
        
        ttk.Label(nn_tab, text="Training epochs:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        e6 = ttk.Entry(nn_tab, textvariable=self.epochs, width=10)
        e6.grid(row=2, column=1, padx=5)
        ToolTip(e6, "Number of passes through training data\nMore epochs = better learning but slower")
        
        return mcts_tab  # Initial focus
    
    def validate(self):
        try:
            # Validate MCTS params
            it = int(self.it.get())
            c = float(self.c.get())
            games = int(self.games.get())
            lr = float(self.lr.get())
            batch = int(self.batch.get())
            epochs = int(self.epochs.get())
            
            if it <= 0 or c < 0 or games <= 0 or lr <= 0 or batch <= 0 or epochs <= 0:
                messagebox.showwarning("Invalid", "All values must be positive.")
                return False
            return True
        except:
            messagebox.showwarning("Invalid", "Enter valid numbers.")
            return False
    
    def apply(self):
        self.result = {
            'mcts': {
                'iterations': int(self.it.get()),
                'C_param': float(self.c.get())
            },
            'training': {
                'games': int(self.games.get())
            },
            'nn_params': {
                'learning_rate': float(self.lr.get()),
                'batch_size': int(self.batch.get()),
                'epochs': int(self.epochs.get())
            }
        }

class TrainDialog(simpledialog.Dialog):
    def __init__(self, master):
        self.e = tk.StringVar(master, str(master.train_games))
        super().__init__(master, "Train NN")
    def body(self, m):
        ttk.Label(m,text="Self‑play games:").grid(row=0,column=0,sticky="w",padx=5,pady=5)
        e = ttk.Entry(m,textvariable=self.e,width=10)
        e.grid(row=0,column=1,padx=5)
        ToolTip(e, "Number of AI vs AI games to play for training")
    def validate(self):
        try:
            return int(self.e.get())>0
        except:
            messagebox.showwarning("Invalid","Enter positive integer.")
            return False
    def apply(self):
        self.result=int(self.e.get())

# ----------------------------------------------------------------------
# GUI – Connect4GUI (updated for new features)
# ----------------------------------------------------------------------
class Connect4GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Connect 4 – AlphaZero Edition (2025)")
        self.resizable(False, False)

        self.mcts_params = {'iterations': DEFAULT_MCTS_ITERATIONS, 'C_param': DEFAULT_PUCT_C}
        self.nn_params = {'learning_rate': 1e-3, 'batch_size': 64, 'epochs': 5}
        self.train_games = DEFAULT_TRAIN_GAMES
        self.training_in_progress = False
        self.training_stop_requested = False
        
        self.nn = NNManager(self.nn_params)
        self._load_cfg()

        self.game = Connect4Game()
        self.players = {RED_PIECE: None, GREEN_PIECE: None}
        self.score = {'red':0,'green':0,'draws':0,'games':0}
        self.turn_count=0; self.auto_job=None; self.game_in_progress=False
        self.is_comp=False; self.paused=False; self.last_hover=None

        main = ttk.Frame(self, padding=10); main.grid(row=0,column=0)
        self.canvas = tk.Canvas(main,width=WIDTH,height=HEIGHT,bg=BLUE,highlightthickness=0)
        self.canvas.grid(row=0,column=0,rowspan=3,padx=(0,10))
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Motion>", self._hover)
        ToolTip(self.canvas, "Click a column to drop a piece")
        
        side = ttk.Frame(main); side.grid(row=0,column=1,rowspan=3,sticky="ns")
        side.grid_rowconfigure(6,weight=1)

        self.learn_var = tk.BooleanVar(self, True)
        self.learn_check = ttk.Checkbutton(side,text="Learn (Yes)",variable=self.learn_var,
            command=self._update_learn_label)
        self.learn_check.grid(row=2,column=0,columnspan=2,sticky="w",padx=5,pady=5)
        ToolTip(self.learn_check, "When enabled, game moves are used to train the neural network")

        self.fullplay_var = tk.BooleanVar(self, False)
        self.fullplay_check = ttk.Checkbutton(side,text="Full Play (Off)",variable=self.fullplay_var,
            command=self._update_fullplay_label)
        self.fullplay_check.grid(row=2,column=2,columnspan=2,sticky="w",padx=5,pady=5)
        ToolTip(self.fullplay_check, "When enabled, AI plays at full strength without exploration noise")
        
        self.status = ttk.Label(side,font=("Helvetica",16,"bold"),width=25)
        self.status.grid(row=0,column=0,columnspan=3,sticky="ew",pady=(0,5))
        ToolTip(self.status, "Game status")
        
        self.score_lbl = ttk.Label(side,font=("Helvetica",12))
        self.score_lbl.grid(row=1,column=0,columnspan=3,sticky="ew",pady=(0,10))
        ToolTip(self.score_lbl, "Score summary")

        # Replace MCTS Config button with Settings cog
        settings_btn = ttk.Button(side, text="⚙", width=3, command=self._settings)
        settings_btn.grid(row=0, column=3, padx=5)
        ToolTip(settings_btn, "Settings (MCTS, Training, Advanced)")
        
        self.train_btn = ttk.Button(side,text="Train NN",command=self._train)
        self.train_btn.grid(row=1,column=3,padx=5)
        ToolTip(self.train_btn, "Run AI self-play games to train the neural network")

        # history text
        hist = ttk.Frame(side); hist.grid(row=3,column=0,columnspan=4,sticky="nsew")
        hist.grid_rowconfigure(1,weight=1); hist.grid_columnconfigure(0,weight=1)
        ttk.Label(hist,text="History:").grid(row=0,column=0,sticky="w")
        self.moves = tk.Text(hist,width=28,height=15,font=("Courier",10),state="disabled")
        scr = ttk.Scrollbar(hist,command=self.moves.yview); self.moves['yscrollcommand']=scr.set
        self.moves.grid(row=1,column=0,sticky="nsew"); scr.grid(row=1,column=1,sticky="ns")
        ToolTip(self.moves, "Game move history")

        ctl = ttk.Frame(side); ctl.grid(row=4,column=0,columnspan=4,pady=5)
        restart_btn = ttk.Button(ctl,text="Restart",command=lambda:self._new_game(False))
        restart_btn.pack(side="left",padx=5)
        ToolTip(restart_btn, "Start a new game")
        
        self.stop_btn = ttk.Button(ctl,text="Stop",state="disabled",command=self._pause)
        self.stop_btn.pack(side="left",padx=5)
        ToolTip(self.stop_btn, "Pause the current game or stop training")
        
        self.go_btn = ttk.Button(ctl,text="Continue",state="disabled",command=self._resume)
        self.go_btn.pack(side="left",padx=5)
        ToolTip(self.go_btn, "Resume the paused game")

        self._draw(); self._set_status("Select players"); self._update_score(); self._choose_players()

    # ----- UI helpers -----
    def _update_learn_label(self):
        self.learn_check.config(text=f"Learn ({'Yes' if self.learn_var.get() else 'No'})")

    def _update_fullplay_label(self):
        self.fullplay_check.config(text=f"Full Play ({'On' if self.fullplay_var.get() else 'Off'})")
        
    # ----- drawing board, hover, click 
    def _mix(self, c1, c2, a):
        r1,g1,b1=[int(c1[i:i+2],16) for i in (1,3,5)]
        r2,g2,b2=[int(c2[i:i+2],16) for i in (1,3,5)]
        return f"#{int(r1*(1-a)+r2*a):02x}{int(g1*(1-a)+g2*a):02x}{int(b1*(1-a)+b2*a):02x}"

    def _draw(self):
        self.canvas.delete("all")
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                x1=c*SQUARESIZE; y1=HEIGHT-(r+1)*SQUARESIZE; x2=x1+SQUARESIZE; y2=y1+SQUARESIZE
                self.canvas.create_rectangle(x1,y1,x2,y2,fill=BLUE,outline=BLACK)
                cx=x1+SQUARESIZE/2; cy=y1+SQUARESIZE/2
                piece=self.game.board[r,c]
                fill=EMPTY_COLOR if piece==EMPTY else COLOR_MAP[piece]
                self.canvas.create_oval(cx-RADIUS,cy-RADIUS,cx+RADIUS,cy+RADIUS,fill=fill,outline=BLACK)
        if self.last_hover is not None and self.game_in_progress and isinstance(self.players[self.game.current_player],HumanPlayer):
            col=self.last_hover; cx=col*SQUARESIZE+SQUARESIZE/2; cy=SQUARESIZE/2
            light=self._mix(COLOR_MAP[self.game.current_player],"#FFFFFF",0.4)
            self.canvas.create_oval(cx-RADIUS,cy-RADIUS,cx+RADIUS,cy+RADIUS,fill=light,outline=BLACK,dash=(3,3))

    def _set_status(self,msg,color=BLACK):
        self.status.config(text=msg,foreground=color)
    def _update_score(self):
        s=self.score; self.score_lbl.config(text=f"Red {s['red']}  Green {s['green']}  Draw {s['draws']}  Games {s['games']}")

    # hover/click 
    def _hover(self,e):
        col=e.x//SQUARESIZE; self.last_hover=col if 0<=col<COLUMN_COUNT and self.game.is_valid(col) else None; self._draw()
    def _click(self,e):
        if not self.game_in_progress or self.game.game_over or self.paused or not isinstance(self.players[self.game.current_player],HumanPlayer):
            return
        col=e.x//SQUARESIZE
        if self.game.is_valid(col):
            self._make_move(col)

    # history log helper
    def _log(self,t):
        self.moves.config(state="normal"); self.moves.insert("end",t+"\n"); self.moves.config(state="disabled"); self.moves.see("end")

    # making a move 
    def _make_move(self,col):
        if not self.game.is_valid(col): return
        state_before=self.game.get_state_copy()
        ok,_=self.game.drop_piece(col)
        if not ok: return
        # Only add example for human/random move when Learn=Yes
        if self.nn and self.learn_var.get() and not isinstance(self.players[self.game.current_player],MCTSComputerPlayer):
            # placeholder – visit_probs will be filled with one‑hot (since no search)
            vp=np.zeros(COLUMN_COUNT,dtype=np.float32); vp[col]=1.0
            self.nn.add_example(state_before,vp)
        mv=(self.turn_count//2)+1
        if self.game.current_player==RED_PIECE:
            if self.game.game_over:
                self._log(f"{mv:>3}. {col+1}")
            else:
                self.moves.config(state="normal"); self.moves.insert("end",f"{mv:>3}. {col+1} -- "); self.moves.config(state="disabled"); self.moves.see("end")
        else:
            self._log(f"{col+1}")
        self.turn_count+=1; self._draw()
        if self.game.game_over:
            self._finish()
        else:
            self.game.switch(); self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn",COLOR_MAP[self.game.current_player]); self.after(30,self._next_turn)

    # AI turns
    def _next_turn(self):
        if self.game.game_over or not self.game_in_progress or self.paused: return
        ply=self.players[self.game.current_player]
        if isinstance(ply,(RandomComputerPlayer,MCTSComputerPlayer)):
            self._set_status(f"{PLAYER_MAP[self.game.current_player]} (AI) thinking…",COLOR_MAP[self.game.current_player])
            threading.Thread(target=lambda:self._ai_play(ply),daemon=True).start()
    def _ai_play(self,ply):
        mv=ply.get_move(self.game,self); self.after(10,lambda:self._make_move(mv))

    # finish & restart 
    def _finish(self):
        self.game_in_progress=False; self.score['games']+=1
        if self.game.winner=='Draw': self.score['draws']+=1; self._set_status("Draw")
        else:
            if self.game.winner==RED_PIECE: self.score['red']+=1; name="Red"
            else: self.score['green']+=1; name="Green"
            self._set_status(f"{name} wins",COLOR_MAP[self.game.winner])
        self._update_score()
        # learning decisions
        if self.is_comp or self.learn_var.get():
            self.nn.finish_game(self.game.winner); self.nn.train()
        else:
            self.nn.pending.clear()
        if self.is_comp and not self.paused:
            self.auto_job=self.after(1200,lambda:self._new_game(True))

    def _new_game(self,keep_players):
        if self.auto_job: self.after_cancel(self.auto_job); self.auto_job=None
        self.game.reset(); self.turn_count=0; self.last_hover=None
        self.moves.config(state="normal"); self.moves.delete("1.0","end"); self.moves.config(state="disabled")
        self.game_in_progress=True; self.paused=False; self._draw()
        if keep_players:
            self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn",COLOR_MAP[self.game.current_player]); self.after(30,self._next_turn)
        else:
            self._choose_players()

    # choose players
    def _choose_players(self):
        dlg=PlayerDialog(self,self.mcts_params,self.nn)
        if not dlg.result: return
        self.players[RED_PIECE]=dlg.result['red']; self.players[GREEN_PIECE]=dlg.result['green']
        self.is_comp=all(isinstance(p,(RandomComputerPlayer,MCTSComputerPlayer)) for p in self.players.values())
        if self.is_comp:
            self.stop_btn['state'] = 'normal'
            self.go_btn['state'] = 'disabled'
        else:
            self.stop_btn['state'] = 'disabled'
            self.go_btn['state'] = 'disabled'
        if self.is_comp:
            self.learn_var.set(True); self.learn_check.state(['disabled'])
        else:
            self.learn_check.state(['!disabled'])
        self._update_learn_label()
        self.game_in_progress=True; self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn",COLOR_MAP[self.game.current_player]); self.after(30,self._next_turn)

    # NEW: Settings dialog replacing MCTS config
    def _settings(self):
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Pause the game before changing settings.")
            return
        
        dlg = SettingsDialog(self, self.mcts_params, self.train_games, self.nn_params)
        if dlg.result:
            self.mcts_params = dlg.result['mcts']
            self.train_games = dlg.result['training']['games']
            old_lr = self.nn_params.get('learning_rate')
            self.nn_params = dlg.result['nn_params']
            
            # Update NN optimizer with new learning rate if it changed
            if old_lr != self.nn_params['learning_rate']:
                for param_group in self.nn.opt.param_groups:
                    param_group['lr'] = self.nn_params['learning_rate']
            
            # Save to config file
            cfg = {
                'mcts': self.mcts_params,
                'training': {'games': self.train_games},
                'nn_params': self.nn_params
            }
            json.dump(cfg, open(MCTS_CONFIG_FILE, "w"), indent=4)
            messagebox.showinfo("Saved", "Settings updated.")
    
    # ENHANCED: Training with stop capability
    def _train(self):
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Finish or pause the current game before training.")
            return
        
        if self.training_in_progress:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        dlg = TrainDialog(self)
        n = getattr(dlg, 'result', None)
        if not n:
            return
        
        self.train_btn['state'] = "disabled"
        self.training_in_progress = True
        self.training_stop_requested = False
        self.stop_btn['state'] = "normal"  # Enable stop button during training
        
        win = tk.Toplevel(self)
        win.title("Training")
        win.resizable(False, False)
        ttk.Label(win, text=f"Running {n} self-play games...").pack(pady=10)
        bar = ttk.Progressbar(win, length=300, mode="determinate", maximum=n)
        bar.pack(pady=5)
        prog = tk.StringVar(value=f"0 / {n}")  # Fixed f-string
        ttk.Label(win, textvariable=prog).pack(pady=5)
        
        def worker():
            aiR = MCTSComputerPlayer(self.mcts_params['iterations'], self.mcts_params['C_param'], self.nn)
            aiG = MCTSComputerPlayer(self.mcts_params['iterations'], self.mcts_params['C_param'], self.nn)
            i = 0
            while i < n and not self.training_stop_requested:
                g = Connect4Game()
                cur = RED_PIECE
                while not g.game_over and not self.training_stop_requested:
                    mv = (aiR if cur == RED_PIECE else aiG).get_move(g)
                    ok, _ = g.drop_piece(mv)
                    if ok and not g.game_over:
                        g.switch()
                        cur = g.current_player
                if not self.training_stop_requested:
                    self.nn.finish_game(g.winner)
                    i += 1
                    self.after(0, lambda idx=i: (bar.configure(value=idx), prog.set(f"{idx} / {n}")))
            
            # Train with data collected so far
            if i > 0:
                self.nn.train(batch_size=self.nn_params.get('batch_size', 64),
                              epochs=self.nn_params.get('epochs', 5))
            
            self.after(0, win.destroy)
            self.after(0, lambda: self.train_btn.config(state="normal"))
            self.after(0, lambda: setattr(self, 'training_in_progress', False))
            if self.training_stop_requested:
                messagebox.showinfo("Stopped", f"Training stopped after {i} games.")
            else:
                messagebox.showinfo("Done", "Training finished.")
            self.training_stop_requested = False
            
            # Reset stop button state if not in computer vs computer game
            if not self.is_comp:
                self.stop_btn['state'] = "disabled"
        
        threading.Thread(target=worker, daemon=True).start()

    # ENHANCED: Pause/stop for both games and training
    def _pause(self):
        if self.training_in_progress:
            self.training_stop_requested = True
            self.stop_btn['state'] = "disabled"
            self._set_status("Training stopping...")
            return
        
        if not self.is_comp: return
        self.paused=True; self.stop_btn['state']="disabled"; self.go_btn['state']="normal"
        if self.auto_job: self.after_cancel(self.auto_job); self.auto_job=None
        self._set_status("Match paused")
        
    def _resume(self):
        if not self.paused: return
        self.paused=False; self.stop_btn['state']="normal"; self.go_btn['state']="disabled"
        if self.game.game_over: self._new_game(True)
        else:
            self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn",COLOR_MAP[self.game.current_player]); self._next_turn()

    # ENHANCED: Config loader for new settings format
    def _load_cfg(self):
        if os.path.exists(MCTS_CONFIG_FILE):
            try:
                cfg = json.load(open(MCTS_CONFIG_FILE))
                # Handle old format
                if 'iterations' in cfg and 'C_param' in cfg:
                    self.mcts_params = cfg
                # Handle new format
                elif 'mcts' in cfg:
                    self.mcts_params = cfg['mcts']
                    if 'training' in cfg and 'games' in cfg['training']:
                        self.train_games = cfg['training']['games']
                    if 'nn_params' in cfg:
                        self.nn_params = cfg['nn_params']
                        # Update NN learning rate
                        for param_group in self.nn.opt.param_groups:
                            param_group['lr'] = self.nn_params['learning_rate']
            except:
                pass

# ----------------------------------------------------------------------
if __name__=='__main__':
    Connect4GUI().mainloop()