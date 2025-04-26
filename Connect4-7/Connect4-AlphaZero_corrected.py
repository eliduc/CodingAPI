#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connect‑4 with AlphaZero‑style self‑play (PUCT MCTS + CNN policy‑value NN)
Corrected & enhanced – April 2025 overhaul.

Improvements implemented
------------------------
1. **CNN architecture** – spatial 3‑plane board encoding → deeper conv net.
2. **Single NN call per simulation** – evaluate leaf only once; value + priors returned together.
3. **Policy targets = root visit distribution** instead of chosen move.
4. **Training loss** – custom cross‑entropy for distribution + value MSE.
5. **UI safeguards** – Training disabled during active games; Learn toggle forced on for AI‑vs‑AI & grayed‑out.
6. **Clearer Learn toggle semantics** – user‑driven games respect switch fully.
7. **Minor clean‑ups** – removed redundant value inversion; small refactors.

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
NN_MODEL_FILE           = "C4_NN.pt"
MCTS_CONFIG_FILE        = "mcts_config.json"
MAX_TRAINING_EXAMPLES   = 30_000

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
    def __init__(self):
        self.net = Connect4CNN()
        self.opt = optim.Adam(self.net.parameters(), lr=1e-3)
        self.data = {'states': [], 'policies': [], 'values': []}
        self.pending = []  # moves from current game
        if os.path.exists(NN_MODEL_FILE):
            ck = torch.load(NN_MODEL_FILE, map_location='cpu')
            self.net.load_state_dict(ck['model_state_dict'])
            self.opt.load_state_dict(ck['optimizer_state_dict'])
            print("Model loaded.")
        else:
            print("No existing model – new weights.")

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
    def train(self, batch_size=64, epochs=5):
        if not self.data['states']:
            return
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
    def __init__(self, iterations, c_puct, nn: NNManager):
        self.I = iterations; self.c = c_puct; self.nn = nn

    def search(self, root_state: 'Connect4Game'):
        root = TreeNode(root_state.copy())
        # initial prior via NN + Dirichlet noise
        prior, _ = self.nn.policy_value(root.state)
        valid = root.state.valid_moves()
        dirichlet = np.random.dirichlet([0.3]*len(valid))
        for i, m in enumerate(valid):
            p = 0.75 * prior[m] + 0.25 * dirichlet[i]
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
    def __init__(self, iters, c, nn: NNManager):
        self.mcts = MCTS(iters, c, nn)
    def get_move(self, state, gui=None):
        start = time.time()
        mv = self.mcts.search(state)
        dt = time.time() - start
        if dt < 0.1:
            time.sleep(0.1 - dt)
        return mv

# ----------------------------------------------------------------------
# Dialogs (updated MCTS dialog unchanged, Player dialog forces learn for AI‑AI)
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
            ttk.Radiobutton(m,text=o,variable=self.p1,value=o).grid(row=0,column=i,sticky="w")
        ttk.Label(m, text="Green:").grid(row=1,column=0,sticky="w", padx=5, pady=5)
        for i,o in enumerate(opts,1):
            ttk.Radiobutton(m,text=o,variable=self.p2,value=o).grid(row=1,column=i,sticky="w")
    def apply(self):
        def mk(sel):
            if sel=="Human": return HumanPlayer()
            if sel=="Computer (Random)": return RandomComputerPlayer()
            return MCTSComputerPlayer(self.mcts['iterations'], self.mcts['C_param'], self.nn)
        self.result={'red':mk(self.p1.get()),'green':mk(self.p2.get())}

class MCTSDialog(simpledialog.Dialog):
    def __init__(self, master, it, c):
        self.it = tk.StringVar(master, str(it))
        self.c  = tk.StringVar(master, f"{c:.2f}")
        super().__init__(master, "MCTS Config")
    def body(self, m):
        ttk.Label(m,text="Iterations:").grid(row=0,column=0,sticky="w",padx=5,pady=5)
        ttk.Entry(m,textvariable=self.it,width=10).grid(row=0,column=1,padx=5)
        ttk.Label(m,text="Exploration C:").grid(row=1,column=0,sticky="w",padx=5,pady=5)
        ttk.Entry(m,textvariable=self.c,width=10).grid(row=1,column=1,padx=5)
    def validate(self):
        try:
            return int(self.it.get())>0 and float(self.c.get())>=0
        except:
            messagebox.showwarning("Invalid","Enter positive numbers.")
            return False
    def apply(self):
        self.result={'iterations':int(self.it.get()),'C_param':float(self.c.get())}

class TrainDialog(simpledialog.Dialog):
    def __init__(self, master):
        self.e = tk.StringVar(master, "200")
        super().__init__(master, "Train NN")
    def body(self, m):
        ttk.Label(m,text="Self‑play games:").grid(row=0,column=0,sticky="w",padx=5,pady=5)
        ttk.Entry(m,textvariable=self.e,width=10).grid(row=0,column=1,padx=5)
    def validate(self):
        try:
            return int(self.e.get())>0
        except:
            messagebox.showwarning("Invalid","Enter positive integer.")
            return False
    def apply(self):
        self.result=int(self.e.get())

# ----------------------------------------------------------------------
# GUI – Connect4GUI (selected sections updated for Learn toggle & concurrency)
# ----------------------------------------------------------------------
class Connect4GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Connect 4 – AlphaZero Edition (2025)")
        self.resizable(False, False)

        self.mcts_params = {'iterations': DEFAULT_MCTS_ITERATIONS, 'C_param': DEFAULT_PUCT_C}
        self.nn = NNManager()
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
        side = ttk.Frame(main); side.grid(row=0,column=1,rowspan=3,sticky="ns")
        side.grid_rowconfigure(6,weight=1)

        self.learn_var = tk.BooleanVar(self, True)
        self.learn_check = ttk.Checkbutton(side,text="Learn (Yes)",variable=self.learn_var,
            command=self._update_learn_label)
        self.learn_check.grid(row=2,column=0,columnspan=4,sticky="w",padx=5,pady=5)

        self.status = ttk.Label(side,font=("Helvetica",16,"bold"),width=25)
        self.status.grid(row=0,column=0,columnspan=3,sticky="ew",pady=(0,5))
        self.score_lbl = ttk.Label(side,font=("Helvetica",12))
        self.score_lbl.grid(row=1,column=0,columnspan=3,sticky="ew",pady=(0,10))

        ttk.Button(side,text="MCTS Config",command=self._config).grid(row=0,column=3,padx=5)
        self.train_btn = ttk.Button(side,text="Train NN",command=self._train)
        self.train_btn.grid(row=1,column=3,padx=5)

        # history text
        hist = ttk.Frame(side); hist.grid(row=3,column=0,columnspan=4,sticky="nsew")
        hist.grid_rowconfigure(1,weight=1); hist.grid_columnconfigure(0,weight=1)
        ttk.Label(hist,text="History:").grid(row=0,column=0,sticky="w")
        self.moves = tk.Text(hist,width=28,height=15,font=("Courier",10),state="disabled")
        scr = ttk.Scrollbar(hist,command=self.moves.yview); self.moves['yscrollcommand']=scr.set
        self.moves.grid(row=1,column=0,sticky="nsew"); scr.grid(row=1,column=1,sticky="ns")

        ctl = ttk.Frame(side); ctl.grid(row=4,column=0,columnspan=4,pady=5)
        ttk.Button(ctl,text="Restart",command=lambda:self._new_game(False)).pack(side="left",padx=5)
        self.stop_btn = ttk.Button(ctl,text="Stop",state="disabled",command=self._pause)
        self.stop_btn.pack(side="left",padx=5)
        self.go_btn   = ttk.Button(ctl,text="Continue",state="disabled",command=self._resume)
        self.go_btn.pack(side="left",padx=5)

        self._draw(); self._set_status("Select players"); self._update_score(); self._choose_players()

    # ----- UI helpers -----
    def _update_learn_label(self):
        self.learn_check.config(text=f"Learn ({'Yes' if self.learn_var.get() else 'No'})")

    # ----- drawing board, hover, click – unchanged except call to _draw() uses CNN
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

    # hover/click unchanged logic
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

    # making a move (updated Learn logic still consistent)
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

    # finish & restart (learning gate clarified)
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

    # choose players – enforce learn=Yes + disable checkbox if AI‑AI
    def _choose_players(self):
        dlg=PlayerDialog(self,self.mcts_params,self.nn)
        if not dlg.result: return
        self.players[RED_PIECE]=dlg.result['red']; self.players[GREEN_PIECE]=dlg.result['green']
        self.is_comp=all(isinstance(p,(RandomComputerPlayer,MCTSComputerPlayer)) for p in self.players.values())
        if self.is_comp:
            self.learn_var.set(True); self.learn_check.state(['disabled'])
        else:
            self.learn_check.state(['!disabled'])
        self._update_learn_label()
        self.game_in_progress=True; self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn",COLOR_MAP[self.game.current_player]); self.after(30,self._next_turn)

    # MCTS config
    def _config(self):
        if self.game_in_progress: messagebox.showinfo("Info","Stop the game before changing MCTS settings."); return
        dlg=MCTSDialog(self,self.mcts_params['iterations'],self.mcts_params['C_param'])
        if dlg.result:
            self.mcts_params=dlg.result; json.dump(self.mcts_params,open(MCTS_CONFIG_FILE,"w"),indent=4); messagebox.showinfo("Saved","MCTS parameters updated.")

    # Training dialog (disabled during game)
    def _train(self):
        if self.game_in_progress:
            messagebox.showinfo("Info","Finish or pause the current game before training."); return
        dlg=TrainDialog(self); n=getattr(dlg,'result',None)
        if not n: return
        self.train_btn['state']="disabled"  # prevent re‑entry
        win = tk.Toplevel(self); win.title("Training"); win.resizable(False,False)
        ttk.Label(win,text=f"Running {n} self‑play games…").pack(pady=10)
        bar=ttk.Progressbar(win,length=300,mode="determinate",maximum=n); bar.pack(pady=5)
        prog=tk.StringVar(value="0 / {n}"); ttk.Label(win,textvariable=prog).pack(pady=5)
        def worker():
            aiR=MCTSComputerPlayer(self.mcts_params['iterations'],self.mcts_params['C_param'],self.nn)
            aiG=MCTSComputerPlayer(self.mcts_params['iterations'],self.mcts_params['C_param'],self.nn)
            for i in range(n):
                g=Connect4Game(); cur=RED_PIECE
                while not g.game_over:
                    mv=(aiR if cur==RED_PIECE else aiG).get_move(g)
                    ok,_=g.drop_piece(mv)
                    if ok and not g.game_over:
                        g.switch(); cur=g.current_player
                self.nn.finish_game(g.winner)
                self.after(0,lambda idx=i: (bar.configure(value=idx+1), prog.set(f"{idx+1} / {n}")))
            self.nn.train(epochs=10)
            self.after(0,win.destroy); self.after(0,lambda:self.train_btn.config(state="normal")); messagebox.showinfo("Done","Training finished.")
        threading.Thread(target=worker,daemon=True).start()

    # pause/resume – unchanged
    def _pause(self):
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

    # cfg loader (unchanged)
    def _load_cfg(self):
        if os.path.exists(MCTS_CONFIG_FILE):
            try:
                cfg=json.load(open(MCTS_CONFIG_FILE));
                if isinstance(cfg.get('iterations'),int) and cfg['iterations']>0 and isinstance(cfg.get('C_param'),(int,float)) and cfg['C_param']>=0:
                    self.mcts_params=cfg
            except: pass

# ----------------------------------------------------------------------
if __name__=='__main__':
    Connect4GUI().mainloop()
