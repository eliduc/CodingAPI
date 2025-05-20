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
9. **Model selection** – Support for selecting different model files for players
10. **Performance optimizations** - GPU acceleration, parallel training, mixed precision

(C) 2025 – released under MIT License.
"""

# ----------------------------------------------------------------------
# Imports & constants
# ----------------------------------------------------------------------
import tkinter as tk
from tkinter import ttk, simpledialog, messagebox, filedialog
import numpy as np, random, math, threading, time, json, os, numba
import torch, torch.nn as nn, torch.optim as optim, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

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
NN_MODEL_FILE           = "C4.pt"
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
    def __init__(self, hyperparams=None, model_path=NN_MODEL_FILE):
        self.hyperparams = hyperparams or {
            'learning_rate': 1e-3, 
            'batch_size': 64, 
            'epochs': 5,
            'policy_weight': 1.0,  # Weight for policy loss
            'value_weight': 1.0,   # Weight for value loss
            'lr_decay': 0.9995     # Learning rate decay factor
        }
        self.model_path = model_path
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.net = Connect4CNN().to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.hyperparams['learning_rate'])
        
        # Setup for mixed precision training if on CUDA
        self.use_mixed_precision = self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            print("Mixed precision training enabled")
        
        self.data = {'states': [], 'policies': [], 'values': []}
        self.pending = []  # moves from current game
        self.train_iterations = 0  # Track number of training iterations
        
        if os.path.exists(model_path):
            try:
                ck = torch.load(model_path, map_location=self.device, weights_only=True)
                self.net.load_state_dict(ck['model_state_dict'])
                self.opt.load_state_dict(ck['optimizer_state_dict'])
                if 'train_iterations' in ck:
                    self.train_iterations = ck['train_iterations']
                if 'hyperparams' in ck:
                    # Keep user settings but update with any new hyperparams from saved model
                    saved_params = ck['hyperparams']
                    for key in saved_params:
                        if key not in self.hyperparams:
                            self.hyperparams[key] = saved_params[key]
                print(f"Model loaded from {model_path}. Training iterations: {self.train_iterations}")
            except Exception as e:
                print(f"Error loading model from {model_path}: {e}")
                print("Using fresh weights.")
        else:
            print(f"Model file {model_path} not found – using new weights.")

    # 3‑plane tensor with device option
    @staticmethod
    def _tensor(state: 'Connect4Game', device=None):
        red_plane   = (state.board == RED_PIECE).astype(np.float32)
        green_plane = (state.board == GREEN_PIECE).astype(np.float32)
        turn_plane  = np.full_like(red_plane, 1.0 if state.current_player == RED_PIECE else 0.0)
        stacked = np.stack([red_plane, green_plane, turn_plane])  # [3,6,7]
        tensor = torch.tensor(stacked, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    # inference with GPU acceleration
    def policy_value(self, state: 'Connect4Game'):
        self.net.eval()
        with torch.no_grad():
            t = self._tensor(state, self.device).unsqueeze(0)  # [1,3,6,7]
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
        
        # Smart buffer management - keep a more diverse set of examples
        if len(self.data['states']) > MAX_TRAINING_EXAMPLES:
            current_size = len(self.data['states'])
            target_size = MAX_TRAINING_EXAMPLES
            
            # Keep all very recent examples (last 20%)
            keep_recent = int(target_size * 0.2)
            # Reserve space for older examples to maintain diversity
            keep_old = int(target_size * 0.2)
            # Fill the middle with evenly sampled examples
            keep_middle = target_size - keep_recent - keep_old
            
            # Calculate sizes
            recent_end = current_size
            recent_start = recent_end - keep_recent
            old_end = recent_start
            old_start = 0
            middle_size = old_end - old_start
            
            # If we have enough examples for this strategy
            if middle_size > keep_middle and keep_old > 0:
                # Keep oldest examples
                oldest_indices = list(range(0, keep_old))
                
                # Sample from middle section (excluding very old and very new)
                if middle_size > keep_middle:
                    # Evenly sample from middle section to maintain diversity
                    step = middle_size / keep_middle
                    middle_indices = [int(old_start + i * step) for i in range(keep_middle)]
                else:
                    # If middle section is smaller than requested, keep all
                    middle_indices = list(range(keep_old, old_end))
                
                # Keep newest examples
                newest_indices = list(range(recent_start, recent_end))
                
                # Combine all indices to keep
                keep_indices = sorted(oldest_indices + middle_indices + newest_indices)
                
                # Apply the filtering
                for k in self.data:
                    self.data[k] = [self.data[k][i] for i in keep_indices]
            else:
                # Fallback to simple trimming if we don't have enough examples
                excess = current_size - target_size
                for k in self.data:
                    self.data[k] = self.data[k][excess:]

    # custom loss with distribution targets and GPU/mixed precision training
    def train(self, batch_size=None, epochs=None):
        if not self.data['states']:
            return
        batch_size = batch_size or self.hyperparams['batch_size']
        epochs = epochs or self.hyperparams['epochs']
        
        # Get loss weights
        policy_weight = self.hyperparams.get('policy_weight', 1.0)
        value_weight = self.hyperparams.get('value_weight', 1.0)
        
        # Apply learning rate decay
        lr_decay = self.hyperparams.get('lr_decay', 1.0)
        if lr_decay < 1.0:
            current_lr = self.opt.param_groups[0]['lr']
            new_lr = max(current_lr * lr_decay, 1e-5)  # Don't go below 1e-5
            for param_group in self.opt.param_groups:
                param_group['lr'] = new_lr
            print(f"Learning rate adjusted: {current_lr:.6f} → {new_lr:.6f}")
        
        # Create dataset and loader
        ds = Connect4Dataset(torch.stack(self.data['states']),
                            torch.stack(self.data['policies']),
                            torch.stack(self.data['values']).squeeze(1))
        
        # Use weighted random sampler to balance training examples
        # We'll sample recent and older examples with equal probability
        num_samples = len(ds)
        half_point = num_samples // 2
        sample_weights = torch.ones(num_samples)
        if num_samples > 1000:  # Only use weighting for substantial datasets
            # Give more weight to older examples (which might be underrepresented)
            sample_weights[:half_point] = 1.2
            # Keep recent examples at normal weight
            sample_weights[half_point:] = 0.8
            
        dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        
        self.net.train()
        for ep in range(epochs):
            p_loss_sum = v_loss_sum = 0.0
            for s, p_target, v_target in dl:
                # Move batch to device
                s, p_target, v_target = s.to(self.device), p_target.to(self.device), v_target.to(self.device)
                
                self.opt.zero_grad()
                
                # Use mixed precision training if on CUDA
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits, v_pred = self.net(s)
                        log_probs = F.log_softmax(logits, dim=1)
                        policy_loss = -(p_target * log_probs).sum(dim=1).mean()
                        value_loss = F.mse_loss(v_pred, v_target)
                        weighted_loss = (policy_weight * policy_loss) + (value_weight * value_loss)
                    
                    self.scaler.scale(weighted_loss).backward()
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    logits, v_pred = self.net(s)
                    log_probs = F.log_softmax(logits, dim=1)
                    policy_loss = -(p_target * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(v_pred, v_target)
                    weighted_loss = (policy_weight * policy_loss) + (value_weight * value_loss)
                    weighted_loss.backward()
                    self.opt.step()
                
                p_loss_sum += policy_loss.item()
                v_loss_sum += value_loss.item()
                
            print(f"[Train] Epoch {ep+1}/{epochs}  policy={p_loss_sum/len(dl):.4f}  value={v_loss_sum/len(dl):.4f}")
        
        # Increment training iterations counter
        self.train_iterations += 1
        
        # Save model with additional information
        torch.save({
            'model_state_dict': self.net.state_dict(), 
            'optimizer_state_dict': self.opt.state_dict(),
            'train_iterations': self.train_iterations,
            'hyperparams': self.hyperparams
        }, self.model_path)

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
    def __init__(self, iters, c, nn: NNManager, explore=True, model_path=None):
        self.nn = nn
        self.model_path = model_path  # Store model path for reference
        self.mcts = MCTS(iters, 0.0 if not explore else c, nn, explore=explore)
    def get_move(self, state, gui=None):
        start = time.time()
        mv = self.mcts.search(state)
        dt = time.time() - start
        if dt < 0.1:
            time.sleep(0.1 - dt)
        return mv

# Function for parallel training game generation
def _play_single_training_game(mcts_iterations, puct_c, nn_manager, use_exploration=True):
    """Self-contained function to play a single training game and return the moves and result"""
    # Create local copies of neural network for thread safety
    nn_copy = NNManager(nn_manager.hyperparams, nn_manager.model_path)
    
    # Create AI players
    ai_red = MCTSComputerPlayer(mcts_iterations, puct_c, nn_copy, explore=use_exploration)
    ai_green = MCTSComputerPlayer(mcts_iterations, puct_c, nn_copy, explore=use_exploration)
    
    # Play the game
    game = Connect4Game()
    game_moves = []
    
    while not game.game_over:
        player = game.current_player
        move = ai_red.get_move(game) if player == RED_PIECE else ai_green.get_move(game)
        
        # Create a training example
        state_before = game.copy()
        ok, _ = game.drop_piece(move)
        
        if ok:
            # Store the game state and move for later training
            game_moves.append({
                'state': state_before,
                'player': player,
                'move': move
            })
            
            if not game.game_over:
                game.switch()
    
    return game_moves, game.winner

# ----------------------------------------------------------------------
# Dialogs (updated to include settings dialog)
# ----------------------------------------------------------------------
class PlayerDialog(simpledialog.Dialog):
    def __init__(self, master, mcts_params, nn):
        self.mcts = mcts_params
        self.nn = nn
        self.p1 = tk.StringVar(master, "Human")
        self.p2 = tk.StringVar(master, "Computer (AI)")
        self.red_model = tk.StringVar(master, NN_MODEL_FILE)    # Default model path for Red
        self.green_model = tk.StringVar(master, NN_MODEL_FILE)  # Default model path for Green
        super().__init__(master, "Select Players")
        
    def _browse_model(self, player_var):
        """Browse for a neural network model file"""
        # Determine which model path variable to update
        if player_var == self.red_model:
            title = "Select Neural Network Model for Red AI"
        else:
            title = "Select Neural Network Model for Green AI"
            
        filename = tk.filedialog.askopenfilename(
            title=title,
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(player_var.get()))
        )
        if filename:
            player_var.set(filename)
            self._update_file_browse_state()
    
    def body(self, m):
        opts = ["Human", "Computer (Random)", "Computer (AI)"]
        
        # Red player row
        ttk.Label(m, text="Red:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        for i, o in enumerate(opts, 1):
            rb = ttk.Radiobutton(m, text=o, variable=self.p1, value=o, 
                              command=lambda: self._update_file_browse_state())
            rb.grid(row=0, column=i, sticky="w")
            ToolTip(rb, f"Select {o} for Red player")
        
        # Red AI model row with browse button and model name in same row
        self.red_browse = ttk.Button(m, text="📂", width=3, 
                             command=lambda: self._browse_model(self.red_model))
        self.red_browse.grid(row=0, column=4, padx=5)
        ToolTip(self.red_browse, "Browse for Red AI neural network model file")
        
        # Red model name label - position in same row, to the right of browse button
        self.red_model_label = ttk.Label(m, text="", foreground="#0000FF", font=("Helvetica", 9, "bold"))
        self.red_model_label.grid(row=0, column=5, sticky="w", padx=2)
        
        # Green player row
        ttk.Label(m, text="Green:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        for i, o in enumerate(opts, 1):
            rb = ttk.Radiobutton(m, text=o, variable=self.p2, value=o,
                              command=lambda: self._update_file_browse_state())
            rb.grid(row=1, column=i, sticky="w")
            ToolTip(rb, f"Select {o} for Green player")
        
        # Green AI model row with browse button and model name in same row
        self.green_browse = ttk.Button(m, text="📂", width=3,
                               command=lambda: self._browse_model(self.green_model))
        self.green_browse.grid(row=1, column=4, padx=5)
        ToolTip(self.green_browse, "Browse for Green AI neural network model file")
        
        # Green model name label - position in same row, to the right of browse button
        self.green_model_label = ttk.Label(m, text="", foreground="#0000FF", font=("Helvetica", 9, "bold"))
        self.green_model_label.grid(row=1, column=5, sticky="w", padx=2)
        
        # Update initial state of browse buttons and model displays
        self._update_file_browse_state()
    
    def _update_file_browse_state(self):
        """Enable/disable model browse buttons based on player selection and update model names"""
        # Update Red browse button and label
        if self.p1.get() == "Computer (AI)":
            self.red_browse["state"] = "normal"
            # Display model name without extension
            model_name = os.path.splitext(os.path.basename(self.red_model.get()))[0]
            self.red_model_label.config(text=model_name)
        else:
            self.red_browse["state"] = "disabled"
            self.red_model_label.config(text="")
            
        # Update Green browse button and label
        if self.p2.get() == "Computer (AI)":
            self.green_browse["state"] = "normal"
            # Display model name without extension
            model_name = os.path.splitext(os.path.basename(self.green_model.get()))[0]
            self.green_model_label.config(text=model_name)
        else:
            self.green_browse["state"] = "disabled"
            self.green_model_label.config(text="")
    
    def apply(self):
        def mk_player(sel, model_path):
            if sel == "Human": 
                return HumanPlayer()
            if sel == "Computer (Random)": 
                return RandomComputerPlayer()
            
            # For AI players, use the selected model file
            if model_path and os.path.exists(model_path):
                # Create a new NNManager with the specified model path
                ai_nn = NNManager(self.nn.hyperparams, model_path)
                return MCTSComputerPlayer(
                    self.mcts['iterations'], 
                    self.mcts['C_param'], 
                    ai_nn,
                    explore=not self.master.fullplay_var.get(),
                    model_path=model_path
                )
            else:
                # Use the default NN if model file doesn't exist
                return MCTSComputerPlayer(
                    self.mcts['iterations'], 
                    self.mcts['C_param'], 
                    self.nn,
                    explore=not self.master.fullplay_var.get()
                )
                
        self.result = {
            'red': mk_player(self.p1.get(), self.red_model.get() if self.p1.get() == "Computer (AI)" else None),
            'green': mk_player(self.p2.get(), self.green_model.get() if self.p2.get() == "Computer (AI)" else None)
        }

class SettingsDialog(simpledialog.Dialog):
    def __init__(self, master, mcts_params, train_games=200, nn_params=None):
        self.it = tk.StringVar(master, str(mcts_params['iterations']))
        self.c = tk.StringVar(master, f"{mcts_params['C_param']:.2f}")
        self.games = tk.StringVar(master, str(train_games))
        
        # Initialize with default values if not provided
        self.nn_params = nn_params or {
            'learning_rate': 1e-3, 
            'batch_size': 64, 
            'epochs': 5,
            'policy_weight': 1.0,
            'value_weight': 1.0,
            'lr_decay': 0.9995
        }
        
        # Set up string vars for all parameters
        self.lr = tk.StringVar(master, str(self.nn_params.get('learning_rate', 1e-3)))
        self.batch = tk.StringVar(master, str(self.nn_params.get('batch_size', 64)))
        self.epochs = tk.StringVar(master, str(self.nn_params.get('epochs', 5)))
        self.policy_weight = tk.StringVar(master, str(self.nn_params.get('policy_weight', 1.0)))
        self.value_weight = tk.StringVar(master, str(self.nn_params.get('value_weight', 1.0)))
        self.lr_decay = tk.StringVar(master, str(self.nn_params.get('lr_decay', 0.9995)))
        
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
        
        # Add training presets to the training tab
        self._add_training_presets(train_tab)
        
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
        
        # Add new parameters in a separate frame
        ttk.Label(nn_tab, text="Learning Balance:").grid(row=3, column=0, columnspan=2, sticky="w", padx=5, pady=(15,5))
        ttk.Label(nn_tab, text="Policy weight:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        e7 = ttk.Entry(nn_tab, textvariable=self.policy_weight, width=10)
        e7.grid(row=4, column=1, padx=5)
        ToolTip(e7, "Weight for policy loss\nHigher values prioritize move prediction\nIncrease if policy loss is rising")
        
        ttk.Label(nn_tab, text="Value weight:").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        e8 = ttk.Entry(nn_tab, textvariable=self.value_weight, width=10)
        e8.grid(row=5, column=1, padx=5)
        ToolTip(e8, "Weight for value loss\nHigher values prioritize outcome prediction")
        
        ttk.Label(nn_tab, text="LR decay factor:").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        e9 = ttk.Entry(nn_tab, textvariable=self.lr_decay, width=10)
        e9.grid(row=6, column=1, padx=5)
        ToolTip(e9, "Learning rate decay per training round\n0.9995 = gentle decay\n1.0 = no decay\nLower for faster decay")
        
        return mcts_tab  # Initial focus
    
    def _add_training_presets(self, train_tab):
        """Add training presets to the settings dialog"""
        # Training presets frame
        preset_frame = ttk.LabelFrame(train_tab, text="Training Presets")
        preset_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
        # Define presets
        self.presets = {
            "Fast": {
                "iterations": 200,
                "batch_size": 128,
                "epochs": 3,
                "description": "Fastest training with reasonable quality"
            },
            "Balanced": {
                "iterations": 400,
                "batch_size": 64, 
                "epochs": 5,
                "description": "Good balance of speed and quality"
            },
            "Quality": {
                "iterations": 800,
                "batch_size": 64,
                "epochs": 10,
                "description": "Highest quality with slower training"
            }
        }
        
        # Preset buttons
        for i, (name, settings) in enumerate(self.presets.items()):
            btn = ttk.Button(
                preset_frame, 
                text=name, 
                command=lambda s=settings: self._apply_preset(s)
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
            ToolTip(btn, f"{name}: {settings['description']}\n" + 
                    f"MCTS: {settings['iterations']} iterations\n" +
                    f"Batch: {settings['batch_size']}, Epochs: {settings['epochs']}")
        
        # Add explanation of presets
        ttk.Label(preset_frame, text="Select a preset to quickly configure training settings",
                  font=("Helvetica", 8)).grid(row=1, column=0, columnspan=3, pady=(0,5))
    
    def _apply_preset(self, settings):
        """Apply a training preset"""
        # Update MCTS iterations
        self.it.set(str(settings['iterations']))
        
        # Update NN parameters
        self.batch.set(str(settings['batch_size']))
        self.epochs.set(str(settings['epochs']))
        
        # Provide feedback
        feedback_label = ttk.Label(self.body_frame, text="Preset applied", foreground="green")
        feedback_label.grid(row=0, column=0, sticky="w", padx=5, pady=5)
        # Auto-remove feedback after 2 seconds
        self.after(2000, lambda: feedback_label.destroy())
    
    def validate(self):
        try:
            # Validate MCTS params
            it = int(self.it.get())
            c = float(self.c.get())
            games = int(self.games.get())
            
            # Validate NN params
            lr = float(self.lr.get())
            batch = int(self.batch.get())
            epochs = int(self.epochs.get())
            p_weight = float(self.policy_weight.get())
            v_weight = float(self.value_weight.get())
            lr_decay = float(self.lr_decay.get())
            
            # Check positive values where needed
            if (it <= 0 or c < 0 or games <= 0 or lr <= 0 or 
                batch <= 0 or epochs <= 0 or p_weight <= 0 or v_weight <= 0):
                messagebox.showwarning("Invalid", "All values must be positive.")
                return False
                
            # Check LR decay specifically
            if lr_decay <= 0 or lr_decay > 1.0:
                messagebox.showwarning("Invalid", "Learning rate decay must be between 0 and 1.0")
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
                'epochs': int(self.epochs.get()),
                'policy_weight': float(self.policy_weight.get()),
                'value_weight': float(self.value_weight.get()),
                'lr_decay': float(self.lr_decay.get())
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
        self.training_model_path = NN_MODEL_FILE  # Default training model path
        
        self.nn = NNManager(self.nn_params, self.training_model_path)
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
        
        # Train row with button, file icon and model name
        train_frame = ttk.Frame(side)
        train_frame.grid(row=1, column=3, padx=5, sticky="w")
        
        self.train_btn = ttk.Button(train_frame, text="Train NN", command=self._train)
        self.train_btn.grid(row=0, column=0)
        ToolTip(self.train_btn, "Run AI self-play games to train the neural network")
        
        self.train_browse = ttk.Button(train_frame, text="📂", width=2, command=self._browse_training_model)
        self.train_browse.grid(row=0, column=1, padx=(2,2))
        ToolTip(self.train_browse, "Select model file to train (default: C4)")
        
        # Display current training model name (without extension)
        model_name = os.path.splitext(os.path.basename(self.training_model_path))[0]
        self.train_model_label = ttk.Label(train_frame, text=model_name, 
                                       font=("Helvetica", 9, "bold"), foreground="#0000FF")
        self.train_model_label.grid(row=0, column=2, padx=(0,5))

        # history text
        hist = ttk.Frame(side); hist.grid(row=3,column=0,columnspan=4,sticky="nsew")
        hist.grid_rowconfigure(1,weight=1); hist.grid_columnconfigure(0,weight=1)
        ttk.Label(hist,text="History:").grid(row=0,column=0,sticky="w")
        self.moves = tk.Text(hist,width=28,height=15,font=("Courier",10),state="disabled")
        scr = ttk.Scrollbar(hist,command=self.moves.yview); self.moves['yscrollcommand']=scr.set
        self.moves.grid(row=1,column=0,sticky="nsew"); scr.grid(row=1,column=1,sticky="ns")
        ToolTip(self.moves, "Game move history")

        # Control buttons
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

        # Training progress area - initially hidden, will appear when Train NN is pressed
        self.separator = ttk.Separator(side, orient='horizontal')
        self.separator.grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)
        self.separator.grid_remove()  # Hide initially
        
        self.train_frame = ttk.LabelFrame(side, text="Training Progress")
        self.train_frame.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=5)
        self.train_frame.grid_columnconfigure(0, weight=1)
        self.train_frame.grid_remove()  # Hide initially
        
        self.train_progress = ttk.Progressbar(self.train_frame, length=250, mode="determinate")
        self.train_progress.grid(row=0, column=0, padx=5, pady=(5,0), sticky="ew")
        
        self.train_status = ttk.Label(self.train_frame, text="Idle", anchor="center")
        self.train_status.grid(row=1, column=0, padx=5, pady=(0,5), sticky="ew")
        
        # Model info label
        self.model_info = ttk.Label(self.train_frame, text="", anchor="center", font=("Helvetica", 8))
        self.model_info.grid(row=2, column=0, padx=5, pady=(0,5), sticky="ew")
        
        self._update_model_info()

        self._draw(); self._set_status("Select players"); self._update_score(); self._choose_players()
        
    def _browse_training_model(self):
        """Browse for neural network model file to train"""
        filename = filedialog.askopenfilename(
            title="Select Neural Network Model to Train",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.training_model_path))
        )
        if filename:
            # Update the training model path
            self.training_model_path = filename
            
            # Reload the neural network with the new model
            self.nn = NNManager(self.nn_params, self.training_model_path)
            
            # Update the model info display
            self._update_model_info()
            
    def _update_model_info(self):
        """Update the model info display in the training frame"""
        # Get filename without extension
        model_name = os.path.splitext(os.path.basename(self.training_model_path))[0]
        self.model_info.config(text=f"Active model: {model_name}")
        
        # Update the label next to the browse button (with bold blue text)
        self.train_model_label.config(text=model_name, foreground="#0000FF", font=("Helvetica", 9, "bold"))
        
        # If the training frame is visible, make the model info visible too
        if str(self.train_frame.grid_info()) != "{}":  # If grid info not empty (means it's visible)
            self.model_info.grid()

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
        
        # Store original settings for comparison
        original_settings = {
            'mcts': dict(self.mcts_params),
            'training': {'games': self.train_games},
            'nn_params': dict(self.nn_params)
        }
        
        dlg = SettingsDialog(self, self.mcts_params, self.train_games, self.nn_params)
        if dlg.result:
            # Get new settings from dialog
            new_mcts = dlg.result['mcts']
            new_train_games = dlg.result['training']['games']
            new_nn_params = dlg.result['nn_params']
            
            # Check if any settings have changed
            settings_changed = False
            
            # Check MCTS params
            if (new_mcts['iterations'] != self.mcts_params['iterations'] or
                new_mcts['C_param'] != self.mcts_params['C_param']):
                settings_changed = True
            
            # Check training games
            if new_train_games != self.train_games:
                settings_changed = True
            
            # Check NN params
            for key, value in new_nn_params.items():
                if key not in self.nn_params or self.nn_params[key] != value:
                    settings_changed = True
                    break
            
            # Update settings if there were changes
            self.mcts_params = new_mcts
            self.train_games = new_train_games
            old_lr = self.nn_params.get('learning_rate')
            self.nn_params = new_nn_params
            
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
            
            # Only show message if settings were actually changed
            if settings_changed:
                messagebox.showinfo("Saved", "Settings updated.")
    
    # ENHANCED: Parallel training with GPU acceleration
    def _train(self):
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Finish or pause the current game before training.")
            return
        
        if self.training_in_progress:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        # Show training progress UI elements
        self.separator.grid()
        self.train_frame.grid()
        self.model_info.grid()
        
        # Use the number of games from settings without asking
        n = self.train_games
        
        self.train_btn['state'] = "disabled"
        self.train_browse['state'] = "disabled"  # Disable model selection during training
        self.training_in_progress = True
        self.training_stop_requested = False
        self.stop_btn['state'] = "normal"  # Enable stop button during training
        
        # Update training progress area
        self.train_progress['maximum'] = n
        self.train_progress['value'] = 0
        self.train_status.config(text=f"Training: 0 / {n} games")
        
        # Make sure model info is updated
        self._update_model_info()
        
        def worker():
            # Check if both players are using the same model - if so, we'll optimize training
            using_same_model = False
            red_model_path = None
            green_model_path = None
            
            if (isinstance(self.players[RED_PIECE], MCTSComputerPlayer) and 
                isinstance(self.players[GREEN_PIECE], MCTSComputerPlayer)):
                red_player = self.players[RED_PIECE]
                green_player = self.players[GREEN_PIECE]
                
                # Check if both players have model_path attribute and they're the same
                if (hasattr(red_player, 'model_path') and hasattr(green_player, 'model_path') and
                    red_player.model_path and green_player.model_path and
                    os.path.abspath(red_player.model_path) == os.path.abspath(green_player.model_path)):
                    using_same_model = True
                else:
                    # Store model paths if they exist
                    red_model_path = getattr(red_player, 'model_path', None)
                    green_model_path = getattr(green_player, 'model_path', None)
            
            # Prepare separate NNs for different models if needed
            red_nn = None
            green_nn = None
            if not using_same_model:
                if red_model_path and os.path.exists(red_model_path):
                    red_nn = NNManager(self.nn_params, red_model_path)
                if green_model_path and os.path.exists(green_model_path):
                    green_nn = NNManager(self.nn_params, green_model_path)
            
            # Determine how many workers to use (leave one core free for UI)
            num_workers = max(1, min(os.cpu_count() - 1, 6))  # Cap at 6 to prevent excessive memory usage
            print(f"Starting parallel training with {num_workers} workers")
            
            # Prepare the function and batch size for parallel execution
            batch_size = min(n, max(1, n // (num_workers * 2)))  # Process in smaller batches
            game_func = partial(
                _play_single_training_game, 
                self.mcts_params['iterations'], 
                self.mcts_params['C_param'], 
                self.nn,
                not self.fullplay_var.get() # exploration param
            )
            
            games_completed = 0
            all_results = []
            
            # Process games in batches until we have enough or are stopped
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                while games_completed < n and not self.training_stop_requested:
                    # Calculate remaining games
                    games_to_run = min(batch_size, n - games_completed)
                    if games_to_run <= 0:
                        break
                    
                    # Submit batch of games
                    futures = [executor.submit(game_func) for _ in range(games_to_run)]
                    
                    # Process results as they complete
                    for future in as_completed(futures):
                        if self.training_stop_requested:
                            break
                        
                        try:
                            game_moves, winner = future.result()
                            all_results.append((game_moves, winner))
                            games_completed += 1
                            
                            # Update progress
                            self.after(0, lambda idx=games_completed: self._update_training_progress(idx, n))
                        except Exception as e:
                            print(f"Error in game generation: {e}")
                    
                    if self.training_stop_requested:
                        break
            
            # Training phase - process all collected games
            if games_completed > 0 and not self.training_stop_requested:
                self.train_status.config(text="Processing game data...")
                
                # Process all games for training
                for game_moves, winner in all_results:
                    # Train the primary model (selected for training)
                    for move in game_moves:
                        state = move['state']
                        player = move['player']
                        move_col = move['move']
                        
                        # Create one-hot vector for the move
                        vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
                        vp[move_col] = 1.0
                        
                        # Add example to primary training model
                        self.nn.add_example(state, vp)
                    
                    # Finalize the game for primary model
                    self.nn.finish_game(winner)
                    
                    # If separate models for red and green, process for them too
                    if not using_same_model:
                        if red_nn and red_model_path != self.training_model_path:
                            for move in game_moves:
                                if move['player'] == RED_PIECE:
                                    state = move['state']
                                    move_col = move['move']
                                    vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
                                    vp[move_col] = 1.0
                                    red_nn.add_example(state, vp)
                            red_nn.finish_game(winner)
                            
                        if green_nn and green_model_path != self.training_model_path:
                            for move in game_moves:
                                if move['player'] == GREEN_PIECE:
                                    state = move['state']
                                    move_col = move['move']
                                    vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
                                    vp[move_col] = 1.0
                                    green_nn.add_example(state, vp)
                            green_nn.finish_game(winner)
                
                # Train models with collected data
                self.train_status.config(text="Training neural network...")
                self.nn.train(batch_size=self.nn_params.get('batch_size', 64),
                              epochs=self.nn_params.get('epochs', 5))
                
                # Train player models if needed
                if not using_same_model:
                    if red_nn and red_model_path != self.training_model_path:
                        red_nn.train(batch_size=self.nn_params.get('batch_size', 64),
                                    epochs=self.nn_params.get('epochs', 5))
                    if green_nn and green_model_path != self.training_model_path:
                        green_nn.train(batch_size=self.nn_params.get('batch_size', 64),
                                      epochs=self.nn_params.get('epochs', 5))
            
            # Clean up and update UI
            self.after(0, lambda: self.train_btn.config(state="normal"))
            self.after(0, lambda: self.train_browse.config(state="normal"))
            self.after(0, lambda: setattr(self, 'training_in_progress', False))
            
            # Final status update
            if self.training_stop_requested:
                self.after(0, lambda: self.train_status.config(text=f"Stopped after {games_completed} games"))
                messagebox.showinfo("Stopped", f"Training stopped after {games_completed} games.")
            else:
                self.after(0, lambda: self.train_status.config(text=f"Completed {games_completed} games"))
                messagebox.showinfo("Done", "Training finished.")
                
            self.training_stop_requested = False
            
            # Reset stop button state if not in computer vs computer game
            if not self.is_comp:
                self.stop_btn['state'] = "disabled"
        
        threading.Thread(target=worker, daemon=True).start()
    
    def _hide_training_ui(self):
        """Hide the training progress UI elements"""
        self.separator.grid_remove()
        self.train_frame.grid_remove()
    
    def _update_training_progress(self, current, total):
        """Update the training progress indicators in the main window"""
        self.train_progress['value'] = current
        self.train_status.config(text=f"Training: {current} / {total} games")

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
    # Enable multiprocessing for PyTorch if available
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    Connect4GUI().mainloop()