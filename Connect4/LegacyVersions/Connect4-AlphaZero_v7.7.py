#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Connect‑4 with AlphaZero‑style self‑play (PUCT MCTS + CNN policy‑value NN)
Version 8.2 - Algorithm fixes for MCTS and training

Improvements implemented
------------------------
1. **CNN architecture** – spatial 3‑plane board encoding → deeper conv net.
2. **Single NN call per simulation** – evaluate leaf only once; value + priors returned together.
3. **Policy targets = root visit distribution** instead of chosen move.
4. **Training loss** – custom cross‑entropy for distribution + value MSE.
5. **UI safeguards** – Training disabled during active games; Learn toggle forced on for AI‑vs‑AI & grayed‑out.
6. **Clearer Learn toggle semantics** – user‑driven games respect switch fully.
7. **Minor clean‑ups** – removed redundant value inversion; small refactors.
8. **UI enhancements** – Settings cog, tooltips, renamed buttons, improved layout
9. **Model selection** – Support for selecting different model files for players
10. **Performance optimizations** - GPU acceleration, parallel training, mixed precision, balanced worker count
11. **Enhanced training output** - Policy/value losses and training time in scrollable training log
12. **UI fixes** - Fixed geometry manager conflicts and improved feedback
13. **Responsive stop** - Better training cancellation with two-stage stop (graceful/force)
14. **Bug fixes** - Fixed training completion issues and hanging on stop
15. **Adjusted presets** - Optimized Quality preset for better performance
16. **Smart AI** - Immediate win/block detection for more responsive play
17. **Training metrics** - Track total games the model has been trained on
18. **Stop confirmation** - Added confirmation dialogs when stopping training or AI games
19. **UI improvements** - Moved Play@Full Strength to Player dialog, renamed History to Game History, clear board on Train NN
20. **AI strength fixes** - Improved threat detection, parameter tuning, fixed fullplay var storage
21. **Dialog memory** - Select Players dialog remembers previous settings
22. **Responsiveness** - Reduced delay for showing human winning moves
23. **Center bias** - Added positional bias to favor center columns
24. **Reduced exploration** - Lowered exploration parameter for stronger play
25. **MCTS algorithm fixes** - Fixed c_puct in full strength mode, proper renormalization of priors, improved training signal
26. **Thread safety** - Made UI updates thread-safe to prevent crashes
27. **Training data correctness** - Fixed example-to-game association for accurate learning
28. **Improved win detection** - Fixed opponent win logic to properly identify safe moves
29. **Settings consistency** - Ensured neural network settings are properly propagated
30. **Error handling** - Added proper error catching and reporting for AI threads

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
from concurrent.futures import ProcessPoolExecutor, as_completed, wait, FIRST_COMPLETED
from functools import partial
import sys

ROW_COUNT, COLUMN_COUNT = 6, 7
SQUARESIZE, RADIUS      = 100, int(100/2 - 5)
WIDTH, HEIGHT           = COLUMN_COUNT * SQUARESIZE, (ROW_COUNT + 1) * SQUARESIZE

BLUE, BLACK = "#0000FF", "#000000"
RED,  GREEN = "#FF0000", "#00FF00"
EMPTY_COLOR = "#CCCCCC"

EMPTY, RED_PIECE, GREEN_PIECE = 0, 1, 2
PLAYER_MAP = {RED_PIECE: "Red", GREEN_PIECE: "Green"}
COLOR_MAP  = {RED_PIECE: RED,   GREEN_PIECE: GREEN}

# Improved default parameters for better AI play
DEFAULT_MCTS_ITERATIONS = 1200  # Increased from 800 for better search depth
DEFAULT_PUCT_C          = 0.8   # Reduced from 1.0 for less exploration
DEFAULT_TRAIN_GAMES     = 200
NN_MODEL_FILE           = "C4.pt"
MCTS_CONFIG_FILE        = "mcts_config.json"
MAX_TRAINING_EXAMPLES   = 30_000

# Center bias for position evaluation - strongly favor center columns
CENTER_BIAS = np.array([0.5, 0.7, 1.0, 1.5, 1.0, 0.7, 0.5])

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
    def __init__(self, channels=96):  # Increased channels from 64 to 96 for more capacity
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
# Core game state 
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
    def __init__(self, hyperparams=None, model_path=NN_MODEL_FILE, quiet=True):
        self.hyperparams = hyperparams or {
            'learning_rate': 5e-4,  # Lowered from 1e-3 for more stable training
            'batch_size': 128,      # Increased from 64 for better training
            'epochs': 10,           # Increased from 5 for more learning iterations
            'policy_weight': 1.5,   # Increased from 1.0 to emphasize policy learning
            'value_weight': 1.0,    # Weight for value loss kept as is
            'lr_decay': 0.9995      # Learning rate decay factor
        }
        self.model_path = model_path
        self.quiet = quiet  # Add quiet flag
        
        # Check for GPU availability
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not quiet:
            print(f"Using device: {self.device}")
        
        self.net = Connect4CNN().to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.hyperparams['learning_rate'])
        
        # Setup for mixed precision training if on CUDA
        self.use_mixed_precision = self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            if not quiet:
                print("Mixed precision training enabled")
        
        self.data = {'states': [], 'policies': [], 'values': []}
        self.pending = []  # moves from current game
        self.train_iterations = 0  # Track number of training iterations
        self.total_games = 0      # Track total number of games trained on
        
        if os.path.exists(model_path):
            try:
                # Use weights_only=True to avoid security warnings
                ck = torch.load(model_path, map_location=self.device, weights_only=True)
                self.net.load_state_dict(ck['model_state_dict'])
                self.opt.load_state_dict(ck['optimizer_state_dict'])
                if 'train_iterations' in ck:
                    self.train_iterations = ck['train_iterations']
                # Load total_games if present, otherwise set to 0 for backward compatibility
                if 'total_games' in ck:
                    self.total_games = ck['total_games']
                else:
                    self.total_games = 0  # Default value for backward compatibility
                if 'hyperparams' in ck:
                    # Keep user settings but update with any new hyperparams from saved model
                    saved_params = ck['hyperparams']
                    for key in saved_params:
                        if key not in self.hyperparams:
                            self.hyperparams[key] = saved_params[key]
                if not quiet:
                    print(f"Model loaded from {model_path}. Training iterations: {self.train_iterations}, Total games: {self.total_games}")
            except Exception as e:
                if not quiet:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using fresh weights.")
        elif not quiet:
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
        # Validate state
        assert state.board.shape == (ROW_COUNT, COLUMN_COUNT), f"Invalid board shape: {state.board.shape}"
        
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
        
        # Balance training data by column
        if len(self.data['states']) > 1000:
            policies = torch.stack(self.data['policies']).cpu().numpy()
            max_indices = np.argmax(policies, axis=1)
            unique_moves, move_counts = np.unique(max_indices, return_counts=True)
            
            # If any column is more than 3x overrepresented
            if np.max(move_counts) > 3 * np.min(move_counts) and len(unique_moves) == COLUMN_COUNT:
                # Find underrepresented columns
                avg_count = np.mean(move_counts)
                under_rep_cols = [col for i, col in enumerate(unique_moves) if move_counts[i] < avg_count * 0.7]
                over_rep_cols = [col for i, col in enumerate(unique_moves) if move_counts[i] > avg_count * 1.3]
                
                if under_rep_cols and over_rep_cols:
                    # Keep more examples of underrepresented columns
                    # Keep fewer examples of overrepresented columns
                    indices_to_keep = []
                    for i, max_idx in enumerate(max_indices):
                        if max_idx in under_rep_cols:
                            indices_to_keep.append(i)  # Keep all under-represented examples
                        elif max_idx in over_rep_cols:
                            # Probabilistically drop over-represented examples
                            drop_prob = (move_counts[np.where(unique_moves == max_idx)[0][0]] / avg_count - 1) * 0.3
                            if random.random() > drop_prob:
                                indices_to_keep.append(i)
                        else:
                            indices_to_keep.append(i)  # Keep normal examples
                    
                    # Apply the filtering
                    for k in self.data:
                        self.data[k] = [self.data[k][i] for i in indices_to_keep]
        
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
    def train(self, batch_size=None, epochs=None, start_time=None, logger=None, num_games=1):
        if not self.data['states']:
            if logger:
                logger("No training data available. Skipping training.")
            else:
                print("No training data available. Skipping training.")
            return
        
        # Helper function for logging
        def log(msg):
            if logger:
                logger(msg)
            else:
                print(msg)
        
        batch_size = batch_size or self.hyperparams['batch_size']
        epochs = epochs or self.hyperparams['epochs']
        
        # Get loss weights
        policy_weight = self.hyperparams.get('policy_weight', 1.5)  # Default to 1.5 now
        value_weight = self.hyperparams.get('value_weight', 1.0)
        
        # Apply learning rate decay
        lr_decay = self.hyperparams.get('lr_decay', 0.9995)
        if lr_decay < 1.0:
            current_lr = self.opt.param_groups[0]['lr']
            new_lr = max(current_lr * lr_decay, 1e-6)  # Lower floor to 1e-6 for finer control
            for param_group in self.opt.param_groups:
                param_group['lr'] = new_lr
            log(f"Learning rate adjusted: {current_lr:.6f} → {new_lr:.6f}")
        
        # Create dataset and loader
        ds = Connect4Dataset(torch.stack(self.data['states']),
                            torch.stack(self.data['policies']),
                            torch.stack(self.data['values']).squeeze(1))
        
        # Create weighted random sampler for better balance
        num_samples = len(ds)
        if num_samples > 1000:  # Only use weighted sampling for substantial datasets
            # Give more weight to older examples (which might be underrepresented)
            sample_weights = torch.ones(num_samples)
            half_point = num_samples // 2
            sample_weights[:half_point] = 1.2
            sample_weights[half_point:] = 0.8
            
            # Use WeightedRandomSampler instead of shuffle=True
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=num_samples,
                replacement=True
            )
            dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), sampler=sampler)
        else:
            # For small datasets, just shuffle
            dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)
        
        # Print a header for the training section
        log("\n" + "="*50)
        log(f"TRAINING NEURAL NETWORK - {num_samples} examples")
        log("="*50)
        
        # Analyze training data quality
        self._analyze_training_data(logger)
        
        self.net.train()
        all_policy_losses = []
        all_value_losses = []
        
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
            
            # Calculate average loss values for this epoch
            policy_loss_avg = p_loss_sum/len(dl)
            value_loss_avg = v_loss_sum/len(dl)
            
            # Store losses for analysis
            all_policy_losses.append(policy_loss_avg)
            all_value_losses.append(value_loss_avg)
            
            # Calculate elapsed time
            elapsed = ""
            if start_time is not None:
                elapsed_seconds = time.time() - start_time
                minutes, seconds = divmod(elapsed_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                elapsed = f" - Time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Print training progress with policy/value losses and elapsed time - make it more visible
            log(f"[Epoch {ep+1}/{epochs}] POLICY={policy_loss_avg:.6f}  VALUE={value_loss_avg:.6f}{elapsed}")
            
            # Force stdout to flush to ensure output appears immediately
            sys.stdout.flush()
        
        # Analyze training progress
        if len(all_policy_losses) > 1:
            policy_improvement = all_policy_losses[0] - all_policy_losses[-1]
            value_improvement = all_value_losses[0] - all_value_losses[-1]
            log(f"Policy loss improvement: {policy_improvement:.6f} ({policy_improvement/all_policy_losses[0]*100:.1f}%)")
            log(f"Value loss improvement: {value_improvement:.6f} ({value_improvement/all_value_losses[0]*100:.1f}%)")
            
            if policy_improvement < 0:
                log("WARNING: Policy loss is increasing! Consider adjusting hyperparameters.")
            if value_improvement < 0:
                log("WARNING: Value loss is increasing! Consider adjusting hyperparameters.")
        
        # Print a line to separate training output
        log("-"*50)
        log(f"Training complete - {self.train_iterations+1} iterations")
        log("-"*50 + "\n")
        
        # Increment training iterations counter
        self.train_iterations += 1
        
        # Update total games counter
        self.total_games += num_games
        
        # Save model with additional information
        torch.save({
            'model_state_dict': self.net.state_dict(), 
            'optimizer_state_dict': self.opt.state_dict(),
            'train_iterations': self.train_iterations,
            'total_games': self.total_games,
            'hyperparams': self.hyperparams
        }, self.model_path)
        
        # Verify model was saved
        if os.path.exists(self.model_path):
            log(f"Model successfully saved to {self.model_path}")
            log(f"Total games trained on: {self.total_games}")
        else:
            log(f"WARNING: Failed to save model to {self.model_path}")
    
    def _analyze_training_data(self, logger=None):
        """Analyze training data quality to help diagnose issues"""
        # Helper function for logging
        def log(msg):
            if logger:
                logger(msg)
            else:
                print(msg)
                
        if not self.data['states'] or len(self.data['states']) == 0:
            log("No training data to analyze.")
            return
            
        # Check value distribution
        values = torch.stack(self.data['values']).numpy().flatten()
        win_count = np.sum(values > 0)
        loss_count = np.sum(values < 0)
        draw_count = np.sum(values == 0)
        total = len(values)
        
        log(f"Training data value distribution:")
        log(f"  Wins: {win_count} ({win_count/total*100:.1f}%)")
        log(f"  Losses: {loss_count} ({loss_count/total*100:.1f}%)")
        log(f"  Draws: {draw_count} ({draw_count/total*100:.1f}%)")
        
        # Check policy distribution (how diverse the move choices are)
        policies = torch.stack(self.data['policies']).cpu().numpy()
        max_indices = np.argmax(policies, axis=1)
        unique_moves, move_counts = np.unique(max_indices, return_counts=True)
        
        log(f"Move distribution (columns 0-6):")
        for move, count in zip(unique_moves, move_counts):
            log(f"  Column {move}: {count} moves ({count/total*100:.1f}%)")
            
        # Check for very lopsided distribution
        if len(unique_moves) < 3 or np.max(move_counts) / total > 0.7:
            log("WARNING: Move distribution is very uneven, which may lead to poor generalization.")
            
        # Analytics summary
        log(f"Total training examples: {total}")
        if total < 1000:
            log("WARNING: Small training dataset may lead to overfitting.")
        elif total > 10000:
            log("Good dataset size for training.")
        log("-" * 30)

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
        self.I = iterations
        self.c = c_puct
        self.nn = nn
        self.explore = explore

    def search(self, root_state: 'Connect4Game'):
        # Debug to verify if explore setting is working
        # print(f"MCTS search with exploration={'enabled' if self.explore else 'disabled'}")
        
        # Quick check for immediate winning or blocking moves
        valid_moves = root_state.valid_moves()
        
        # First check if we can win immediately
        for move in valid_moves:
            test_state = root_state.copy()
            test_state.drop_piece(move)
            if test_state.game_over and test_state.winner == root_state.current_player:
                # If we can win immediately, return this move with minimal computation
                vp = np.zeros(COLUMN_COUNT, dtype=np.float32); vp[move] = 1.0
                self.nn.add_example(root_state, vp)
                return move
        
        # IMPROVED: Check if opponent can win immediately with their next move
        opponent = GREEN_PIECE if root_state.current_player == RED_PIECE else RED_PIECE
        for move in valid_moves:
            test_state = root_state.copy()
            test_state.current_player = opponent  # Set to opponent
            test_state.drop_piece(move)
            if test_state.game_over and test_state.winner == opponent:
                # Block this immediate threat
                vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
                vp[move] = 1.0
                self.nn.add_example(root_state, vp)
                return move
        
        # Find moves that don't allow opponent to win on next turn
        safe_moves = []
        for move in valid_moves:
            test_state = root_state.copy()
            test_state.drop_piece(move)
            test_state.switch()
            
            # Check if this move allows the opponent to win
            opponent_can_win = False
            for opp_move in test_state.valid_moves():
                test_state2 = test_state.copy()
                test_state2.drop_piece(opp_move)
                if test_state2.game_over and test_state2.winner == opponent:
                    opponent_can_win = True
                    break
            
            if not opponent_can_win:
                # This move is safe - opponent cannot win next turn
                safe_moves.append(move)

        # If we found any safe moves, use one (prioritize center columns if possible)
        if safe_moves:
            # Try to pick a central column if available in safe moves
            center_moves = [m for m in safe_moves if m == COLUMN_COUNT // 2]  # Center column
            if center_moves:
                chosen_move = center_moves[0]  # Use center column
            else:
                chosen_move = random.choice(safe_moves)  # Otherwise, pick randomly
            
            vp = np.zeros(COLUMN_COUNT, dtype=np.float32); vp[chosen_move] = 1.0
            self.nn.add_example(root_state, vp)
            return chosen_move
        
        # No immediate win or block needed, proceed with normal MCTS
        root = TreeNode(root_state.copy())
        # initial prior via NN + Dirichlet noise if explore is True
        prior, _ = self.nn.policy_value(root.state)
        valid = root.state.valid_moves()
        
        # Initialize array for child priors with center bias
        child_priors = np.zeros(COLUMN_COUNT, dtype=np.float32)
        
        # Only add Dirichlet noise if explore is True (not full strength)
        if self.explore:
            dirichlet = np.random.dirichlet([0.3]*len(valid))
            
            # Apply center bias and mix with Dirichlet noise
            for i, m in enumerate(valid):
                child_priors[m] = 0.75 * prior[m] * CENTER_BIAS[m] + 0.25 * dirichlet[i]
        else:
            # When in full strength mode (explore=False), use pure policy from NN with center bias
            for m in valid:
                child_priors[m] = prior[m] * CENTER_BIAS[m]
        
        # FIX 4: Renormalize priors after applying center bias and Dirichlet noise
        valid_priors_sum = sum(child_priors[m] for m in valid)
        if valid_priors_sum > 0:  # Guard against division by zero
            for m in valid:
                child_priors[m] /= valid_priors_sum
        else:
            # If all priors are zero (shouldn't happen), use uniform distribution
            for m in valid:
                child_priors[m] = 1.0 / len(valid)
        
        # Create child nodes with properly normalized priors
        for m in valid:
            ns = root.state.copy()
            ns.drop_piece(m)
            if not ns.game_over:
                ns.switch()
            root.children[m] = TreeNode(ns, parent=root, move=m, prior=child_priors[m])

        for _ in range(self.I):
            node = root
            # SELECT
            while node.children:
                node = node.best_child(self.c)
            # EXPAND & EVALUATE (if game not over)
            if not node.state.game_over:
                probs, value = self.nn.policy_value(node.state)
                valid = node.state.valid_moves()
                
                # FIX 4: Apply center bias and renormalize for expansion
                child_priors = np.zeros(COLUMN_COUNT, dtype=np.float32)
                for m in valid:
                    child_priors[m] = probs[m] * CENTER_BIAS[m]
                
                # Renormalize
                valid_priors_sum = sum(child_priors[m] for m in valid)
                if valid_priors_sum > 0:
                    for m in valid:
                        child_priors[m] /= valid_priors_sum
                else:
                    # Fallback to uniform if all zero (shouldn't happen)
                    for m in valid:
                        child_priors[m] = 1.0 / len(valid)
                
                # Create children with normalized priors
                for m in valid:
                    if m not in node.children:
                        ns = node.state.copy()
                        ns.drop_piece(m)
                        if not ns.game_over:
                            ns.switch()
                        node.children[m] = TreeNode(ns, parent=node, move=m, prior=child_priors[m])
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
        # FIX 1: When explore=False (Play @ Full Strength), keep c_puct small but positive
        # Use a reduced c_puct (0.3) rather than 0.0 to maintain policy influence
        self.mcts = MCTS(iters, c if explore else 0.3, nn, explore=explore)
        
    def get_move(self, state, gui=None):
        # FIX 5: Removed immediate win/block detection - now handled in MCTS.search()
        # Just call MCTS search directly
        start = time.time()
        mv = self.mcts.search(state)
        dt = time.time() - start
        if dt < 0.1:
            time.sleep(0.1 - dt)
        return mv

# Function to collect training examples from self-play - returns full visit probabilities
def _play_single_training_game(mcts_iterations, puct_c, nn_manager_config):
    """
    Play a single training game and return serializable data.
    """
    try:
        # Create local copies of neural network for thread safety
        model_path = nn_manager_config.get('model_path', NN_MODEL_FILE)
        hyperparams = nn_manager_config.get('hyperparams', None)
        nn_copy = NNManager(hyperparams, model_path, quiet=True)
        
        # Create AI players with exploration enabled for training
        ai_red = MCTSComputerPlayer(mcts_iterations, puct_c, nn_copy, explore=True)
        ai_green = MCTSComputerPlayer(mcts_iterations, puct_c, nn_copy, explore=True)
        
        # Play the game and collect training examples
        game = Connect4Game()
        serialized_examples = []
        
        while not game.game_over:
            player = game.current_player
            player_ai = ai_red if player == RED_PIECE else ai_green
            
            # Save state before the move
            state_before = game.copy()
            board_before = state_before.board.copy().tolist()
            player_before = state_before.current_player
            
            # Get move from MCTS
            move = player_ai.get_move(game)
            
            # Get the policy from the most recent example
            if nn_copy.pending:
                policy_data = nn_copy.pending[-1]['policy'].numpy().tolist()
                
                # Create serializable example
                example_data = {
                    'board': board_before,
                    'player': player_before,
                    'policy': policy_data
                }
                serialized_examples.append(example_data)
            
            # Make the move
            game.drop_piece(move)
            
            if not game.game_over:
                game.switch()
        
        # Return simple Python types that can be safely serialized
        return serialized_examples, game.winner
    except Exception as e:
        # Print any errors that occur
        print(f"Error in _play_single_training_game: {e}")
        import traceback
        traceback.print_exc()
        raise

# ----------------------------------------------------------------------
# Dialogs (updated to include settings dialog)
# ----------------------------------------------------------------------
class PlayerDialog(simpledialog.Dialog):
    def __init__(self, master, mcts_params, nn):
        self.mcts = mcts_params
        self.nn = nn
        
        # Set default values from master if available, otherwise use defaults
        self.p1 = tk.StringVar(master, master.last_p1 if hasattr(master, 'last_p1') else "Human")
        self.p2 = tk.StringVar(master, master.last_p2 if hasattr(master, 'last_p2') else "Computer (AI)")
        self.red_model = tk.StringVar(master, master.last_red_model if hasattr(master, 'last_red_model') else NN_MODEL_FILE)
        self.green_model = tk.StringVar(master, master.last_green_model if hasattr(master, 'last_green_model') else NN_MODEL_FILE)
        self.continuous_play = tk.BooleanVar(master, master.last_continuous_play if hasattr(master, 'last_continuous_play') else False)
        self.fullplay_var = tk.BooleanVar(master, master.last_fullplay if hasattr(master, 'last_fullplay') else False)
        
        super().__init__(master, "Results:") # Changed title to "Results:"
        
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
        
        # Play @ Full Strength checkbox moved from main window to here
        ttk.Separator(m, orient="horizontal").grid(row=2, column=0, columnspan=6, sticky="ew", pady=10)
        
        options_frame = ttk.Frame(m)
        options_frame.grid(row=3, column=0, columnspan=6, sticky="w", padx=5, pady=5)
        
        # Play @ Full Strength checkbox
        self.fullplay_check = ttk.Checkbutton(options_frame, text="Play @ Full Strength", variable=self.fullplay_var)
        self.fullplay_check.grid(row=0, column=0, sticky="w", padx=5)
        ToolTip(self.fullplay_check, "When enabled, AI plays at full strength without exploration noise")
        
        # Add Continuous Play checkbox next to Play @ Full Strength
        self.continuous_check = ttk.Checkbutton(options_frame, text="Continuous Play", variable=self.continuous_play)
        self.continuous_check.grid(row=0, column=1, sticky="w", padx=25)
        ToolTip(self.continuous_check, "When enabled, computer vs computer games will play continuously\nuntil Stop is pressed or max games is reached")
        
        # Initial state update for all UI elements
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
            
        # Check if any player is AI to determine if Play @ Full Strength should be enabled
        has_ai = (self.p1.get() == "Computer (AI)" or self.p2.get() == "Computer (AI)")
        if has_ai:
            self.fullplay_check.state(['!disabled'])  # Enable
        else:
            self.fullplay_check.state(['disabled'])  # Disable
            
        # Update Continuous Play checkbox state
        is_comp = all(p != "Human" for p in [self.p1.get(), self.p2.get()])
        if is_comp:
            self.continuous_check.state(['!disabled'])  # Enable
            self.continuous_play.set(True)  # Set to True by default for AI vs AI
        else:
            self.continuous_check.state(['disabled'])  # Disable
            self.continuous_play.set(False)  # Reset to False
    
    def apply(self):
        # Store selected values in the master window for next time
        self.master.last_p1 = self.p1.get()
        self.master.last_p2 = self.p2.get()
        self.master.last_red_model = self.red_model.get()
        self.master.last_green_model = self.green_model.get()
        self.master.last_continuous_play = self.continuous_play.get()
        self.master.last_fullplay = self.fullplay_var.get()
        
        # Store the full_strength value in master's fullplay_var
        self.master.fullplay_var = tk.BooleanVar(self.master, self.fullplay_var.get())
        
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
                    explore=not self.fullplay_var.get(),
                    model_path=model_path
                )
            else:
                # Use the default NN if model file doesn't exist
                return MCTSComputerPlayer(
                    self.mcts['iterations'], 
                    self.mcts['C_param'], 
                    self.nn,
                    explore=not self.fullplay_var.get()
                )
                
        self.result = {
            'red': mk_player(self.p1.get(), self.red_model.get() if self.p1.get() == "Computer (AI)" else None),
            'green': mk_player(self.p2.get(), self.green_model.get() if self.p2.get() == "Computer (AI)" else None),
            'continuous_play': self.continuous_play.get(),  # Include continuous play setting in result
            'full_strength': self.fullplay_var.get()  # Include full_strength setting in result
        }

class SettingsDialog(simpledialog.Dialog):
# SettingsDialog.__init__ method
    def __init__(self, master, mcts_params, train_games=200, max_cc_games=100, cc_train_interval=50, cc_delay=500, nn_params=None):
        self.it = tk.StringVar(master, str(mcts_params['iterations']))
        self.c = tk.StringVar(master, f"{mcts_params['C_param']:.2f}")
        self.games = tk.StringVar(master, str(train_games))
        self.max_cc = tk.StringVar(master, str(max_cc_games))
        self.cc_train_interval = tk.StringVar(master, str(cc_train_interval))  # NEW: C-C games before training
        self.cc_delay = tk.StringVar(master, str(cc_delay))  # NEW: Delay in C-C games
        
        # Initialize with default values if not provided
        self.nn_params = nn_params or {
            'learning_rate': 5e-4, 
            'batch_size': 128, 
            'epochs': 10,
            'policy_weight': 1.5,
            'value_weight': 1.0,
            'lr_decay': 0.9995
        }
    
        # Set up string vars for all parameters
        self.lr = tk.StringVar(master, str(self.nn_params.get('learning_rate', 5e-4)))
        self.batch = tk.StringVar(master, str(self.nn_params.get('batch_size', 128)))
        self.epochs = tk.StringVar(master, str(self.nn_params.get('epochs', 10)))
        self.policy_weight = tk.StringVar(master, str(self.nn_params.get('policy_weight', 1.5)))
        self.value_weight = tk.StringVar(master, str(self.nn_params.get('value_weight', 1.0)))
        self.lr_decay = tk.StringVar(master, str(self.nn_params.get('lr_decay', 0.9995)))
        
        super().__init__(master, "Settings")
            
    # Modified SettingsDialog.body method - adding new fields for C-C game settings
    def body(self, m):
        # Create a frame for feedback messages at the top
        self.feedback_frame = ttk.Frame(m)
        self.feedback_frame.pack(fill='x', pady=5)
        
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
        
        # Training tab - updated field names
        train_tab = ttk.Frame(nb)
        nb.add(train_tab, text="Training")
        
        # Changed field name from "Self-play games:" to "Games in Train NN mode:"
        ttk.Label(train_tab, text="Games in Train NN mode:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        e3 = ttk.Entry(train_tab, textvariable=self.games, width=10)
        e3.grid(row=0, column=1, padx=5)
        ToolTip(e3, "Number of self-play games for training when using Train NN button")
        
        # Changed field name from "Max C-C games:" to "Games in AI-AI mode:"
        ttk.Label(train_tab, text="Games in AI-AI mode:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        e3b = ttk.Entry(train_tab, textvariable=self.max_cc, width=10)
        e3b.grid(row=1, column=1, padx=5)
        ToolTip(e3b, "Maximum number of games in continuous Computer-Computer play")
        
        # NEW: Add field for C-C games before training
        ttk.Label(train_tab, text="C-C games before training:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        e3c = ttk.Entry(train_tab, textvariable=self.cc_train_interval, width=10)
        e3c.grid(row=2, column=1, padx=5)
        ToolTip(e3c, "Number of Computer-Computer games to play before training the neural network")
        
        # NEW: Add field for delay in C-C games
        ttk.Label(train_tab, text="Delay in C-C games (ms):").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        e3d = ttk.Entry(train_tab, textvariable=self.cc_delay, width=10)
        e3d.grid(row=3, column=1, padx=5)
        ToolTip(e3d, "Delay in milliseconds between moves in Computer-Computer games")
        
        # Add training presets to the training tab
        self._add_training_presets(train_tab)
        
        # Advanced NN tab
        nn_tab = ttk.Frame(nb)
        nb.add(nn_tab, text="Advanced")
        
        ttk.Label(nn_tab, text="Learning rate:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        e4 = ttk.Entry(nn_tab, textvariable=self.lr, width=10)
        e4.grid(row=0, column=1, padx=5)
        ToolTip(e4, "Neural network learning rate\nTypical values: 0.0005-0.0001")
        
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
    
    #def _add_training_presets(self, train_tab):
    def _add_training_presets(self, train_tab):
        """Add training presets to the settings dialog"""
        # Training presets frame - CHANGED ROW FROM 2 TO 4 to avoid conflict with fields
        preset_frame = ttk.LabelFrame(train_tab, text="Training Presets")
        preset_frame.grid(row=4, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
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
                "iterations": 600,  # Reduced from 800 to be more responsive
                "batch_size": 64,
                "epochs": 8,  # Reduced from a10 to be more efficient
                "description": "High quality with reasonable speed"
            }
        }
        
        # Store button references for styling
        self.preset_buttons = {}
        
        # Preset buttons
        for i, (name, settings) in enumerate(self.presets.items()):
            # Create a style for the button
            style_name = f"{name}.TButton"
            if not hasattr(self, 'button_style_created'):
                ttk.Style().configure(style_name, background="#E0E0E0")
                
            btn = ttk.Button(
                preset_frame, 
                text=name,
                style=style_name,
                command=lambda n=name, s=settings: self._apply_preset(n, s)
            )
            btn.grid(row=0, column=i, padx=5, pady=5)
            self.preset_buttons[name] = btn
            ToolTip(btn, f"{name}: {settings['description']}\n" + 
                    f"MCTS: {settings['iterations']} iterations\n" +
                    f"Batch: {settings['batch_size']}, Epochs: {settings['epochs']}")
        
        # Add explanation of presets
        ttk.Label(preset_frame, text="Select a preset to quickly configure training settings",
                  font=("Helvetica", 8)).grid(row=1, column=0, columnspan=3, pady=(0,5))
        
        # Store the currently active preset
        self.active_preset = None
    
    def _apply_preset(self, preset_name, settings):
        """Apply a training preset"""
        # Update MCTS iterations
        self.it.set(str(settings['iterations']))
        
        # Update NN parameters
        self.batch.set(str(settings['batch_size']))
        self.epochs.set(str(settings['epochs']))
        
        # Clear any existing feedback
        for widget in self.feedback_frame.winfo_children():
            widget.destroy()
            
        # Provide feedback using Pack geometry manager
        feedback_label = ttk.Label(self.feedback_frame, text=f"{preset_name} preset applied", foreground="green")
        feedback_label.pack(side="left", padx=5)
        
        # Update button styles to highlight the active preset
        for name, btn in self.preset_buttons.items():
            style_name = f"{name}.TButton"
            if name == preset_name:
                # Highlight active preset with blue background
                ttk.Style().configure(style_name, background="#CCE5FF")
                self.active_preset = preset_name
            else:
                # Reset other buttons to default
                ttk.Style().configure(style_name, background="#E0E0E0")
        
        # Auto-remove feedback after 2 seconds
        self.after(2000, lambda: feedback_label.destroy())
        
    # SettingsDialog.validate method
    def validate(self):
        try:
            # Validate MCTS params
            it = int(self.it.get())
            c = float(self.c.get())
            games = int(self.games.get())
            max_cc = int(self.max_cc.get())
            cc_train_interval = int(self.cc_train_interval.get())  # NEW: Validate C-C games before training
            cc_delay = int(self.cc_delay.get())  # NEW: Validate delay in C-C games
            
            # Validate NN params
            lr = float(self.lr.get())
            batch = int(self.batch.get())
            epochs = int(self.epochs.get())
            p_weight = float(self.policy_weight.get())
            v_weight = float(self.value_weight.get())
            lr_decay = float(self.lr_decay.get())
            
            # Check positive values where needed
            if (it <= 0 or c < 0 or games <= 0 or max_cc <= 0 or 
                cc_train_interval <= 0 or cc_delay < 0 or  # NEW: Check new fields
                lr <= 0 or batch <= 0 or epochs <= 0 or 
                p_weight <= 0 or v_weight <= 0):
                messagebox.showwarning("Invalid", "All values must be positive (delay can be zero).")
                return False
                    
            # Check LR decay specifically
            if lr_decay <= 0 or lr_decay > 1.0:
                messagebox.showwarning("Invalid", "Learning rate decay must be between 0 and 1.0")
                return False
                    
            return True
        except:
            messagebox.showwarning("Invalid", "Enter valid numbers.")
            return False
        
    # SettingsDialog.apply method
    def apply(self):
        self.result = {
            'mcts': {
                'iterations': int(self.it.get()),
                'C_param': float(self.c.get())
            },
            'training': {
                'games': int(self.games.get()),
                'max_cc_games': int(self.max_cc.get()),
                'cc_train_interval': int(self.cc_train_interval.get()),  # NEW: C-C games before training
                'cc_delay': int(self.cc_delay.get())  # NEW: Delay in C-C games
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
# Stop Confirmation Dialog
# ----------------------------------------------------------------------
class StopConfirmationDialog(simpledialog.Dialog):
    def __init__(self, master, mode="training"):
        self.mode = mode  # "training" or "continuous"
        super().__init__(master, "Confirm Stop")
        
    def body(self, m):
        if self.mode == "training":
            question = "Are you sure you want to stop the training?"
        else:  # continuous
            question = "Are you sure you want to stop the continuous play?"
            
        ttk.Label(m, text=question, font=("Helvetica", 11)).grid(row=0, column=0, columnspan=2, pady=10, padx=20)
        return None  # No specific widget to get initial focus
        
    def buttonbox(self):
        box = ttk.Frame(self)
        
        stop_btn = ttk.Button(box, text="Stop", width=10, command=self.ok)
        stop_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        continue_btn = ttk.Button(box, text="Continue", width=10, command=self.cancel)
        continue_btn.pack(side=tk.LEFT, padx=5, pady=5)
        
        self.bind("<Return>", self.ok)
        self.bind("<Escape>", self.cancel)
        
        box.pack()
        
    def apply(self):
        # Called when OK button (Stop) is clicked
        self.result = True  # Set result to True for Stop

# ----------------------------------------------------------------------
# GUI – Connect4GUI (updated for new features)
# ----------------------------------------------------------------------

class Connect4GUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Connect 4 – AlphaZero Edition (2025)")
        
        # Make window resizable
        self.resizable(True, True)  # Changed from (False, False) to (True, True)
        
        # Configure window to use grid weights for resizing
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.mcts_params = {'iterations': DEFAULT_MCTS_ITERATIONS, 'C_param': DEFAULT_PUCT_C}
        self.nn_params = {
            'learning_rate': 5e-4,  # Lowered from 1e-3 
            'batch_size': 128,      # Increased from 64
            'epochs': 10,           # Increased from 5
            'policy_weight': 1.5,   # Increased from 1.0
            'value_weight': 1.0,    # Weight for value loss kept as is
            'lr_decay': 0.9995      # Learning rate decay factor
        }
        self.train_games = DEFAULT_TRAIN_GAMES
        self.max_cc_games = 100  # Default maximum number of Computer-Computer games
        self.cc_train_interval = 50  # NEW: Default C-C games before training
        self.cc_delay = 500  # NEW: Default delay in C-C games (ms)
        self.continuous_play = False  # Flag for continuous play mode
        self.training_in_progress = False
        self.training_stop_requested = False
        self.training_model_path = NN_MODEL_FILE  # Default training model path
        self.play_till_end = False  # NEW: Flag for playing till end of game
        self.games_since_training = 0  # NEW: Counter for games since last training
        self.train_blink_job = None  # NEW: For blinking "Training NN" text
        
        # Initialize last player settings
        self.last_p1 = "Human"
        self.last_p2 = "Computer (AI)"
        self.last_red_model = NN_MODEL_FILE
        self.last_green_model = NN_MODEL_FILE
        self.last_continuous_play = False
        self.last_fullplay = False
        
        # Create fullplay_var (now moved to PlayerDialog but we need this for initialization)
        self.fullplay_var = tk.BooleanVar(self, False)
        
        self.nn = NNManager(self.nn_params, self.training_model_path)
        self._load_cfg()

        self.game = Connect4Game()
        self.players = {RED_PIECE: None, GREEN_PIECE: None}
        self.score = {'red':0,'green':0,'draws':0,'games':0}
        self.turn_count=0; self.auto_job=None; self.game_in_progress=False
        self.is_comp=False; self.paused=False; self.last_hover=None
        
        # Flag to track if players have been selected
        self.players_selected = False
        
        # For thread-safe logging
        self.log_buffer = []
        self.log_buffer_lock = threading.Lock()
        self.log_last_update = time.time()
        self.log_update_interval = 0.5  # Update at most every 0.5 seconds

        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0, sticky="nsew")  # Make main frame expand with window
        
        # Configure main frame rows and columns to expand
        main.grid_columnconfigure(1, weight=1)  # Side panel can expand
        main.grid_rowconfigure(0, weight=1)  # Make content expand vertically
        
        self.canvas = tk.Canvas(main, width=WIDTH, height=HEIGHT, bg=BLUE, highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=3, padx=(0,10), sticky="nsew")  # Make canvas expand
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Motion>", self._hover)
        ToolTip(self.canvas, "Click a column to drop a piece")
        
        side = ttk.Frame(main)
        side.grid(row=0, column=1, rowspan=3, sticky="nsew")  # Make side panel expand
        side.grid_rowconfigure(6, weight=1)  # Training log area can expand
        side.grid_columnconfigure(0, weight=1)  # Make columns expandable

        # Show the game status - renamed to "Results:"
        self.status = ttk.Label(side, font=("Helvetica",16,"bold"), width=20, text="Results:")
        self.status.grid(row=0, column=0, columnspan=3, sticky="ew", pady=(0,5))
        ToolTip(self.status, "Game status")
        
        self.score_lbl = ttk.Label(side, font=("Helvetica",12))
        self.score_lbl.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0,10))
        ToolTip(self.score_lbl, "Score summary")

        # Moved settings icon to the right
        settings_btn = ttk.Button(side, text="⚙", width=3, command=self._settings)
        settings_btn.grid(row=0, column=3, padx=5, sticky="e")
        ToolTip(settings_btn, "Settings (MCTS, Training, Advanced)")
        
        # Create new row for checkboxes and Train NN button
        control_options = ttk.Frame(side)
        control_options.grid(row=2, column=0, columnspan=4, sticky="ew", pady=5)
        
        # Learn @ Play checkbox
        self.learn_var = tk.BooleanVar(self, True)
        self.learn_check = ttk.Checkbutton(control_options, text="Learn @ Play", variable=self.learn_var)
        self.learn_check.grid(row=0, column=0, sticky="w", padx=5)
        self.learn_check.state(['disabled'])  # Disabled until players are selected
        ToolTip(self.learn_check, "When enabled, game moves are used to train the neural network")

        # Play @ Full Strength checkbox removed from here (moved to Player Dialog)
        
        # Train NN button and model selector moved to same row as checkboxes
        train_frame = ttk.Frame(control_options)
        train_frame.grid(row=0, column=1, padx=5, sticky="e")
        
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

        # History text - renamed from "History:" to "Game History:"
        hist = ttk.Frame(side); hist.grid(row=3, column=0, columnspan=4, sticky="nsew")
        hist.grid_rowconfigure(1, weight=1); hist.grid_columnconfigure(0, weight=1)
        ttk.Label(hist, text="Game History:").grid(row=0, column=0, sticky="w")
        self.moves = tk.Text(hist, width=28, height=15, font=("Courier",10), state="disabled")
        scr = ttk.Scrollbar(hist, command=self.moves.yview); self.moves['yscrollcommand']=scr.set
        self.moves.grid(row=1, column=0, sticky="nsew"); scr.grid(row=1, column=1, sticky="ns")
        ToolTip(self.moves, "Game move history")

        # Control buttons: New Game and Select Players
        control_frame = ttk.Frame(side); control_frame.grid(row=4, column=0, columnspan=4, pady=5)
        
        new_game_btn = ttk.Button(control_frame, text="New Game", command=self._new_game)
        new_game_btn.pack(side="left", padx=5)
        ToolTip(new_game_btn, "Start a new game with current players")
        
        select_players_btn = ttk.Button(control_frame, text="Select Players", command=self._choose_players)
        select_players_btn.pack(side="left", padx=5)
        ToolTip(select_players_btn, "Choose new players")
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", state="disabled", command=self._pause)
        self.stop_btn.pack(side="left", padx=5)
        ToolTip(self.stop_btn, "Stop the current game, training, or continuous play")

        # Training progress area - initially hidden, will appear when Train NN is pressed
        self.separator = ttk.Separator(side, orient='horizontal')
        self.separator.grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)
        self.separator.grid_remove()  # Hide initially
        
        self.train_frame = ttk.LabelFrame(side, text="Training Progress")
        self.train_frame.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=5)
        self.train_frame.grid_columnconfigure(0, weight=1)
        self.train_frame.grid_rowconfigure(3, weight=1)  # Make training log area expandable
        self.train_frame.grid_remove()  # Hide initially
        
        self.train_progress = ttk.Progressbar(self.train_frame, length=250, mode="determinate")
        self.train_progress.grid(row=0, column=0, padx=5, pady=(5,0), sticky="ew")
        
        self.train_status = ttk.Label(self.train_frame, text="Idle", anchor="center")
        self.train_status.grid(row=1, column=0, padx=5, pady=(0,5), sticky="ew")
        
        # Model info label
        self.model_info = ttk.Label(self.train_frame, text="", anchor="center", font=("Helvetica", 8))
        self.model_info.grid(row=2, column=0, padx=5, pady=(0,5), sticky="ew")
        
        # Add training log area
        self.train_log_frame = ttk.LabelFrame(self.train_frame, text="Training Log")
        self.train_log_frame.grid(row=3, column=0, padx=5, pady=(0,5), sticky="nsew")
        self.train_log_frame.grid_columnconfigure(0, weight=1)
        self.train_log_frame.grid_rowconfigure(0, weight=1)
        
        self.train_log = tk.Text(self.train_log_frame, width=28, height=10, font=("Courier", 9), 
                                 wrap="word", state="disabled")
        self.train_log_scr = ttk.Scrollbar(self.train_log_frame, command=self.train_log.yview)
        self.train_log['yscrollcommand'] = self.train_log_scr.set
        self.train_log.grid(row=0, column=0, sticky="nsew", padx=(2,0))
        self.train_log_scr.grid(row=0, column=1, sticky="ns")
        
        # Dictionary to track training loss over time
        self.training_losses = {"policy": [], "value": []}
        
        # Set up periodic log update
        self.after(100, self._update_log_from_buffer)
        
        self._update_model_info()

        # Initialize the board but don't select players yet
        self._draw()
        self._update_score()
        self._set_status("Results:")  # Set default status
        
        # Add window state and size settings
        self.state('normal')  # Start in normal mode - can be maximized
        
        # Set minimum window size
        width = WIDTH + 350  # Canvas width + side panel width
        height = HEIGHT + 50  # Add some margin
        self.minsize(width, height)
        
        # Set initial size slightly larger than minimum
        self.geometry(f"{width+50}x{height+50}")    
    
    def _blink_training_text(self):
        if not self.training_in_progress:
            # Stop blinking if training is no longer in progress
            self.train_status.config(text="Training completed")
            self.train_status.config(foreground="black")  # Reset to normal color
            return
        
        # Toggle text visibility/color
        current_color = self.train_status.cget("foreground")
        if current_color == "red":
            self.train_status.config(foreground="black")
        else:
            self.train_status.config(foreground="red")
        
        # Call this method again after 500ms to create blinking effect
        self.train_blink_job = self.after(500, self._blink_training_text)

    
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
        
        # Include total games trained on in the model info
        total_games = self.nn.total_games
        self.model_info.config(text=f"Active model: {model_name} (Trained on {total_games} games)")
        
        # Update the label next to the browse button (with bold blue text)
        self.train_model_label.config(text=model_name, foreground="#0000FF", font=("Helvetica", 9, "bold"))
        
        # If the training frame is visible, make the model info visible too
        if str(self.train_frame.grid_info()) != "{}":  # If grid info not empty (means it's visible)
            self.model_info.grid()

    # Modified to check if players have been selected, if not, prompt to select them
    def _new_game(self):
        if self.auto_job: 
            self.after_cancel(self.auto_job)
            self.auto_job = None
                
        # Check if players have been selected, if not, prompt to select them first
        if not self.players_selected:
            self._choose_players()
            return
        
        # Reset play_till_end flag when starting a new game
        self.play_till_end = False
        
        # Only reset games counter when not in continuous play mode or when explicitly starting a new game
        if not self.continuous_play or self.score['games'] == 0:
            self.games_since_training = 0
                
        self.game.reset()
        self.turn_count = 0
        self.last_hover = None
        self.moves.config(state="normal")
        self.moves.delete("1.0", "end")
        self.moves.config(state="disabled")
        self.game_in_progress = True
        self.paused = False
        self._draw()
        
        # Make sure Stop button is enabled for Computer vs Computer games
        if self.is_comp:
            self.stop_btn['state'] = "normal"
        
        # If this is an AI vs AI game and the training log is visible, 
        # add a separator for the new game
        if self.is_comp and str(self.train_frame.grid_info()) != "{}":
            game_number = self.score['games'] + 1
            self.log_to_training(f"\n{'-'*20}\nGame #{game_number}\n{'-'*20}\n")
        
        self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", COLOR_MAP[self.game.current_player])
        self.after(30, self._next_turn)    
    # Choose players dialog - now with full strength option moved here
    def _choose_players(self):
        # Create the dialog with last settings remembered
        dlg = PlayerDialog(self, self.mcts_params, self.nn)
        if not dlg.result: 
            return
            
        self.players[RED_PIECE] = dlg.result['red']
        self.players[GREEN_PIECE] = dlg.result['green']
        self.continuous_play = dlg.result['continuous_play']  # Get continuous play setting
        self.is_comp = all(isinstance(p, (RandomComputerPlayer, MCTSComputerPlayer)) for p in self.players.values())
        
        # Set the players_selected flag
        self.players_selected = True
        
        # Check if any player is an AI player
        has_ai = any(isinstance(p, MCTSComputerPlayer) for p in self.players.values())
        
        if self.is_comp:
            self.stop_btn['state'] = 'normal'
            # Force Learn on for Computer vs Computer
            self.learn_var.set(True)
            self.learn_check.state(['disabled'])
        else:
            # Not computer vs computer
            self.stop_btn['state'] = 'disabled'
            
            # Enable/disable checkboxes based on whether any AI players exist
            if has_ai:
                self.learn_check.state(['!disabled'])  # Enable learn
            else:
                self.learn_check.state(['disabled'])  # Disable learn
        
        # Show training UI for games with AI players
        if has_ai:
            self.separator.grid()
            self.train_frame.grid()
            self.model_info.grid()
            
            # Update model info
            self._update_model_info()
            
            # Reset training log for new session
            self.train_log.config(state="normal")
            self.train_log.delete("1.0", "end")
            
            # Add header about game type
            log_text = ""
            if self.is_comp:
                log_text += "Computer (AI) vs Computer (AI) Game\n"
            else:
                log_text += "Human vs Computer (AI) Game\n"
                
            log_text += "-" * 50 + "\n"
            
            # Add info about the models being used
            red_model = "default"
            green_model = "default"
            
            # Get model names if available
            red_player = self.players[RED_PIECE]
            green_player = self.players[GREEN_PIECE]
            
            if isinstance(red_player, MCTSComputerPlayer) and hasattr(red_player, 'model_path') and red_player.model_path:
                red_model = os.path.splitext(os.path.basename(red_player.model_path))[0]
            
            if isinstance(green_player, MCTSComputerPlayer) and hasattr(green_player, 'model_path') and green_player.model_path:
                green_model = os.path.splitext(os.path.basename(green_player.model_path))[0]
            
            log_text += f"Red Model: {red_model}\n"
            log_text += f"Green Model: {green_model}\n"
            log_text += f"MCTS iterations: {self.mcts_params['iterations']}\n"
            log_text += f"Exploration: {'Disabled' if dlg.result['full_strength'] else 'Enabled'}\n"
            
            if self.is_comp:
                log_text += f"Continuous Play: {'Enabled' if self.continuous_play else 'Disabled'}\n"
                log_text += f"Max games: {self.max_cc_games}\n"
            
            log_text += "-" * 50 + "\n\n"
            self.log_to_training(log_text)
            
            # Setup progress bar for continuous play
            if self.is_comp and self.continuous_play:
                self.train_progress['maximum'] = self.max_cc_games
                self.train_progress['value'] = 0
                self.train_status.config(text=f"Games: 0 / {self.max_cc_games}")
            
            # Reset the loss tracking for new session
            self.training_losses = {"policy": [], "value": []}
        else:
            # Hide training UI if there are no AI players
            self.separator.grid_remove()
            self.train_frame.grid_remove()
        
        self.game_in_progress = True
        self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", COLOR_MAP[self.game.current_player])
        self.after(30, self._next_turn)
    
    # Thread-safe logging function - adds to buffer instead of directly updating UI
    def log_to_training(self, *args, **kwargs):
        """Thread-safe logging - add messages to buffer for later UI update"""
        # Format the message
        message = " ".join(str(arg) for arg in args)
        if "end" in kwargs:
            message += kwargs["end"]
        else:
            message += "\n"
        
        # Print to console
        print(*args, **kwargs)
        
        # Add to buffer with thread safety
        with self.log_buffer_lock:
            self.log_buffer.append(message)
            
    # Periodic UI update from log buffer - runs in main thread
    def _update_log_from_buffer(self):
        """Process buffered log messages on the main thread"""
        # Check if we should update yet
        current_time = time.time()
        if current_time - self.log_last_update > self.log_update_interval:
            # Get all messages from buffer with thread safety
            messages = []
            with self.log_buffer_lock:
                if self.log_buffer:
                    messages = self.log_buffer.copy()
                    self.log_buffer.clear()
            
            # Process messages if any
            if messages and hasattr(self, 'train_log') and self.train_log.winfo_exists():
                self.train_log.config(state="normal")
                for message in messages:
                    self.train_log.insert("end", message)
                    
                    # Highlight policy and value loss lines for easier tracking
                    if "POLICY=" in message:
                        # Extract policy and value losses
                        try:
                            policy_loss = float(message.split("POLICY=")[1].split()[0])
                            value_loss = float(message.split("VALUE=")[1].split()[0])
                            
                            # Add to tracking dictionary
                            self.training_losses["policy"].append(policy_loss)
                            self.training_losses["value"].append(value_loss)
                            
                            # Update summary at the top of the log
                            self.update_loss_summary()
                        except:
                            pass
                            
                        # Highlight the line with policy/value losses
                        last_line = self.train_log.index("end-1c linestart")
                        self.train_log.tag_add("highlight", last_line, f"{last_line} lineend")
                        self.train_log.tag_config("highlight", background="#E0F0FF")
                
                self.train_log.see("end")
                self.train_log.config(state="disabled")
                self.log_last_update = current_time
        
        # Schedule next check
        self.after(100, self._update_log_from_buffer)
    
    def update_loss_summary(self):
        """Update the loss summary at the top of the training log"""
        if len(self.training_losses["policy"]) > 0:
            # Create summary text
            current_policy = self.training_losses["policy"][-1]
            current_value = self.training_losses["value"][-1]
            
            # Calculate improvement if we have at least 2 data points
            if len(self.training_losses["policy"]) > 1:
                policy_change = current_policy - self.training_losses["policy"][0]
                value_change = current_value - self.training_losses["value"][0]
                
                summary = (f"Current Loss - Policy: {current_policy:.6f}, Value: {current_value:.6f}\n"
                          f"Change - Policy: {policy_change:.6f}, Value: {value_change:.6f}\n"
                          f"{'Improving ✓' if policy_change < 0 else 'Not improving ✗'}\n"
                          f"{'-' * 50}\n")
            else:
                summary = (f"Current Loss - Policy: {current_policy:.6f}, Value: {current_value:.6f}\n"
                          f"{'-' * 50}\n")
            
            # Replace existing summary
            self.train_log.config(state="normal")
            summary_end = self.train_log.search("-" * 20, "1.0", stopindex="end")
            if summary_end:
                self.train_log.delete("1.0", summary_end + "+1l")
            else:
                # No existing summary, insert at beginning
                self.train_log.insert("1.0", summary)
            self.train_log.config(state="disabled")
        
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

    # making a move - improved to reduce delay for human moves
    def _make_move(self, col):
        if not self.game.is_valid(col): return
        state_before = self.game.get_state_copy()
        ok, _ = self.game.drop_piece(col)
        if not ok: return
        
        # Only add example for human/random move when Learn=Yes AND not using RandomComputerPlayer
        if (self.nn and self.learn_var.get() and 
            not isinstance(self.players[self.game.current_player], MCTSComputerPlayer) and
            not isinstance(self.players[self.game.current_player], RandomComputerPlayer)):
            # placeholder – visit_probs will be filled with one‑hot (since no search)
            vp = np.zeros(COLUMN_COUNT, dtype=np.float32); vp[col] = 1.0
            self.nn.add_example(state_before, vp)
            
        mv = (self.turn_count//2)+1
        if self.game.current_player == RED_PIECE:
            if self.game.game_over:
                self._log(f"{mv:>3}. {col+1}")
            else:
                self.moves.config(state="normal"); self.moves.insert("end",f"{mv:>3}. {col+1} -- "); self.moves.config(state="disabled"); self.moves.see("end")
        else:
            self._log(f"{col+1}")
        self.turn_count += 1; self._draw()
        
        # Check right away if the game is over to reduce delay for human's winning move
        if self.game.game_over:
            self._finish()
        else:
            self.game.switch()
            # MODIFIED: Don't update status if we're in play_till_end mode
            if not self.play_till_end:
                self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn", COLOR_MAP[self.game.current_player])
            
            # For computer's next move, we can keep the small 30ms delay
            self.after(30, self._next_turn)

    # AI turns
    def _next_turn(self):
        if self.game.game_over or not self.game_in_progress or self.paused: return
        ply = self.players[self.game.current_player]
        if isinstance(ply, (RandomComputerPlayer, MCTSComputerPlayer)):
            # MODIFIED: Don't update status if we're in play_till_end mode
            if not self.play_till_end:
                self._set_status(f"{PLAYER_MAP[self.game.current_player]} (AI) thinking…", COLOR_MAP[self.game.current_player])
            threading.Thread(target=lambda: self._ai_play(ply), daemon=True).start()
    
    def _ai_play(self, ply):
        try:
            mv = ply.get_move(self.game, self)
            
            # Add delay for Computer vs Computer games based on setting
            if self.is_comp and hasattr(self, 'cc_delay') and self.cc_delay > 0:
                delay = self.cc_delay  # Use the configured delay
            else:
                delay = 10  # Default minimal delay
            
            self.after(delay, lambda: self._make_move(mv))
        except Exception as e:
            # Handle AI errors gracefully
            error_msg = f"AI error: {str(e)}"
            self.after(0, lambda: messagebox.showerror("AI Error", error_msg))
            self.after(0, lambda: self._set_status(f"AI error occurred", "red"))
            # Log error to training log if visible
            if str(self.train_frame.grid_info()) != "{}":
                self.log_to_training(f"ERROR: {error_msg}")

    # finish & restart with training info display for AI vs AI games
    def _finish(self):
        self.game_in_progress = False
        self.score['games'] += 1
        
        # Increment games since training counter
        self.games_since_training += 1
        
        if self.game.winner == 'Draw': 
            self.score['draws'] += 1
            self._set_status("Draw")
        else:
            if self.game.winner == RED_PIECE: 
                self.score['red'] += 1
                name = "Red"
            else: 
                self.score['green'] += 1
                name = "Green"
            self._set_status(f"{name} wins", COLOR_MAP[self.game.winner])
        self._update_score()
        
        # Log the game result in the training log if it's AI vs AI
        if self.is_comp and str(self.train_frame.grid_info()) != "{}":
            if self.game.winner == 'Draw':
                self.log_to_training("Result: Draw")
            else:
                winner_name = "Red" if self.game.winner == RED_PIECE else "Green"
                self.log_to_training(f"Result: {winner_name} wins")
            self.log_to_training(f"Current score - Red: {self.score['red']}, Green: {self.score['green']}, Draws: {self.score['draws']}")
        
        # Update training progress bar for continuous play
        if self.is_comp and self.continuous_play and str(self.train_frame.grid_info()) != "{}":
            self.train_progress['maximum'] = self.max_cc_games
            self.train_progress['value'] = self.score['games']
            self.train_status.config(text=f"Games: {self.score['games']} / {self.max_cc_games}")
        
        # Force immediate UI update before any training
        self.update_idletasks()
        
        # Check if we should train - either regular interval reached or play_till_end is True
        should_train = (self.games_since_training >= self.cc_train_interval) or self.play_till_end
        
        # If play_till_end is True, we need to train and then reset flag
        if self.play_till_end:
            self.log_to_training(f"Game ended after stop request. Training on {self.games_since_training} accumulated games.")
            self.play_till_end = False  # Reset flag
            self.paused = True  # Set paused flag to stop continuous play
            self.stop_btn['state'] = "disabled"  # Disable stop button
            
            # Schedule training to happen after UI has been refreshed
            self.after(50, self._perform_end_game_training)
            
            # Don't start a new game after training when stopped
            return
        
        # Regular training interval check
        if should_train and self.is_comp and not self.paused:
            self.log_to_training(f"Training interval of {self.cc_train_interval} games reached.")
            self.games_since_training = 0  # Reset counter
            
            # Schedule training to happen after UI has been refreshed
            self.after(50, self._perform_end_game_training)
        elif not should_train and self.is_comp and not self.paused:
            # No training needed yet
            self.log_to_training(f"Games since last training: {self.games_since_training}/{self.cc_train_interval}")
        
        # Check if we should continue with another game (for AI vs AI only)
        if self.is_comp and not self.paused:
            # Update status after training
            if str(self.train_frame.grid_info()) != "{}":
                self.train_status.config(text=f"Games: {self.score['games']} / {self.max_cc_games} - Training complete")
            
            # Check if continuous play is enabled and we haven't reached max games
            if self.continuous_play and self.score['games'] < self.max_cc_games:
                self.auto_job = self.after(1200, lambda: self._new_game())
            else:
                # Stop after one game if continuous play is disabled
                # or we've reached the maximum number of games
                if self.continuous_play and self.score['games'] >= self.max_cc_games:
                    self.log_to_training(f"Maximum number of games ({self.max_cc_games}) reached. Stopping continuous play.")
                    
                    # Show message box that max games have been reached
                    self.after(500, lambda: messagebox.showinfo("Continuous Play Complete", 
                                              f"Maximum number of games ({self.max_cc_games}) reached.\n\n"
                                              f"Final score:\nRed: {self.score['red']}\nGreen: {self.score['green']}\nDraws: {self.score['draws']}"))

    # SETTINGS
    
    
    # Add a new method that handles the end-game training separately
    def _perform_end_game_training(self):
        """Perform training after game ends, separated to allow UI to update first"""
        # Check if any player is RandomComputerPlayer - skip training if so
        has_random_player = any(isinstance(p, RandomComputerPlayer) for p in self.players.values())
        
        # learning decisions - don't train if Random player is present
        if (not has_random_player) and (self.is_comp or self.learn_var.get()):
            # ADDED: Update main status to show training is in progress
            self._set_status("Training NN...", "#990000")  # Dark red color
            
            # If it's AI vs AI, update the status before training
            if self.is_comp and str(self.train_frame.grid_info()) != "{}":
                self.train_status.config(text="Training neural network...")
                self.train_status.config(foreground="red")  # Set initial color for blinking
                self.log_to_training("Training neural network...")
                
                # Start blinking effect
                self._blink_training_text()
            
            # CRITICAL FIX: For AI vs AI games, we need to collect training examples from both AI players
            if self.is_comp:
                red_player = self.players[RED_PIECE]
                green_player = self.players[GREEN_PIECE]
                
                # If the players have their own neural network managers, transfer their examples
                if isinstance(red_player, MCTSComputerPlayer) and red_player.nn is not self.nn:
                    # Copy pending examples from red player's nn to the main nn
                    for ex in red_player.nn.pending:
                        self.nn.pending.append(ex)
                    red_player.nn.pending.clear()
                    
                if isinstance(green_player, MCTSComputerPlayer) and green_player.nn is not self.nn:
                    # Copy pending examples from green player's nn to the main nn
                    for ex in green_player.nn.pending:
                        self.nn.pending.append(ex)
                    green_player.nn.pending.clear()
                    
                # Log the number of collected examples
                if str(self.train_frame.grid_info()) != "{}":
                    self.log_to_training(f"Collected {len(self.nn.pending)} training examples")
            
            self.nn.finish_game(self.game.winner)
            
            # Process any buffered logs
            self._force_process_log_buffer()
            
            # Create a custom logger that forces processing after each log
            def immediate_logger(msg):
                self.log_to_training(msg)
                self._force_process_log_buffer()
            
            # Use correct hyperparameters from settings with the immediate logger
            self.nn.train(
                batch_size=self.nn_params['batch_size'], 
                epochs=self.nn_params['epochs'],
                logger=immediate_logger, 
                num_games=1
            )
            
            # Stop blinking when training is done
            if self.train_blink_job:
                self.after_cancel(self.train_blink_job)
                self.train_blink_job = None
            self.train_status.config(foreground="black")  # Reset to normal color
            
            # Ensure all remaining logs are processed
            self._force_process_log_buffer()
            
            # Update model info after training to show new total games
            self._update_model_info()
            
            # FIXED: Changed self.game_over to self.game.game_over
            if self.game.game_over:
                if self.game.winner == 'Draw':
                    self._set_status("Draw")
                else:
                    winner_name = "Red" if self.game.winner == RED_PIECE else "Green"
                    self._set_status(f"{winner_name} wins", COLOR_MAP[self.game.winner])
            else:
                self._set_status(f"Training complete", "green")
        else:
            # No training - just clear pending examples
            self.nn.pending.clear()
            
            # Log if training was skipped due to Random player
            if has_random_player and self.is_comp and str(self.train_frame.grid_info()) != "{}":
                self.log_to_training("Training skipped - Random Computer player present")
                self._force_process_log_buffer()
            
    def _force_process_log_buffer(self):
        """Force processing of all buffered log messages immediately"""
        messages = []
        with self.log_buffer_lock:
            if self.log_buffer:
                messages = self.log_buffer.copy()
                self.log_buffer.clear()
        
        if messages and hasattr(self, 'train_log') and self.train_log.winfo_exists():
            self.train_log.config(state="normal")
            for message in messages:
                self.train_log.insert("end", message)
                
                # Highlight policy and value loss lines for easier tracking
                if "POLICY=" in message:
                    # Extract policy and value losses
                    try:
                        policy_loss = float(message.split("POLICY=")[1].split()[0])
                        value_loss = float(message.split("VALUE=")[1].split()[0])
                        
                        # Add to tracking dictionary
                        self.training_losses["policy"].append(policy_loss)
                        self.training_losses["value"].append(value_loss)
                        
                        # Update summary at the top of the log
                        self.update_loss_summary()
                    except:
                        pass
                        
                    # Highlight the line with policy/value losses
                    last_line = self.train_log.index("end-1c linestart")
                    self.train_log.tag_add("highlight", last_line, f"{last_line} lineend")
                    self.train_log.tag_config("highlight", background="#E0F0FF")
            
            self.train_log.see("end")
            self.train_log.config(state="disabled")
            # Force update to make changes visible immediately
            self.update_idletasks()
        
    def _settings(self):
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Pause the game before changing settings.")
            return
        
        # Store original settings for comparison
        original_settings = {
            'mcts': dict(self.mcts_params),
            'training': {
                'games': self.train_games, 
                'max_cc_games': self.max_cc_games,
                'cc_train_interval': self.cc_train_interval, 
                'cc_delay': self.cc_delay
            },
            'nn_params': dict(self.nn_params)
        }
        
        dlg = SettingsDialog(self, self.mcts_params, self.train_games, self.max_cc_games, 
                              self.cc_train_interval, self.cc_delay, self.nn_params)
        if dlg.result:
            # Get new settings from dialog
            new_mcts = dlg.result['mcts']
            new_train_games = dlg.result['training']['games']
            new_max_cc_games = dlg.result['training']['max_cc_games']
            new_cc_train_interval = dlg.result['training']['cc_train_interval']
            new_cc_delay = dlg.result['training']['cc_delay']
            new_nn_params = dlg.result['nn_params']
            
            # Check if any settings have changed
            settings_changed = False
            
            # Check MCTS params
            if (new_mcts['iterations'] != self.mcts_params['iterations'] or
                new_mcts['C_param'] != self.mcts_params['C_param']):
                settings_changed = True
            
            # Check training games
            if (new_train_games != self.train_games or
                new_max_cc_games != self.max_cc_games or
                new_cc_train_interval != self.cc_train_interval or
                new_cc_delay != self.cc_delay):
                settings_changed = True
            
            # Check NN params
            for key, value in new_nn_params.items():
                if key not in self.nn_params or self.nn_params[key] != value:
                    settings_changed = True
                    break
            
            # Update settings if there were changes
            self.mcts_params = new_mcts
            self.train_games = new_train_games
            self.max_cc_games = new_max_cc_games
            self.cc_train_interval = new_cc_train_interval
            self.cc_delay = new_cc_delay
            old_lr = self.nn_params.get('learning_rate')
            self.nn_params = new_nn_params
            
            # Update NN settings both in optimizer and NNManager's hyperparams
            if settings_changed:
                # Update learning rate in optimizer
                if old_lr != self.nn_params['learning_rate']:
                    for param_group in self.nn.opt.param_groups:
                        param_group['lr'] = self.nn_params['learning_rate']
                
                # Make sure hyperparams are synchronized
                for key, value in self.nn_params.items():
                    self.nn.hyperparams[key] = value
            
            # Save to config file
            cfg = {
                'mcts': self.mcts_params,
                'training': {
                    'games': self.train_games, 
                    'max_cc_games': self.max_cc_games,
                    'cc_train_interval': self.cc_train_interval,
                    'cc_delay': self.cc_delay
                },
                'nn_params': self.nn_params
            }
            json.dump(cfg, open(MCTS_CONFIG_FILE, "w"), indent=4)
            
            # Only show message if settings were actually changed
            if settings_changed:
                messagebox.showinfo("Saved", "Settings updated.")
    
    # TRAINING NN - Modified to clear the board when training starts
    def _train(self):
        """Train neural network using self-play games"""
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Finish or pause the current game before training.")
            return
        
        if self.training_in_progress:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        # Clear the game board and history when starting training
        self.game.reset()
        self.last_hover = None
        self.moves.config(state="normal")
        self.moves.delete("1.0", "end")
        self.moves.config(state="disabled")
        self._draw()
        
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
        
        # ADDED: Update main status area to show training is in progress
        self._set_status("Training NN...", "#990000")  # Dark red color
        
        # Make sure model info is updated
        self._update_model_info()
        
        # Clear the log buffer
        with self.log_buffer_lock:
            self.log_buffer.clear()
        
        # Clean training log
        self.train_log.config(state="normal")
        self.train_log.delete("1.0", "end")
        
        # Add training start information to the log
        self.log_to_training(f"Starting training with {n} self-play games")
        self.log_to_training(f"Model: {os.path.basename(self.training_model_path)}")
        self.log_to_training(f"MCTS iterations: {self.mcts_params['iterations']}")
        self.log_to_training(f"Batch size: {self.nn_params['batch_size']}, Epochs: {self.nn_params['epochs']}")
        self.log_to_training("-" * 50 + "\n")
        
        # Create a dedicated thread for the worker
        self.training_thread = threading.Thread(target=self._training_worker_thread, args=(n,), daemon=True)
        self.training_thread.start()
    
    # FIXED: Updated _training_worker_thread to use the corrected _play_single_training_game function
    def _training_worker_thread(self, n):
        try:
            # Record start time for training
            start_time = time.time()
            
            # Add training start information to the log
            self.log_to_training(f"Started training for {n} games")
            self.log_to_training(f"MCTS iterations: {self.mcts_params['iterations']}")
            self.log_to_training(f"Neural network model: {os.path.basename(self.training_model_path)}")
            self.log_to_training(f"Batch size: {self.nn_params['batch_size']}, Epochs: {self.nn_params['epochs']}")
            self.log_to_training("-" * 50)
            
            # Clear training loss tracking for new session
            self.training_losses = {"policy": [], "value": []}
            
            # Determine how many workers to use
            num_workers = max(1, min(os.cpu_count() - 1, 4))
            self.log_to_training(f"Starting parallel training with {num_workers} workers - Elapsed: 0h 0m 0s")
            
            # Prepare the function and batch size for parallel execution
            batch_size = min(n, max(1, n // (num_workers * 2)))
            
            # Prepare NN manager config to pass to workers
            nn_config = {
                'model_path': self.training_model_path,
                'hyperparams': self.nn_params
            }
            
            game_func = partial(
                _play_single_training_game, 
                self.mcts_params['iterations'], 
                self.mcts_params['C_param'], 
                nn_config
            )
            
            games_completed = 0
            all_examples = []  # Store game examples separately
            all_game_results = []  # Store game results separately
            executor = None
            
            try:
                # Process games in batches
                executor = ProcessPoolExecutor(max_workers=num_workers)
                futures = []
                
                while games_completed < n and not self.training_stop_requested:
                    # Calculate remaining games
                    games_to_run = min(batch_size, n - games_completed)
                    if games_to_run <= 0:
                        break
                    
                    # Submit batch of games
                    batch_futures = [executor.submit(game_func) for _ in range(games_to_run)]
                    futures.extend(batch_futures)
                    
                    # Process results with a timeout
                    pending = batch_futures.copy()
                    
                    while pending and not self.training_stop_requested:
                        # Wait for the first future to complete
                        done, pending = wait(pending, timeout=0.5, return_when=FIRST_COMPLETED)
                        
                        if self.training_stop_requested:
                            for future in pending:
                                future.cancel()
                            break
                            
                        # Process completed futures
                        for future in done:
                            if future.cancelled():
                                continue
                                
                            try:
                                examples, winner = future.result()
                                # Store examples and result as a complete game
                                all_examples.append(examples)
                                all_game_results.append(winner)
                                
                                games_completed += 1
                                
                                # Calculate elapsed time
                                elapsed_seconds = time.time() - start_time
                                minutes, seconds = divmod(elapsed_seconds, 60)
                                hours, minutes = divmod(minutes, 60)
                                elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                                
                                # Update UI thread safely
                                self.after(0, lambda p=games_completed, n=n, t=elapsed_str: 
                                          self._update_training_progress(p, n, t))
                                
                                # Print progress periodically
                                if games_completed % max(1, n // 10) == 0 or games_completed % 5 == 0:
                                    self.log_to_training(f"Training progress: {games_completed}/{n} games - Elapsed: {elapsed_str}")
                                    # Log game outcome statistics
                                    red_wins = sum(1 for w in all_game_results if w == RED_PIECE)
                                    green_wins = sum(1 for w in all_game_results if w == GREEN_PIECE)
                                    draws = sum(1 for w in all_game_results if w == 'Draw')
                                    self.log_to_training(f"Game outcomes: Red: {red_wins}, Green: {green_wins}, Draws: {draws}")
                            except Exception as ex:
                                self.log_to_training(f"Error in game generation: {str(ex)}")
                        
                    # Check if we need to stop
                    if self.training_stop_requested:
                        break
                            
                # Cancel any remaining futures
                for future in futures:
                    if not future.done():
                        future.cancel()
            finally:
                # Make sure to shutdown the executor
                if executor:
                    executor.shutdown(wait=False)
            
            # Process collected games and train models
            if games_completed > 0:
                elapsed_seconds = time.time() - start_time
                minutes, seconds = divmod(elapsed_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                
                status_msg = ""
                if self.training_stop_requested:
                    status_msg = f"Training stopped, still processing {games_completed} completed games - {elapsed_str}"
                    self.log_to_training(status_msg)
                else:
                    status_msg = f"Game generation complete: {games_completed}/{n} games - {elapsed_str}"
                    self.log_to_training(status_msg)
                
                # Update UI with processing status
                self.after(0, lambda msg=status_msg: self.train_status.config(text=msg))
                
                # Process each game separately - keep examples and results aligned by game
                for i, (game_examples, winner) in enumerate(zip(all_examples, all_game_results)):
                    # Skip processing if explicitly requested to stop
                    if hasattr(self, 'force_stop_training') and self.force_stop_training:
                        break
                        
                    try:
                        # Process all examples for this single game
                        for example_data in game_examples:
                            # Reconstruct game state from serialized data
                            game_state = Connect4Game()
                            game_state.board = np.array(example_data['board'], dtype=np.int8)
                            game_state.current_player = example_data['player']
                            policy = np.array(example_data['policy'], dtype=np.float32)
                            
                            # Add example to training data
                            self.nn.add_example(game_state, policy)
                        
                        # Call finish_game only once per complete game
                        self.nn.finish_game(winner)
                        
                        # Log progress occasionally
                        if i % 10 == 0 and i > 0:
                            self.log_to_training(f"Processed {i}/{len(all_game_results)} games")
                            
                    except Exception as proc_error:
                        self.log_to_training(f"Error processing game {i}: {str(proc_error)}")
                
                # Add a summary of data before training
                self.log_to_training(f"Training data collected: {len(self.nn.data['states'])} examples")
                
                # Skip neural network training if user requested complete stop
                if not hasattr(self, 'force_stop_training') or not self.force_stop_training:
                    # Calculate time for status
                    elapsed_seconds = time.time() - start_time
                    minutes, seconds = divmod(elapsed_seconds, 60)
                    hours, minutes = divmod(minutes, 60)
                    elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                    
                    status_text = "Training neural network..."
                    if self.training_stop_requested:
                        status_text = f"Training with {games_completed} games..."
                    
                    final_status = f"{status_text} - {elapsed_str}"
                    self.after(0, lambda status=final_status: self.train_status.config(text=status))
                    
                    self.log_to_training(f"Starting neural network training - Elapsed: {elapsed_str}")
                    self.log_to_training(f"Training iterations so far: {self.nn.train_iterations}")
                    
                    # Train the neural network
                    self.nn.train(
                        batch_size=self.nn_params['batch_size'], 
                        epochs=self.nn_params['epochs'], 
                        start_time=start_time,
                        logger=self.log_to_training,
                        num_games=games_completed
                    )
                    
                    # Add post-training information
                    self.log_to_training(f"Training completed - Model saved to {self.training_model_path}")
                    self.log_to_training(f"New training iterations total: {self.nn.train_iterations}")
                    self.log_to_training(f"Total games trained on: {self.nn.total_games}")
                    self.log_to_training("-" * 50)
                    
                    # Update model info display
                    self.after(0, lambda: self._update_model_info())
            
            # Clean up and update UI
            self.after(0, lambda: self.train_btn.config(state="normal"))
            self.after(0, lambda: self.train_browse.config(state="normal"))
            self.after(0, lambda: setattr(self, 'training_in_progress', False))
            
            # Calculate total training time
            elapsed_seconds = time.time() - start_time
            minutes, seconds = divmod(elapsed_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Prepare final status messages
            if hasattr(self, 'force_stop_training') and self.force_stop_training:
                final_text = f"Terminated - {elapsed_str}"
                final_msg = f"Training terminated after {games_completed} games.\nTotal time: {elapsed_str}"
                self.after(0, lambda txt=final_text: self.train_status.config(text=txt))
                self.after(0, lambda msg=final_msg: messagebox.showinfo("Terminated", msg))
            elif self.training_stop_requested:
                final_text = f"Stopped after {games_completed} games - {elapsed_str}"
                final_msg = f"Training stopped after {games_completed} games.\nTotal time: {elapsed_str}"
                self.after(0, lambda txt=final_text: self.train_status.config(text=txt))
                self.after(0, lambda msg=final_msg: messagebox.showinfo("Stopped", msg))
            else:
                final_text = f"Completed {games_completed} games - {elapsed_str}"
                final_msg = f"Training finished.\nTotal time: {elapsed_str}"
                self.after(0, lambda txt=final_text: self.train_status.config(text=txt))
                self.after(0, lambda msg=final_msg: messagebox.showinfo("Done", msg))
            
            # Reset flags
            if hasattr(self, 'force_stop_training'):
                delattr(self, 'force_stop_training')
            self.training_stop_requested = False
            
            # Reset stop button state if not in computer vs computer game
            if not self.is_comp:
                self.after(0, lambda: self.stop_btn.config(state="disabled"))
        
        except Exception as error:
            # Log the error for debugging
            error_message = f"Error in training worker: {error}"
            self.log_to_training(error_message)
            import traceback
            traceback.print_exc()
            
            # Update UI on error
            self.after(0, lambda: self.train_btn.config(state="normal"))
            self.after(0, lambda: self.train_browse.config(state="normal"))
            self.after(0, lambda: setattr(self, 'training_in_progress', False))
            self.after(0, lambda: self.stop_btn.config(state="disabled"))
            self.after(0, lambda: self.train_status.config(text="Error during training"))
            
            # Create a local copy of the error message for the lambda
            error_dialog_msg = f"An error occurred during training:\n{str(error)}"
            self.after(0, lambda msg=error_dialog_msg: messagebox.showerror("Error", msg))
            self.after(0, lambda: self._set_status("Results:", "black"))

    def _hide_training_ui(self):
        """Hide the training progress UI elements"""
        self.separator.grid_remove()
        self.train_frame.grid_remove()

    def _update_training_progress(self, current, total, elapsed_time=""):
        """Update the training progress indicators in the main window"""
        self.train_progress['value'] = current
        self.train_status.config(text=f"Training: {current} / {total} games - {elapsed_time}")

    def _pause(self):
        # Case 1: Training is in progress
        if self.training_in_progress:
            # If stop was already requested, change behavior to immediate termination
            if self.training_stop_requested:
                # Second press - force immediate termination
                self.force_stop_training = True
                self.stop_btn['state'] = "disabled"
                self._set_status("Training terminating... (please wait)")
                self.train_status.config(text="Terminating immediately...")
                
                # Try to interrupt any running processes more aggressively
                if hasattr(self, 'training_thread') and self.training_thread.is_alive():
                    self.log_to_training("Force stopping training...")
                
                # Update the UI to show we're force stopping
                self.after(100, self._check_force_stop_progress)
            else:
                # First press - show confirmation dialog
                dialog = StopConfirmationDialog(self, mode="training")
                if dialog.result:  # User clicked Stop
                    # Request graceful stop
                    self.training_stop_requested = True
                    self._set_status("Training stopping... (press Stop again to force)")
                    
                    # Schedule a rapid UI update to show stopping progress
                    self.after(100, self._check_training_stop_progress)
            return
                
        # Case 2: Computer vs Computer game is in progress
        if self.is_comp and self.game_in_progress:
            # If game is already in "play till end" mode, do nothing
            if self.play_till_end:
                return
                
            # First, cancel any scheduled auto job to prevent pending new games
            if self.auto_job:
                self.after_cancel(self.auto_job)
                self.auto_job = None
            
            # Set "play till end" flag
            self.play_till_end = True
            
            # Update status - CHANGED FROM "Playing till end of game" to "Finishing game"
            self._set_status("Finishing game", "orange")
            
            # Log to training log if visible
            if str(self.train_frame.grid_info()) != "{}":
                self.log_to_training("\nStop requested - Playing until game ends.")
            
            return
        
        # If we got here, there's nothing to pause
        return

    # Add a new method to update the UI while waiting for training to stop
    def _check_training_stop_progress(self):
        if self.training_in_progress:
            # Update the status with a "moving" indicator
            current_text = self.train_status.cget("text")
            if "stopping" in current_text.lower():
                # Add a dot to show progress
                if current_text.count('.') > 5:  # Reset after too many dots
                    self.train_status.config(text="Stopping")
                else:
                    self.train_status.config(text=current_text + ".")
            else:
                self.train_status.config(text="Stopping.")
            
            # Schedule another check
            self.after(300, self._check_training_stop_progress)

    # New method for force stop progress indication
    def _check_force_stop_progress(self):
        if self.training_in_progress:
            # Update with a more urgent indicator
            current_text = self.train_status.cget("text")
            if "terminating" in current_text.lower():
                # Add an exclamation mark to show urgency
                if current_text.count('!') > 3:
                    self.train_status.config(text="Terminating immediately")
                else:
                    self.train_status.config(text=current_text + "!")
            else:
                self.train_status.config(text="Terminating immediately!")
            
            # Schedule another check
            self.after(200, self._check_force_stop_progress)
        
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
                    if 'training' in cfg:
                        if 'games' in cfg['training']:
                            self.train_games = cfg['training']['games']
                        if 'max_cc_games' in cfg['training']:
                            self.max_cc_games = cfg['training']['max_cc_games']
                        if 'cc_train_interval' in cfg['training']:
                            self.cc_train_interval = cfg['training']['cc_train_interval']
                        if 'cc_delay' in cfg['training']:
                            self.cc_delay = cfg['training']['cc_delay']
                    if 'nn_params' in cfg:
                        self.nn_params = cfg['nn_params']
                        # Update NN with loaded parameters
                        for key, value in self.nn_params.items():
                            self.nn.hyperparams[key] = value
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