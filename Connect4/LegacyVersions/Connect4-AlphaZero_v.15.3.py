#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Connect‑4 with AlphaZero‑style self‑play (PUCT MCTS + CNN policy‑value NN)
Version a13.2 - Major algorithm improvements and bug fixes

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
31. **Human-AI Learning Flow** - AI players now use the updated neural network after training
32. **Consistent Value Target** - Fixed value target perspective in edge cases
33. **Tree Reuse Between Moves** - Added the ability to reuse search trees between moves for better performance
34. **Improved Exploration/Exploitation** - Better handling of Dirichlet noise for exploration
35. **Residual Network Architecture** - Added optional ResNet architecture
36. **Prioritized Experience Replay** - Added prioritization based on surprise factor
37. **Stochastic Weight Averaging** - Added SWA for more robust training
38. **Progressive Temperature Decay** - Implemented temperature annealing during game play
39. **Curriculum Learning** - Added adaptive MCTS iterations based on win rate
40. **Visual Heatmaps** - Added policy and value visualization for better understanding
41. **Win Probability Display** - Shows win probability estimates during gameplay
42. **Model Versioning** - Added automatic versioning and model management
43. **Persistent Training Log** - Training log persists through app lifetime and saves to disk on exit
44. **Customizable Column Bias** - Added ability to configure center column bias weights
45. **Integrated Visualization** - Policy/value visualization now embedded in main window instead of popup
46. **Tree-Reuse Bias Fix** - Normalized visit counts when reusing subtrees to prevent exploration bias
47. **Improved Full-Strength Mode** - Increased exploration in full strength mode for better tactical search
48. **Smart Safe Move Selection** - Use neural network to evaluate and select the best safe move
49. **Optimized Training Sampling** - Prioritize newer, higher quality training examples
50. **Balanced Column Pruning** - Less aggressive pruning of center moves for better positional learning
51. **Fixed Center-Bias Application** - Applied center bias once in MCTS to avoid over-prioritizing center columns
52. **Improved Tree-Reuse Normalization** - More adaptive approach to prevent over-dampening reused nodes
53. **Enhanced Temperature Decay** - Better late-game deterministic play with lower temperature floor
54. **Consistent Full-Strength Parameters** - Fixed c_puct value for full-strength mode across the application
55. **Iteration Boost Cap** - Added upper bound to curriculum learning iteration scaling
56. **Safe Move Selection Logic** - Improved safe move selection to be more position-aware
57. **Fixed Parameter Drift** - Consistent parameters between training and play modes
58. **Temperature Reset Fix** - Reset temperature when starting new games
59. **Fixed Perspective Error in Safe Move Evaluation** - Added player switching before evaluating states to ensure correct perspective
60. **Fresh Dirichlet Noise for Tree Reuse** - Added fresh exploration noise when reusing subtrees to ensure proper exploration
61. **Deeper Tactical Evaluation** - Added multi-ply lookahead for detecting traps in safe move evaluation
62. **Occasional Safe Move Bypassing** - Added small probability to bypass safe move logic in self-play for better tactical learning
63. **Enhanced Training Signal for Non-Draws** - Increased the value of wins/losses in training compared to draws
64. **Tactical Position Curriculum** - Added injection of tactical training positions to train critical skills
65. **Persistent Training Log** - Fixed training log to ensure it never clears between training sessions
66. **Log File Export** - Ensured training log is properly saved as C4.log file on application exit
67. **Fixed CNN Architecture** - Corrected padding and added dynamic shape calculation for proper flattening
68. **Fixed First PUCT Selection** - Added epsilon to PUCT calculation to ensure priors influence first simulation
69. **Enhanced Exploration** - Added sampling by visit proportion for initial moves instead of always using argmax
70. **True Prioritized Experience Replay** - Fixed weighted sampling to properly prioritize high-error training examples
71. **Fixed Value Target Range** - Capped value targets to ±1 to match tanh activation range
72. **LR Scheduler Integration** - Properly stepped the learning rate scheduler using training loss
73. **Configurable Safe-move Bypass** - Made safe-move bypass probability a configurable parameter
74. **Fading Center Bias** - Added decay to center bias that fades as the game progresses
75. **Batch Evaluation** - Added batch processing for position evaluation at game end
76. **Policy Loss Masking** - Added masking of invalid moves during policy loss calculation
77. **Fixed Training Log Display and Persistence** - Fixed issues with log truncation during NN training and ensured logs are properly saved to disk
78. **Clear UI on Train NN** - Clear the board, Game History, and Policy/Value visualization when starting a new training session
79. **Fixed Window Resizing During Training** - Fixed the issue with window shrinking when focus changes during training
80. **Improved Button Sizes** - Increased the size of main control buttons for better usability
81. **Optimized Visualization Layout** - Moved the Value box to the upper-right corner within the visualization frame
Version v13.3 – Minor UI/Value-sync fixes  
82.  Value/Win% now always computed from side-to-move  
83.  Policy-Value pane uses the correct NN when sides load different models
Version 14.0 - Major algorithm and architecture improvements

Improvements implemented
------------------------
84. **Enhanced PUCT formula** - Improved numerical stability and exploration-exploitation balance
85. **Advanced Tree Reuse** - More sophisticated visit count normalization for better performance
86. **Data Augmentation** - Added horizontal flipping for more efficient learning
87. **Dynamic Loss Weighting** - Automatic adjustment of policy/value loss weights based on learning trends
88. **Neural Network Enhancements** - Improved architecture with better value head and residual connections
89. **Advanced Learning Rate Scheduling** - Optional cosine annealing for better convergence
90. **Full-Strength Optimization** - Better parameter settings for maximum playing strength

Version 14.1 - Major algorithm improvements based on audit recommendations
Improvements implemented
------------------------
91. **Safe-move heuristic annealing** - Safe-move logic now decays from 100% to 10% over training
92. **Increased exploration parameters** - Raised Dirichlet noise to 0.4, c_puct to 1.2, later temperature decay
93. **Fixed shared NNManager issue** - Each player now has its own independent NNManager instance
94. **Improved dynamic loss weighting** - More stable gradient-based weighting system
95. **Fixed learning rate scheduler conflicts** - Unified scheduler approach, removed conflicting methods
96. **Adaptive safe-move bypass** - Bypass probability now adapts based on win rate (15% to 1%)
97. **Enhanced SWA batch norm update** - Proper full dataset BN recalibration after SWA
98. **Column imbalance pruning improvements** - More sophisticated KL-divergence based pruning
99. **Tree-reuse prior invalidation** - Invalidate stale priors when neural network weights change
100. **Performance optimizations** - Batched UI value calculations to reduce lag

Version 15.0 - Enhanced stability and performance improvements

Improvements implemented
------------------------
101. **Fixed stale neural network prior detection** - TreeNode now properly initializes nn_version during creation to prevent using outdated priors after training
102. **Added depth bonus for terminal evaluations** - Games now slightly prefer longer paths to defeat when all moves lose, reducing perceived blunders (0.001 * depth bonus)  
103. **Pre-computed legal move masks for training** - Eliminated heavyweight game object reconstruction during batch training, improving performance on fast GPUs
104. **Statistical tree reuse normalization** - Replaced ad-hoc thresholds with exponential smoothing and target-ratio based visit count adjustment
105. **Configurable curriculum learning cap** - Added max_iteration_scale parameter to prevent UI lag on strong models (default 2.0x cap)
106. **Configurable AI move delay** - Made the 0.1s delay configurable per player to optimize performance in different scenarios
107. **Win probability visualization** - UI now displays win percentages instead of raw values for clearer interpretation
108. **Rolling window training log** - Limited in-memory log to 10,000 lines to prevent memory bloat during long training sessions
109. **Fixed default dirichlet noise mismatch** - Settings dialog now defaults to 0.4 to match hard-coded value
110. **Simplified data management** - Unified priority and recency sampling into single approach, removing KL-based pruning complexity
111. **Code hygiene improvements** - Removed unused variables and standardized parameter defaults throughout codebase
112. **Enhanced visualization display** - Policy visualization now clearly shows win probabilities for both players as percentages

Version 15.1 - Enhanced stability and performance improvements

Improvements implemented
------------------------
113. **Fixed stale neural network prior detection - TreeNode now properly initializes nn_version 
       during creation and updates it when evaluating nodes to prevent using outdated priors after training
114. **Aligned dirichlet noise default values - Settings dialog now defaults to 0.4 to match hardcoded value
115. **Code hygiene improvements - Cleaned up remaining inconsistencies for cleaner codebase

Version 15.2 - Enhanced stability and performance improvements

Improvements implemented
------------------------
113. **Fixed Policy&Value visualization diagram after choosing AI-AI in the Select Players window

Version 15.3 – Spec-compliance & UI consistency fixes

Improvements implemented
------------------------
116. **Draw = 50 / 50** – Draw results now show 50 % win chance each instead of 0 %.
117. **Raw NN value exposed** – “Value:” label shows the raw network output for side-to-move.
118. **Single evaluation source of truth** – Visualization & top-bar reuse the same NN evaluation; duplicate formulas removed.
119. **Immediate-win heuristic toned down** – Removed stand-alone win-in-1 override to keep results consistent with the NN.
120. **Helper `_compute_win_probs`** – Centralised conversion from raw value to Red/Green win %.
121. **Unified advantage colouring** – Threshold moved to constant `ADV_THRESHOLD`.

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
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from torch.optim.swa_utils import AveragedModel, SWALR
import uuid
import datetime
import re

ROW_COUNT, COLUMN_COUNT = 6, 7
SQUARESIZE, RADIUS      = 100, int(100/2 - 5)
WIDTH, HEIGHT           = COLUMN_COUNT * SQUARESIZE, (ROW_COUNT + 1) * SQUARESIZE

BLUE, BLACK = "#0000FF", "#000000"
RED,  GREEN = "#FF0000", "#00FF00"
EMPTY_COLOR = "#CCCCCC"
LIGHT_BLUE  = "#99CCFF"  # This is a lighter blue 

EMPTY, RED_PIECE, GREEN_PIECE = 0, 1, 2
PLAYER_MAP = {RED_PIECE: "Red", GREEN_PIECE: "Green"}
COLOR_MAP  = {RED_PIECE: RED,   GREEN_PIECE: GREEN}

DEFAULT_MCTS_ITERATIONS = 1200
DEFAULT_PUCT_C          = 0.8
DEFAULT_TRAIN_GAMES     = 200
NN_MODEL_FILE           = "C4.pt"
MCTS_CONFIG_FILE        = "mcts_config.json"
MAX_TRAINING_EXAMPLES   = 30_000

CENTER_BIAS = np.array([0.5, 0.7, 1.0, 1.5, 1.0, 0.7, 0.5])

ADV_THRESHOLD = 0.6  # colour-change threshold for win-prob display

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
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        
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
class ResidualBlock(nn.Module):
    def __init__(self, channels, dropout_rate=0.1):
        super().__init__()
        # Enhanced block with consistent padding and normalization
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout_rate)
        
    def forward(self, x):
        residual = x
        # First convolution block with batch norm and activation
        x = self.bn1(self.conv1(x))
        x = F.relu(x)  # Activation after batch norm
        x = self.dropout(x)
        
        # Second convolution block with batch norm
        x = self.bn2(self.conv2(x))
        
        # Add residual connection
        x += residual
        
        # Apply activation after addition
        x = F.relu(x)
        return x
        
class Connect4CNN(nn.Module):
    def __init__(self, channels=128, dropout_rate=0.2, fc_dropout_rate=0.4):  # Increased channels
        super().__init__()
        # Enhanced CNN architecture with proper padding and kernel sizes
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)  # Changed kernel size from 4 to 3
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.bn3 = nn.BatchNorm2d(channels)
        self.bn4 = nn.BatchNorm2d(channels)
        
        # Add spatial dropout after convolutions
        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        # Add dropout for fully connected layers
        self.fc_dropout = nn.Dropout(fc_dropout_rate)

        # Calculate flattened size dynamically
        self.forward_dummy = True

        # Enhanced policy head with more capacity
        self.fc_pol = nn.Sequential(
            nn.Linear(channels * ROW_COUNT * COLUMN_COUNT, 256), 
            nn.ReLU(), 
            nn.Dropout(fc_dropout_rate),
            nn.Linear(256, COLUMN_COUNT)
        )
        
        # Enhanced value head with additional layer for better prediction
        self.fc_val = nn.Sequential(
            nn.Linear(channels * ROW_COUNT * COLUMN_COUNT, 256), 
            nn.ReLU(), 
            nn.Dropout(fc_dropout_rate),
            nn.Linear(256, 64),  # Added intermediate layer
            nn.ReLU(),
            nn.Dropout(fc_dropout_rate * 0.5),  # Less dropout in final layers
            nn.Linear(64, 1), 
            nn.Tanh()
        )
        
        # Initialize to ensure flat_size is calculated on first call
        self.flat_size = None

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.spatial_dropout(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.spatial_dropout(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.spatial_dropout(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.spatial_dropout(x)
        
        # If this is the first forward pass, recalculate the flatten dimensions
        if self.flat_size is None:
            self.flat_size = x.size(1) * x.size(2) * x.size(3)
            # Update linear layer input dimensions
            self.fc_pol[0] = nn.Linear(self.flat_size, 256)
            self.fc_val[0] = nn.Linear(self.flat_size, 256)
        
        x = x.view(x.size(0), -1)
        x = self.fc_dropout(x)
        p = self.fc_pol(x)
        v = self.fc_val(x)
        return p, v.squeeze(1)

class Connect4ResNet(nn.Module):
    def __init__(self, channels=128, num_res_blocks=10, dropout_rate=0.15, fc_dropout_rate=0.4):  # Adjusted blocks
        super().__init__()
        # Initial processing
        self.conv_in = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.bn_in = nn.BatchNorm2d(channels)
        
        # Residual blocks with improved structure
        self.res_blocks = nn.ModuleList([ResidualBlock(channels, dropout_rate) for _ in range(num_res_blocks)])
        
        # Spatial dropout
        self.spatial_dropout = nn.Dropout2d(dropout_rate)
        
        # Regular dropout for fully connected layers
        self.fc_dropout = nn.Dropout(fc_dropout_rate)
        
        # Policy head with improved structure
        self.conv_pol = nn.Conv2d(channels, 64, kernel_size=1)
        self.bn_pol = nn.BatchNorm2d(64)
        self.fc_pol = nn.Linear(64 * ROW_COUNT * COLUMN_COUNT, COLUMN_COUNT)
        
        # Value head with additional layer for better prediction
        self.conv_val = nn.Conv2d(channels, 64, kernel_size=1)
        self.bn_val = nn.BatchNorm2d(64)
        self.fc_val1 = nn.Linear(64 * ROW_COUNT * COLUMN_COUNT, 256)
        self.fc_dropout2 = nn.Dropout(fc_dropout_rate)
        self.fc_val2 = nn.Linear(256, 64)  # Additional intermediate layer
        self.fc_dropout3 = nn.Dropout(fc_dropout_rate * 0.5)  # Less dropout in final layers
        self.fc_val3 = nn.Linear(64, 1)
    
    def forward(self, x):
        # Initial processing
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.spatial_dropout(x)
        
        # Residual blocks
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # Policy head
        pol = F.relu(self.bn_pol(self.conv_pol(x)))
        pol = self.spatial_dropout(pol)
        pol = pol.view(-1, 64 * ROW_COUNT * COLUMN_COUNT)
        pol = self.fc_dropout(pol)
        pol = self.fc_pol(pol)
        
        # Value head with additional layer
        val = F.relu(self.bn_val(self.conv_val(x)))
        val = self.spatial_dropout(val)
        val = val.view(-1, 64 * ROW_COUNT * COLUMN_COUNT)
        val = self.fc_dropout(val)
        val = F.relu(self.fc_val1(val))
        val = self.fc_dropout2(val)
        val = F.relu(self.fc_val2(val))  # Process through additional layer
        val = self.fc_dropout3(val)
        val = torch.tanh(self.fc_val3(val))
        
        return pol, val.squeeze(1)
        
class Connect4Dataset(Dataset):
    def __init__(self, s, p, v, priorities=None):
        self.s, self.p, self.v = s, p, v
        self.priorities = priorities
        
        # If priorities aren't provided, create a recency-based weighting
        if priorities is None:
            # Linear weighting that favors more recent examples
            n = len(s)
            if n > 100:  # Only apply for reasonably sized datasets
                # Create weights that increase with index (newer examples)
                self.weights = np.linspace(0.8, 1.2, n)
            else:
                self.weights = None
        else:
            self.weights = None
            
    def __len__(self):
        return len(self.s)
            
    def __getitem__(self, i):
        return self.s[i], self.p[i], self.v[i]
            
    def get_weights(self):
        if self.priorities is not None:
            # Use priorities when available - fix to return the actual priorities
            weights = np.array(self.priorities)
            return weights / weights.sum() if weights.sum() > 0 else None
        elif hasattr(self, 'weights') and self.weights is not None:
            # Use recency-based weights otherwise
            return self.weights / self.weights.sum()
        else:
            return None
            
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
    def __init__(self, hyperparams=None, model_path=NN_MODEL_FILE, quiet=True, use_resnet=False):
        self.hyperparams = hyperparams or {
            'learning_rate': 1e-2,
            'batch_size': 128,
            'epochs': 10,
            'policy_weight': 1.5,
            'value_weight': 1.0,
            'lr_decay': 0.9995,
            'use_resnet': use_resnet,
            'use_swa': False,
            'momentum': 0.9,
            'weight_decay': 1e-4
        }
        self.model_path = model_path
        self.quiet = quiet
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if not quiet:
            print(f"Using device: {self.device}")
        
        if self.hyperparams.get('use_resnet', False):
            self.net = Connect4ResNet().to(self.device)
        else:
            self.net = Connect4CNN().to(self.device)
            
        momentum = self.hyperparams.get('momentum', 0.9)
        weight_decay = self.hyperparams.get('weight_decay', 1e-4)
        
        self.opt = optim.SGD(
            self.net.parameters(),
            lr=self.hyperparams['learning_rate'],
            momentum=momentum,
            weight_decay=weight_decay
        )
        
        # Use only ReduceLROnPlateau - remove conflicting schedulers
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.opt, 
            mode='min',
            factor=0.5,
            patience=3,
            cooldown=10,
            min_lr=1e-6,
        )
        
        self.use_swa = self.hyperparams.get('use_swa', False)
        if self.use_swa:
            self.swa_model = AveragedModel(self.net)
            for param_group in self.opt.param_groups:
                param_group['swa_lr'] = 1e-4
            self.swa_scheduler = SWALR(self.opt, swa_lr=1e-4)
            self.swa_start = 5
            
        self.use_mixed_precision = self.device.type == 'cuda'
        if self.use_mixed_precision:
            self.scaler = torch.cuda.amp.GradScaler()
            if not quiet:
                print("Mixed precision training enabled")
        
        self.data = {'states': [], 'policies': [], 'values': [], 'priorities': [], 'legal_masks': []}  # Pre-store masks
        self.pending = []
        self.train_iterations = 0
        self.total_games = 0
        self.version = "1.0.0"
        self.win_rate = 0.5
        
        # Track gradient magnitudes for dynamic weighting
        self.policy_grad_norms = []
        self.value_grad_norms = []
        
        # Rolling window for log (Issue #11 fix)
        self.training_log_lines = []
        self.max_log_lines = 10000
        
        if os.path.exists(model_path):
            # Load existing model...
            try:
                # Try loading with weights_only=True first, then fallback
                try:
                    ck = torch.load(model_path, map_location=self.device, weights_only=True)
                except Exception as e:
                    if "weights_only" in str(e).lower() or "unsupported global" in str(e).lower():
                        if not quiet:
                            print(f"Warning: Loading {model_path} with weights_only=False due to compatibility")
                        ck = torch.load(model_path, map_location=self.device, weights_only=False)
                    else:
                        raise e
                
                if isinstance(self.net, Connect4ResNet) and not isinstance(ck['model_state_dict'].get('conv_in.weight', None), torch.Tensor):
                    if not quiet:
                        print(f"Model architecture mismatch: Expected ResNet but found CNN in {model_path}. Using fresh weights.")
                elif isinstance(self.net, Connect4CNN) and isinstance(ck['model_state_dict'].get('conv_in.weight', None), torch.Tensor):
                    if not quiet:
                        print(f"Model architecture mismatch: Expected CNN but found ResNet in {model_path}. Using fresh weights.")
                else:
                    self.net.load_state_dict(ck['model_state_dict'])
                    
                    try:
                        self.opt.load_state_dict(ck['optimizer_state_dict'])
                        
                        for param_group in self.opt.param_groups:
                            if 'momentum' not in param_group:
                                param_group['momentum'] = momentum
                            if 'weight_decay' not in param_group:
                                param_group['weight_decay'] = weight_decay
                        
                    except Exception as opt_error:
                        if not quiet:
                            print(f"Could not load optimizer state: {opt_error}. Using fresh optimizer.")
                        self.opt = optim.SGD(
                            self.net.parameters(),
                            lr=self.hyperparams['learning_rate'],
                            momentum=momentum,
                            weight_decay=weight_decay
                        )
                    
                    if 'scheduler_state_dict' in ck:
                        try:
                            self.scheduler.load_state_dict(ck['scheduler_state_dict'])
                        except Exception as sched_error:
                            if not quiet:
                                print(f"Could not load scheduler state: {sched_error}. Using fresh scheduler.")
                            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                                self.opt, 
                                mode='min',
                                factor=0.5,
                                patience=3,
                                cooldown=10,
                                min_lr=1e-6,
                            )
                    
                    if self.use_swa:
                        for param_group in self.opt.param_groups:
                            if 'swa_lr' not in param_group:
                                param_group['swa_lr'] = 1e-4
                    
                    if 'train_iterations' in ck:
                        self.train_iterations = ck['train_iterations']
                    if 'total_games' in ck:
                        self.total_games = ck['total_games']
                    else:
                        self.total_games = 0
                    if 'version' in ck:
                        self.version = ck['version']
                    if 'win_rate' in ck:
                        self.win_rate = ck['win_rate']
                    if 'hyperparams' in ck:
                        saved_params = ck['hyperparams']
                        for key in saved_params:
                            if key not in self.hyperparams:
                                self.hyperparams[key] = saved_params[key]
                    
                    if self.use_swa and 'swa_model_state_dict' in ck:
                        self.swa_model.load_state_dict(ck['swa_model_state_dict'])
                    
                    # Load gradient norms if available
                    if 'policy_grad_norms' in ck:
                        self.policy_grad_norms = ck['policy_grad_norms']
                    if 'value_grad_norms' in ck:
                        self.value_grad_norms = ck['value_grad_norms']
                    
                    if not quiet:
                        print(f"Model loaded from {model_path}. Training iterations: {self.train_iterations}, Total games: {self.total_games}, Version: {self.version}")
                    
            except Exception as e:
                if not quiet:
                    print(f"Error loading model from {model_path}: {e}")
                    print("Using fresh weights.")
        elif not quiet:
            print(f"Model file {model_path} not found – using new weights.")

    @staticmethod
    def _tensor(state: 'Connect4Game', device=None):
        red_plane   = (state.board == RED_PIECE).astype(np.float32)
        green_plane = (state.board == GREEN_PIECE).astype(np.float32)
        turn_plane  = np.full_like(red_plane, 1.0 if state.current_player == RED_PIECE else 0.0)
        stacked = np.stack([red_plane, green_plane, turn_plane])
        tensor = torch.tensor(stacked, dtype=torch.float32)
        if device is not None:
            tensor = tensor.to(device)
        return tensor

    def policy_value(self, state: 'Connect4Game'):
        assert state.board.shape == (ROW_COUNT, COLUMN_COUNT), f"Invalid board shape: {state.board.shape}"
        
        self.net.eval()
        with torch.no_grad():
            t = self._tensor(state, self.device).unsqueeze(0)
            logits, v = self.net(t)
            logits = logits.squeeze(0)
            valid = state.valid_moves()
            mask = torch.full_like(logits, -1e9)
            mask[valid] = 0.0
            logits = logits + mask
            probs = F.softmax(logits, dim=0).cpu().numpy()
            return probs, float(v.item())

    def add_example(self, state, visit_probs):
        # Pre-compute legal mask (Issue #5 fix)
        legal_mask = torch.zeros(COLUMN_COUNT, dtype=torch.bool)
        valid_moves = state.valid_moves()
        legal_mask[valid_moves] = True
        
        self.pending.append({
            'state': self._tensor(state), 
            'policy': torch.tensor(visit_probs, dtype=torch.float32),
            'player': state.current_player,
            'legal_mask': legal_mask
        })
        
    def log_training(self, message):
        """Add to training log with rolling window (Issue #11 fix)"""
        self.training_log_lines.append(message)
        if len(self.training_log_lines) > self.max_log_lines:
            # Keep only recent lines
            self.training_log_lines = self.training_log_lines[-self.max_log_lines:]
    
    def finish_game(self, winner):
        win_count = 0
        loss_count = 0
        draw_count = 0
        
        if not self.pending:
            return
        
        # Batch evaluation
        batch_states = []
        batch_indices = []
        for i, ex in enumerate(self.pending):
            game_state = self._tensor_to_game(ex['state'])
            batch_states.append(game_state)
            batch_indices.append(i)
        
        batch_values = self.batch_policy_value(batch_states)
        
        for i, ex in enumerate(self.pending):
            if winner == 'Draw':
                value = 0.0
                draw_count += 1
            elif winner == ex['player']:
                value = 1.0
                win_count += 1
            else:
                value = -1.0
                loss_count += 1
                    
            predicted_value = batch_values[i]
            surprise = abs(value - predicted_value)
            
            if value < 0:
                surprise *= 2.0
            
            if winner != 'Draw':
                surprise *= 1.5
            
            value_tensor = torch.tensor([value], dtype=torch.float32)
            
            self.data['states'].append(ex['state'])
            self.data['policies'].append(ex['policy'])
            self.data['values'].append(value_tensor)
            self.data['priorities'].append(surprise)
            self.data['legal_masks'].append(ex['legal_mask'])  # Store pre-computed mask
        
        if len(self.pending) > 0:
            print(f"Added examples: {win_count} wins, {loss_count} losses, {draw_count} draws")
            
        self.pending.clear()
        
        # Simplified data management (Issue #8 fix)
        if len(self.data['states']) > MAX_TRAINING_EXAMPLES:
            self._manage_data_size()

    def _manage_data_size(self):
        """Simplified data management with priority-based retention"""
        if len(self.data['states']) <= MAX_TRAINING_EXAMPLES:
            return
            
        current_size = len(self.data['states'])
        keep_size = int(MAX_TRAINING_EXAMPLES * 0.8)  # Keep 80% of max
        
        # Weighted sampling based on priorities and recency
        priorities = np.array(self.data['priorities'])
        # Add recency bonus (newer examples get slight boost)
        recency_bonus = np.linspace(0, 0.2, len(priorities))
        combined_weights = priorities + recency_bonus
        combined_weights = combined_weights / combined_weights.sum()
        
        # Select indices to keep
        keep_indices = np.random.choice(
            current_size, 
            size=keep_size, 
            replace=False, 
            p=combined_weights
        )
        keep_indices = sorted(keep_indices)
        
        # Keep selected examples
        for key in self.data:
            self.data[key] = [self.data[key][i] for i in keep_indices]

    def _prioritized_sampling(self):
        """Prioritized sampling to manage dataset size"""
        if hasattr(self, 'data') and 'priorities' in self.data and self.data['priorities']:
            priorities = np.array(self.data['priorities'])
            normalized_priorities = priorities / priorities.sum()
            
            target_size = MAX_TRAINING_EXAMPLES
            # Keep some of the most surprising examples
            top_k = int(target_size * 0.3)
            top_indices = np.argpartition(priorities, -top_k)[-top_k:]
            
            # Sample the rest based on priorities
            remaining = target_size - top_k
            remaining_indices = np.random.choice(
                len(priorities), 
                size=remaining, 
                replace=False, 
                p=normalized_priorities
            )
            
            keep_indices = np.concatenate([top_indices, remaining_indices])
            keep_indices = np.unique(keep_indices)
            
            # If we got fewer than expected, pad with random indices
            if len(keep_indices) < target_size:
                missing = target_size - len(keep_indices)
                all_indices = set(range(len(priorities)))
                available = list(all_indices - set(keep_indices))
                extra = np.random.choice(available, size=missing, replace=False)
                keep_indices = np.concatenate([keep_indices, extra])
            
            # Bias keeping non-draw examples
            if 'values' in self.data:
                values = torch.stack(self.data['values']).numpy().flatten()
                draw_indices = [i for i, v in enumerate(values) if abs(v) < 0.1]
                non_draw_indices = [i for i, v in enumerate(values) if abs(v) >= 0.1]
                
                # If we have too many draws, replace some with non-draws
                draw_count = sum(1 for i in keep_indices if i in draw_indices)
                if draw_count > target_size * 0.3 and non_draw_indices:
                    excess_draws = int(draw_count - target_size * 0.3)
                    draws_to_remove = random.sample([i for i in keep_indices if i in draw_indices], excess_draws)
                    
                    # Find available non-draws not already in keep_indices
                    available_non_draws = [i for i in non_draw_indices if i not in keep_indices]
                    
                    if available_non_draws:
                        # Replace with non-draws
                        replacements = random.sample(available_non_draws, min(excess_draws, len(available_non_draws)))
                        keep_indices = np.array([i for i in keep_indices if i not in draws_to_remove] + replacements)
            
            for k in self.data:
                self.data[k] = [self.data[k][i] for i in keep_indices]
        else:
            # Fall back to improved sampling strategy (prioritize NEWER data)
            current_size = len(self.data['states'])
            target_size = MAX_TRAINING_EXAMPLES
            
            keep_recent = int(target_size * 0.2)
            keep_old = int(target_size * 0.2)
            keep_middle = target_size - keep_recent - keep_old
            
            recent_end = current_size
            recent_start = recent_end - keep_recent
            old_end = recent_start
            old_start = 0
            middle_size = old_end - old_start
            
            if middle_size > keep_middle and keep_old > 0:
                oldest_indices = list(range(0, keep_old))
                
                if middle_size > keep_middle:
                    # Weighted sampling from middle - prefer newer examples
                    middle_weights = np.linspace(1.0, 2.0, middle_size)
                    middle_weights = middle_weights / middle_weights.sum()
                    middle_indices = np.random.choice(
                        range(keep_old, old_end),
                        size=keep_middle,
                        replace=False,
                        p=middle_weights
                    )
                else:
                    middle_indices = list(range(keep_old, old_end))
                
                newest_indices = list(range(recent_start, recent_end))
                
                keep_indices = sorted(oldest_indices + list(middle_indices) + newest_indices)
                
                for k in self.data:
                    self.data[k] = [self.data[k][i] for i in keep_indices]
            else:
                excess = current_size - target_size
                for k in self.data:
                    self.data[k] = self.data[k][excess:]

    def batch_policy_value(self, states):
        """Evaluate multiple states in a single batch for efficiency"""
        if not states:
            return []
        
        self.net.eval()
        with torch.no_grad():
            # Create a batch of all states
            batch_tensors = []
            for state in states:
                batch_tensors.append(self._tensor(state, self.device))
            
            batch_input = torch.stack(batch_tensors)
            
            # Single forward pass for the whole batch
            _, values = self.net(batch_input)
            
            return values.cpu().numpy()

    def _tensor_to_game(self, tensor):
        """Convert a tensor back to a game state for prediction"""
        game = Connect4Game()
        tensor_np = tensor.cpu().numpy()
        
        red_plane = tensor_np[0]
        green_plane = tensor_np[1]
        turn_plane = tensor_np[2]
        
        for r in range(ROW_COUNT):
            for c in range(COLUMN_COUNT):
                if red_plane[r, c] > 0.5:
                    game.board[r, c] = RED_PIECE
                elif green_plane[r, c] > 0.5:
                    game.board[r, c] = GREEN_PIECE
        
        # Set the current player based on the turn plane
        if turn_plane[0, 0] > 0.5:
            game.current_player = RED_PIECE
        else:
            game.current_player = GREEN_PIECE
            
        return game

    def train(self, batch_size=None, epochs=None, start_time=None, logger=None, num_games=1):
        if not self.data['states']:
            if logger:
                logger("No training data available. Skipping training.")
            else:
                print("No training data available. Skipping training.")
            return
        
        def log(msg):
            self.log_training(msg)  # Use rolling window logger
            if logger:
                logger(msg)
            else:
                print(msg)
        
        batch_size = batch_size or self.hyperparams['batch_size']
        epochs = epochs or self.hyperparams['epochs']
        
        # Enhanced training with pre-computed masks (Issue #5 fix)
        policy_weight = self.hyperparams.get('policy_weight', 1.5)
        value_weight = self.hyperparams.get('value_weight', 1.0)
        
        # Data augmentation with horizontal flipping
        use_data_augmentation = self.hyperparams.get('use_data_augmentation', True)
        original_data_size = len(self.data['states'])
        
        if use_data_augmentation:
            aug_states = []
            aug_policies = []
            aug_values = []
            aug_priorities = []
            aug_masks = []
            
            for i in range(original_data_size):
                # Original sample
                aug_states.append(self.data['states'][i])
                aug_policies.append(self.data['policies'][i])
                aug_values.append(self.data['values'][i])
                if 'priorities' in self.data and self.data['priorities']:
                    aug_priorities.append(self.data['priorities'][i])
                aug_masks.append(self.data['legal_masks'][i])
                
                # Flipped sample
                state = self.data['states'][i]
                flipped_state = torch.flip(state, [2])
                
                policy = self.data['policies'][i]
                flipped_policy = torch.flip(policy, [0])
                
                value = self.data['values'][i]
                mask = self.data['legal_masks'][i]
                flipped_mask = torch.flip(mask, [0])
                
                aug_states.append(flipped_state)
                aug_policies.append(flipped_policy)
                aug_values.append(value)
                if 'priorities' in self.data and self.data['priorities']:
                    aug_priorities.append(self.data['priorities'][i])
                aug_masks.append(flipped_mask)
                    
            self.data['states'] = aug_states
            self.data['policies'] = aug_policies
            self.data['values'] = aug_values
            if 'priorities' in self.data and self.data['priorities']:
                self.data['priorities'] = aug_priorities
            self.data['legal_masks'] = aug_masks
                
            log(f"Applied data augmentation: {original_data_size} → {len(aug_states)} examples")
        
        # Dataset preparation with pre-computed masks
        values = torch.stack(self.data['values']).numpy().flatten()
        log(f"Value distribution - min: {values.min():.3f}, max: {values.max():.3f}, mean: {values.mean():.3f}")
        
        priorities = None
        if 'priorities' in self.data and self.data['priorities']:
            priorities = torch.tensor(self.data['priorities'], dtype=torch.float32)
        
        # Custom dataset that includes masks
        class Connect4DatasetWithMasks(Dataset):
            def __init__(self, states, policies, values, masks, priorities=None):
                self.states = states
                self.policies = policies
                self.values = values
                self.masks = masks
                self.priorities = priorities
            
            def __len__(self):
                return len(self.states)
            
            def __getitem__(self, i):
                return self.states[i], self.policies[i], self.values[i], self.masks[i]
            
            def get_weights(self):
                if self.priorities is not None:
                    weights = np.array(self.priorities)
                    return weights / weights.sum() if weights.sum() > 0 else None
                return None
        
        ds = Connect4DatasetWithMasks(
            torch.stack(self.data['states']),
            torch.stack(self.data['policies']),
            torch.stack(self.data['values']).squeeze(1),
            torch.stack(self.data['legal_masks']),
            priorities
        )
        
        num_samples = len(ds)
        
        # Prioritized sampling
        weights = ds.get_weights()
        if weights is not None and len(weights) == num_samples:
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=num_samples,
                replacement=True
            )
            dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), sampler=sampler)
        else:
            dl = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True)        
            
        log("\n" + "="*50)
        log(f"TRAINING NEURAL NETWORK - {num_samples} examples")
        log("="*50)
        
        self._analyze_training_data(logger)
        
        # SWA setup
        use_swa_this_round = self.use_swa and self.train_iterations >= self.swa_start
        if use_swa_this_round:
            log(f"Using Stochastic Weight Averaging (SWA)")
            
            for param_group in self.opt.param_groups:
                if 'swa_lr' not in param_group:
                    param_group['swa_lr'] = 1e-4
        
        self.net.train()
        all_policy_losses = []
        all_value_losses = []
        total_losses = []
        
        # Track gradients for dynamic weighting
        policy_gradients = []
        value_gradients = []
            
        for ep in range(epochs):
            p_loss_sum = v_loss_sum = 0.0
            batch_idx = 0
            
            for s, p_target, v_target, masks in dl:
                # Check for training interruption
                if hasattr(self, 'training_interrupted') and self.training_interrupted:
                    log(f"Training interrupted after epoch {ep+1}")
                    return
                    
                s, p_target, v_target, masks = s.to(self.device), p_target.to(self.device), v_target.to(self.device), masks.to(self.device)
                
                
                self.opt.zero_grad()
                
                if self.use_mixed_precision:
                    with torch.cuda.amp.autocast():
                        logits, v_pred = self.net(s)
                        
                        # Use pre-computed masks (Issue #5 fix)
                        masked_logits = logits.clone()
                        masked_logits[~masks] = -1e9
                        
                        log_probs = F.log_softmax(masked_logits, dim=1)
                        policy_loss = -(p_target * log_probs).sum(dim=1).mean()
                        value_loss = F.mse_loss(v_pred, v_target)
                        
                        # Dynamic weighting based on gradient magnitudes
                        if len(self.policy_grad_norms) > 0 and len(self.value_grad_norms) > 0:
                            # Use running average of gradient norms
                            avg_policy_grad = np.mean(self.policy_grad_norms[-10:])
                            avg_value_grad = np.mean(self.value_grad_norms[-10:])
                            
                            # Adjust weights to balance gradients
                            if avg_policy_grad > 0 and avg_value_grad > 0:
                                ratio = avg_value_grad / avg_policy_grad
                                policy_weight = min(3.0, max(0.5, policy_weight * ratio))
                                value_weight = min(3.0, max(0.5, value_weight / ratio))
                        
                        weighted_loss = (policy_weight * policy_loss) + (value_weight * value_loss)
                    
                    self.scaler.scale(weighted_loss).backward()
                    
                    # Track gradient norms before stepping
                    policy_grad_norm = 0.0
                    value_grad_norm = 0.0
                    for name, param in self.net.named_parameters():
                        if param.grad is not None:
                            if 'fc_pol' in name:
                                policy_grad_norm += param.grad.norm().item() ** 2
                            elif 'fc_val' in name:
                                value_grad_norm += param.grad.norm().item() ** 2
                    
                    self.policy_grad_norms.append(policy_grad_norm ** 0.5)
                    self.value_grad_norms.append(value_grad_norm ** 0.5)
                    
                    # Keep only recent gradients
                    if len(self.policy_grad_norms) > 100:
                        self.policy_grad_norms = self.policy_grad_norms[-50:]
                    if len(self.value_grad_norms) > 100:
                        self.value_grad_norms = self.value_grad_norms[-50:]
                    
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    # Non-mixed precision version with pre-computed masks
                    logits, v_pred = self.net(s)
                    
                    # Use pre-computed masks (Issue #5 fix)
                    masked_logits = logits.clone()
                    masked_logits[~masks] = -1e9
                    
                    log_probs = F.log_softmax(masked_logits, dim=1)
                    policy_loss = -(p_target * log_probs).sum(dim=1).mean()
                    value_loss = F.mse_loss(v_pred, v_target)
                    
                    # Dynamic weighting
                    if len(self.policy_grad_norms) > 0 and len(self.value_grad_norms) > 0:
                        avg_policy_grad = np.mean(self.policy_grad_norms[-10:])
                        avg_value_grad = np.mean(self.value_grad_norms[-10:])
                        
                        if avg_policy_grad > 0 and avg_value_grad > 0:
                            ratio = avg_value_grad / avg_policy_grad
                            policy_weight = min(3.0, max(0.5, policy_weight * ratio))
                            value_weight = min(3.0, max(0.5, value_weight / ratio))
                    
                    weighted_loss = (policy_weight * policy_loss) + (value_weight * value_loss)
                    weighted_loss.backward()
                    
                    # Track gradients
                    policy_grad_norm = 0.0
                    value_grad_norm = 0.0
                    for name, param in self.net.named_parameters():
                        if param.grad is not None:
                            if 'fc_pol' in name:
                                policy_grad_norm += param.grad.norm().item() ** 2
                            elif 'fc_val' in name:
                                value_grad_norm += param.grad.norm().item() ** 2
                    
                    self.policy_grad_norms.append(policy_grad_norm ** 0.5)
                    self.value_grad_norms.append(value_grad_norm ** 0.5)
                    
                    if len(self.policy_grad_norms) > 100:
                        self.policy_grad_norms = self.policy_grad_norms[-50:]
                    if len(self.value_grad_norms) > 100:
                        self.value_grad_norms = self.value_grad_norms[-50:]
                    
                    self.opt.step()
                
                # Update SWA
                if use_swa_this_round:
                    self.swa_model.update_parameters(self.net)
                    
                    if all('swa_lr' in group for group in self.opt.param_groups):
                        self.swa_scheduler.step()
                    else:
                        for param_group in self.opt.param_groups:
                            if 'swa_lr' not in param_group:
                                param_group['swa_lr'] = 1e-4
                        self.swa_scheduler.step()
                
                p_loss_sum += policy_loss.item()
                v_loss_sum += value_loss.item()
                batch_idx += 1
            
            policy_loss_avg = p_loss_sum/len(dl)
            value_loss_avg = v_loss_sum/len(dl)
            total_loss_avg = policy_loss_avg + value_loss_avg
            
            # Step scheduler with total loss
            self.scheduler.step(total_loss_avg)
            
            all_policy_losses.append(policy_loss_avg)
            all_value_losses.append(value_loss_avg)
            total_losses.append(total_loss_avg)
            
            elapsed = ""
            if start_time is not None:
                elapsed_seconds = time.time() - start_time
                minutes, seconds = divmod(elapsed_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                elapsed = f" - Time: {int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Log with current weights
            log(f"[Epoch {ep+1}/{epochs}] POLICY={policy_loss_avg:.6f}  VALUE={value_loss_avg:.6f} (pw={policy_weight:.2f}, vw={value_weight:.2f}){elapsed}")
            
            sys.stdout.flush()
        
        # Store loss trends
        if not hasattr(self, 'training_losses'):
            self.training_losses = {'policy': [], 'value': []}
        
        self.training_losses['policy'].extend(all_policy_losses)
        self.training_losses['value'].extend(all_value_losses)
        
        max_history = 50
        if len(self.training_losses['policy']) > max_history:
            self.training_losses['policy'] = self.training_losses['policy'][-max_history:]
        if len(self.training_losses['value']) > max_history:
            self.training_losses['value'] = self.training_losses['value'][-max_history:]
        
        # Log improvements
        if len(all_policy_losses) > 1:
            policy_improvement = all_policy_losses[0] - all_policy_losses[-1]
            value_improvement = all_value_losses[0] - all_value_losses[-1]
            log(f"Policy loss improvement: {policy_improvement:.6f} ({policy_improvement/all_policy_losses[0]*100:.1f}%)")
            
            if all_value_losses[0] > 0:
                log(f"Value loss improvement: {value_improvement:.6f} ({value_improvement/all_value_losses[0]*100:.1f}%)")
            else:
                log(f"Value loss improvement: {value_improvement:.6f} (N/A - initial value was 0)")
            
            if policy_improvement < 0:
                log("WARNING: Policy loss is increasing! Consider adjusting hyperparameters.")
            if value_improvement < 0:
                log("WARNING: Value loss is increasing! Consider adjusting hyperparameters.")
        
        log("-"*50)
        log(f"Training complete - {self.train_iterations+1} iterations")
        log("-"*50 + "\n")
        
        self.train_iterations += 1
        self.total_games += num_games
        
        # Update version
        version_parts = self.version.split('.')
        if len(version_parts) == 3:
            major, minor, patch = map(int, version_parts)
            patch += 1
            if patch > 99:
                patch = 0
                minor += 1
                if minor > 99:
                    minor = 0
                    major += 1
            self.version = f"{major}.{minor}.{patch}"
            log(f"Updated model version to {self.version}")
        
        # Enhanced SWA handling
        if use_swa_this_round:
            log("Applying SWA model weights to main model")
            
            # Full dataset batch norm update as suggested in audit
            full_dataset_loader = DataLoader(ds, batch_size=16, shuffle=True)
            
            self.swa_model.eval()
            with torch.no_grad():
                # Update batch norm statistics with full dataset
                from torch.optim.swa_utils import update_bn
                update_bn(full_dataset_loader, self.swa_model)
            
            # Copy SWA parameters to main model
            self.net.load_state_dict(self.swa_model.module.state_dict())
            log("Completed full dataset BN recalibration for SWA model")
        
        # Save model
        save_dict = {
            'model_state_dict': self.net.state_dict(), 
            'optimizer_state_dict': self.opt.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_iterations': self.train_iterations,
            'total_games': self.total_games,
            'version': self.version,
            'win_rate': self.win_rate,
            'hyperparams': self.hyperparams,
            'training_losses': self.training_losses,
            'policy_grad_norms': self.policy_grad_norms,
            'value_grad_norms': self.value_grad_norms
        }
        
        if self.use_swa:
            save_dict['swa_model_state_dict'] = self.swa_model.state_dict()
        
        try:
            torch.save(save_dict, self.model_path)
            log(f"Model successfully saved to {self.model_path}")
            
            # Save versioned copy
            model_dir = os.path.dirname(os.path.abspath(self.model_path))
            model_name = os.path.basename(self.model_path)
            base, ext = os.path.splitext(model_name)
            versioned_name = f"{base}_v{self.version}{ext}"
            versioned_path = os.path.join(model_dir, versioned_name)
            
            torch.save(save_dict, versioned_path)
            log(f"Versioned model saved as {versioned_name}")
            log(f"Total games trained on: {self.total_games}")
        except Exception as e:
            log(f"WARNING: Error saving model: {str(e)}")
        
    def _analyze_training_data(self, logger=None):
        def log(msg):
            if logger:
                logger(msg)
            else:
                print(msg)
                
        if not self.data['states'] or len(self.data['states']) == 0:
            log("No training data to analyze.")
            return
            
        values = torch.stack(self.data['values']).numpy().flatten()
        win_count = np.sum(values > 0)
        loss_count = np.sum(values < 0)
        draw_count = np.sum(values == 0)
        total = len(values)
        
        # Add detailed value analysis
        log(f"Value statistics - min: {values.min():.3f}, max: {values.max():.3f}, mean: {values.mean():.3f}")
        log(f"Unique value targets: {np.unique(values)}")
        
        # Update the win rate for curriculum learning
        win_rate = win_count / total if total > 0 else 0.5
        self.win_rate = 0.9 * self.win_rate + 0.1 * win_rate  # Smoothed update
        
        log(f"Training data value distribution:")
        log(f"  Wins: {win_count} ({win_count/total*100:.1f}%)")
        log(f"  Losses: {loss_count} ({loss_count/total*100:.1f}%)")
        log(f"  Draws: {draw_count} ({draw_count/total*100:.1f}%)")
        log(f"  Win rate (smoothed): {self.win_rate:.3f}")
        
        policies = torch.stack(self.data['policies']).cpu().numpy()
        max_indices = np.argmax(policies, axis=1)
        unique_moves, move_counts = np.unique(max_indices, return_counts=True)
        
        log(f"Move distribution (columns 0-6):")
        for move, count in zip(unique_moves, move_counts):
            log(f"  Column {move}: {count} moves ({count/total*100:.1f}%)")
            
        if len(unique_moves) < 3 or np.max(move_counts) / total > 0.7:
            log("WARNING: Move distribution is very uneven, which may lead to poor generalization.")
            
        log(f"Total training examples: {total}")
        if total < 1000:
            log("WARNING: Small training dataset may lead to overfitting.")
        elif total > 10000:
            log("Good dataset size for training.")
            
        # Analyze priorities if available
        if 'priorities' in self.data and self.data['priorities']:
            priorities = np.array(self.data['priorities'])
            log(f"Priority statistics:")
            log(f"  Mean: {priorities.mean():.4f}")
            log(f"  Min: {priorities.min():.4f}, Max: {priorities.max():.4f}")
            log(f"  Std dev: {priorities.std():.4f}")
            
        log("-" * 30)
        
    def visualize_policy(self, state):
        """Generate a visual representation of the policy and value predictions"""
        probs, value = self.policy_value(state)
        
        fig, ax = plt.subplots(figsize=(7, 3))
        column_labels = [f"Col {i}" for i in range(7)]
        
        # Create the heatmap
        im = ax.imshow(probs.reshape(1, -1), cmap='YlOrRd', aspect='auto')
        
        # Add column labels
        ax.set_xticks(np.arange(len(column_labels)))
        ax.set_xticklabels(column_labels)
        ax.set_yticks([])
        
        # Add a colorbar
        cbar = fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label('Move Probability')
        
        # Add value prediction as text
        fig.text(0.5, 0.01, f"Value Prediction: {value:.3f}", ha='center', fontsize=12)
        
        ax.set_title(f"Policy Distribution - {PLAYER_MAP[state.current_player]}'s Turn")
        fig.tight_layout()
        
        return fig
            
# ----------------------------------------------------------------------
# MCTS with single NN call per simulation, storing visit counts
# ----------------------------------------------------------------------
class TreeNode:
    __slots__ = ('state', 'parent', 'move', 'prior', 'children', 'visits', 'value_sum', 'player', 'original_prior', 'nn_version')
    def __init__(self, state: 'Connect4Game', parent=None, move=None, prior=0.0, original_prior=None, nn_version=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.prior = prior
        self.original_prior = prior if original_prior is None else original_prior
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.player = state.current_player
        # Set nn_version properly from the start
        self.nn_version = nn_version if nn_version is not None else (parent.nn_version if parent else None)

    def q(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def u(self, c_puct, total_visits):
        epsilon = 1e-6
        return c_puct * self.prior * math.sqrt(total_visits + epsilon) / (1 + self.visits)

    def best_child(self, c_puct):
        total = max(1, self.visits)
        return max(self.children.values(), key=lambda n: n.q() + n.u(c_puct, total))
        
class MCTS:
    def __init__(self, iterations, c_puct, nn: NNManager, explore=True, dirichlet_noise=0.4, safe_bypass_prob=0.15):
        self.I = iterations
        self.c = c_puct
        self.nn = nn
        self.explore = explore
        self.dirichlet_noise = dirichlet_noise
        self.safe_bypass_prob = safe_bypass_prob
        self.root = None
        self.temperature = 1.0
        # Track games played for annealing
        self.games_played = 0
        # Add configurable max curriculum scaling
        self.max_iteration_scale = 2.0  # Configurable cap

    def search(self, root_state: 'Connect4Game', previous_root=None, move_number=0):
        # Reset temperature for new games
        if move_number == 0:
            self.temperature = 1.0
        
        # Enhanced temperature decay - starts later (move 10 instead of 8)
        if move_number > 10:  # Start decay later as suggested
            decay_factor = min(0.95, 0.98 - move_number * 0.005)
            self.temperature = max(0.05, self.temperature * decay_factor)
            
        valid_moves = root_state.valid_moves()
        
        # Quick win detection
        for move in valid_moves:
            test_state = root_state.copy()
            test_state.drop_piece(move)
            if test_state.game_over and test_state.winner == root_state.current_player:
                vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
                vp[move] = 1.0
                self.nn.add_example(root_state, vp)
                return move
        
        # Opponent win block detection
        opponent = GREEN_PIECE if root_state.current_player == RED_PIECE else RED_PIECE
        for move in valid_moves:
            test_state = root_state.copy()
            test_state.current_player = opponent
            test_state.drop_piece(move)
            if test_state.game_over and test_state.winner == opponent:
                vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
                vp[move] = 1.0
                self.nn.add_example(root_state, vp)
                return move
        
        # Enhanced safety probability based on accuracy instead of just games played
        safety_prob = self._calculate_safety_probability()
        bypass_safe_logic = self.explore and random.random() < (1.0 - safety_prob)
        
        # Safe move detection remains the same but with accuracy-based annealing
        safe_moves = []
        if not bypass_safe_logic and random.random() < safety_prob:
            # Safe move logic unchanged but optimized detection would go here
            # For now keeping existing logic but noting optimization opportunity
            for move in valid_moves:
                test_state = root_state.copy()
                test_state.drop_piece(move)
                test_state.switch()
                
                # First level - immediate win detection
                opponent_can_win = False
                for opp_move in test_state.valid_moves():
                    test_state2 = test_state.copy()
                    test_state2.drop_piece(opp_move)
                    if test_state2.game_over and test_state2.winner == opponent:
                        opponent_can_win = True
                        break
                
                # Second level - trap detection
                if not opponent_can_win and len(test_state.valid_moves()) > 0:
                    for opp_move in test_state.valid_moves():
                        test_state2 = test_state.copy()
                        test_state2.drop_piece(opp_move)
                        if not test_state2.game_over:
                            test_state2.switch()
                            
                            forced_move = None
                            for our_move in test_state2.valid_moves():
                                test_state3 = test_state2.copy()
                                test_state3.drop_piece(our_move)
                                test_state3.switch()
                                
                                can_win_after = False
                                for final_move in test_state3.valid_moves():
                                    test_state4 = test_state3.copy()
                                    test_state4.drop_piece(final_move)
                                    if test_state4.game_over and test_state4.winner == opponent:
                                        can_win_after = True
                                        break
                                        
                                if not can_win_after:
                                    forced_move = our_move
                                    break
                            
                            if forced_move is None:
                                opponent_can_win = True
                                break
                
                if not opponent_can_win:
                    safe_moves.append(move)
        
        # Handle safe moves
        if safe_moves and not bypass_safe_logic:
            if len(safe_moves) > 1:
                best_value = -float('inf')
                best_move = safe_moves[0]
                base_policy, _ = self.nn.policy_value(root_state)
                
                for move in safe_moves:
                    test_state = root_state.copy()
                    test_state.drop_piece(move)
                    test_state.switch()
                    _, value = self.nn.policy_value(test_state)
                    value = -value
                    move_score = value + 0.1 * base_policy[move]
                    
                    if move_score > best_value:
                        best_value = move_score
                        best_move = move
                
                chosen_move = best_move
            else:
                chosen_move = safe_moves[0]
            
            # Store both the safe move and alternatives to help learning
            vp = np.zeros(COLUMN_COUNT, dtype=np.float32)
            if safety_prob < 1.0:  # If not fully safety-mode, store contrasting moves
                vp[chosen_move] = 0.8
                # Add small probability to other valid moves
                other_moves = [m for m in valid_moves if m != chosen_move]
                if other_moves:
                    for m in other_moves:
                        vp[m] = 0.2 / len(other_moves)
            else:
                vp[chosen_move] = 1.0
            
            self.nn.add_example(root_state, vp)
            return chosen_move
        
        # Increase exploration for training (c_puct = 1.2 instead of 0.8)
        if not self.explore:
            adaptive_iterations = int(self.I * 1.5)
            c_puct_value = 0.8  # Keep 0.8 for full-strength
        else:
            adaptive_iterations = self.I
            c_puct_value = 1.2  # Increased from 0.8 to 1.2 for better exploration
            
            # Curriculum learning with cap
            if hasattr(self.nn, 'win_rate'):
                win_rate = min(0.9, max(0.1, self.nn.win_rate))
                iter_scale = 0.8 + (win_rate - 0.1) * 1.5
                iter_scale = min(iter_scale, self.max_iteration_scale)  # Apply cap
                adaptive_iterations = int(self.I * iter_scale)
                adaptive_iterations = min(adaptive_iterations, self.I * self.max_iteration_scale)  # Hard cap
        
        # Tree reuse with proper nn_version checking
        if (previous_root is not None and 
            hasattr(previous_root, 'children') and 
            previous_root.children):
            
            opponent_last_move = None
            for i in range(COLUMN_COUNT):
                if (root_state.board != previous_root.state.board).any():
                    for r in range(ROW_COUNT):
                        if (i < COLUMN_COUNT and 
                            r < ROW_COUNT and 
                            root_state.board[r, i] != previous_root.state.board[r, i] and
                            root_state.board[r, i] != EMPTY):
                            opponent_last_move = i
                            break
                    if opponent_last_move is not None:
                        break
                        
            if (opponent_last_move is not None and 
                opponent_last_move in previous_root.children and 
                previous_root.children[opponent_last_move] is not None and
                hasattr(previous_root.children[opponent_last_move], 'children') and
                previous_root.children[opponent_last_move].children):
                
                self.root = previous_root.children[opponent_last_move]
                if (self.root.state.board == root_state.board).all():
                    # Check if neural network weights have changed
                    if not self._priors_are_stale():
                        self._normalize_reused_root()
                        
                        # Fresh Dirichlet noise
                        if self.explore:
                            valid = root_state.valid_moves()
                            dirichlet = np.random.dirichlet([self.dirichlet_noise] * len(valid))
                            
                            for i, m in enumerate(valid):
                                if m in self.root.children:
                                    original_prior = self.root.children[m].original_prior
                                    if original_prior is None:
                                        original_prior = self.root.children[m].prior
                                    self.root.children[m].prior = 0.75 * original_prior + 0.25 * dirichlet[i]
                            
                            prior_sum = sum(self.root.children[m].prior for m in valid if m in self.root.children)
                            if prior_sum > 0:
                                for m in valid:
                                    if m in self.root.children:
                                        self.root.children[m].prior /= prior_sum
                    else:
                        # Invalidate stale tree
                        self.root = TreeNode(root_state.copy())
                else:
                    self.root = TreeNode(root_state.copy())
            else:
                self.root = TreeNode(root_state.copy())
        else:
            self.root = TreeNode(root_state.copy())
        
        # Get network policy and initialize - SET NN_VERSION HERE (Issue #1 fix)
        prior, _ = self.nn.policy_value(self.root.state)
        # FIX: Properly set nn_version when evaluating root
        self.root.nn_version = getattr(self.nn, 'version', None)
        
        valid = self.root.state.valid_moves()
        
        # Apply center bias with fading
        bias_strength = max(0.0, 1.0 - (move_number / 20.0))
        
        biased_priors = np.zeros(COLUMN_COUNT, dtype=np.float32)
        for m in valid:
            if bias_strength > 0:
                biased_priors[m] = prior[m] * (1.0 + bias_strength * (CENTER_BIAS[m] - 1.0))
            else:
                biased_priors[m] = prior[m]
            
        valid_priors_sum = sum(biased_priors[m] for m in valid)
        if valid_priors_sum > 0:
            for m in valid:
                biased_priors[m] /= valid_priors_sum
            
        original_priors = biased_priors.copy()
            
        # Apply Dirichlet noise
        child_priors = np.zeros(COLUMN_COUNT, dtype=np.float32)
        if self.explore:
            dirichlet = np.random.dirichlet([self.dirichlet_noise] * len(valid))
            
            for i, m in enumerate(valid):
                child_priors[m] = 0.75 * biased_priors[m] + 0.25 * dirichlet[i]
        else:
            child_priors = biased_priors.copy()
        
        valid_priors_sum = sum(child_priors[m] for m in valid)
        if valid_priors_sum > 0:
            for m in valid:
                child_priors[m] /= valid_priors_sum
        else:
            for m in valid:
                child_priors[m] = 1.0 / len(valid)
        
        # Create child nodes with nn_version set - FIX: Pass nn_version to children
        for m in valid:
            if m not in self.root.children:
                ns = self.root.state.copy()
                ns.drop_piece(m)
                if not ns.game_over:
                    ns.switch()
                self.root.children[m] = TreeNode(
                    ns, 
                    parent=self.root, 
                    move=m, 
                    prior=child_priors[m],
                    original_prior=original_priors[m],
                    nn_version=getattr(self.nn, 'version', None)  # Fix for issue #1
                )

        # Run MCTS simulations
        for _ in range(adaptive_iterations):
            node = self.root
            path = [node]  # Track path for depth bonus
            
            while node.children:
                node = node.best_child(c_puct_value)
                path.append(node)
                
            if not node.state.game_over:
                probs, value = self.nn.policy_value(node.state)
                # FIX: Set nn_version when expanding node
                node.nn_version = getattr(self.nn, 'version', None)
                
                valid = node.state.valid_moves()
                
                child_priors = np.zeros(COLUMN_COUNT, dtype=np.float32)
                for m in valid:
                    child_priors[m] = probs[m] * CENTER_BIAS[m]
                
                valid_priors_sum = sum(child_priors[m] for m in valid)
                if valid_priors_sum > 0:
                    for m in valid:
                        child_priors[m] /= valid_priors_sum
                else:
                    for m in valid:
                        child_priors[m] = 1.0 / len(valid)
                
                for m in valid:
                    if m not in node.children:
                        ns = node.state.copy()
                        ns.drop_piece(m)
                        if not ns.game_over:
                            ns.switch()
                        node.children[m] = TreeNode(
                            ns, 
                            parent=node, 
                            move=m, 
                            prior=child_priors[m],
                            nn_version=getattr(self.nn, 'version', None)  # Fix for issue #1
                        )
            else:
                if node.state.winner == 'Draw':
                    value = 0.0
                else:
                    # Add depth bonus for longer games (Issue #4 fix)
                    base_value = 1.0 if node.state.winner == node.player else -1.0
                    depth_bonus = 0.001 * len(path)  # Small bonus for longer games
                    value = base_value + (depth_bonus if base_value > 0 else -depth_bonus)
            
            # Backpropagation with depth adjustment
            cur = node
            while cur:
                cur.visits += 1
                cur.value_sum += value if cur.player == node.player else -value
                cur = cur.parent

        # Calculate visit counts and probabilities
        visit_counts = np.zeros(COLUMN_COUNT, dtype=np.float32)
        for m, child in self.root.children.items():
            visit_counts[m] = child.visits
            
        if self.temperature != 1.0:
            visit_counts = np.power(visit_counts, 1/self.temperature)
            
        if visit_counts.sum() == 0:
            for m in valid:
                visit_counts[m] = 1.0
                
        visit_probs = visit_counts / visit_counts.sum()
        self.nn.add_example(root_state, visit_probs)
        
        # Choose move with exploration early in game
        if move_number < 8 and self.explore:
            valid_moves = [m for m in valid if visit_counts[m] > 0]
            if valid_moves:
                move_probs = visit_counts[valid_moves] / visit_counts[valid_moves].sum()
                best_move = int(np.random.choice(valid_moves, p=move_probs))
            else:
                best_move = int(visit_counts.argmax())
        else:
            best_move = int(visit_counts.argmax())
        
        return best_move

    def _calculate_safety_probability(self):
        # Get accuracy metrics from neural network if available
        if hasattr(self.nn, 'recent_tactical_accuracy'):
            # Use measured tactical accuracy to determine safety probability
            accuracy = self.nn.recent_tactical_accuracy
            # Higher accuracy = lower need for safety override
            return max(0.1, 1.0 - accuracy)
        
        # Fallback to game-based annealing if no accuracy metric
        total_games = 0
        if hasattr(self.nn, 'total_games'):
            total_games = self.nn.total_games
        
        # Anneal from 100% to 10% over 5000 games
        max_games = 5000
        start_prob = 1.0
        end_prob = 0.1
        
        if total_games >= max_games:
            return end_prob
        
        # Linear annealing
        progress = total_games / max_games
        return start_prob - progress * (start_prob - end_prob)

    def _priors_are_stale(self):
        """Check if stored priors are from an older version of the neural network"""
        if not hasattr(self.nn, 'version') or not hasattr(self, 'root'):
            return False
        
        # Check if any child node has stored version info
        if hasattr(self.root, 'nn_version'):
            return self.root.nn_version != self.nn.version
        
        # If no version stored, assume fresh
        return False


    def _normalize_reused_root(self):
        if not hasattr(self, 'root') or self.root is None:
            return
        
        child_visit_sum = sum(child.visits for child in self.root.children.values())
        
        if self.root.visits <= 0 or child_visit_sum <= 0:
            return
        
        # Statistical normalization - aim for 70% of visits in children
        target_ratio = 0.7
        desired_child_visits = self.root.visits * target_ratio
        
        if child_visit_sum > 0:
            scale_factor = desired_child_visits / child_visit_sum
            
            # Apply exponential smoothing instead of hard scaling
            alpha = min(0.5, 1.0 / max(1, math.log(self.root.visits + 1)))
            
            for child in self.root.children.values():
                old_visits = child.visits
                new_visits = int(old_visits * scale_factor)
                # Exponential smoothing
                child.visits = int(old_visits * (1 - alpha) + new_visits * alpha)
                
                # Adjust value_sum proportionally
                if old_visits > 0:
                    child.value_sum *= child.visits / old_visits
        
        # Adjust root visits to maintain consistency
        new_child_sum = sum(child.visits for child in self.root.children.values())
        if new_child_sum > 0:
            self.root.visits = int(new_child_sum / target_ratio)
            q_value = self.root.value_sum / max(1, self.root.visits)
            self.root.value_sum = q_value * self.root.visits
                    
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
    def __init__(self, iters, c, nn: NNManager, explore=True, model_path=None, dirichlet_noise=0.4, safe_bypass_prob=0.15, ai_delay=0.1):
        # Create independent neural network manager for this player
        if model_path is not None and model_path != nn.model_path:
            # Load independent model for this player
            hyperparams = nn.hyperparams.copy() if hasattr(nn, 'hyperparams') else {}
            use_resnet = hyperparams.get('use_resnet', False)
            self.nn = NNManager(hyperparams, model_path, use_resnet=use_resnet)
        else:
            # Use shared neural network
            self.nn = nn
        
        self.model_path = model_path
        # Use higher c_puct for exploration during training
        mcts_c = c if not explore else max(c, 1.2)
        self.mcts = MCTS(iters, mcts_c, self.nn, explore=explore, dirichlet_noise=dirichlet_noise, safe_bypass_prob=safe_bypass_prob)
        self.root = None
        self.move_number = 0
        self.ai_delay = ai_delay  # Configurable delay (Issue #9 fix)
    
    def reset(self):
        """Reset the AI player state between games"""
        self.root = None
        self.move_number = 0
        if hasattr(self.mcts, 'root'):
            self.mcts.root = None
        
    def get_move(self, state, gui=None):
        start = time.time()
        mv = self.mcts.search(state, previous_root=self.root, move_number=self.move_number)
        
        if hasattr(self.mcts, 'root') and self.mcts.root is not None:
            if hasattr(self.mcts.root, 'children') and mv in self.mcts.root.children:
                self.root = self.mcts.root.children[mv]
                if self.root is not None:
                    self.root.parent = None
                    # Store neural network version for staleness check
                    if hasattr(self.nn, 'version'):
                        self.root.nn_version = self.nn.version
            else:
                self.root = None
        else:
            self.root = None
                
        self.move_number += 1
        
        # Configurable delay (Issue #9 fix)
        dt = time.time() - start
        if self.ai_delay > 0 and dt < self.ai_delay:
            time.sleep(self.ai_delay - dt)
        return mv

# ----------------------------------------------------------------------
# Global training function
# ----------------------------------------------------------------------

def _play_single_training_game(mcts_iterations, puct_c, nn_manager_config, dirichlet_noise=0.4):
    try:
        model_path = nn_manager_config.get('model_path', NN_MODEL_FILE)
        hyperparams = nn_manager_config.get('hyperparams', None)
        use_resnet = nn_manager_config.get('use_resnet', False)
        nn_copy = NNManager(hyperparams, model_path, quiet=True, use_resnet=use_resnet)
        
        # Calculate adaptive safe-move bypass based on current training state
        total_games = nn_copy.total_games if hasattr(nn_copy, 'total_games') else 0
        
        # Adaptive bypass: start at 15%, decay to 1% based on training progress
        max_games = 5000
        start_bypass = 0.15
        end_bypass = 0.01
        
        if total_games >= max_games:
            safe_bypass_prob = end_bypass
        else:
            progress = total_games / max_games
            safe_bypass_prob = start_bypass - progress * (start_bypass - end_bypass)
        
        # Create AI players with adaptive bypass probability
        ai_red = MCTSComputerPlayer(mcts_iterations, puct_c, nn_copy, explore=True, 
                                   dirichlet_noise=dirichlet_noise, safe_bypass_prob=safe_bypass_prob)
        ai_green = MCTSComputerPlayer(mcts_iterations, puct_c, nn_copy, explore=True, 
                                     dirichlet_noise=dirichlet_noise, safe_bypass_prob=safe_bypass_prob)
        
        game = Connect4Game()
        serialized_examples = []
        
        while not game.game_over:
            player = game.current_player
            player_ai = ai_red if player == RED_PIECE else ai_green
            
            state_before = game.copy()
            board_before = state_before.board.copy().tolist()
            player_before = state_before.current_player
            
            move = player_ai.get_move(game)
            
            if nn_copy.pending:
                policy_data = nn_copy.pending[-1]['policy'].numpy().tolist()
                
                example_data = {
                    'board': board_before,
                    'player': player_before,
                    'policy': policy_data
                }
                serialized_examples.append(example_data)
            
            game.drop_piece(move)
            
            if not game.game_over:
                game.switch()
        
        return serialized_examples, game.winner
    except Exception as e:
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
        
        self.p1 = tk.StringVar(master, master.last_p1 if hasattr(master, 'last_p1') else "Human")
        self.p2 = tk.StringVar(master, master.last_p2 if hasattr(master, 'last_p2') else "Computer (AI)")
        self.red_model = tk.StringVar(master, master.last_red_model if hasattr(master, 'last_red_model') else NN_MODEL_FILE)
        self.green_model = tk.StringVar(master, master.last_green_model if hasattr(master, 'last_green_model') else NN_MODEL_FILE)
        self.continuous_play = tk.BooleanVar(master, master.last_continuous_play if hasattr(master, 'last_continuous_play') else False)
        self.fullplay_var = tk.BooleanVar(master, master.last_fullplay if hasattr(master, 'last_fullplay') else False)
        
        super().__init__(master, "Select Players")
        
    def _browse_model(self, player_var):
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
        
        ttk.Label(m, text="Red:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        for i, o in enumerate(opts, 1):
            rb = ttk.Radiobutton(m, text=o, variable=self.p1, value=o, 
                              command=lambda: self._update_file_browse_state())
            rb.grid(row=0, column=i, sticky="w")
            ToolTip(rb, f"Select {o} for Red player")
        
        self.red_browse = ttk.Button(m, text="📂", width=3, 
                             command=lambda: self._browse_model(self.red_model))
        self.red_browse.grid(row=0, column=4, padx=5)
        ToolTip(self.red_browse, "Browse for Red AI neural network model file")
        
        self.red_model_label = ttk.Label(m, text="", foreground="#0000FF", font=("Helvetica", 9, "bold"))
        self.red_model_label.grid(row=0, column=5, sticky="w", padx=2)
        
        ttk.Label(m, text="Green:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        for i, o in enumerate(opts, 1):
            rb = ttk.Radiobutton(m, text=o, variable=self.p2, value=o,
                              command=lambda: self._update_file_browse_state())
            rb.grid(row=1, column=i, sticky="w")
            ToolTip(rb, f"Select {o} for Green player")
        
        self.green_browse = ttk.Button(m, text="📂", width=3,
                               command=lambda: self._browse_model(self.green_model))
        self.green_browse.grid(row=1, column=4, padx=5)
        ToolTip(self.green_browse, "Browse for Green AI neural network model file")
        
        self.green_model_label = ttk.Label(m, text="", foreground="#0000FF", font=("Helvetica", 9, "bold"))
        self.green_model_label.grid(row=1, column=5, sticky="w", padx=2)
        
        ttk.Separator(m, orient="horizontal").grid(row=2, column=0, columnspan=6, sticky="ew", pady=10)
        
        options_frame = ttk.Frame(m)
        options_frame.grid(row=3, column=0, columnspan=6, sticky="w", padx=5, pady=5)
        
        self.fullplay_check = ttk.Checkbutton(options_frame, text="Play @ Full Strength", variable=self.fullplay_var)
        self.fullplay_check.grid(row=0, column=0, sticky="w", padx=5)
        ToolTip(self.fullplay_check, "When enabled, AI plays at full strength without exploration noise")
        
        self.continuous_check = ttk.Checkbutton(options_frame, text="Continuous Play", variable=self.continuous_play)
        self.continuous_check.grid(row=0, column=1, sticky="w", padx=25)
        ToolTip(self.continuous_check, "When enabled, computer vs computer games will play continuously\nuntil Stop is pressed or max games is reached")
        
        self._update_file_browse_state()
        
        return None
    
    def _update_file_browse_state(self):
        if self.p1.get() == "Computer (AI)":
            self.red_browse["state"] = "normal"
            model_name = os.path.splitext(os.path.basename(self.red_model.get()))[0]
            self.red_model_label.config(text=model_name)
        else:
            self.red_browse["state"] = "disabled"
            self.red_model_label.config(text="")
            
        if self.p2.get() == "Computer (AI)":
            self.green_browse["state"] = "normal"
            model_name = os.path.splitext(os.path.basename(self.green_model.get()))[0]
            self.green_model_label.config(text=model_name)
        else:
            self.green_browse["state"] = "disabled"
            self.green_model_label.config(text="")
            
        has_ai = (self.p1.get() == "Computer (AI)" or self.p2.get() == "Computer (AI)")
        if has_ai:
            self.fullplay_check.state(['!disabled'])
        else:
            self.fullplay_check.state(['disabled'])
            
        is_comp = all(p != "Human" for p in [self.p1.get(), self.p2.get()])
        if is_comp:
            self.continuous_check.state(['!disabled'])
            self.continuous_play.set(True)
        else:
            self.continuous_check.state(['disabled'])
            self.continuous_play.set(False)
            
    def apply(self):
        red_player = self.p1.get()
        green_player = self.p2.get()
        
        # Use a consistent full-strength c_puct value
        full_strength_c_puct = 0.8  # Matches the value in MCTSComputerPlayer
        
        # Create player objects based on selection
        if red_player == "Human":
            red = HumanPlayer()
        elif red_player == "Computer (Random)":
            red = RandomComputerPlayer()
        else:  # Computer (AI)
            red = MCTSComputerPlayer(
                self.mcts['iterations'], 
                self.mcts['C_param'] if not self.fullplay_var.get() else full_strength_c_puct, 
                self.nn, 
                explore=not self.fullplay_var.get(),
                model_path=self.red_model.get(),
                dirichlet_noise=self.mcts.get('dirichlet_noise', 0.3)
            )
        
        if green_player == "Human":
            green = HumanPlayer()
        elif green_player == "Computer (Random)":
            green = RandomComputerPlayer()
        else:  # Computer (AI)
            green = MCTSComputerPlayer(
                self.mcts['iterations'], 
                self.mcts['C_param'] if not self.fullplay_var.get() else full_strength_c_puct, 
                self.nn, 
                explore=not self.fullplay_var.get(),
                model_path=self.green_model.get(),
                dirichlet_noise=self.mcts.get('dirichlet_noise', 0.3)
            )
        
        # Save last selections for next time
        self.master.last_p1 = red_player
        self.master.last_p2 = green_player
        self.master.last_red_model = self.red_model.get()
        self.master.last_green_model = self.green_model.get()
        self.master.last_continuous_play = self.continuous_play.get()
        self.master.last_fullplay = self.fullplay_var.get()
        
        self.result = {
            'red': red,
            'green': green,
            'continuous_play': self.continuous_play.get(),
            'full_strength': self.fullplay_var.get()
        }
        
class SettingsDialog(simpledialog.Dialog):
    
    def __init__(self, master, mcts_params, train_games=200, max_cc_games=100, cc_train_interval=50, 
             cc_delay=500, games_before_training=20, nn_params=None):
        self.it = tk.StringVar(master, str(mcts_params['iterations']))
        self.c = tk.StringVar(master, f"{mcts_params['C_param']:.2f}")
        self.games = tk.StringVar(master, str(train_games))
        self.max_cc = tk.StringVar(master, str(max_cc_games))
        self.cc_train_interval = tk.StringVar(master, str(cc_train_interval))
        self.cc_delay = tk.StringVar(master, str(cc_delay))

        if isinstance(games_before_training, dict):
            games_before_training = min(100, train_games)
        self.games_before_training = tk.StringVar(master, str(games_before_training))

        self.nn_params = nn_params or {
            'learning_rate': 5e-4, 
            'batch_size': 128, 
            'epochs': 10,
            'policy_weight': 1.5,
            'value_weight': 1.0,
            'lr_decay': 0.9995,
            'use_resnet': False,
            'use_swa': False
        }

        self.lr = tk.StringVar(master, str(self.nn_params.get('learning_rate', 5e-4)))
        self.batch = tk.StringVar(master, str(self.nn_params.get('batch_size', 128)))
        self.epochs = tk.StringVar(master, str(self.nn_params.get('epochs', 10)))
        self.policy_weight = tk.StringVar(master, str(self.nn_params.get('policy_weight', 1.5)))
        self.value_weight = tk.StringVar(master, str(self.nn_params.get('value_weight', 1.0)))
        self.lr_decay = tk.StringVar(master, str(self.nn_params.get('lr_decay', 0.9995)))
        self.use_resnet = tk.BooleanVar(master, self.nn_params.get('use_resnet', False))
        self.use_swa = tk.BooleanVar(master, self.nn_params.get('use_swa', False))

        # FIX ISSUE #13: Default dirichlet_noise should be 0.4 to match hardcoded value
        self.dirichlet_noise = tk.StringVar(master, str(mcts_params.get('dirichlet_noise', 0.4)))

        # Initialize column bias variables
        # Get current CENTER_BIAS values (first 4)
        current_bias = list(CENTER_BIAS[:4])
        self.col_bias_1 = tk.StringVar(master, str(current_bias[0]))
        self.col_bias_2 = tk.StringVar(master, str(current_bias[1]))
        self.col_bias_3 = tk.StringVar(master, str(current_bias[2]))
        self.col_bias_4 = tk.StringVar(master, str(current_bias[3]))
        
        # Add AI delay setting (Issue #9 fix)
        self.ai_delay = tk.StringVar(master, "0.1")

        super().__init__(master, "Settings")
            
    def body(self, m):
        self.feedback_frame = ttk.Frame(m)
        self.feedback_frame.pack(fill='x', pady=5)

        nb = ttk.Notebook(m)
        nb.pack(fill='both', expand=True, padx=5, pady=5)

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
        
        # Add curriculum scaling cap (Issue #7 fix)
        ttk.Label(mcts_tab, text="Max iteration scaling:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.max_iter_scale = tk.StringVar(master=m, value="2.0")
        e_scale = ttk.Entry(mcts_tab, textvariable=self.max_iter_scale, width=10)
        e_scale.grid(row=2, column=1, padx=5)
        ToolTip(e_scale, "Maximum multiplier for MCTS iterations in curriculum learning\nPrevents UI lag on strong models")

        train_tab = ttk.Frame(nb)
        nb.add(train_tab, text="Training")

        ttk.Label(train_tab, text="Games in Train NN mode:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        e3 = ttk.Entry(train_tab, textvariable=self.games, width=10)
        e3.grid(row=0, column=1, padx=5)
        ToolTip(e3, "Number of self-play games for training when using Train NN button")

        ttk.Label(train_tab, text="Games before training:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        e3a = ttk.Entry(train_tab, textvariable=self.games_before_training, width=10)
        e3a.grid(row=1, column=1, padx=5)
        ToolTip(e3a, "Train the neural network after this many games during Train NN mode")

        ttk.Separator(train_tab, orient='horizontal').grid(row=2, column=0, columnspan=2, sticky="ew", padx=5, pady=10)

        ttk.Label(train_tab, text="Games in AI-AI mode:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        e3b = ttk.Entry(train_tab, textvariable=self.max_cc, width=10)
        e3b.grid(row=3, column=1, padx=5)
        ToolTip(e3b, "Maximum number of games in continuous Computer-Computer play")

        ttk.Label(train_tab, text="AI-AI games before training:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        e3c = ttk.Entry(train_tab, textvariable=self.cc_train_interval, width=10)
        e3c.grid(row=4, column=1, padx=5)
        ToolTip(e3c, "Number of AI vs AI games to play before training the neural network\nMust be less than or equal to Games in AI-AI mode")

        ttk.Label(train_tab, text="Delay in AI-AI games (ms):").grid(row=5, column=0, sticky="w", padx=5, pady=5)
        e3d = ttk.Entry(train_tab, textvariable=self.cc_delay, width=10)
        e3d.grid(row=5, column=1, padx=5)
        ToolTip(e3d, "Delay in milliseconds between moves in AI vs AI games\nHigher values slow down the game to make it more visible")
        
        # Add AI delay for all games (Issue #9 fix)
        ttk.Label(train_tab, text="AI move delay (s):").grid(row=6, column=0, sticky="w", padx=5, pady=5)
        e_delay = ttk.Entry(train_tab, textvariable=self.ai_delay, width=10)
        e_delay.grid(row=6, column=1, padx=5)
        ToolTip(e_delay, "Minimum delay between AI moves in all games\nSet to 0 for no delay")

        self._add_training_presets(train_tab)

        nn_tab = ttk.Frame(nb)
        nb.add(nn_tab, text="Neural Network")

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

        advanced_tab = ttk.Frame(nb)
        nb.add(advanced_tab, text="Advanced")

        ttk.Label(advanced_tab, text="Network Architecture:").grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=(10,5))

        resnet_check = ttk.Checkbutton(advanced_tab, text="Use ResNet Architecture", variable=self.use_resnet)
        resnet_check.grid(row=1, column=0, columnspan=2, sticky="w", padx=20, pady=5)
        ToolTip(resnet_check, "Use ResNet architecture instead of standard CNN\nMore powerful but slower to train\nWarning: Changing will create a new model")

        ttk.Label(advanced_tab, text="Training Techniques:").grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=(15,5))

        swa_check = ttk.Checkbutton(advanced_tab, text="Use Stochastic Weight Averaging (SWA)", variable=self.use_swa)
        swa_check.grid(row=3, column=0, columnspan=2, sticky="w", padx=20, pady=5)
        ToolTip(swa_check, "Stochastic Weight Averaging averages models from\nlater epochs for better generalization\nMakes training more robust but slightly slower")

        # Add Dirichlet noise parameter  
        ttk.Label(advanced_tab, text="Exploration Parameters:").grid(row=4, column=0, columnspan=2, sticky="w", padx=5, pady=(15,5))
                
        ttk.Label(advanced_tab, text="Dirichlet Noise:").grid(row=5, column=0, sticky="w", padx=20, pady=5)
        e_noise = ttk.Entry(advanced_tab, textvariable=self.dirichlet_noise, width=10)
        e_noise.grid(row=5, column=1, padx=5, pady=5)
        ToolTip(e_noise, "Dirichlet noise for root node exploration\nHigher values increase exploration\nRecommended range: 0.1-0.5")

        # Add column bias tab
        bias_tab = ttk.Frame(nb)
        nb.add(bias_tab, text="Column Bias")

        ttk.Label(bias_tab, text="Column Positional Bias:", font=("Helvetica", 10, "bold")).grid(
            row=0, column=0, columnspan=4, sticky="w", padx=5, pady=(10,5))

        ttk.Label(bias_tab, text="These weights determine how much the AI favors each column.").grid(
            row=1, column=0, columnspan=4, sticky="w", padx=5, pady=(0,10))

        ttk.Label(bias_tab, text="Higher values mean stronger preference for that column.").grid(
            row=2, column=0, columnspan=4, sticky="w", padx=5, pady=(0,10))

        ttk.Label(bias_tab, text="Columns 5-7 are set symmetrically (1=7, 2=6, 3=5)").grid(
            row=3, column=0, columnspan=4, sticky="w", padx=5, pady=(0,10))

        # Create a frame for the bias inputs
        bias_frame = ttk.Frame(bias_tab)
        bias_frame.grid(row=4, column=0, columnspan=4, padx=5, pady=10)

        # Column 1
        ttk.Label(bias_frame, text="Column 1:").grid(row=0, column=0, sticky="e", padx=(0,5), pady=5)
        col1_entry = ttk.Entry(bias_frame, textvariable=self.col_bias_1, width=8)
        col1_entry.grid(row=0, column=1, padx=(0,15), pady=5)
        ToolTip(col1_entry, "Bias weight for column 1 (and 7)")

        # Column 2
        ttk.Label(bias_frame, text="Column 2:").grid(row=0, column=2, sticky="e", padx=(0,5), pady=5)
        col2_entry = ttk.Entry(bias_frame, textvariable=self.col_bias_2, width=8)
        col2_entry.grid(row=0, column=3, padx=(0,15), pady=5)
        ToolTip(col2_entry, "Bias weight for column 2 (and 6)")

        # Column 3
        ttk.Label(bias_frame, text="Column 3:").grid(row=1, column=0, sticky="e", padx=(0,5), pady=5)
        col3_entry = ttk.Entry(bias_frame, textvariable=self.col_bias_3, width=8)
        col3_entry.grid(row=1, column=1, padx=(0,15), pady=5)
        ToolTip(col3_entry, "Bias weight for column 3 (and 5)")

        # Column 4
        ttk.Label(bias_frame, text="Column 4:").grid(row=1, column=2, sticky="e", padx=(0,5), pady=5)
        col4_entry = ttk.Entry(bias_frame, textvariable=self.col_bias_4, width=8)
        col4_entry.grid(row=1, column=3, padx=(0,15), pady=5)
        ToolTip(col4_entry, "Bias weight for column 4 (center)")

        # Preview of the bias
        ttk.Label(bias_tab, text="Preview:").grid(row=5, column=0, sticky="w", padx=5, pady=(15,5))
        self.preview_canvas = tk.Canvas(bias_tab, width=280, height=40, bg="white")
        self.preview_canvas.grid(row=6, column=0, columnspan=4, padx=5, pady=5)

        # Update preview when values change
        self.col_bias_1.trace_add("write", lambda *args: self._update_bias_preview())
        self.col_bias_2.trace_add("write", lambda *args: self._update_bias_preview())
        self.col_bias_3.trace_add("write", lambda *args: self._update_bias_preview())
        self.col_bias_4.trace_add("write", lambda *args: self._update_bias_preview())

        # Add a Reset button to restore default values
        reset_btn = ttk.Button(bias_tab, text="Reset to Defaults", 
                             command=self._reset_column_bias)
        reset_btn.grid(row=7, column=0, columnspan=4, padx=5, pady=15)

        self._update_bias_preview()

        return mcts_tab
        
    def _reset_column_bias(self):
        default_bias = [0.5, 0.7, 1.0, 1.5]
        self.col_bias_1.set(str(default_bias[0]))
        self.col_bias_2.set(str(default_bias[1]))
        self.col_bias_3.set(str(default_bias[2]))
        self.col_bias_4.set(str(default_bias[3]))
        
    def _update_bias_preview(self):
        try:
            # Get the current bias values
            try:
                bias_1 = float(self.col_bias_1.get())
                bias_2 = float(self.col_bias_2.get())
                bias_3 = float(self.col_bias_3.get())
                bias_4 = float(self.col_bias_4.get())
                
                # Create the full bias array with symmetry
                bias_values = [bias_1, bias_2, bias_3, bias_4, bias_3, bias_2, bias_1]
            except:
                # Use default values if parsing fails
                bias_values = [0.5, 0.7, 1.0, 1.5, 1.0, 0.7, 0.5]
                
            # Clear the canvas
            self.preview_canvas.delete("all")
            
            # Draw column labels
            for i in range(7):
                self.preview_canvas.create_text(
                    20 + i * 40, 10, 
                    text=str(i+1), 
                    font=("Helvetica", 8)
                )
            
            # Normalize bias values for visualization
            max_bias = max(bias_values)
            if max_bias > 0:
                normalized = [b / max_bias for b in bias_values]
            else:
                normalized = [0] * 7
                
            # Draw bars
            for i, (bias, norm) in enumerate(zip(bias_values, normalized)):
                height = norm * 20
                self.preview_canvas.create_rectangle(
                    10 + i * 40, 38 - height,
                    30 + i * 40, 38,
                    fill="#8888FF", outline="#000000"
                )
                # Add value text
                self.preview_canvas.create_text(
                    20 + i * 40, 38 - height - 5,
                    text=f"{bias:.1f}",
                    font=("Helvetica", 7)
                )
        except:
            # Ignore visualization errors
            pass
    
    def _add_training_presets(self, train_tab):
        preset_frame = ttk.LabelFrame(train_tab, text="Training Presets")
        preset_frame.grid(row=7, column=0, columnspan=2, sticky="ew", padx=5, pady=10)
        
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
                "iterations": 600,
                "batch_size": 64,
                "epochs": 8,
                "description": "High quality with reasonable speed"
            },
            "Deep": {
                "iterations": 800,
                "batch_size": 64,
                "epochs": 10,
                "use_resnet": True,
                "description": "Deep training with ResNet architecture"
            }
        }
        
        self.preset_buttons = {}
        
        for i, (name, settings) in enumerate(self.presets.items()):
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
            
            tooltip_text = f"{name}: {settings['description']}\n" + \
                        f"MCTS: {settings['iterations']} iterations\n" + \
                        f"Batch: {settings['batch_size']}, Epochs: {settings['epochs']}"
                        
            if 'use_resnet' in settings and settings['use_resnet']:
                tooltip_text += "\nUses ResNet architecture"
                
            ToolTip(btn, tooltip_text)
        
        ttk.Label(preset_frame, text="Select a preset to quickly configure training settings",
                  font=("Helvetica", 8)).grid(row=1, column=0, columnspan=4, pady=(0,5))
        
        self.active_preset = None
    
    def _apply_preset(self, preset_name, settings):
        self.it.set(str(settings['iterations']))
        
        self.batch.set(str(settings['batch_size']))
        self.epochs.set(str(settings['epochs']))
        
        if 'use_resnet' in settings:
            self.use_resnet.set(settings['use_resnet'])
            
        if 'use_swa' in settings:
            self.use_swa.set(settings['use_swa'])
        
        for widget in self.feedback_frame.winfo_children():
            widget.destroy()
            
        feedback_label = ttk.Label(self.feedback_frame, text=f"{preset_name} preset applied", foreground="green")
        feedback_label.pack(side="left", padx=5)
        
        for name, btn in self.preset_buttons.items():
            style_name = f"{name}.TButton"
            if name == preset_name:
                ttk.Style().configure(style_name, background="#CCE5FF")
                self.active_preset = preset_name
            else:
                ttk.Style().configure(style_name, background="#E0E0E0")
        
        self.after(2000, lambda: feedback_label.destroy())
            
    def validate(self):
        try:
            it = int(self.it.get())
            c = float(self.c.get())
            games = int(self.games.get())
            games_before_training = int(self.games_before_training.get())
            max_cc = int(self.max_cc.get())
            cc_train_interval = int(self.cc_train_interval.get())
            cc_delay = int(self.cc_delay.get())
            
            lr = float(self.lr.get())
            batch = int(self.batch.get())
            epochs = int(self.epochs.get())
            p_weight = float(self.policy_weight.get())
            v_weight = float(self.value_weight.get())
            lr_decay = float(self.lr_decay.get())
            dirichlet_noise = float(self.dirichlet_noise.get())
            ai_delay = float(self.ai_delay.get())
            max_iter_scale = float(self.max_iter_scale.get())
            
            # Validate column bias values
            col_bias_1 = float(self.col_bias_1.get())
            col_bias_2 = float(self.col_bias_2.get())
            col_bias_3 = float(self.col_bias_3.get())
            col_bias_4 = float(self.col_bias_4.get())
            
            if (it <= 0 or c < 0 or games <= 0 or games_before_training <= 0 or max_cc <= 0 or 
                cc_train_interval <= 0 or cc_delay < 0 or
                lr <= 0 or batch <= 0 or epochs <= 0 or 
                p_weight <= 0 or v_weight <= 0 or dirichlet_noise <= 0 or ai_delay < 0 or
                max_iter_scale < 1.0):
                messagebox.showwarning("Invalid", "All values must be positive (delay can be zero, max scale >= 1.0).")
                return False
            
            # Check column bias values
            if col_bias_1 < 0 or col_bias_2 < 0 or col_bias_3 < 0 or col_bias_4 < 0:
                messagebox.showwarning("Invalid", "Column bias values must be non-negative.")
                return False
                    
            if lr_decay <= 0 or lr_decay > 1.0:
                messagebox.showwarning("Invalid", "Learning rate decay must be between 0 and 1.0")
                return False
            
            if games_before_training > games:
                messagebox.showwarning("Invalid", "Games before training cannot be greater than Games in Train NN mode.")
                return False
                    
            if cc_train_interval > max_cc:
                messagebox.showwarning("Invalid", "AI-AI games before training cannot be greater than Games in AI-AI mode.")
                return False
                    
            return True
        except:
            messagebox.showwarning("Invalid", "Enter valid numbers.")
            return False
        
    def apply(self):
        # Get column bias values
        try:
            col_bias_1 = float(self.col_bias_1.get())
            col_bias_2 = float(self.col_bias_2.get())
            col_bias_3 = float(self.col_bias_3.get())
            col_bias_4 = float(self.col_bias_4.get())
            
            # Update the global CENTER_BIAS with symmetry
            global CENTER_BIAS
            CENTER_BIAS = np.array([col_bias_1, col_bias_2, col_bias_3, col_bias_4, col_bias_3, col_bias_2, col_bias_1])
        except:
            # Keep existing values if there's an error
            pass
        
        self.result = {
            'mcts': {
                'iterations': int(self.it.get()),
                'C_param': float(self.c.get()),
                'dirichlet_noise': float(self.dirichlet_noise.get()),
                'max_iteration_scale': float(self.max_iter_scale.get())  # Issue #7 fix
            },
            'training': {
                'games': int(self.games.get()),
                'games_before_training': int(self.games_before_training.get()),
                'max_cc_games': int(self.max_cc.get()),
                'cc_train_interval': int(self.cc_train_interval.get()),
                'cc_delay': int(self.cc_delay.get()),
                'ai_delay': float(self.ai_delay.get())  # Issue #9 fix
            },
            'nn_params': {
                'learning_rate': float(self.lr.get()),
                'batch_size': int(self.batch.get()),
                'epochs': int(self.epochs.get()),
                'policy_weight': float(self.policy_weight.get()),
                'value_weight': float(self.value_weight.get()),
                'lr_decay': float(self.lr_decay.get()),
                'use_resnet': self.use_resnet.get(),
                'use_swa': self.use_swa.get()
            },
            'column_bias': list(CENTER_BIAS)  # Save the column bias settings
        }

# ----------------------------------------------------------------------
# Model Management Dialog
# ----------------------------------------------------------------------
class ModelManagementDialog(simpledialog.Dialog):
    def __init__(self, master):
        self.master = master
        self.model_dir = os.path.dirname(os.path.abspath(master.nn_model_path))  # Fixed: was master.training_model_path
        super().__init__(master, "Model Management")
        
    def body(self, frame):
        self.models = []
        self.model_versions = {}
        self.model_info = {}
        self.selected_model = None
        
        ttk.Label(frame, text="Available Models:", font=("Helvetica", 11, "bold")).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        
        # Create a frame for models with scrollbar
        list_frame = ttk.Frame(frame)
        list_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)
        
        self.models_list = tk.Listbox(list_frame, width=50, height=15, font=("Helvetica", 10))
        scrollbar = ttk.Scrollbar(list_frame, command=self.models_list.yview)
        self.models_list.config(yscrollcommand=scrollbar.set)
        
        self.models_list.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        
        self.models_list.bind('<<ListboxSelect>>', self._on_model_select)
        
        # Info display area
        info_frame = ttk.LabelFrame(frame, text="Model Information")
        info_frame.grid(row=2, column=0, sticky="ew", padx=5, pady=5)
        
        self.info_text = tk.Text(info_frame, width=50, height=8, font=("Helvetica", 9), wrap="word")
        info_scroll = ttk.Scrollbar(info_frame, command=self.info_text.yview)
        self.info_text.config(yscrollcommand=info_scroll.set, state="disabled")
        
        self.info_text.grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        info_scroll.grid(row=0, column=1, sticky="ns")
        info_frame.columnconfigure(0, weight=1)
        info_frame.rowconfigure(0, weight=1)
        
        # Action buttons
        button_frame = ttk.Frame(frame)
        button_frame.grid(row=3, column=0, sticky="ew", padx=5, pady=5)
        
        load_btn = ttk.Button(button_frame, text="Load Selected Model", command=self._load_selected)
        load_btn.grid(row=0, column=0, padx=5)
        ToolTip(load_btn, "Load the selected model as the active training model")
        
        compare_btn = ttk.Button(button_frame, text="Compare Models", command=self._compare_models)
        compare_btn.grid(row=0, column=1, padx=5)
        ToolTip(compare_btn, "Set up a game between two models")
        
        delete_btn = ttk.Button(button_frame, text="Delete", command=self._delete_model)
        delete_btn.grid(row=0, column=2, padx=5)
        ToolTip(delete_btn, "Delete the selected model version (cannot be undone)")
        
        refresh_btn = ttk.Button(button_frame, text="Refresh", command=self._load_models)
        refresh_btn.grid(row=0, column=3, padx=5)
        ToolTip(refresh_btn, "Refresh the model list")
        
        self._load_models()
        
        return frame

    def _safe_load_model(self, model_path, map_location=None):
        """
        Safely load a PyTorch model file, trying weights_only=True first, then falling back to weights_only=False
        """
        if map_location is None:
            map_location = torch.device('cpu')
        
        try:
            # First attempt with weights_only=True (safer)
            return torch.load(model_path, map_location=map_location, weights_only=True)
        except Exception as e:
            if "weights_only" in str(e).lower() or "unsupported global" in str(e).lower():
                # If it fails due to weights_only restriction, try without it
                try:
                    return torch.load(model_path, map_location=map_location, weights_only=False)
                except Exception as e2:
                    raise e2
            else:
                raise e
                
    def _load_models(self):
        self.models_list.delete(0, tk.END)
        self.models = []
        self.model_versions = {}
        self.model_info = {}
        
        # Get all .pt files in the model directory
        model_files = [f for f in os.listdir(self.model_dir) if f.endswith('.pt')]
        
        # Group by base name and extract version information
        for file in model_files:
            if '_v' in file:
                # Versioned model file
                base_name, version_ext = file.split('_v', 1)
                version = os.path.splitext(version_ext)[0]
                if base_name not in self.model_versions:
                    self.model_versions[base_name] = []
                self.model_versions[base_name].append(version)
            else:
                # Base model file
                base_name = os.path.splitext(file)[0]
                if base_name not in self.models:
                    self.models.append(base_name)
        
        # Load model information
        for base_name in self.models:
            # Try to load the main model file
            main_file = f"{base_name}.pt"
            main_path = os.path.join(self.model_dir, main_file)
            
            if os.path.exists(main_path):
                try:
                    # Use the safe loading method
                    model_data = self._safe_load_model(main_path)
                    info = {
                        'version': model_data.get('version', "unknown"),
                        'iterations': model_data.get('train_iterations', 0),
                        'games': model_data.get('total_games', 0),
                        'win_rate': model_data.get('win_rate', 0.5),
                        'architecture': 'ResNet' if model_data.get('hyperparams', {}).get('use_resnet', False) else 'CNN',
                        'last_modified': datetime.datetime.fromtimestamp(os.path.getmtime(main_path)).strftime('%Y-%m-%d %H:%M:%S')
                    }
                    self.model_info[main_file] = info
                    
                    # Add to listbox
                    self.models_list.insert(tk.END, f"{base_name} (Current: v{info['version']})")
                except Exception as e:
                    self.models_list.insert(tk.END, f"{base_name} (Error loading info)")
                    print(f"Error loading {main_path}: {e}")
                
            # Load version information
            if base_name in self.model_versions:
                for version in sorted(self.model_versions[base_name], key=self._version_key):
                    version_file = f"{base_name}_v{version}.pt"
                    version_path = os.path.join(self.model_dir, version_file)
                    
                    try:
                        # Use the safe loading method
                        model_data = self._safe_load_model(version_path)
                        info = {
                            'version': version,
                            'iterations': model_data.get('train_iterations', 0),
                            'games': model_data.get('total_games', 0),
                            'win_rate': model_data.get('win_rate', 0.5),
                            'architecture': 'ResNet' if model_data.get('hyperparams', {}).get('use_resnet', False) else 'CNN',
                            'last_modified': datetime.datetime.fromtimestamp(os.path.getmtime(version_path)).strftime('%Y-%m-%d %H:%M:%S')
                        }
                        self.model_info[version_file] = info
                        
                        # Add to listbox with indentation
                        self.models_list.insert(tk.END, f"    v{version}")
                    except Exception as e:
                        self.models_list.insert(tk.END, f"    v{version} (Error loading info)")
                        print(f"Error loading {version_path}: {e}")
        
        # If there are no models, show a message
        if not self.models:
            self.models_list.insert(tk.END, "No models found")
    
    def _version_key(self, version):
        try:
            return tuple(map(int, version.split('.')))
        except:
            return (0, 0, 0)
        
    def _on_model_select(self, event):
        selection = self.models_list.curselection()
        if not selection:
            return
        
        index = selection[0]
        selected_text = self.models_list.get(index)
        
        # Clear info display
        self.info_text.config(state="normal")
        self.info_text.delete(1.0, tk.END)
        
        # Determine the selected model file
        if selected_text.startswith("    v"):
            # This is a version
            version = selected_text.strip().lstrip("v")
            # Find the base model for this version
            # We need to look backwards in the list to find the base model
            base_model_name = None
            for i in range(index - 1, -1, -1):
                list_item = self.models_list.get(i)
                if not list_item.startswith("    "):
                    # Found the base model
                    match = re.match(r"^(.*?) \(Current:", list_item)
                    if match:
                        base_model_name = match.group(1)
                        break
            
            if base_model_name:
                self.selected_model = f"{base_model_name}_v{version}.pt"
            else:
                self.selected_model = None
        else:
            # This is a base model
            match = re.match(r"^(.*?) \(Current:", selected_text)
            if match:
                base_name = match.group(1)
                self.selected_model = f"{base_name}.pt"
            else:
                # Handle case where model has no version info
                # Extract just the model name before any parentheses
                base_name = selected_text.split('(')[0].strip()
                if base_name:
                    self.selected_model = f"{base_name}.pt"
                else:
                    self.selected_model = None
        
        # Show model info
        if self.selected_model and self.selected_model in self.model_info:
            info = self.model_info[self.selected_model]
            self.info_text.insert(tk.END, f"Model: {self.selected_model}\n")
            self.info_text.insert(tk.END, f"Version: {info['version']}\n")
            self.info_text.insert(tk.END, f"Training Iterations: {info['iterations']}\n")
            self.info_text.insert(tk.END, f"Total Games: {info['games']}\n")
            self.info_text.insert(tk.END, f"Win Rate: {info['win_rate']:.3f}\n")
            self.info_text.insert(tk.END, f"Architecture: {info['architecture']}\n")
            self.info_text.insert(tk.END, f"Last Modified: {info['last_modified']}\n")
        else:
            self.info_text.insert(tk.END, "No detailed information available.")
            if self.selected_model:
                self.info_text.insert(tk.END, f"\nSelected: {self.selected_model}")
            
        self.info_text.config(state="disabled")
            
    def _load_selected(self):
        if not self.selected_model:
            messagebox.showwarning("No Selection", "Please select a model first.")
            return
            
        selected_path = os.path.join(self.model_dir, self.selected_model)
        if not os.path.exists(selected_path):
            messagebox.showerror("Error", f"Model file not found: {selected_path}")
            return
            
        # Ask for confirmation
        confirm = messagebox.askyesno("Confirm", f"Set {self.selected_model} as the active training model?")
        if not confirm:
            return
            
        # Update the master's training model path
        self.master.nn_model_path = selected_path
        
        # Re-initialize the neural network
        try:
            # Check architecture type - use safe loading method
            model_data = self._safe_load_model(selected_path)
            use_resnet = model_data.get('hyperparams', {}).get('use_resnet', False)
            
            # Update master's nn_params
            self.master.nn_params['use_resnet'] = use_resnet
            
            # Reload neural network
            self.master.nn = NNManager(self.master.nn_params, self.master.nn_model_path)
            
            # Update model info display
            self.master._update_model_info()
            
            messagebox.showinfo("Success", f"Successfully loaded {self.selected_model} as the active model.")
            self.destroy()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
    
    def _compare_models(self):
        if not self.models or len(self.models) < 1:
            messagebox.showwarning("No Models", "Need at least one model to compare.")
            return
            
        # Create a dialog to select models to compare
        compare_dialog = ModelCompareDialog(self, self.model_dir, self.models, self.model_versions)
        if compare_dialog.result:
            red_model, green_model = compare_dialog.result
            
            # Close this dialog
            self.destroy()
            
            # Set up a game between these models
            self.master.last_p1 = "Computer (AI)"
            self.master.last_p2 = "Computer (AI)"
            self.master.last_red_model = red_model
            self.master.last_green_model = green_model
            self.master.last_continuous_play = True
            self.master.last_fullplay = True
            
            # Force the players to be updated
            self.master._choose_players()
            
            # Start the game
            self.master._new_game()
    
    def _delete_model(self):
        if not self.selected_model:
            messagebox.showwarning("No Selection", "Please select a model first.")
            return
            
        # Don't allow deleting the active model
        if self.selected_model == os.path.basename(self.master.training_model_path):
            messagebox.showwarning("Cannot Delete", "Cannot delete the currently active model.")
            return
            
        # Extra safeguard for base models
        if "_v" not in self.selected_model:
            confirm = messagebox.askyesno("Confirm Delete", 
                                          f"Are you ABSOLUTELY SURE you want to delete the base model {self.selected_model}?\n\n"
                                          "This could make version management difficult.", 
                                          icon="warning")
        else:
            confirm = messagebox.askyesno("Confirm Delete", 
                                        f"Are you sure you want to delete {self.selected_model}?", 
                                        icon="warning")
            
        if not confirm:
            return
            
        # Delete the file
        try:
            os.remove(os.path.join(self.model_dir, self.selected_model))
            messagebox.showinfo("Success", f"Deleted {self.selected_model}")
            self._load_models()  # Refresh list
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete model: {str(e)}")

class ModelCompareDialog(simpledialog.Dialog):
    
        def __init__(self, parent, model_dir, models, model_versions):
            self.model_dir = model_dir
            self.models = models
            self.model_versions = model_versions
            self.all_model_files = self._get_all_model_files()
            
            super().__init__(parent, "Compare Models")
            
        def _get_all_model_files(self):
            model_files = []
            for base_name in self.models:
                model_files.append(f"{base_name}.pt")
                for version in self.model_versions.get(base_name, []):
                    model_files.append(f"{base_name}_v{version}.pt")
            return model_files
            
        def body(self, frame):
            ttk.Label(frame, text="Select models to compare:", font=("Helvetica", 10, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", padx=5, pady=5)
            
            ttk.Label(frame, text="Red player:").grid(row=1, column=0, sticky="e", padx=5, pady=5)
            self.red_model_var = tk.StringVar(frame)
            red_combo = ttk.Combobox(frame, textvariable=self.red_model_var, width=40)
            red_combo['values'] = self.all_model_files
            red_combo.grid(row=1, column=1, sticky="w", padx=5, pady=5)
            if self.all_model_files:
                red_combo.current(0)
            
            ttk.Label(frame, text="Green player:").grid(row=2, column=0, sticky="e", padx=5, pady=5)
            self.green_model_var = tk.StringVar(frame)
            green_combo = ttk.Combobox(frame, textvariable=self.green_model_var, width=40)
            green_combo['values'] = self.all_model_files
            green_combo.grid(row=2, column=1, sticky="w", padx=5, pady=5)
            if len(self.all_model_files) > 1:
                green_combo.current(1)
            elif self.all_model_files:
                green_combo.current(0)
            
            ttk.Label(frame, text="This will set up a continuous play session\nbetween the selected models.", 
                     font=("Helvetica", 9)).grid(row=3, column=0, columnspan=2, padx=5, pady=10)
            
            return frame
        
        def validate(self):
            red_model = self.red_model_var.get()
            green_model = self.green_model_var.get()
            
            if not red_model or not green_model:
                messagebox.showwarning("Invalid Selection", "Please select both models.")
                return False
                
            red_path = os.path.join(self.model_dir, red_model)
            green_path = os.path.join(self.model_dir, green_model)
            
            if not os.path.exists(red_path):
                messagebox.showwarning("File Not Found", f"Red model file not found: {red_model}")
                return False
                
            if not os.path.exists(green_path):
                messagebox.showwarning("File Not Found", f"Green model file not found: {green_model}")
                return False
            
            return True
        
        def apply(self):
            red_model = os.path.join(self.model_dir, self.red_model_var.get())
            green_model = os.path.join(self.model_dir, self.green_model_var.get())
            self.result = (red_model, green_model)

    # ----------------------------------------------------------------------
    # GUI – Connect4GUI (updated for new features)
    # ----------------------------------------------------------------------

class Connect4GUI(tk.Tk):
        
    def __init__(self):
        super().__init__()
        self.title("Connect 4 – AlphaZero Edition (2025)")
        
        self.resizable(True, True)
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Set up minimum size first
        width = WIDTH + 350
        height = HEIGHT + 50
        self.minsize(width, height)
        
        # Maximize the window immediately at startup
        self.update_idletasks()  # Process pending events
        
        # Try different maximization approaches for cross-platform compatibility
        try:
            self.state('zoomed')  # Windows
        except:
            try:
                self.attributes('-zoomed', True)  # Linux (some window managers)
            except:
                try:
                    self.wm_state('zoomed')  # Alternative method
                except:
                    # If all else fails, use a reasonable large size
                    screen_width = self.winfo_screenwidth()
                    screen_height = self.winfo_screenheight()
                    self.geometry(f"{screen_width-100}x{screen_height-100}+50+50")
        
        # Force an update to apply the maximized state
        self.update_idletasks()

        # Initialize visualization attributes
        self.visualization_visible = True  # Changed to True by default
        self.policy_fig = None
        self.policy_canvas = None
        self.current_vis_player = RED_PIECE  # Track which player's data is being shown

        # Create style for bold labels
        style = ttk.Style()
        style.configure('Bold.TLabel', font=('Helvetica', 10, 'bold'))
        style.configure('Bold.TLabelframe.Label', font=('Helvetica', 10, 'bold'))
        style.configure('BoldStatus.TLabel', font=('Helvetica', 16, 'bold'))
        style.configure('Bold.TButton', font=('Helvetica', 9, 'bold'))
        
        # Apply bold style to labelframe titles
        style.configure('TLabelframe.Label', font=('Helvetica', 10, 'bold'))

        # Initialize variables
        self.mcts_params = {'iterations': DEFAULT_MCTS_ITERATIONS, 'C_param': DEFAULT_PUCT_C, 'dirichlet_noise': 0.3}
        self.nn_params = {
            'learning_rate': 1e-2,  # Changed for SGD
            'batch_size': 128,
            'epochs': 10,
            'policy_weight': 1.5,
            'value_weight': 1.0,
            'lr_decay': 0.9995,
            'use_resnet': False,
            'use_swa': False,
            'momentum': 0.9,        # Added for SGD
            'weight_decay': 1e-4    # Added for SGD
        }
        self.train_games = DEFAULT_TRAIN_GAMES
        
        self.games_before_training = min(100, self.train_games)
        
        self.max_cc_games = 100
        self.cc_train_interval = 50
        self.cc_delay = 500
        self.continuous_play = False
        self.training_in_progress = False
        self.training_stop_requested = False
        
        # Rename the variable to avoid collision with tkinter internals
        self.nn_model_path = NN_MODEL_FILE  # Changed from self.training_model_path
        
        self.play_till_end = False
        self.games_since_training = 0
        self.train_blink_job = None
        self.status_blink_job = None
        self.show_win_probability = True

        self.last_p1 = "Human"
        self.last_p2 = "Computer (AI)"
        self.last_red_model = NN_MODEL_FILE
        self.last_green_model = NN_MODEL_FILE
        self.last_continuous_play = False
        self.last_fullplay = False
        
        self.fullplay_var = tk.BooleanVar(self, False)
        
        # Create the NNManager with ResNet option from settings
        self.nn = NNManager(self.nn_params, self.nn_model_path)  # Changed to use nn_model_path
        self._load_cfg()

        self.game = Connect4Game()
        self.players = {RED_PIECE: None, GREEN_PIECE: None}
        self.score = {'red':0,'green':0,'draws':0,'games':0}
        self.turn_count=0; self.auto_job=None; self.game_in_progress=False
        self.is_comp=False; self.paused=False; self.last_hover=None
        
        self.players_selected = False
                
        self.log_buffer = []
        self.log_buffer_lock = threading.Lock()
        self.log_last_update = time.time()
        self.log_update_interval = 0.5
        self.full_log_content = []
        
        # Persistent training log content
        self.training_log_content = []
        
        self.nn_training_phase = False
        self.performing_training = False

        main = ttk.Frame(self, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        
        main.grid_columnconfigure(1, weight=1)
        main.grid_rowconfigure(0, weight=1)
                        
        self.canvas = tk.Canvas(main, width=WIDTH, height=HEIGHT + 60, bg=LIGHT_BLUE, highlightthickness=0)
        self.canvas.grid(row=0, column=0, rowspan=3, padx=(0,10), sticky="nsew")
        self.canvas.bind("<Button-1>", self._click)
        self.canvas.bind("<Motion>", self._hover)
        ToolTip(self.canvas, "Click a column to drop a piece")
        
        side = ttk.Frame(main)
        side.grid(row=0, column=1, rowspan=3, sticky="nsew")
        side.grid_rowconfigure(6, weight=1)
        side.grid_columnconfigure(0, weight=1)

        # Title frame with fixed layout to prevent overlapping
        title_frame = ttk.Frame(side)
        title_frame.grid(row=0, column=0, columnspan=4, sticky="ew", pady=(0,5))
        title_frame.grid_columnconfigure(0, weight=1)
        title_frame.grid_columnconfigure(1, weight=0)  # Don't expand win prob
        title_frame.grid_columnconfigure(2, weight=0)  # Don't expand settings
        title_frame.grid_propagate(False)  # Prevent propagation of size changes
        title_frame.config(height=60, width=400)  # Set both height and width
        
        # Increased width significantly for status label and prevent wrapping
        self.status = ttk.Label(title_frame, style='BoldStatus.TLabel', text="Results:")
        self.status.grid(row=0, column=0, sticky="w", padx=(0,10))  # Added padding
        self.status.config(wraplength=800)  # Very large wraplength to prevent any wrapping
        ToolTip(self.status, "Game status")
        
        # Win probability display - moved to its own column with proper spacing
        self.win_prob_var = tk.StringVar(value="Win: ---%")
        self.win_prob_label = ttk.Label(title_frame, textvariable=self.win_prob_var, 
                                      font=("Helvetica", 12))
        self.win_prob_label.grid(row=0, column=1, padx=(10,20), sticky="e")  # Added more padding
        ToolTip(self.win_prob_label, "Estimated win probability for current player")
        
        # Move settings frame to its own column
        settings_frame = ttk.Frame(title_frame)
        settings_frame.grid(row=0, column=2, padx=(0,5), sticky="e")
        
        settings_btn = ttk.Button(settings_frame, text="⚙", width=3, command=self._settings)
        settings_btn.pack(side="left", pady=(0, 0), padx=(0,5))
        ToolTip(settings_btn, "Settings (MCTS, Training, Neural Network)")
        
        vis_btn = ttk.Button(settings_frame, text="👁", width=3, command=self._toggle_visualization)
        vis_btn.pack(side="left")
        ToolTip(vis_btn, "Toggle policy/value visualization")
        
        # Score frame
        score_frame = ttk.Frame(side)
        score_frame.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(0,10))
        score_frame.grid_columnconfigure(0, weight=1)
        
        self.score_lbl = ttk.Label(score_frame, font=("Helvetica",12))
        self.score_lbl.grid(row=0, column=0, sticky="w")
        ToolTip(self.score_lbl, "Score summary")
        
        # Use bold style for Reset Score button
        reset_score_btn = ttk.Button(score_frame, text="Reset Score", style='Bold.TButton', command=self._reset_score)
        reset_score_btn.grid(row=0, column=1, sticky="e")
        ToolTip(reset_score_btn, "Reset all scores to zero")
        
        control_options = ttk.Frame(side)
        control_options.grid(row=2, column=0, columnspan=4, sticky="ew", pady=5)
        control_options.grid_propagate(False)  # Prevent propagation
        control_options.config(height=40)  # Fixed height
        
        self.learn_var = tk.BooleanVar(self, True)
        self.learn_check = ttk.Checkbutton(control_options, text="Learn @ Play", variable=self.learn_var)
        self.learn_check.grid(row=0, column=0, sticky="w", padx=5)
        self.learn_check.state(['disabled'])
        ToolTip(self.learn_check, "When enabled, game moves are used to train the neural network")

        train_frame = ttk.Frame(control_options)
        train_frame.grid(row=0, column=1, padx=5, sticky="e")
        
        self.train_btn = ttk.Button(train_frame, text="Train NN", style='Bold.TButton', command=self._train)
        self.train_btn.grid(row=0, column=0)
        ToolTip(self.train_btn, "Run AI self-play games to train the neural network")
        
        ttk.Label(train_frame, text="Model:").grid(row=0, column=1, padx=(5,2))
        
        # Use nn_model_path instead of training_model_path
        model_name = os.path.splitext(os.path.basename(self.nn_model_path))[0]
        self.train_model_label = ttk.Label(train_frame, text=model_name, 
                                           font=("Helvetica", 9, "bold"), foreground="#0000FF")
        self.train_model_label.grid(row=0, column=2, padx=(0,5))
        
        model_frame = ttk.Frame(train_frame)
        model_frame.grid(row=0, column=3)
        
        self.train_browse = ttk.Button(model_frame, text="📂", width=2, command=self._browse_training_model)
        self.train_browse.grid(row=0, column=0, padx=(2,2))
        ToolTip(self.train_browse, "Select model file to train (default: C4)")
        
        self.manage_models_btn = ttk.Button(model_frame, text="🔄", width=2, command=self._manage_models)
        self.manage_models_btn.grid(row=0, column=1, padx=(0,2))
        ToolTip(self.manage_models_btn, "Manage model versions")

        hist = ttk.Frame(side)
        hist.grid(row=3, column=0, columnspan=4, sticky="nsew")
        hist.grid_rowconfigure(1, weight=1)
        hist.grid_columnconfigure(0, weight=1)
        hist.grid_columnconfigure(1, weight=0)  # Scrollbar
        hist.grid_columnconfigure(2, weight=5)  # Visualization pane (wider than moves)
        hist.grid_propagate(False)  # Prevent propagation
        hist.config(height=300)  # Set fixed height
        
        # Use bold font for labels
        ttk.Label(hist, text="Game History:", style='Bold.TLabel').grid(row=0, column=0, sticky="w")
        self.moves = tk.Text(hist, width=20, height=15, font=("Courier",10), state="disabled")
        scr = ttk.Scrollbar(hist, command=self.moves.yview)
        self.moves['yscrollcommand']=scr.set
        self.moves.grid(row=1, column=0, sticky="nsew")
        scr.grid(row=1, column=1, sticky="ns")
        ToolTip(self.moves, "Game move history")
        
        # Add visualization frame directly in the main window - MODIFIED FOR BETTER SIZING
        vis_frame = ttk.Frame(hist)
        vis_frame.grid(row=0, column=2, rowspan=2, sticky="nsew", padx=(10,0))
        vis_frame.grid_rowconfigure(0, weight=0)  # Header doesn't expand
        vis_frame.grid_rowconfigure(1, weight=1)  # Visualization expands
        vis_frame.grid_columnconfigure(0, weight=1)
        
        # Add header with controls for visualization
        vis_header = ttk.Frame(vis_frame)
        vis_header.grid(row=0, column=0, sticky="ew")
        vis_header.grid_columnconfigure(0, weight=1)  # Label expands
        vis_header.grid_columnconfigure(1, weight=0)  # Buttons don't expand
        
        # Use bold font for label
        ttk.Label(vis_header, text="Policy & Value Visualization:", style='Bold.TLabel').grid(row=0, column=0, sticky="w", padx=5)
        
        # Create a frame for the controls to keep them together
        controls_frame = ttk.Frame(vis_header)
        controls_frame.grid(row=0, column=1, sticky="e")
        
        # Switch button instead of update
        self.switch_player_btn = ttk.Button(controls_frame, text="Switch to Green", style='Bold.TButton', command=self._switch_visualization_player)
        self.switch_player_btn.pack(side=tk.LEFT, padx=5)
        ToolTip(self.switch_player_btn, "Switch between Red and Green player visualization")
        
        # Auto-update checkbox
        self.auto_update_var = tk.BooleanVar(value=True)
        auto_update_check = ttk.Checkbutton(controls_frame, text="Auto-update", variable=self.auto_update_var)
        auto_update_check.pack(side=tk.LEFT, padx=5)
        ToolTip(auto_update_check, "Automatically update visualization after each move")
        
        # Create visualization container frame to allow proper expansion
        vis_container = ttk.Frame(vis_frame)
        vis_container.grid(row=1, column=0, sticky="nsew")
        vis_container.grid_rowconfigure(0, weight=1)
        vis_container.grid_columnconfigure(0, weight=1)
        
        # Create matplotlib figure and canvas with adjusted dimensions
        self.policy_fig = plt.Figure(figsize=(8, 7), dpi=100)  # Increased height for better proportions
        self.policy_canvas = FigureCanvasTkAgg(self.policy_fig, master=vis_container)
        self.policy_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew", padx=2, pady=2)
        
        # Set visualization to be visible by default
        self.visualization_frame = vis_frame
        self.visualization_visible = True  # Make visible by default

        # Fixed height control frame to prevent duplication
        control_frame = ttk.Frame(side)
        control_frame.grid(row=4, column=0, columnspan=4, pady=5)
        control_frame.grid_propagate(False)  # Prevent propagation
        control_frame.config(height=60)  # Increased height for larger buttons
        
        # Use bold text for buttons and make them larger
        button_style = {'width': 12, 'padding': (5, 7)}  # Wider and taller
        
        new_game_btn = ttk.Button(control_frame, text="New Game", style='Bold.TButton', command=self._new_game, **button_style)
        new_game_btn.pack(side="left", padx=7)
        ToolTip(new_game_btn, "Start a new game with current players")
        
        select_players_btn = ttk.Button(control_frame, text="Select Players", style='Bold.TButton', command=self._choose_players, **button_style)
        select_players_btn.pack(side="left", padx=7)
        ToolTip(select_players_btn, "Choose new players")
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", style='Bold.TButton', state="disabled", command=self._pause, **button_style)
        self.stop_btn.pack(side="left", padx=7)
        ToolTip(self.stop_btn, "Stop the current game, training, or continuous play")
        
        exit_btn = ttk.Button(control_frame, text="Exit", style='Bold.TButton', command=self.destroy, **button_style)
        exit_btn.pack(side="right", padx=7)
        ToolTip(exit_btn, "Exit the application")

        self.separator = ttk.Separator(side, orient='horizontal')
        self.separator.grid(row=5, column=0, columnspan=4, sticky="ew", pady=10)
        self.separator.grid_remove()
        
        # Fixed height training frame to prevent jumping - using bold font
        self.train_frame = ttk.LabelFrame(side, text="Training Progress")  # Style will be applied globally
        self.train_frame.grid(row=6, column=0, columnspan=4, sticky="nsew", pady=5)
        self.train_frame.grid_columnconfigure(0, weight=1)
        self.train_frame.grid_rowconfigure(3, weight=1)
        self.train_frame.grid_propagate(False)  # Prevent propagation of size changes
        self.train_frame.config(height=300)  # Set fixed height
        self.train_frame.grid_remove()
        
        self.train_progress = ttk.Progressbar(self.train_frame, length=250, mode="determinate")
        self.train_progress.grid(row=0, column=0, padx=5, pady=(5,0), sticky="ew")
        
        self.train_status = ttk.Label(self.train_frame, text="Idle", style='Bold.TLabel', anchor="center")
        self.train_status.grid(row=1, column=0, padx=5, pady=(0,5), sticky="ew")
        
        self.model_info = ttk.Label(self.train_frame, text="", anchor="center", font=("Helvetica", 8))
        self.model_info.grid(row=2, column=0, padx=5, pady=(0,5), sticky="ew")
        
        self.train_log_frame = ttk.LabelFrame(self.train_frame, text="Training Log")  # Style will be applied globally
        self.train_log_frame.grid(row=3, column=0, padx=5, pady=(0,5), sticky="nsew")
        self.train_log_frame.grid_columnconfigure(0, weight=1)
        self.train_log_frame.grid_rowconfigure(0, weight=1)
        self.train_log_frame.grid_propagate(False)  # Prevent propagation
        
        self.train_log = tk.Text(self.train_log_frame, width=28, height=10, font=("Courier", 9), 
                                 wrap="word", state="disabled")
        self.train_log_scr = ttk.Scrollbar(self.train_log_frame, command=self.train_log.yview)
        self.train_log['yscrollcommand'] = self.train_log_scr.set
        self.train_log.grid(row=0, column=0, sticky="nsew", padx=(2,0))
        self.train_log_scr.grid(row=0, column=1, sticky="ns")
        
        self.training_losses = {"policy": [], "value": []}
        
        self.after(100, self._update_log_from_buffer)
        
        self._update_model_info()

        self._draw()
        self._update_score()
        self._set_status("Ready to play...")
        
        # Create default players to be shown on launch (Human vs Computer AI)
        # This will make players visible under the board from the start
        self._choose_players()
        
        # Ensure window stays maximized
        self.update_idletasks()
        
        # Add protocol handler for application exit to save log
        self.protocol("WM_DELETE_WINDOW", self._save_log_on_exit)

    def _maintain_window_size(self, target_width, target_height):
        """Maintain window size during training to prevent unwanted resizing"""
        def check_and_fix_size():
            if self.winfo_exists():
                current_width = self.winfo_width()
                current_height = self.winfo_height()
                
                # Allow some tolerance for minor variations
                width_diff = abs(current_width - target_width)
                height_diff = abs(current_height - target_height)
                
                if width_diff > 20 or height_diff > 20:
                    self.geometry(f"{target_width}x{target_height}")
                
                # Continue checking during training
                if hasattr(self, 'training_in_progress') and self.training_in_progress:
                    self.after(1000, check_and_fix_size)
        
        # Start the periodic size checking
        self.after(100, check_and_fix_size)

    def _blink_training_text(self):
        if not hasattr(self, 'train_blink_job'):
            self.train_blink_job = None
        
        if self.train_blink_job:
            self.after_cancel(self.train_blink_job)
        
        self._blink_train_status()

    def _blink_train_status(self):
        if not hasattr(self, 'train_blink_job'):
            self.train_blink_job = None
            
        current_color = self.train_status.cget("foreground")
        
        if current_color == "red":
            self.train_status.config(foreground="darkred")
        else:
            self.train_status.config(foreground="red")
        
        # Continue blinking if training is still in progress
        if (hasattr(self, 'performing_training') and self.performing_training) or \
           (hasattr(self, 'nn_training_phase') and self.nn_training_phase):
            if not hasattr(self, 'train_blink_job') or self.train_blink_job is None:
                self.train_blink_job = self.after(300, self._blink_train_status)
            else:
                # Cancel previous job and create new one
                self.after_cancel(self.train_blink_job)
                self.train_blink_job = self.after(300, self._blink_train_status)

    def _toggle_visualization(self):
        try:
            if self.visualization_frame.winfo_ismapped():
                # Hide the visualization frame
                self.visualization_frame.grid_remove()
                self.visualization_visible = False
            else:
                # Show the visualization frame
                self.visualization_frame.grid()
                self.visualization_visible = True
                self._update_visualization()
        except Exception as e:
            print(f"Error toggling visualization: {e}")

    def _load_cfg(self):
        if os.path.exists(MCTS_CONFIG_FILE):
            try:
                cfg = json.load(open(MCTS_CONFIG_FILE))
                if 'iterations' in cfg and 'C_param' in cfg:
                    self.mcts_params = cfg
                elif 'mcts' in cfg:
                    self.mcts_params = cfg['mcts']
                    if 'training' in cfg:
                        if 'games' in cfg['training']:
                            self.train_games = cfg['training']['games']
                        if 'games_before_training' in cfg['training']:
                            self.games_before_training = cfg['training']['games_before_training']
                        else:
                            self.games_before_training = min(100, self.train_games)
                        if 'max_cc_games' in cfg['training']:
                            self.max_cc_games = cfg['training']['max_cc_games']
                        if 'cc_train_interval' in cfg['training']:
                            self.cc_train_interval = cfg['training']['cc_train_interval']
                        if 'cc_delay' in cfg['training']:
                            self.cc_delay = cfg['training']['cc_delay']
                    if 'nn_params' in cfg:
                        self.nn_params = cfg['nn_params']
                        
                        # Check if there are new parameters that weren't in the config
                        if 'use_resnet' not in self.nn_params:
                            self.nn_params['use_resnet'] = False
                        if 'use_swa' not in self.nn_params:
                            self.nn_params['use_swa'] = False
                            
                        # Recreate NN manager with updated parameters
                        self.nn = NNManager(self.nn_params, self.nn_model_path, 
                                          use_resnet=self.nn_params.get('use_resnet', False))
                    
                    # Load column bias if present
                    if 'column_bias' in cfg:
                        global CENTER_BIAS
                        CENTER_BIAS = np.array(cfg['column_bias'])
                    
                    # Ensure dirichlet_noise is set with a default if not present
                    if 'dirichlet_noise' not in self.mcts_params:
                        self.mcts_params['dirichlet_noise'] = 0.3
                        
                    if self.games_before_training > self.train_games:
                        self.games_before_training = self.train_games
            except:
                pass

    def _reset_score(self):
        self.score = {'red':0, 'green':0, 'draws':0, 'games':0}
        self._update_score()
        
        if str(self.train_frame.grid_info()) != "{}":
            self.log_to_training("Score reset to zero.")
            
    def _switch_visualization_player(self):
        if self.current_vis_player == RED_PIECE:
            self.current_vis_player = GREEN_PIECE
            self.switch_player_btn.config(text="Switch to Red")
        else:
            self.current_vis_player = RED_PIECE
            self.switch_player_btn.config(text="Switch to Green")
        
        # Create a temporary copy of the game state to show the other player's perspective
        temp_game = self.game.copy()
        if temp_game.current_player != self.current_vis_player:
            temp_game.current_player = self.current_vis_player
        
        # Store the original game
        original_game = self.game
        
        # Temporarily replace the game with our modified copy
        self.game = temp_game
        
        # Update the visualization
        self._update_visualization()
        
        # Restore the original game
        self.game = original_game

    # ──────────────────────────────────────────────────────────────
    # Helper - convert raw NN value → absolute win probabilities
    # ──────────────────────────────────────────────────────────────
    def _compute_win_probs(self, value):
        """
        Convert raw NN value (-1…+1 for side-to-move) into
        (red_win_prob, green_win_prob, value) tuple.
        """
        current_player_win = (value + 1.0) / 2.0
        if self.game.current_player == RED_PIECE:
            red_win = current_player_win
            green_win = 1.0 - red_win
        else:
            green_win = current_player_win
            red_win = 1.0 - green_win
        return red_win, green_win, value


    def _update_switch_button_text(self):
        if self.current_vis_player == RED_PIECE:
            self.switch_player_btn.config(text="Switch to Green")
        else:
            self.switch_player_btn.config(text="Switch to Red")
                
    def _show_visualization(self):
        try:
            if not self.game_in_progress:
                messagebox.showinfo("Visualization", "Start a game to see policy and value visualization.")
                return
            
            # Create new visualization window
            self.visualization_window = tk.Toplevel()
            self.visualization_window.title("Policy & Value Visualization")
            self.visualization_window.geometry("720x400")  # Slightly taller
            self.visualization_window.protocol("WM_DELETE_WINDOW", self._toggle_visualization)
            
            # Create figure and canvas
            from matplotlib.figure import Figure
            self.policy_fig = Figure(figsize=(7, 4), dpi=100)
            self.policy_canvas = FigureCanvasTkAgg(self.policy_fig, master=self.visualization_window)
            self.policy_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)
            
            # Add a toolbar
            toolbar_frame = ttk.Frame(self.visualization_window)
            toolbar_frame.pack(fill=tk.X)
            
            # Add buttons
            update_btn = ttk.Button(toolbar_frame, text="Update Visualization", 
                                  command=self._update_visualization)
            update_btn.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Auto-update checkbox
            self.auto_update_var = tk.BooleanVar(value=True)
            auto_update_check = ttk.Checkbutton(toolbar_frame, text="Auto-update on move", 
                                             variable=self.auto_update_var)
            auto_update_check.pack(side=tk.LEFT, padx=5, pady=5)
            
            # Explanation text
            ttk.Label(toolbar_frame, text="Shows neural network's policy (move probabilities) and value prediction",
                   font=("Helvetica", 8)).pack(side=tk.RIGHT, padx=5, pady=5)
            
            # Initialize win probabilities if needed
            if not hasattr(self, 'current_red_win_prob'):
                self.current_red_win_prob = 0.5
                self.current_green_win_prob = 0.5
                self.current_draw_prob = 0.0
                
            # Show placeholder initially
            ax = self.policy_fig.add_subplot(111)
            ax.text(0.5, 0.5, "Initializing visualization...", 
                 ha='center', va='center', fontsize=12)
            ax.axis('off')
            self.policy_canvas.draw()
            
            # Schedule the real update after a short delay
            self.after(300, self._update_visualization)
        except Exception as e:
            print(f"Error showing visualization: {e}")
            import traceback
            traceback.print_exc()
                                                
    # ──────────────────────────────────────────────────────────────
    # Policy & value visualisation pane
    # ──────────────────────────────────────────────────────────────
    def _update_visualization(self, force_game_over=False):
        try:
            if not getattr(self, 'visualization_frame', None) \
               or not self.visualization_frame.winfo_ismapped():
                return
            if not getattr(self, 'policy_fig', None):
                return

            self.policy_fig.clear()
            ax = self.policy_fig.add_subplot(111)

            # ── terminal / forced draw rendering ─────────────────
            if self.game.game_over or force_game_over:
                if self.game.winner == 'Draw':
                    red_win_prob = green_win_prob = 0.5
                    value = 0.0
                    ax.set_title("Game Ended in Draw", fontsize=14,
                                 color='black', pad=10)
                elif self.game.winner == RED_PIECE:
                    red_win_prob, green_win_prob, value = 1.0, 0.0, +1.0
                    ax.set_title("Red Wins", fontsize=14, color='red', pad=10)
                else:
                    red_win_prob, green_win_prob, value = 0.0, 1.0, +1.0
                    ax.set_title("Green Wins", fontsize=14, color='green', pad=10)
                probs = np.zeros(COLUMN_COUNT, dtype=np.float32)

            # ── live position: single NN evaluation ───────────────
            else:
                player     = self.players[self.game.current_player]
                nn_manager = player.nn if isinstance(player, MCTSComputerPlayer) else self.nn
                probs, value = nn_manager.policy_value(self.game)
                red_win_prob, green_win_prob, _ = self._compute_win_probs(value)

            # ── draw policy bars ──────────────────────────────────
            bar_positions = np.arange(1, COLUMN_COUNT + 1)
            bar_color     = 'red' if self.game.current_player == RED_PIECE else 'green'
            bars = ax.bar(bar_positions, probs, color=bar_color, width=0.6)

            for bar, prob in zip(bars, probs):
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width()/2.0,
                            h + 0.01,
                            f"{prob:.2f}",
                            ha='center', va='bottom', fontsize=9)

            max_h = max(0.2, (probs.max() if probs.any() else 0.0) * 1.2)
            ax.set_ylim(0.0, max_h)
            ax.set_xlim(0.25, COLUMN_COUNT + 0.75)
            ax.set_xticks(bar_positions)
            ax.set_xticklabels([str(i) for i in range(1, COLUMN_COUNT+1)])
            ax.set_ylabel('Prob', fontsize=10, labelpad=0)
            ax.grid(True, linestyle='--', alpha=0.5, axis='y')

            # ── overlay raw value + win percentages ───────────────
            y_txt = -0.22 * max_h
            ax.text(4.0,  y_txt, f"Value: {value:+.3f}",
                    ha='center', va='center', fontsize=10)
            ax.text(1.5, y_txt, f"Red: {red_win_prob:.0%}",
                    ha='center', va='center', fontsize=10, color='red')
            ax.text(6.5, y_txt, f"Green: {green_win_prob:.0%}",
                    ha='center', va='center', fontsize=10, color='green')

            # ── store & render ─────────────────────────────────────
            self.current_red_win_prob   = red_win_prob
            self.current_green_win_prob = green_win_prob
            self.current_draw_prob      = 0.0

            self.policy_fig.tight_layout()
            self.policy_fig.subplots_adjust(bottom=0.25)
            self.policy_canvas.draw()

        except Exception as e:
            print(f"Error updating visualization: {e}")
  
            
    def _browse_training_model(self):
        filename = filedialog.askopenfilename(
            title="Select Neural Network Model to Train",
            filetypes=[("PyTorch Models", "*.pt"), ("All files", "*.*")],
            initialdir=os.path.dirname(os.path.abspath(self.nn_model_path))
        )
        if filename:
            self.nn_model_path = filename
            
            # Check if the model uses ResNet
            try:
                model_data = torch.load(filename, map_location=torch.device('cpu'), weights_only=True)
                use_resnet = model_data.get('hyperparams', {}).get('use_resnet', False)
                self.nn_params['use_resnet'] = use_resnet
            except:
                pass
            
            self.nn = NNManager(self.nn_params, self.nn_model_path)
            
            self._update_model_info()

    def destroy(self):
        try:
            # Set shutdown flag immediately
            self.shutting_down = True
            
            # Stop all ongoing operations
            if hasattr(self, 'training_in_progress') and self.training_in_progress:
                self.training_stop_requested = True
                self.force_stop_training = True
            
            # Stop any AI games in progress
            if hasattr(self, 'is_comp') and self.is_comp and hasattr(self, 'game_in_progress') and self.game_in_progress:
                self.paused = True
                self.play_till_end = True
            
            # Cancel any scheduled tasks
            if hasattr(self, 'auto_job') and self.auto_job:
                self.after_cancel(self.auto_job)
            if hasattr(self, 'train_blink_job') and self.train_blink_job:
                self.after_cancel(self.train_blink_job)
            if hasattr(self, 'status_blink_job') and self.status_blink_job:
                self.after_cancel(self.status_blink_job)
            
            # Force shutdown ProcessPoolExecutor and kill child processes
            if hasattr(self, 'training_executor') and self.training_executor:
                self.training_executor.shutdown(wait=False)
                self.training_executor = None
            
            # Kill all child processes more aggressively
            import psutil
            import os
            current_process = psutil.Process(os.getpid())
            children = current_process.children(recursive=True)
            for child in children:
                try:
                    child.terminate()
                except psutil.NoSuchProcess:
                    pass
            
            # Wait a moment for processes to terminate
            time.sleep(0.2)
            
            # Kill any remaining children forcefully
            for child in children:
                try:
                    if child.is_running():
                        child.kill()
                except psutil.NoSuchProcess:
                    pass
            
            # Save log without calling super().destroy() yet
            if not getattr(self, "_log_saved", False):
                self._log_saved = True
                # Flush any messages still in the buffer
                self._force_process_log_buffer()
                
                # Get log content
                complete_log_content = "".join(getattr(self, "full_log_content", []))
                if not complete_log_content and hasattr(self, "train_log") and self.train_log.winfo_exists():
                    self.train_log.config(state="normal")
                    complete_log_content = self.train_log.get("1.0", "end-1c")
                    self.train_log.config(state="disabled")
                
                # Write to disk
                if complete_log_content:
                    with open("c4.log", "w", encoding="utf-8") as f:
                        f.write(complete_log_content)
            
            # Now destroy the window
            super().destroy()
            
        except Exception as e:
            print(f"Error during destroy: {e}")
        finally:
            # Force exit the entire process
            import sys
            sys.exit(0)

    def _save_log_on_exit(self):
        try:
            # Prevent double-saving if destroy() calls us again
            if getattr(self, "_log_saved", False):
                super().destroy()
                return
            self._log_saved = True

            # Flush any messages still in the buffer
            self._force_process_log_buffer()

            # 1️⃣  Prefer the persistent in-memory copy
            complete_log_content = "".join(getattr(self, "full_log_content", []))

            # 2️⃣  Fallback – pull everything currently displayed in the GUI
            if not complete_log_content and hasattr(self, "train_log") and self.train_log.winfo_exists():
                self.train_log.config(state="normal")
                complete_log_content = self.train_log.get("1.0", "end-1c")
                self.train_log.config(state="disabled")

            # Write to disk
            if complete_log_content:
                with open("c4.log", "w", encoding="utf-8") as f:
                    f.write(complete_log_content)
                print(f"[INFO] Training log saved to c4.log ({len(complete_log_content)} bytes)")
            else:
                print("[WARN] No training log content to save")

            # Final shut-down
            super().destroy()

        except Exception as e:
            print(f"[FATAL] Error while saving log on exit: {e}")
            # Ensure that the application still quits
            super().destroy()

    def _manage_models(self):
        model_dialog = ModelManagementDialog(self)
        # Dialog will handle updates to the model path and manager
        # No need for additional code here
        
    def _update_model_info(self):
        model_name = os.path.splitext(os.path.basename(self.nn_model_path))[0]
        
        total_games = self.nn.total_games
        version = self.nn.version if hasattr(self.nn, 'version') else "unknown"
        architecture = "ResNet" if self.nn_params.get('use_resnet', False) else "CNN"
        
        self.model_info.config(text=f"Active model: {model_name} v{version} ({architecture}, Trained on {total_games} games)")
        
        self.train_model_label.config(text=model_name, foreground="#0000FF", font=("Helvetica", 9, "bold"))
        
        if str(self.train_frame.grid_info()) != "{}":
            self.model_info.grid()
            
            # Add model info to log if it's the first time showing
            if not self.full_log_content:
                self.log_to_training(f"Loaded model: {model_name} v{version}")
                self.log_to_training(f"Architecture: {architecture}")
                self.log_to_training(f"Trained on: {total_games} games")
                self.log_to_training(f"MCTS iterations: {self.mcts_params['iterations']}")
                self.log_to_training("-" * 50 + "\n")

    def _new_game(self):
        if self.auto_job: 
            self.after_cancel(self.auto_job)
            self.auto_job = None
                
        if not self.players_selected:
            self._choose_players()
            return
        
        self.play_till_end = False
        
        if not self.continuous_play or self.score['games'] == 0:
            self.games_since_training = 0
        
        # Reset AI players for new game
        for player_id, player in self.players.items():
            if isinstance(player, MCTSComputerPlayer):
                player.reset()
                
                # Refresh neural network if needed after training
                if hasattr(self, 'nn') and player.model_path is not None:
                    # Load same model but with potentially updated weights
                    player.nn = NNManager(self.nn_params, player.model_path, 
                                         use_resnet=self.nn_params.get('use_resnet', False))
                    player.mcts = MCTS(player.mcts.I, player.mcts.c, player.nn, player.mcts.explore,
                                     dirichlet_noise=self.mcts_params.get('dirichlet_noise', 0.4),
                                     safe_bypass_prob=player.mcts.safe_bypass_prob)
                    player.mcts.root = None  # Ensure root is properly reset
                    
        # Reset game state
        self.game.reset()
        self.turn_count = 0
        self.last_hover = None
        self.moves.config(state="normal")
        self.moves.delete("1.0", "end")
        self.moves.config(state="disabled")
        self.game_in_progress = True
        self.paused = False
        
        # Initialize win probabilities - start with equal chances
        self.current_red_win_prob = 0.5
        self.current_green_win_prob = 0.5
        self.current_draw_prob = 0.0
        self.win_prob_var.set(f"Win Probs: Red: {self.current_red_win_prob:.0%}/Green: {self.current_green_win_prob:.0%}")
        self.win_prob_label.config(foreground="black")
        
        # Initialize UI
        self._draw()
        
        # Update win probability based on neural network if there's an AI player
        has_ai = any(isinstance(p, MCTSComputerPlayer) for p in self.players.values())
        if has_ai:
            self.after(50, self._update_win_probability)
        
        if self.is_comp:
            self.stop_btn['state'] = "normal"
        
        if self.is_comp and str(self.train_frame.grid_info()) != "{}":
            game_number = self.score['games'] + 1
            self.log_to_training(f"\n{'-'*20}\nGame #{game_number}\n{'-'*20}\n")
        
        if isinstance(self.players[self.game.current_player], HumanPlayer):
            self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn...", COLOR_MAP[self.game.current_player])
        else:
            self._set_status(f"{PLAYER_MAP[self.game.current_player]} is thinking...", COLOR_MAP[self.game.current_player])
        
        # Ensure visualization is visible and updated
        if hasattr(self, 'visualization_frame'):
            self.visualization_frame.grid()  # Make sure it's visible
            self.visualization_visible = True
            # Update the visualization after initial setup
            if has_ai:
                self.after(100, self._update_visualization)
        
        # Start game
        self.after(30, self._next_turn)
        
    def _choose_players(self):
            # Store current window and widget dimensions to try and restore them later if needed
            current_width = self.winfo_width()
            current_height = self.winfo_height()
            hist_width = 0
            if hasattr(self, 'moves') and self.moves.winfo_exists():
                # Estimate current width of moves widget in characters
                try:
                    char_pixel_width = self.moves.winfo_pixels("1m") if self.moves.winfo_pixels("1m") > 0 else 8 # typical char width
                    hist_width = self.moves.winfo_width() // char_pixel_width
                except tk.TclError: # Widget might not be fully initialized or visible
                    hist_width = 20 # Default

            dlg = PlayerDialog(self, self.mcts_params, self.nn) # self (Connect4GUI) is passed as master
            if not dlg.result:
                self.lift()
                self.focus_force()
                return

            # PlayerDialog.apply() has already set self.last_p1, self.last_p2,
            # self.last_red_model, self.last_green_model, etc. on self (Connect4GUI instance)
            # dlg.result contains the actual player objects and other dialog choices.
            dlg_result_data = dlg.result

            red_player_type_str = self.last_p1 # Correctly get from Connect4GUI's attribute
            green_player_type_str = self.last_p2 # Correctly get from Connect4GUI's attribute

            red_player = dlg_result_data['red']
            green_player = dlg_result_data['green']

            # Handle Red player neural network
            if isinstance(red_player, MCTSComputerPlayer):
                # MCTSComputerPlayer's __init__ in PlayerDialog's apply method already handles NN setup
                # We just need to ensure its attributes are correct if using shared self.nn
                if not (hasattr(red_player, 'model_path') and red_player.model_path and red_player.model_path != self.nn_model_path):
                    red_player.nn = self.nn # Ensure it points to the main nn if not custom
                    if hasattr(red_player, 'mcts'): red_player.mcts.nn = self.nn

            # Handle Green player neural network
            if isinstance(green_player, MCTSComputerPlayer):
                if not (hasattr(green_player, 'model_path') and green_player.model_path and green_player.model_path != self.nn_model_path):
                    green_player.nn = self.nn # Ensure it points to the main nn if not custom
                    if hasattr(green_player, 'mcts'): green_player.mcts.nn = self.nn

            self.players[RED_PIECE] = red_player
            self.players[GREEN_PIECE] = green_player
            self.continuous_play = dlg_result_data['continuous_play']
            self.fullplay_var.set(dlg_result_data['full_strength'])

            self.is_comp = all(isinstance(p, (RandomComputerPlayer, MCTSComputerPlayer)) for p in self.players.values())
            self.players_selected = True

            has_ai = any(isinstance(p, MCTSComputerPlayer) for p in self.players.values())

            if self.is_comp:
                self.stop_btn['state'] = 'normal'
                self.learn_var.set(True)
                self.learn_check.state(['disabled'])
            else:
                self.stop_btn['state'] = 'disabled'
                if has_ai:
                    self.learn_check.state(['!disabled'])
                else:
                    self.learn_check.state(['disabled'])
                    self.learn_var.set(False)

            # --- Layout updates ---
            # 1. Remove UI elements that depend on AI presence
            self.separator.grid_remove()
            self.train_frame.grid_remove() # model_info is inside train_frame

            # 2. Crucial step: Force the parent 'side' frame to update its layout after removals
            if hasattr(self, 'side') and self.side.winfo_exists():
                 self.side.update_idletasks()

            # 3. Conditionally add back UI elements if AI is present
            if has_ai:
                self.separator.grid()
                self.train_frame.grid()
                self._update_model_info()

                log_text = "\n" + "=" * 50 + "\n"
                if self.is_comp:
                    log_text += "Computer (AI) vs Computer (AI) Game Setup\n"
                else:
                    log_text += f"{red_player_type_str} vs {green_player_type_str} Game Setup\n"
                log_text += "-" * 50 + "\n"

                red_model_desc = "Human/Random"
                if isinstance(red_player, MCTSComputerPlayer):
                    # Use the model path stored in the player object by PlayerDialog
                    red_model_path = getattr(red_player, 'model_path', self.nn_model_path)
                    red_model_name = os.path.splitext(os.path.basename(red_model_path))[0]
                    # Access version from the player's specific nn manager
                    red_version = getattr(red_player.nn, 'version', "N/A") if hasattr(red_player, 'nn') else "N/A"
                    red_model_desc = f"{red_model_name} v{red_version}"

                green_model_desc = "Human/Random"
                if isinstance(green_player, MCTSComputerPlayer):
                    green_model_path = getattr(green_player, 'model_path', self.nn_model_path)
                    green_model_name = os.path.splitext(os.path.basename(green_model_path))[0]
                    green_version = getattr(green_player.nn, 'version', "N/A") if hasattr(green_player, 'nn') else "N/A"
                    green_model_desc = f"{green_model_name} v{green_version}"

                log_text += f"Red Player: {red_player_type_str} (Model: {red_model_desc})\n"
                log_text += f"Green Player: {green_player_type_str} (Model: {green_model_desc})\n"
                if isinstance(red_player, MCTSComputerPlayer) or isinstance(green_player, MCTSComputerPlayer):
                     log_text += f"MCTS iterations: {self.mcts_params['iterations']}\n"
                     log_text += f"Exploration (Full Strength Off): {'Enabled' if not self.fullplay_var.get() else 'Disabled'}\n"

                if self.is_comp:
                    log_text += f"Continuous Play: {'Enabled' if self.continuous_play else 'Disabled'}\n"
                    if self.continuous_play: log_text += f"Max games for continuous play: {self.max_cc_games}\n"
                log_text += "-" * 50 + "\n\n"
                self.log_to_training(log_text)

                if self.is_comp and self.continuous_play:
                    self.train_progress['maximum'] = self.max_cc_games
                    self.train_progress['value'] = 0
                    self.train_status.config(text=f"Games: 0 / {self.max_cc_games}")
                self.training_losses = {"policy": [], "value": []}
            # No 'else' for separator/train_frame if not has_ai, as they are already removed.

            # 4. Ensure core elements like hist and P&V frame are correctly configured/visible.
            if hasattr(self, 'hist') and self.hist.winfo_exists():
                self.hist.grid()
                self.hist.grid_columnconfigure(2, weight=5)
                if hasattr(self, 'visualization_frame') and self.visualization_frame.winfo_exists():
                    if self.visualization_visible:
                        self.visualization_frame.grid()
                    else:
                        self.visualization_frame.grid_remove()

            # 5. Update main board display and status message
            self._draw()
            self._set_status("Ready to play...")

            # 6. Final global update to process all accumulated UI changes
            self.update_idletasks()

            # Attempt to restore window size if it changed drastically
            if current_width > 100 and current_height > 100: # Check if initial size was meaningful
                # Allow for minor variations from window manager decorations etc.
                if abs(self.winfo_width() - current_width) > 30 or abs(self.winfo_height() - current_height) > 30:
                    self.geometry(f"{current_width}x{current_height}")
            
            # Attempt to restore moves widget width
            if hist_width >= 5 and hasattr(self, 'moves') and self.moves.winfo_exists(): # Check if initial width was meaningful
                try:
                    current_char_width = self.moves.winfo_width() // (self.moves.winfo_pixels("1m") if self.moves.winfo_pixels("1m") > 0 else 8)
                    if abs(current_char_width - hist_width) > 2 :
                        self.moves.config(width=hist_width)
                except tk.TclError:
                    pass # Widget might not be ready

            self.lift()
            self.focus_force()
            self.update_idletasks()                            
    def log_to_training(self, *args, **kwargs):
        message = " ".join(str(arg) for arg in args)
        if "end" in kwargs:
            message += kwargs["end"]
        else:
            message += "\n"
        
        # Print to terminal first
        print(*args, **kwargs)
        
        # Store in persistent log content directly
        if not hasattr(self, 'full_log_content'):
            self.full_log_content = []
        self.full_log_content.append(message)
        
        # Then add to buffer for GUI updates
        with self.log_buffer_lock:
            self.log_buffer.append(message)
        
        # Force immediate processing for critical messages
        # Expanded the list of important message types to catch
        important_message = (
            "Training" in message or 
            "Epoch" in message or 
            "POLICY=" in message or 
            "VALUE=" in message or 
            "statistics" in message or 
            "distribution" in message or
            "Added examples" in message or
            "Processing" in message or
            "Batch" in message or
            "progress" in message or
            "games" in message
        )
        
        if important_message:
            self.after(10, self._force_process_log_buffer)
                                        
    def _update_log_from_buffer(self):
        try:
            current_time = time.time()
            
            # Process buffer if time elapsed or buffer has content
            if self.log_buffer or (current_time - self.log_last_update > self.log_update_interval):
                messages = []
                with self.log_buffer_lock:
                    if self.log_buffer:
                        # Take all messages from buffer
                        messages = self.log_buffer.copy()
                        self.log_buffer.clear()
                
                if messages and hasattr(self, 'train_log') and self.train_log.winfo_exists():
                    # Process all messages in batch
                    self.train_log.config(state="normal")
                    
                    # Limit batch size to avoid GUI freezing
                    batch_size = 100
                    
                    for i in range(0, len(messages), batch_size):
                        batch = messages[i:i+batch_size]
                        for message in batch:
                            self.train_log.insert("end", message)
                            
                            if "POLICY=" in message:
                                try:
                                    policy_loss = float(message.split("POLICY=")[1].split()[0])
                                    value_loss = float(message.split("VALUE=")[1].split()[0])
                                    
                                    self.training_losses["policy"].append(policy_loss)
                                    self.training_losses["value"].append(value_loss)
                                    
                                    self.update_loss_summary()
                                except Exception as e:
                                    print(f"Error parsing loss values: {e}")
                                    
                                last_line = self.train_log.index("end-1c linestart")
                                self.train_log.tag_add("highlight", last_line, f"{last_line} lineend")
                                self.train_log.tag_config("highlight", background="#E0F0FF")
                        
                        # Update display after each batch
                        self.train_log.see("end")
                        self.update_idletasks()
                    
                    self.train_log.config(state="disabled")
                    self.log_last_update = current_time
                
                # Process GUI events to ensure display updates
                self.update_idletasks()
            
            # Schedule next update with priority based on buffer state
            if self.log_buffer:
                # Process immediately if buffer still has content
                self.after(10, self._update_log_from_buffer)
            else:
                # Normal scheduling
                self.after(100, self._update_log_from_buffer)
        except Exception as e:
            print(f"Error in _update_log_from_buffer: {e}")
            # Ensure the loop continues even on error
            self.after(100, self._update_log_from_buffer)
    
    def update_loss_summary(self):
        # Only run once at least one datapoint is available
        if not self.training_losses["policy"]:
            return

        current_policy = self.training_losses["policy"][-1]
        current_value  = self.training_losses["value"][-1]

        if len(self.training_losses["policy"]) > 1:
            policy_change = current_policy - self.training_losses["policy"][0]
            value_change  = current_value  - self.training_losses["value"][0]

            summary = (
                f"Current Loss - Policy: {current_policy:.6f}, Value: {current_value:.6f}\n"
                f"Change        - Policy: {policy_change:+.6f}, Value: {value_change:+.6f}\n"
                f"{'Improving ✓' if policy_change < 0 else 'Not improving ✗'}\n"
                f"{'-'*50}\n"
            )
        else:
            summary = (
                f"Current Loss - Policy: {current_policy:.6f}, Value: {current_value:.6f}\n"
                f"{'-'*50}\n"
            )

        # --- GUI update ---
        self.train_log.config(state="normal")

        # Look for an *existing* “Current Loss” header to replace.
        summary_start = self.train_log.search(r"^Current Loss", "1.0", regexp=True)
        if summary_start:
            # Delete only the old summary block (up to the first dashed separator)
            summary_end = self.train_log.search(r"\n-{20,}\n", summary_start, regexp=True)
            if not summary_end:                                     # extremely rare fallback
                summary_end = f"{summary_start} lineend +1c"
            self.train_log.delete(summary_start, summary_end)

        # Prepend fresh summary (top of the text widget) and re-lock the widget
        self.train_log.insert("1.0", summary)
        self.train_log.config(state="disabled")
        
    def _mix(self, c1, c2, a):
        r1,g1,b1=[int(c1[i:i+2],16) for i in (1,3,5)]
        r2,g2,b2=[int(c2[i:i+2],16) for i in (1,3,5)]
        return f"#{int(r1*(1-a)+r2*a):02x}{int(g1*(1-a)+g2*a):02x}{int(b1*(1-a)+b2*a):02x}"

    def _draw(self):
        self.canvas.delete("all")
        
        # Draw LIGHT_BLUE backgrounds first (before any game elements)
        # Background for the area above the board
        self.canvas.create_rectangle(0, 0, WIDTH, SQUARESIZE, fill=LIGHT_BLUE, outline="")
        
        # Background for text area below the board
        text_bg_y = HEIGHT + 5
        text_bg_height = 80
        self.canvas.create_rectangle(0, text_bg_y, WIDTH, text_bg_y + text_bg_height, fill=LIGHT_BLUE, outline="")
        
        # Draw the game board with standard BLUE
        for c in range(COLUMN_COUNT):
            for r in range(ROW_COUNT):
                x1=c*SQUARESIZE; y1=HEIGHT-(r+1)*SQUARESIZE; x2=x1+SQUARESIZE; y2=y1+SQUARESIZE
                self.canvas.create_rectangle(x1,y1,x2,y2,fill=BLUE,outline=BLACK)
                cx=x1+SQUARESIZE/2; cy=y1+SQUARESIZE/2
                piece=self.game.board[r,c]
                fill=EMPTY_COLOR if piece==EMPTY else COLOR_MAP[piece]
                self.canvas.create_oval(cx-RADIUS,cy-RADIUS,cx+RADIUS,cy+RADIUS,fill=fill,outline=BLACK)
        
        # Draw hover piece for human players (AFTER the backgrounds)
        if self.last_hover is not None and self.game_in_progress and isinstance(self.players[self.game.current_player],HumanPlayer):
            col=self.last_hover; cx=col*SQUARESIZE+SQUARESIZE/2; cy=SQUARESIZE/2
            light=self._mix(COLOR_MAP[self.game.current_player],"#FFFFFF",0.4)
            self.canvas.create_oval(cx-RADIUS,cy-RADIUS,cx+RADIUS,cy+RADIUS,fill=light,outline=BLACK,dash=(3,3))
        
        # Position text within visible canvas area
        y_pos = HEIGHT + 45
        
        # Determine what text to display based on the current state
        if hasattr(self, 'training_in_progress') and self.training_in_progress:
            if hasattr(self, 'nn_training_phase') and self.nn_training_phase:
                self.canvas.create_text(WIDTH//2, y_pos, text="Training on self-played games...", 
                                       font=("Helvetica", 22, "bold"), fill="white")
            else:
                self.canvas.create_text(WIDTH//2, y_pos, text="Self-play headless mode...", 
                                       font=("Helvetica", 22, "bold"), fill="white")
        elif hasattr(self, 'performing_training') and self.performing_training:
            self.canvas.create_text(WIDTH//2, y_pos, text="Training on the last game...", 
                                   font=("Helvetica", 22, "bold"), fill="white")
        elif self.is_comp:
            self.canvas.create_text(WIDTH//2 - 100, y_pos, text="AI", 
                                   font=("Helvetica", 22, "bold"), fill=RED)
            
            self.canvas.create_text(WIDTH//2, y_pos, text="vs", 
                                   font=("Helvetica", 18, "bold"), fill="white")
            
            self.canvas.create_text(WIDTH//2 + 100, y_pos, text="AI", 
                                   font=("Helvetica", 22, "bold"), fill=GREEN)
        elif hasattr(self, 'players') and self.players[RED_PIECE] is not None and self.players[GREEN_PIECE] is not None:
            red_type = "Human" if isinstance(self.players[RED_PIECE], HumanPlayer) else \
                      "Random" if isinstance(self.players[RED_PIECE], RandomComputerPlayer) else "AI"
            green_type = "Human" if isinstance(self.players[GREEN_PIECE], HumanPlayer) else \
                        "Random" if isinstance(self.players[GREEN_PIECE], RandomComputerPlayer) else "AI"
            
            self.canvas.create_text(WIDTH//2 - 100, y_pos, text=red_type, 
                                   font=("Helvetica", 22, "bold"), fill=RED)
            
            self.canvas.create_text(WIDTH//2, y_pos, text="vs", 
                                   font=("Helvetica", 18, "bold"), fill="white")
            
            self.canvas.create_text(WIDTH//2 + 100, y_pos, text=green_type, 
                                   font=("Helvetica", 22, "bold"), fill=GREEN) 
                                           
    def _set_status(self, msg, color=BLACK, blink=False):
        # Make status message more concise to avoid wrapping
        if msg == "Ready to play...":
            msg = "Ready to play... Press New Game"
            
        # Configure the status label with appropriate width and make sure text is visible
        self.status.config(text=msg, foreground=color)
        
        # Prevent wrapping by setting wraplength to a very large value
        self.status.config(wraplength=800)  # Significantly increased to prevent any wrapping
        
        if hasattr(self, 'status_blink_job') and self.status_blink_job:
            self.after_cancel(self.status_blink_job)
            self.status_blink_job = None
        
        if blink:
            if not hasattr(self, 'status_blink_job'):
                self.status_blink_job = None
            self._blink_status(msg, color)
            
    def _blink_status(self, msg, color):
        current_color = self.status.cget("foreground")
        
        if current_color == color:
            self.status.config(foreground="#777777")
        else:
            self.status.config(foreground=color)
        
        if self.status.cget("text") == msg:
            self.status_blink_job = self.after(500, lambda: self._blink_status(msg, color))

                
    def _update_score(self):
        s = self.score
        # Update score label with a simple and clear format
        self.score_lbl.config(text=f"Red: {s['red']}  Green: {s['green']}  Draw: {s['draws']}  Games: {s['games']}")
    
    def _hover(self,e):
        col=e.x//SQUARESIZE; self.last_hover=col if 0<=col<COLUMN_COUNT and self.game.is_valid(col) else None; self._draw()
    def _click(self,e):
        if not self.game_in_progress or self.game.game_over or self.paused or not isinstance(self.players[self.game.current_player],HumanPlayer):
            return
        col=e.x//SQUARESIZE
        if self.game.is_valid(col):
            self._make_move(col)

    def _log(self,t):
        self.moves.config(state="normal"); self.moves.insert("end",t+"\n"); self.moves.config(state="disabled"); self.moves.see("end")

    # ──────────────────────────────────────────────────────────────────
    # Win-probability / value label (top-right corner)
    # ──────────────────────────────────────────────────────────────────
    def _update_win_probability(self):
        if not hasattr(self, 'players') or self.game.current_player not in self.players:
            return

        # choose the NN that’s actually playing this side
        player     = self.players[self.game.current_player]
        nn_manager = player.nn if isinstance(player, MCTSComputerPlayer) else self.nn

        # ── terminal positions ───────────────────────────────────
        if self.game.game_over:
            if self.game.winner == 'Draw':
                red_win_prob = green_win_prob = 0.5
                value = 0.0
            elif self.game.winner == RED_PIECE:
                red_win_prob, green_win_prob, value = 1.0, 0.0, +1.0
            else:
                red_win_prob, green_win_prob, value = 0.0, 1.0, +1.0

        # ── live positions ───────────────────────────────────────
        elif nn_manager is not None:
            _, value = nn_manager.policy_value(self.game)
            red_win_prob, green_win_prob, _ = self._compute_win_probs(value)

        # ── fallback (no NN) ──────────────────────────────────────
        else:
            red_win_prob = green_win_prob = 0.5
            value = 0.0

        # store for potential external queries
        self.current_red_win_prob   = red_win_prob
        self.current_green_win_prob = green_win_prob
        self.current_draw_prob      = 0.0

        # update on-screen label (raw value plus percentages)
        self.win_prob_var.set(
            f"Value: {value:+.3f}   Win Probs: Red {red_win_prob:.0%} | "
            f"Green {green_win_prob:.0%}"
        )

        # colour-coding by advantage
        if   red_win_prob   > ADV_THRESHOLD: self.win_prob_label.config(foreground="red")
        elif green_win_prob > ADV_THRESHOLD: self.win_prob_label.config(foreground="green")
        else:                                self.win_prob_label.config(foreground="black")

        # auto-refresh policy/ value pane
        if getattr(self, 'visualization_visible', False) and \
           getattr(self, 'auto_update_var', tk.BooleanVar()).get():
            self._update_visualization()



    def _make_move(self, col):
        if not self.game.is_valid(col): return
        state_before = self.game.get_state_copy()
        ok, _ = self.game.drop_piece(col)
        if not ok: return
        
        if (self.nn and self.learn_var.get() and 
            not isinstance(self.players[self.game.current_player], MCTSComputerPlayer) and
            not isinstance(self.players[self.game.current_player], RandomComputerPlayer)):
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
        
        if self.game.game_over:
            # For game over, update win probability before finishing
            self._update_win_probability()
            self._finish()
        else:
            # KEY FIX: Switch player BEFORE updating win probability and visualization
            # This ensures visualization shows data for the correct player's turn
            self.game.switch()
            
            # Now update win probability with the new current player
            self._update_win_probability()
            
            if not self.play_till_end:
                if isinstance(self.players[self.game.current_player], HumanPlayer):
                    self._set_status(f"{PLAYER_MAP[self.game.current_player]}'s turn...", COLOR_MAP[self.game.current_player])
                else:
                    self._set_status(f"{PLAYER_MAP[self.game.current_player]} is thinking...", COLOR_MAP[self.game.current_player])
            
            self.after(30, self._next_turn)



    def _next_turn(self):
        if self.game.game_over or not self.game_in_progress or self.paused: return
        ply = self.players[self.game.current_player]
        if isinstance(ply, (RandomComputerPlayer, MCTSComputerPlayer)):
            if not self.play_till_end:
                self._set_status(f"{PLAYER_MAP[self.game.current_player]} is thinking...", COLOR_MAP[self.game.current_player])
            threading.Thread(target=lambda: self._ai_play(ply), daemon=True).start()
                
    def _ai_play(self, ply):
        try:
            # Check for application shutdown
            if hasattr(self, 'shutting_down') and self.shutting_down:
                return
                
            mv = ply.get_move(self.game, self)
            
            # Check again after getting the move
            if hasattr(self, 'shutting_down') and self.shutting_down:
                return
            
            if self.is_comp and hasattr(self, 'cc_delay') and self.cc_delay > 0:
                delay = self.cc_delay
            else:
                delay = 10
            
            self.after(delay, lambda: self._make_move(mv))
        except Exception as e:
            error_msg = f"AI error: {str(e)}"
            self.after(0, lambda: messagebox.showerror("AI Error", error_msg))
            self.after(0, lambda: self._set_status(f"AI error occurred", "red"))
            if str(self.train_frame.grid_info()) != "{}":
                self.log_to_training(f"ERROR: {error_msg}")

    def _process_collected_games(self, all_examples, all_game_results, start_game, end_game, elapsed_str):
        try:
            if hasattr(self, 'force_stop_training') and self.force_stop_training:
                return False
            
            # Check for stop request before processing
            if self.training_stop_requested:
                self.log_to_training("Stop requested during game processing")
                return False
                    
            self.log_to_training(f"Processing games {start_game} to {end_game}")
            
            for i, (game_examples, winner) in enumerate(zip(all_examples[start_game-1:end_game], 
                                                           all_game_results[start_game-1:end_game])):
                # Check for stop request during processing
                if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                    self.log_to_training(f"Stop requested during game processing at game {i}")
                    return False
                    
                try:
                    for example_data in game_examples:
                        game_state = Connect4Game()
                        game_state.board = np.array(example_data['board'], dtype=np.int8)
                        game_state.current_player = example_data['player']
                        policy = np.array(example_data['policy'], dtype=np.float32)
                        
                        self.nn.add_example(game_state, policy)
                    
                    self.nn.finish_game(winner)
                    
                    if i % 10 == 0 and i > 0:
                        self.log_to_training(f"Processed {i}/{end_game-start_game+1} games")
                        
                except Exception as proc_error:
                    self.log_to_training(f"Error processing game {i+start_game}: {str(proc_error)}")
            
            # Check for stop request before training
            if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                self.log_to_training("Stop requested before neural network training")
                return False
            
            if not hasattr(self, 'force_stop_training') or not self.force_stop_training:
                status_text = f"Training neural network... - {elapsed_str}"
                # Check if window still exists before scheduling GUI update
                if self.winfo_exists():
                    self.after(0, lambda status=status_text: self.train_status.config(text=status))
                
                self.nn_training_phase = True
                if self.winfo_exists():
                    self.after(0, lambda: self._draw())
                self.update_idletasks()
                
                self.log_to_training(f"\nTraining neural network on {len(self.nn.data['states'])} examples")
                
                # Set training interrupted flag before training
                if hasattr(self, 'nn'):
                    self.nn.training_interrupted = False
                
                # Check for interruption before starting training
                if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                    self.log_to_training("Stop requested just before neural network training")
                    self.nn_training_phase = False
                    if self.winfo_exists():
                        self.after(0, lambda: self._draw())
                    return False
                
                self.nn.train(
                    batch_size=self.nn_params['batch_size'], 
                    epochs=self.nn_params['epochs'], 
                    start_time=None,
                    logger=self.log_to_training,
                    num_games=end_game-start_game+1
                )
                
                if self.winfo_exists():
                    self.after(0, lambda: self._update_model_info())
                
                self.nn_training_phase = False
                if self.winfo_exists():
                    self.after(0, lambda: self._draw())
                self.update_idletasks()
                
                self.log_to_training(f"Training completed - Continuing with game generation\n")
                return True
            return False
        except Exception as error:
            self.log_to_training(f"Error in _process_collected_games: {str(error)}")
            self.nn_training_phase = False
            if self.winfo_exists():
                self.after(0, lambda: self._draw())
            return False

    def _finish(self):
        self.game_in_progress = False
        self.score['games'] += 1
        
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
                    
        # Update win rate for curriculum learning
        if hasattr(self, 'nn'):
            if self.game.winner == RED_PIECE:
                red_player = self.players[RED_PIECE]
                if isinstance(red_player, MCTSComputerPlayer) and hasattr(red_player.nn, 'win_rate'):
                    new_rate = 0.9 * red_player.nn.win_rate + 0.1 * 1.0
                    red_player.nn.win_rate = new_rate
                        
            elif self.game.winner == GREEN_PIECE:
                green_player = self.players[GREEN_PIECE]
                if isinstance(green_player, MCTSComputerPlayer) and hasattr(green_player.nn, 'win_rate'):
                    new_rate = 0.9 * green_player.nn.win_rate + 0.1 * 1.0
                    green_player.nn.win_rate = new_rate
            
        self._update_score()
        
        if self.is_comp and str(self.train_frame.grid_info()) != "{}":
            if self.game.winner == 'Draw':
                self.log_to_training("Result: Draw")
            else:
                winner_name = "Red" if self.game.winner == RED_PIECE else "Green"
                self.log_to_training(f"Result: {winner_name} wins")
            self.log_to_training(f"Current score - Red: {self.score['red']}, Green: {self.score['green']}, Draws: {self.score['draws']}")
        
        if self.is_comp and self.continuous_play and str(self.train_frame.grid_info()) != "{}":
            self.train_progress['maximum'] = self.max_cc_games
            self.train_progress['value'] = self.score['games']
            self.train_status.config(text=f"Games: {self.score['games']} / {self.max_cc_games}")
        
        self.update_idletasks()
        
        should_train = (self.games_since_training >= self.cc_train_interval) or self.play_till_end
        
        if self.play_till_end:
            self.log_to_training(f"Game ended after stop request. Training on {self.games_since_training} accumulated games.")
            self.play_till_end = False
            self.paused = True
            self.stop_btn['state'] = "disabled"
            
            # FIXED: Always train when play_till_end is True, regardless of player types
            self.after(50, self._perform_end_game_training)
            
            self.after(1200, lambda: self._set_status("Ready to play..."))
            
            return
        
        if self.is_comp and should_train and not self.paused:
            self.log_to_training(f"Training interval of {self.cc_train_interval} games reached.")
            self.games_since_training = 0
            
            # FIXED: Always train when training interval is reached, regardless of player types
            self.after(50, self._perform_end_game_training)
        elif not self.is_comp and self.learn_var.get() and not self.paused:
            has_ai = any(isinstance(p, MCTSComputerPlayer) for p in self.players.values())
            if has_ai:
                if str(self.train_frame.grid_info()) != "{}":
                    self.log_to_training("Learn@Play is enabled - Training on this game.")
                self.games_since_training = 0
                
                self.after(50, self._perform_end_game_training)
        elif self.is_comp and not should_train and not self.paused:
            self.log_to_training(f"Games since last training: {self.games_since_training}/{self.cc_train_interval}")
        
        if self.is_comp and not self.paused:
            if str(self.train_frame.grid_info()) != "{}":
                self.train_status.config(text=f"Games: {self.score['games']} / {self.max_cc_games} - Training complete")
            
            if self.continuous_play and self.score['games'] < self.max_cc_games:
                self.auto_job = self.after(1200, lambda: self._new_game())
            else:
                if self.continuous_play and self.score['games'] >= self.max_cc_games:
                    self.log_to_training(f"Maximum number of games ({self.max_cc_games}) reached. Stopping continuous play.")
                    
                    # FIX: Check if there are games since last training before final training
                    if self.games_since_training > 0:
                        self.log_to_training(f"Training on {self.games_since_training} games since last training.")
                        # FIX: Don't call _perform_end_game_training which expects a single game
                        # Instead, log that training will happen at the end
                        self.log_to_training("Final training will be performed...")
                        
                        # Show completion message after training
                        self.after(50, lambda: self._perform_final_training())
                    else:
                        # No games to train on
                        self.after(500, lambda: messagebox.showinfo("Continuous Play Complete", 
                                                    f"Maximum number of games ({self.max_cc_games}) reached.\n\n"
                                                    f"Final score:\nRed: {self.score['red']}\nGreen: {self.score['green']}\nDraws: {self.score['draws']}"))
                        
                        self.after(1200, lambda: self._set_status("Ready to play..."))                            

    def _perform_final_training(self):
        """Perform final training when maximum games is reached in continuous play"""
        if not self.is_comp:
            return
            
        self.log_to_training("Performing final training on accumulated games...")
        
        # Log statistics about accumulated training data
        if hasattr(self.nn, 'data') and 'states' in self.nn.data:
            num_examples = len(self.nn.data['states'])
            self.log_to_training(f"Training on {num_examples} accumulated examples")
            
            if num_examples > 0:
                # Save current window size before training
                current_width = self.winfo_width()
                current_height = self.winfo_height()
                
                self._set_status("NN is training...", "#990000")
                
                self.performing_training = True
                self._draw()
                
                # Ensure window doesn't resize
                self.update_idletasks()
                self.geometry(f"{current_width}x{current_height}")
                
                self.train_status.config(text="Training neural network...")
                self.train_status.config(foreground="red")
                
                # Start blinking
                if not hasattr(self, 'train_blink_job'):
                    self.train_blink_job = None
                
                if self.train_blink_job:
                    self.after_cancel(self.train_blink_job)
                
                self._blink_train_status()
                
                def immediate_logger(msg):
                    self.log_to_training(msg)
                    self._force_process_log_buffer()
                    # Force geometry to stay consistent
                    self.geometry(f"{current_width}x{current_height}")
                
                # Train on all accumulated examples
                self.nn.train(
                    batch_size=self.nn_params['batch_size'], 
                    epochs=self.nn_params['epochs'],
                    logger=immediate_logger, 
                    num_games=self.games_since_training
                )
                
                # Stop blinking
                if self.train_blink_job:
                    self.after_cancel(self.train_blink_job)
                    self.train_blink_job = None
                self.train_status.config(foreground="black")
                
                self._force_process_log_buffer()
                self._update_model_info()
                
                self.performing_training = False
                self._draw()
                
                # Update AI players with the newly trained network
                for player_id, player in self.players.items():
                    if isinstance(player, MCTSComputerPlayer):
                        player.nn = self.nn
                        player.mcts = MCTS(player.mcts.I, player.mcts.c, player.nn, player.mcts.explore)
                        player.mcts.root = None
                        self.log_to_training(f"Updated {PLAYER_MAP[player_id]} AI with newly trained network")
                
                # Ensure window size is maintained after training
                self.update_idletasks()
                self.geometry(f"{current_width}x{current_height}")
                
                self.log_to_training("Final training completed.")
                
                self.after(1000, lambda: messagebox.showinfo("Continuous Play Complete", 
                          f"Maximum number of games ({self.max_cc_games}) reached.\n\n"
                          f"Final score:\nRed: {self.score['red']}\nGreen: {self.score['green']}\nDraws: {self.score['draws']}\n\n"
                          f"Neural network has been trained on all accumulated examples."))
            else:
                self.log_to_training("No training examples available for final training.")
                self.after(500, lambda: messagebox.showinfo("Continuous Play Complete", 
                          f"Maximum number of games ({self.max_cc_games}) reached.\n\n"
                          f"Final score:\nRed: {self.score['red']}\nGreen: {self.score['green']}\nDraws: {self.score['draws']}"))
        else:
            self.log_to_training("No training data found.")
            self.after(500, lambda: messagebox.showinfo("Continuous Play Complete", 
                      f"Maximum number of games ({self.max_cc_games}) reached.\n\n"
                      f"Final score:\nRed: {self.score['red']}\nGreen: {self.score['green']}\nDraws: {self.score['draws']}"))
        
        self.after(1200, lambda: self._set_status("Ready to play..."))

                            
    def _perform_end_game_training(self):
        # Save current window size before training
        current_width = self.winfo_width()
        current_height = self.winfo_height()
        
        has_random_player = any(isinstance(p, RandomComputerPlayer) for p in self.players.values())
        has_ai_player = any(isinstance(p, MCTSComputerPlayer) for p in self.players.values())
        
        # FIXED: Check if AI players have pending data and transfer it to main neural network
        # This is the key fix - we need to ensure that training data from AI players is collected
        if self.is_comp and has_ai_player:
            # For AI vs AI games, make sure we collect training examples from both players
            red_player = self.players[RED_PIECE]
            green_player = self.players[GREEN_PIECE]
            
            # Transfer pending examples from AI players to main neural network
            if isinstance(red_player, MCTSComputerPlayer) and red_player.nn.pending:
                if str(self.train_frame.grid_info()) != "{}":
                    self.log_to_training(f"Transferring {len(red_player.nn.pending)} training examples from Red AI player")
                for ex in red_player.nn.pending:
                    self.nn.pending.append(ex)
                red_player.nn.pending.clear()
            
            if isinstance(green_player, MCTSComputerPlayer) and green_player.nn.pending:
                if str(self.train_frame.grid_info()) != "{}":
                    self.log_to_training(f"Transferring {len(green_player.nn.pending)} training examples from Green AI player")
                for ex in green_player.nn.pending:
                    self.nn.pending.append(ex)
                green_player.nn.pending.clear()
            
            # Finish the game in the main neural network to assign win/loss values
            self.nn.finish_game(self.game.winner)
        
        # FIXED: Removed the condition that excluded training for computer vs computer games
        # Old code: if (not has_random_player) and (self.learn_var.get() or self.is_comp):
        # New code: Always train if there are AI players and we're not skipping due to random players
        if not has_random_player:
            self._set_status("NN is training...", "#990000")
            
            self.performing_training = True
            self._draw()  # This will now show "Training on the last game..."
            
            # Ensure window doesn't resize
            self.update_idletasks()
            self.geometry(f"{current_width}x{current_height}")
            
            if self.is_comp and str(self.train_frame.grid_info()) != "{}":
                self.train_status.config(text="Training neural network...")
                self.train_status.config(foreground="red")
                self.log_to_training("Training neural network...")
                
                # Fixed: Use proper blinking method
                if not hasattr(self, 'train_blink_job'):
                    self.train_blink_job = None
                
                if self.train_blink_job:
                    self.after_cancel(self.train_blink_job)
                
                self._blink_train_status()
            
            # FIXED: For human vs AI, make sure we transfer examples from the AI player
            if not self.is_comp and has_ai_player:
                ai_player = next(p for p in self.players.values() if isinstance(p, MCTSComputerPlayer))
                if ai_player.nn is not self.nn:
                    for ex in ai_player.nn.pending:
                        self.nn.pending.append(ex)
                    ai_player.nn.pending.clear()
                    
                    if str(self.train_frame.grid_info()) != "{}":
                        self.log_to_training(f"Transferred {len(self.nn.pending)} training examples from AI player")
            
            # FIXED: Log more information when Learn@Play is active
            if not self.is_comp and self.learn_var.get():
                self.log_to_training(
                    f"Learn@Play is enabled - Training on game result: "
                    f"{'Draw' if self.game.winner == 'Draw' else 'Red wins' if self.game.winner == RED_PIECE else 'Green wins'}")
            
            # Complete the game and set win/loss values
            self.nn.finish_game(self.game.winner)
            
            self._force_process_log_buffer()
            
            def immediate_logger(msg):
                self.log_to_training(msg)
                self._force_process_log_buffer()
                # Force geometry to stay consistent
                self.geometry(f"{current_width}x{current_height}")
            
            # Log training process for Human vs AI game
            if not self.is_comp and str(self.train_frame.grid_info()) != "{}":
                self.log_to_training("Training on the last game...")
            
            # FIXED: Increase epochs for human vs AI learning to improve retention
            epochs = self.nn_params['epochs']
            if not self.is_comp and self.learn_var.get():
                # Use more epochs when learning from human example to prevent forgetting
                epochs = min(20, epochs * 2)
                self.log_to_training(f"Using {epochs} epochs for human vs. AI learning")
            
            # Check if there's actually data to train on
            training_occurred = False
            if len(self.nn.data['states']) > 0:
                self.nn.train(
                    batch_size=self.nn_params['batch_size'], 
                    epochs=epochs,
                    logger=immediate_logger, 
                    num_games=1
                )
                training_occurred = True
            else:
                self.log_to_training("No training data available. Skipping training.")
            
            if self.train_blink_job:
                self.after_cancel(self.train_blink_job)
                self.train_blink_job = None
            self.train_status.config(foreground="black")
            
            self._force_process_log_buffer()
            
            self._update_model_info()
            
            self.performing_training = False
            self._draw()
            
            # FIXED: Only update AI players if training actually occurred
            if has_ai_player and training_occurred:
                for player_id, player in self.players.items():
                    if isinstance(player, MCTSComputerPlayer):
                        # Update the AI with the trained network
                        player.nn = self.nn
                        player.mcts = MCTS(player.mcts.I, player.mcts.c, player.nn, player.mcts.explore)
                        player.mcts.root = None
                        self.log_to_training(f"Updated {PLAYER_MAP[player_id]} AI with newly trained network")
            elif has_ai_player and not training_occurred:
                self.log_to_training("No neural network updates performed - no training data was available")
            
            # Ensure window size is maintained after training
            self.update_idletasks()
            self.geometry(f"{current_width}x{current_height}")
            
            if self.game.game_over:
                if self.game.winner == 'Draw':
                    self._set_status("Draw")
                else:
                    winner_name = "Red" if self.game.winner == RED_PIECE else "Green"
                    self._set_status(f"{winner_name} wins", COLOR_MAP[self.game.winner])
            else:
                self._set_status("Ready to play...")
        else:
            self.nn.pending.clear()
            
            if has_random_player and self.is_comp and str(self.train_frame.grid_info()) != "{}":
                self.log_to_training("Training skipped - Random Computer player present")
                self._force_process_log_buffer()
                        
    def _force_process_log_buffer(self):
        try:
            messages = []
            with self.log_buffer_lock:
                if self.log_buffer:
                    messages = self.log_buffer.copy()
                    self.log_buffer.clear()
            
            if messages and hasattr(self, 'train_log') and self.train_log.winfo_exists():
                self.train_log.config(state="normal")
                
                # Process messages in smaller batches to avoid GUI freezing
                batch_size = 50
                
                for i in range(0, len(messages), batch_size):
                    batch = messages[i:i+batch_size]
                    for message in batch:
                        self.train_log.insert("end", message)
                        
                        if "POLICY=" in message:
                            try:
                                policy_loss = float(message.split("POLICY=")[1].split()[0])
                                value_loss = float(message.split("VALUE=")[1].split()[0])
                                
                                self.training_losses["policy"].append(policy_loss)
                                self.training_losses["value"].append(value_loss)
                                
                                self.update_loss_summary()
                            except Exception as e:
                                print(f"Error parsing loss: {e}")
                                
                            last_line = self.train_log.index("end-1c linestart")
                            self.train_log.tag_add("highlight", last_line, f"{last_line} lineend")
                            self.train_log.tag_config("highlight", background="#E0F0FF")
                    
                    # Update display after each batch and process events
                    self.train_log.see("end")
                    self.update_idletasks()
                
                self.train_log.config(state="disabled")
                
                # Explicitly process GUI events to update the display
                self.update_idletasks()
        except Exception as e:
            print(f"Error in _force_process_log_buffer: {e}")
            
    def _settings(self):
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Pause the game before changing settings.")
            return
        
        original_settings = {
            'mcts': dict(self.mcts_params),
            'training': {
                'games': self.train_games, 
                'games_before_training': self.games_before_training,
                'max_cc_games': self.max_cc_games,
                'cc_train_interval': self.cc_train_interval, 
                'cc_delay': self.cc_delay
            },
            'nn_params': dict(self.nn_params)
        }
        
        dlg = SettingsDialog(self, self.mcts_params, self.train_games, self.max_cc_games, 
                             self.cc_train_interval, self.cc_delay, self.games_before_training, self.nn_params)
        if dlg.result:
            new_mcts = dlg.result['mcts']
            new_train_games = dlg.result['training']['games']
            new_games_before_training = dlg.result['training']['games_before_training']
            new_max_cc_games = dlg.result['training']['max_cc_games']
            new_cc_train_interval = dlg.result['training']['cc_train_interval']
            new_cc_delay = dlg.result['training']['cc_delay']
            new_nn_params = dlg.result['nn_params']
            
            settings_changed = False
            model_arch_changed = False
            
            if (new_mcts['iterations'] != self.mcts_params['iterations'] or
                new_mcts['C_param'] != self.mcts_params['C_param']):
                settings_changed = True
            
            if (new_train_games != self.train_games or
                new_games_before_training != self.games_before_training or
                new_max_cc_games != self.max_cc_games or
                new_cc_train_interval != self.cc_train_interval or
                new_cc_delay != self.cc_delay):
                settings_changed = True
            
            for key, value in new_nn_params.items():
                if key not in self.nn_params or self.nn_params[key] != value:
                    settings_changed = True
                    if key == 'use_resnet' and self.nn_params.get('use_resnet', False) != value:
                        model_arch_changed = True
                    break
            
            self.mcts_params = new_mcts
            self.train_games = new_train_games
            self.games_before_training = new_games_before_training
            self.max_cc_games = new_max_cc_games
            self.cc_train_interval = new_cc_train_interval
            self.cc_delay = new_cc_delay
            old_lr = self.nn_params.get('learning_rate')
            self.nn_params = new_nn_params
            
            if settings_changed:
                if model_arch_changed:
                    # Architecture changed, recreate neural network
                    answer = messagebox.askyesno("Architecture Change", 
                                              "Changing between CNN and ResNet requires creating a new model.\n\n"
                                              "Do you want to create a new model with the selected architecture?")
                    if answer:
                        # Generate a new model name
                        model_dir = os.path.dirname(os.path.abspath(self.nn_model_path))
                        model_name = os.path.basename(self.nn_model_path)
                        base, ext = os.path.splitext(model_name)
                        
                        # If it already has a ResNet suffix, remove it
                        if base.endswith("_resnet"):
                            base = base[:-7]
                            
                        # Add suffix based on architecture
                        if self.nn_params['use_resnet']:
                            new_base = f"{base}_resnet"
                        else:
                            new_base = base
                            
                        new_model_path = os.path.join(model_dir, f"{new_base}{ext}")
                        
                        # Ask for confirmation with new name
                        confirm = messagebox.askyesno("Confirm New Model", 
                                                   f"Create new model as:\n{new_model_path}?")
                        if confirm:
                            self.nn_model_path = new_model_path
                            self.nn = NNManager(self.nn_params, self.nn_model_path,
                                              use_resnet=self.nn_params['use_resnet'])
                            messagebox.showinfo("Success", "New model created with the selected architecture.")
                    else:
                        # Revert the architecture change
                        self.nn_params['use_resnet'] = not self.nn_params['use_resnet']
                else:
                    # Update learning rate if it changed
                    if old_lr != self.nn_params['learning_rate']:
                        for param_group in self.nn.opt.param_groups:
                            param_group['lr'] = self.nn_params['learning_rate']
                    
                    # Update neural network hyperparameters
                    for key, value in self.nn_params.items():
                        self.nn.hyperparams[key] = value
            
            cfg = {
                'mcts': self.mcts_params,
                'training': {
                    'games': self.train_games, 
                    'games_before_training': self.games_before_training,
                    'max_cc_games': self.max_cc_games,
                    'cc_train_interval': self.cc_train_interval,
                    'cc_delay': self.cc_delay
                },
                'nn_params': self.nn_params,
                'column_bias': list(CENTER_BIAS)  # Add column bias to the config
            }
            json.dump(cfg, open(MCTS_CONFIG_FILE, "w"), indent=4)
            
            self._update_model_info()
            
            if settings_changed:
                messagebox.showinfo("Saved", "Settings updated.")

        
    def _train(self):
        if self.game_in_progress and not self.paused:
            messagebox.showinfo("Info", "Finish or pause the current game before training.")
            return
        
        if self.training_in_progress:
            messagebox.showinfo("Info", "Training already in progress.")
            return
        
        # Save current window dimensions
        current_width = self.winfo_width()
        current_height = self.winfo_height()
        
        # Reset the game (board)
        self.game.reset()
        self.last_hover = None
        
        # Clear Game History
        self.moves.config(state="normal")
        self.moves.delete("1.0", "end")
        self.moves.config(state="disabled")
        
        # Clear Policy & Value visualization if it exists
        if hasattr(self, 'policy_fig') and self.policy_fig is not None:
            self.policy_fig.clear()
            if hasattr(self, 'policy_canvas') and self.policy_canvas is not None:
                self.policy_canvas.draw()
        
        self.nn_training_phase = False
        self.performing_training = False
        self.training_in_progress = True
        
        # Redraw board (now cleared)
        self._draw()
        
        self.separator.grid()
        self.train_frame.grid()
        self.model_info.grid()
        
        n = self.train_games
        
        self.train_btn['state'] = "disabled"
        self.train_browse['state'] = "disabled"
        self.manage_models_btn['state'] = "disabled"
        self.training_stop_requested = False
        self.stop_btn['state'] = "normal"
        
        self.train_progress['maximum'] = n
        self.train_progress['value'] = 0
        self.train_status.config(text=f"Training: 0 / {n} games")
        
        self._set_status("Training NN...", "#990000")
        
        self._update_model_info()
        
        # Ensure window size is maintained
        self.update_idletasks()
        self.geometry(f"{current_width}x{current_height}")
        
        # Add a section delimiter rather than clearing the log
        self.log_to_training("\n" + "=" * 50)
        self.log_to_training(f"Starting new training session with {n} self-play games")
        self.log_to_training(f"Model: {os.path.basename(self.nn_model_path)} ({self.nn.version})")
        self.log_to_training(f"MCTS iterations: {self.mcts_params['iterations']}")
        self.log_to_training(f"Architecture: {'ResNet' if self.nn_params.get('use_resnet', False) else 'CNN'}")
        self.log_to_training(f"Batch size: {self.nn_params['batch_size']}, Epochs: {self.nn_params['epochs']}")
        self.log_to_training(f"Training after every {self.games_before_training} games")
        self.log_to_training("-" * 50 + "\n")
        
        self.training_losses = {"policy": [], "value": []}
        
        # Start the training thread
        self.training_thread = threading.Thread(target=self._training_worker_thread, args=(n,), daemon=True)
        self.training_thread.start()
        
        # Set up a recurring task to maintain window dimensions
        self._maintain_window_size(current_width, current_height)
    
    def _update_status_during_training(self, current, total, results):
        self._set_status("Training NN...", "#990000")
        self.score_lbl.config(text=f"{results}  Games: {current}/{total}")

    def generate_tactical_position(self):
        game = Connect4Game()
        
        # Create a random mid-game position (10-16 moves in)
        moves_count = random.randint(10, 16)
        
        # Ensure we don't get stuck in an infinite loop
        max_attempts = 100
        attempts = 0
        
        while attempts < max_attempts and not game.game_over and sum(1 for r in range(ROW_COUNT) for c in range(COLUMN_COUNT) if game.board[r, c] != EMPTY) < moves_count:
            attempts += 1
            move = random.choice(game.valid_moves())
            game.drop_piece(move)
            if not game.game_over:
                game.switch()
        
        # Analyze if one player has a forced win or needs to block a win
        current_player = game.current_player
        opponent = GREEN_PIECE if current_player == RED_PIECE else RED_PIECE
        
        # Try to set up a two-move win scenario
        for move1 in game.valid_moves():
            test_game1 = game.copy()
            test_game1.drop_piece(move1)
            
            if test_game1.game_over and test_game1.winner == current_player:
                # Direct win found, this is the tactical position we want
                return game, move1, "win"
            
            if not test_game1.game_over:
                test_game1.switch()
                
                # Check if opponent could win after our move
                for move2 in test_game1.valid_moves():
                    test_game2 = test_game1.copy()
                    test_game2.drop_piece(move2)
                    
                    if test_game2.game_over and test_game2.winner == opponent:
                        # Return the position where we need to block next move
                        # Compute the blocking move
                        for blocking_move in game.valid_moves():
                            test_block = game.copy()
                            test_block.drop_piece(blocking_move)
                            if test_block.game_over and test_block.winner == current_player:
                                return game, blocking_move, "win"
                            
                            if not test_block.game_over:
                                test_block.switch()
                                
                                # Check if this move prevents opponent's win
                                can_win = False
                                for opp_move in test_block.valid_moves():
                                    test_block2 = test_block.copy()
                                    test_block2.drop_piece(opp_move)
                                    if test_block2.game_over and test_block2.winner == opponent:
                                        can_win = True
                                        break
                                
                                if not can_win:
                                    return game, blocking_move, "block"
        
        # If no tactical position found, return the random position
        return game, None, "random"

                                        
    def _training_worker_thread(self, n):
        try:
            # Store reference to executor for proper cleanup
            self.training_executor = None
            
            self.nn_training_phase = False
            self.after(0, lambda: self._draw())
            self.update_idletasks()
            
            start_time = time.time()
            
            self.log_to_training(f"Started training for {n} games")
            self.log_to_training(f"MCTS iterations: {self.mcts_params['iterations']}")
            self.log_to_training(f"Neural network model: {os.path.basename(self.nn_model_path)}")
            self.log_to_training(f"Batch size: {self.nn_params['batch_size']}, Epochs: {self.nn_params['epochs']}")
            self.log_to_training(f"Training after every {self.games_before_training} games")
            self.log_to_training("-" * 50)
            
            self.training_losses = {"policy": [], "value": []}
            
            num_workers = max(1, min(os.cpu_count() - 1, 4))
            self.log_to_training(f"Starting parallel training with {num_workers} workers - Elapsed: 0h 0m 0s")
            
            batch_size = min(n, max(1, n // (num_workers * 2)))
            
            nn_config = {
                'model_path': self.nn_model_path,
                'hyperparams': self.nn_params,
                'use_resnet': self.nn_params.get('use_resnet', False)
            }
            
            game_func = partial(
                _play_single_training_game,
                self.mcts_params['iterations'], 
                self.mcts_params['C_param'], 
                nn_config,
                self.mcts_params.get('dirichlet_noise', 0.3)
            )
            
            games_completed = 0
            all_examples = []
            all_game_results = []
            
            games_since_last_training = 0
            executor = None
            
            tactical_games = 0
            tactical_injection_frequency = 20
            
            try:
                executor = ProcessPoolExecutor(max_workers=num_workers)
                self.training_executor = executor  # Store reference for cleanup
                futures = []
                
                while games_completed < n and not self.training_stop_requested and not (hasattr(self, 'force_stop_training') and self.force_stop_training):
                    # Check for application shutdown
                    if hasattr(self, 'shutting_down') and self.shutting_down:
                        self.log_to_training("Application is shutting down - stopping training")
                        break
                        
                    games_to_run = min(batch_size, n - games_completed)
                    if games_to_run <= 0:
                        break
                    
                    # Check for stop request more frequently
                    if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                        self.log_to_training("Stop requested during game generation")
                        break
                    
                    if games_completed > 0 and games_completed % tactical_injection_frequency == 0:
                        self.log_to_training(f"Injecting tactical training position")
                        tactical_position, best_move, position_type = self.generate_tactical_position()
                        
                        if best_move is not None:
                            tactical_games += 1
                            
                            game_state = tactical_position
                            
                            policy = np.zeros(COLUMN_COUNT, dtype=np.float32)
                            
                            if position_type == "win":
                                policy[best_move] = 0.9
                                valid_moves = game_state.valid_moves()
                                if len(valid_moves) > 1:
                                    for m in valid_moves:
                                        if m != best_move:
                                            policy[m] = 0.1 / (len(valid_moves) - 1)
                                value = 1.2
                            else:  # "block"
                                policy[best_move] = 0.8
                                valid_moves = game_state.valid_moves()
                                if len(valid_moves) > 1:
                                    for m in valid_moves:
                                        if m != best_move:
                                            policy[m] = 0.2 / (len(valid_moves) - 1)
                                value = 0.8
                            
                            example_data = {
                                'board': game_state.board.tolist(),
                                'player': game_state.current_player,
                                'policy': policy.tolist()
                            }
                            
                            all_examples.append([example_data])
                            all_game_results.append(game_state.current_player)
                            
                            games_completed += 1
                            games_since_last_training += 1
                            
                            elapsed_seconds = time.time() - start_time
                            minutes, seconds = divmod(elapsed_seconds, 60)
                            hours, minutes = divmod(minutes, 60)
                            elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                            
                            # Check if window still exists before scheduling GUI update
                            if not self.winfo_exists():
                                return
                            
                            self.after(0, lambda p=games_completed, n=n, t=elapsed_str: 
                                      self._update_training_progress(p, n, t))
                            
                            self.log_to_training(f"Added tactical {position_type} position - Move: {best_move+1}")
                    
                    # Cancel futures if stop requested - more aggressive cancellation
                    if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                        self.log_to_training("Cancelling pending game futures")
                        # Cancel all existing futures first
                        for future in futures:
                            if not future.done():
                                future.cancel()
                        
                        # If force stopping, terminate executor immediately
                        if hasattr(self, 'force_stop_training') and self.force_stop_training:
                            self.log_to_training("Force terminating executor...")
                            executor.shutdown(wait=False)
                            break
                        
                        break
                    
                    batch_futures = [executor.submit(game_func) for _ in range(games_to_run)]
                    futures.extend(batch_futures)
                    
                    pending = batch_futures.copy()
                    
                    while pending and not self.training_stop_requested and not (hasattr(self, 'force_stop_training') and self.force_stop_training):
                        # Check for application shutdown
                        if hasattr(self, 'shutting_down') and self.shutting_down:
                            self.log_to_training("Application is shutting down during game execution")
                            for future in pending:
                                future.cancel()
                            break
                        
                        # Use shorter timeout for more responsive stopping
                        done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
                        
                        if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                            self.log_to_training("Stop requested during game execution")
                            for future in pending:
                                future.cancel()
                            break
                            
                        for future in done:
                            if future.cancelled():
                                continue
                                
                            try:
                                examples, winner = future.result()
                                all_examples.append(examples)
                                all_game_results.append(winner)
                                
                                games_completed += 1
                                games_since_last_training += 1
                                
                                elapsed_seconds = time.time() - start_time
                                minutes, seconds = divmod(elapsed_seconds, 60)
                                hours, minutes = divmod(minutes, 60)
                                elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
                                
                                # Check if window still exists before scheduling GUI update
                                if not self.winfo_exists():
                                    return
                                
                                self.after(0, lambda p=games_completed, n=n, t=elapsed_str: 
                                          self._update_training_progress(p, n, t))
                                
                                red_wins = sum(1 for w in all_game_results if w == RED_PIECE)
                                green_wins = sum(1 for w in all_game_results if w == GREEN_PIECE)
                                draws = sum(1 for w in all_game_results if w == 'Draw')
                                results = f"Red: {red_wins} Green: {green_wins} Draw: {draws}"
                                
                                # Check if window still exists before scheduling GUI update
                                if self.winfo_exists():
                                    self.after(0, lambda p=games_completed, n=n, r=results: 
                                              self._update_status_during_training(p, n, r))
                                
                                if games_completed % max(1, n // 10) == 0 or games_completed % 5 == 0:
                                    self.log_to_training(f"Training progress: {games_completed}/{n} games - Elapsed: {elapsed_str}")
                                    self.log_to_training(f"Game outcomes: Red: {red_wins}, Green: {green_wins}, Draws: {draws}")
                                    self.log_to_training(f"Games since last training: {games_since_last_training}/{self.games_before_training}")
                                    if tactical_games > 0:
                                        self.log_to_training(f"Tactical positions injected: {tactical_games}")
                                
                                # Check for stop before processing games
                                if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                                    self.log_to_training("Stop requested before processing games")
                                    break
                                
                                if games_since_last_training >= self.games_before_training and not self.training_stop_requested:
                                    start_game = games_completed - games_since_last_training + 1
                                    end_game = games_completed
                                    
                                    self.log_to_training(f"\nTraining interval of {self.games_before_training} games reached.")
                                    self.log_to_training(f"Training on games {start_game} to {end_game}")
                                    
                                    if self._process_collected_games(all_examples, all_game_results, 
                                                              start_game, end_game, elapsed_str):
                                        games_since_last_training = 0
                                    
                            except Exception as ex:
                                self.log_to_training(f"Error in game generation: {str(ex)}")
                        
                        # Check for stop again after processing done futures
                        if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                            break
                    
                    if self.training_stop_requested or (hasattr(self, 'force_stop_training') and self.force_stop_training):
                        break
                            
                # Cancel all remaining futures
                for future in futures:
                    if not future.done():
                        future.cancel()
                        
            finally:
                if executor:
                    # More aggressive shutdown with better logging
                    self.log_to_training("Shutting down ProcessPoolExecutor...")
                    executor.shutdown(wait=False)
                    
                    # Clear the reference
                    if hasattr(self, 'training_executor'):
                        self.training_executor = None
                    
                    # Give a moment for processes to terminate
                    time.sleep(0.1)
                    
                    # Log completion of shutdown
                    self.log_to_training("ProcessPoolExecutor shutdown complete")
            
            # Calculate final elapsed time
            elapsed_seconds = time.time() - start_time
            minutes, seconds = divmod(elapsed_seconds, 60)
            hours, minutes = divmod(minutes, 60)
            elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            
            # Ensure final logging happens
            self.log_to_training(f"Training thread completed - Total time: {elapsed_str}")
            
            # Check for stop before final training
            if not self.training_stop_requested and not (hasattr(self, 'force_stop_training') and self.force_stop_training):
                if games_completed > 0 and games_since_last_training > 0:
                    status_msg = f"Training on remaining {games_since_last_training} games - {elapsed_str}"
                    self.log_to_training(status_msg)
                    
                    # Check if window still exists before scheduling GUI update
                    if self.winfo_exists():
                        self.after(0, lambda msg=status_msg: self.train_status.config(text=msg))
                    
                    start_game = games_completed - games_since_last_training + 1
                    end_game = games_completed
                    
                    self._process_collected_games(all_examples, all_game_results,
                                           start_game, end_game, elapsed_str)
            
            # Final GUI updates - ensure these happen
            self.log_to_training(f"Updating GUI with final state...")
            
            # Check if window still exists before scheduling GUI updates
            if self.winfo_exists():
                self.after(0, lambda: self.train_btn.config(state="normal"))
                self.after(0, lambda: self.train_browse.config(state="normal"))
                self.after(0, lambda: self.manage_models_btn.config(state="normal"))
                self.after(0, lambda: setattr(self, 'training_in_progress', False))
                
                self.after(0, lambda: self._update_score())
            
            # Check if window still exists before scheduling GUI update
            if self.winfo_exists():
                self.after(0, lambda: self._set_status("Ready to play..."))
            
            # Handle different completion scenarios with better logging
            if hasattr(self, 'force_stop_training') and self.force_stop_training:
                final_text = f"Terminated - {elapsed_str}"
                final_msg = f"Training terminated after {games_completed} games.\nTotal time: {elapsed_str}"
                self.log_to_training(f"Training force-terminated after {games_completed} games")
                if self.winfo_exists():
                    self.after(0, lambda txt=final_text: self.train_status.config(text=txt))
                    self.after(0, lambda msg=final_msg: messagebox.showinfo("Terminated", msg))
            elif self.training_stop_requested:
                final_text = f"Stopped after {games_completed} games - {elapsed_str}"
                final_msg = f"Training stopped after {games_completed} games.\nTotal time: {elapsed_str}"
                self.log_to_training(f"Training stopped after {games_completed} games")
                if self.winfo_exists():
                    self.after(0, lambda txt=final_text: self.train_status.config(text=txt))
                    self.after(0, lambda msg=final_msg: messagebox.showinfo("Stopped", msg))
            else:
                final_text = f"Completed {games_completed} games - {elapsed_str}"
                final_msg = f"Training finished.\nTotal time: {elapsed_str}"
                self.log_to_training(f"Training completed normally after {games_completed} games")
                if self.winfo_exists():
                    self.after(0, lambda txt=final_text: self.train_status.config(text=txt))
                    self.after(0, lambda msg=final_msg: messagebox.showinfo("Done", msg))
            
            # Clean up flags
            self.log_to_training("Cleaning up training flags...")
            if hasattr(self, 'force_stop_training'):
                delattr(self, 'force_stop_training')
            self.training_stop_requested = False
            
            if not self.is_comp and self.winfo_exists():
                self.after(0, lambda: self.stop_btn.config(state="disabled"))
                
            self.log_to_training("Training thread completed successfully")
        
        except Exception as error:
            error_message = f"Error in training worker: {error}"
            self.log_to_training(error_message)
            import traceback
            traceback.print_exc()
            
            # Calculate elapsed time for error case
            if 'start_time' in locals():
                elapsed_seconds = time.time() - start_time
                minutes, seconds = divmod(elapsed_seconds, 60)
                hours, minutes = divmod(minutes, 60)
                elapsed_str = f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
            else:
                elapsed_str = "unknown"
            
            # Check if window still exists before scheduling GUI updates
            if self.winfo_exists():
                self.after(0, lambda: self.train_btn.config(state="normal"))
                self.after(0, lambda: self.train_browse.config(state="normal"))
                self.after(0, lambda: self.manage_models_btn.config(state="normal"))
                self.after(0, lambda: setattr(self, 'training_in_progress', False))
                self.after(0, lambda: self.stop_btn.config(state="disabled"))
                self.after(0, lambda: self.train_status.config(text=f"Error during training - {elapsed_str}"))
                
                self.after(0, lambda: self._update_score())
                
                error_dialog_msg = f"An error occurred during training:\n{str(error)}"
                self.after(0, lambda msg=error_dialog_msg: messagebox.showerror("Error", msg))
                
                self.after(0, lambda: self._set_status("Ready to play..."))
        
        finally:
            # Final cleanup with logging
            self.log_to_training("Entering final cleanup phase...")
            
            # Ensure executor is cleaned up
            if hasattr(self, 'training_executor') and self.training_executor:
                self.log_to_training("Final cleanup of ProcessPoolExecutor...")
                self.training_executor.shutdown(wait=False)
                self.training_executor = None
                
            self.log_to_training("Training thread cleanup completed")
    def _update_training_progress(self, current, total, elapsed_time=""):
        self.train_progress['value'] = current
        self.train_status.config(text=f"Training: {current} / {total} games - {elapsed_time}")

    def _pause(self):
        if self.training_in_progress:
            if self.training_stop_requested:
                # Force stop - set stronger interrupt signals
                self.force_stop_training = True
                self.stop_btn['state'] = "disabled"
                self._set_status("Training terminating... (please wait)")
                self.train_status.config(text="Terminating immediately...")
                
                # Try to interrupt the training process more aggressively
                if hasattr(self, 'nn') and hasattr(self.nn, 'training_interrupted'):
                    self.nn.training_interrupted = True
                
                # Force shutdown the executor reference
                if hasattr(self, 'training_executor') and self.training_executor:
                    self.log_to_training("Force shutting down ProcessPoolExecutor...")
                    try:
                        self.training_executor.shutdown(wait=False)
                        self.training_executor = None
                    except Exception as e:
                        self.log_to_training(f"Error shutting down executor: {e}")
                
                # Set global shutdown flag
                self.shutting_down = True
                
                self.after(100, self._check_force_stop_progress)
            else:
                # Ask for confirmation to stop training
                confirm = messagebox.askyesno("Stop Training", 
                                            "Do you want to stop the training?\n\n"
                                            "• Current training session will be interrupted\n"
                                            "• Any progress will be lost\n"
                                            "• You can click Stop again to force immediate termination",
                                            icon="warning")
                if confirm:
                    self.training_stop_requested = True
                    self._set_status("Training stopping... (press Stop again to force)")
                    
                    # Set a flag to interrupt training
                    if hasattr(self, 'nn'):
                        self.nn.training_interrupted = True
                    
                    # Log the stop request
                    self.log_to_training("Stop requested during training...")
                    
                    self.after(100, self._check_training_stop_progress)
            return
                
        if self.is_comp and self.game_in_progress:
            if self.play_till_end:
                return
                
            if self.auto_job:
                self.after_cancel(self.auto_job)
                self.auto_job = None
            
            self.play_till_end = True
            
            self._set_status("Finishing game...", "orange", blink=True)
            
            if str(self.train_frame.grid_info()) != "{}":
                self.log_to_training("\nStop requested - Playing until game ends.")
            
            return
        
        return

    def _check_training_stop_progress(self):
        if self.training_in_progress:
            current_text = self.train_status.cget("text")
            if "stopping" in current_text.lower():
                if current_text.count('.') > 5:
                    self.train_status.config(text="Stopping")
                else:
                    self.train_status.config(text=current_text + ".")
            else:
                self.train_status.config(text="Stopping.")
            
            self.after(300, self._check_training_stop_progress)

    def _check_force_stop_progress(self):
        if self.training_in_progress:
            current_text = self.train_status.cget("text")
            if "terminating" in current_text.lower():
                if current_text.count('!') > 3:
                    self.train_status.config(text="Terminating immediately")
                else:
                    self.train_status.config(text=current_text + "!")
            else:
                self.train_status.config(text="Terminating immediately!")
            
            self.after(200, self._check_force_stop_progress)
            
    def _load_cfg(self):
        if os.path.exists(MCTS_CONFIG_FILE):
            try:
                cfg = json.load(open(MCTS_CONFIG_FILE))
                if 'iterations' in cfg and 'C_param' in cfg:
                    self.mcts_params = cfg
                elif 'mcts' in cfg:
                    self.mcts_params = cfg['mcts']
                    if 'training' in cfg:
                        if 'games' in cfg['training']:
                            self.train_games = cfg['training']['games']
                        if 'games_before_training' in cfg['training']:
                            self.games_before_training = cfg['training']['games_before_training']
                        else:
                            self.games_before_training = min(100, self.train_games)
                        if 'max_cc_games' in cfg['training']:
                            self.max_cc_games = cfg['training']['max_cc_games']
                        if 'cc_train_interval' in cfg['training']:
                            self.cc_train_interval = cfg['training']['cc_train_interval']
                        if 'cc_delay' in cfg['training']:
                            self.cc_delay = cfg['training']['cc_delay']
                    if 'nn_params' in cfg:
                        self.nn_params = cfg['nn_params']
                        
                        # Check if there are new parameters that weren't in the config
                        if 'use_resnet' not in self.nn_params:
                            self.nn_params['use_resnet'] = False
                        if 'use_swa' not in self.nn_params:
                            self.nn_params['use_swa'] = False
                            
                        # Recreate NN manager with updated parameters
                        self.nn = NNManager(self.nn_params, self.nn_model_path, 
                                          use_resnet=self.nn_params.get('use_resnet', False))
                    
                    # Load column bias if present
                    if 'column_bias' in cfg:
                        global CENTER_BIAS
                        CENTER_BIAS = np.array(cfg['column_bias'])
                    
                    # Ensure dirichlet_noise is set with a default if not present
                    if 'dirichlet_noise' not in self.mcts_params:
                        self.mcts_params['dirichlet_noise'] = 0.3
                        
                    if self.games_before_training > self.train_games:
                        self.games_before_training = self.train_games
            except:
                pass
                    
                
# ----------------------------------------------------------------------
if __name__=='__main__':
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    
    Connect4GUI().mainloop()