#!/usr/bin/env python3
"""
CodingAPI - Enhanced version 9.2

A comprehensive library for generating code using multiple LLM providers with
improved error handling, security, performance, and user experience.
Updated with current models and optimized parameter settings based on 2025 landscape.

Key features:
- Modular provider architecture with wrapper pattern
- Enhanced error handling with retries and fallbacks
- Secure credential management
- Input validation and sanitization
- Asynchronous processing with caching
- Improved UI with progress indicators
- Enhanced prompting templates
- Maximum reasoning capability for supported models
- API key validation with model availability checking (new in v7.1)
- Support for environment variables for API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY) (new in v7.2)
- Improved UI for API key configuration (new in v7.3)
- Added Exit button for easier application closing (new in v7.4)
- Check Models functionality to dynamically discover and select available models (new in v7.5)
- Added Clear button for Program Description field (new in v7.6)
- Fixed phantom LLMs bug in Multiple Creation mode (new in v7.6)
- Added Run app button to execute and view Python applications (new in v8.0)
- Improved button organization and file naming for final iteration (new in v8.1)
- Enhanced Run app dialog window sizing (new in v8.1.1)
- Replaced JSON format with Delimiter-Based Format for code exchange (new in v8.3)
- Added tooltips to all buttons and field labels for better usability (new in v8.4)
- Fixed tooltip display for listbox widgets (new in v8.4.1)
- Improved iteration display to show progress (current/total) (new in v8.4.1)
- Renamed "Program file name" field to "Project name" for clarity (new in v8.4.1)
- Added unified Settings dialog with Files and API Keys configuration (new in v8.5)
- Added database storage for all generated code, audits and project information (new in v9.0)
- Changed config, settings and database storage to /data/ subdirectory (new in v9.1)

Changes in v9.2.1 (May 23, 2025):
- Fixed dynamic model registration to only log new models on first discovery
- Added persistent storage for dynamically discovered models
- Prevents repeated logging of already-known dynamic models on subsequent launches

Changes in v9.2 (May 22, 2025):
- Added API call to list available Claude models using client.models.list()
- Removed hardcoded Claude model testing in favor of dynamic model discovery
- Aligned Claude model discovery with other providers for consistency
- Split model selection in Settings into separate Coding Models and Audit Models tabs
- Independent checkbox selection for coding and auditing models per provider
- Added model ordering by release date (newest first)
- Fixed UI to properly reflect separate coding and auditing model selections
- Added dynamic model registration for discovered models
- Store model ID mappings when discovering models from APIs
- Added special token limit handling for Claude Opus 4.0 (32000 tokens)

Changes in v9.1 (May 20, 2025), v9.1.1 (May 21, 2025):
- Removed debugging comments for a streamlined production release
- Updated version string across the application

- Modified path for all configuration files to be stored in /data/ subdirectory of current directory
- Improved application portability by not relying on user's home directory
- Ensured consistent path handling for settings, models, and database storage

Changes in v9.0 (May 19, 2025):
- Added SQLite database to store all project data, code generations, and audit results
- Implemented automatic tracking of all coding projects with timestamps
- Stored detailed information for both Multiple Correction and Multiple Creation modes
- Captured code, audits, error counts, and fix statistics for easy analysis and retrieval

Changes in v8.5 (May 19, 2025):
- Added unified Settings dialog with a cog icon button
- Created Files tab for configuring output directory
- Combined API Keys and Models configuration into a single settings tab
- Simplified main UI by removing separate API Keys and Check Models buttons

Changes in v8.4.1 (May 19, 2025):
- Fixed tooltip display for listbox widgets
- Improved iteration display to show progress (current/total)
- Renamed "Program file name" field to "Project name" for clarity


Changes in v8.4 (May 19, 2025):
- Added tooltips to all buttons and field labels with a 0.5-second delay
- Improved usability with descriptive tooltip text for all UI elements
- Created Tooltip class to handle tooltip display and management

Changes in v8.3 (May 19, 2025):
- Replaced JSON structures with Delimiter-Based Format for exchange between LLMs
- Improved parsing reliability by eliminating JSON escaping and quoting issues
- Updated correction prompts to use ###CODE### and ###CORRECTIONS### delimiters
- Enhanced extraction logic to handle the new format consistently

Changes in v8.1.1 (May 17, 2025):
- Increased Run app dialog window size to properly accommodate all interface elements

Changes in v8.1 (May 17, 2025):
- Reorganized button layout for improved usability
- Changed the naming convention for final iteration files to use "-Final" suffix 
- UI improvements for workflow efficiency

Changes in v8.0 (May 17, 2025):
- Added Run app button to execute Python applications
- Implemented file browser to select .py or .trm files
- Added file viewer functionality for viewing code without editing
- Added subprocess execution for running Python applications
- Terminal output is automatically saved to .trm files after execution

Changes in v7.6 (May 16, 2025):
- Added Clear button to Program Description field for easy content clearing
- Fixed bug where previously selected models appeared as phantoms in Multiple Creation mode

Changes in v7.5 (May 16, 2025):
- Added "Check Models" button to dynamically discover available models from all providers
- Users can now select which models to use without modifying code
- Models are grouped by vendor in selection dialog
- Automatic discovery of new models when vendors release them
- Persistent storage of user model selections

Changes in v7.4 (May 11, 2025):
- Added Exit button to the main window for easier application closing

Changes in v7.3 (May 11, 2025):
- Renamed "Configure API Keys" button to "API Keys and Models" for clarity
- Removed manual API key entry from the configuration dialog
- Now API keys can only be set via environment variables or the APIKeys file
- Suppressed httpx and google_genai log messages in terminal output
- Added printing of available models for each provider

Changes in v7.2 (May 11, 2025):
- Added support for retrieving API keys from environment variables
- Now supports OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, DEEPSEEK_API_KEY
- Environment variables take precedence over keys stored in the APIKeys file

Changes in v7.1 (May 10, 2025):
- Added functionality to check which LLMs are available for each API key
- Modified UI to only show LLMs that are available with the current API key
- Added API key validation for each provider
- Improved error handling for API key validation


"""

import os
import re
import sys
import json
import time
import logging
import hashlib
import argparse
import asyncio
import requests
import configparser
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from string import Template
from concurrent.futures import ThreadPoolExecutor
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import threading
import subprocess

# Try to import optional dependencies
try:
    import keyring
    KEYRING_AVAILABLE = True
except ImportError:
    KEYRING_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    FERNET_AVAILABLE = True
except ImportError:
    FERNET_AVAILABLE = False

try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
    from rich.syntax import Syntax
    from rich.panel import Panel
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

try:
    import diskcache
    DISKCACHE_AVAILABLE = True
except ImportError:
    DISKCACHE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("codingapi")

# Silence httpx and google_genai loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)

# Global Configuration Constants
MAX_OUTPUT_TOKENS = 50000  # Maximum number of output tokens for all LLM request
THINKING_BUDGET_TOKENS = 32000    # Thinking budget for models that support thinking/reasoning capabilities

# LLM Provider Constants - Updated with current models as of May 2025
LLM_MAP = {
    # OpenAI models - Current as of 2025
    "OpenAI o3":            {"model": "o3-2025-04-16", "family": "OpenAI", "temperature_allowed": False},
    "OpenAI o4-mini":       {"model": "o4-mini", "family": "OpenAI", "temperature_allowed": False},
    "OpenAI GPT4o":         {"model": "gpt-4o", "family": "OpenAI", "temperature_allowed": True},
    "OpenAI GPT-4.1":       {"model": "gpt-4.1", "family": "OpenAI", "temperature_allowed": True},
    "OpenAI GPT-4.1-mini":  {"model": "gpt-4.1-mini", "family": "OpenAI", "temperature_allowed": True},
    
    # Claude models - Current as of 2025
    "Claude 3.7 Sonnet":    {"model": "claude-3-7-sonnet-20250219", "family": "Claude", "temperature_allowed": True},
    
    # Gemini models - Current as of 2025
    "Gemini 2.5 Pro":       {"model": "gemini-2.5-pro-preview-03-25", "family": "Gemini", "temperature_allowed": True},
    "Gemini 2.0 Flash":     {"model": "gemini-2.0-flash", "family": "Gemini", "temperature_allowed": True},
    
    # DeepSeek models
    "DeepSeek R1":          {"model": "deepseek-reasoner", "family": "DeepSeek", "temperature_allowed": True},
}

# Model grouping by family - Updated with current models
OPENAI_MODELS = [
    "OpenAI o4-mini",
    "OpenAI o3",
    "OpenAI GPT4o",
    "OpenAI GPT-4.1",
    "OpenAI GPT-4.1-mini"
]

CLAUDE_MODELS = [
    "Claude 3.7 Sonnet"
]

GEMINI_MODELS = [
    "Gemini 2.5 Pro",
    "Gemini 2.0 Flash"
]

DEEPSEEK_MODELS = [
    "DeepSeek R1"
]

DYNAMIC_MODEL_REGISTRY = {}


# Grouped models for coding dropdown (OpenAI first)
CODING_LLM_OPTIONS = OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS + DEEPSEEK_MODELS

# Grouped models for auditing dropdown (Claude first)
AUDITING_LLM_OPTIONS = CLAUDE_MODELS + OPENAI_MODELS + GEMINI_MODELS + DEEPSEEK_MODELS

# Optimal Parameter Constants for different model families - Updated based on 2025 landscape
# Based on research for code generation tasks

# OpenAI Parameters - Updated for o-series and GPT models
OPENAI_PARAMS = {
    # o3 parameters - Uses reasoning parameter instead of temperature
    "o3-2025-04-16": {
        "web_app": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
        "data_science": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
        "algorithm": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
        "default": {"reasoning_effort": "high", "reasoning_summary": "detailed"}
    },
    # o4-mini parameters - Uses reasoning parameter instead of temperature
    "o4-mini": {
        "web_app": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
        "data_science": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
        "algorithm": {"reasoning_effort": "high", "reasoning_summary": "detailed"},
        "default": {"reasoning_effort": "high", "reasoning_summary": "detailed"}
    },
    # GPT-4.1 parameters - Uses temperature
    "gpt-4.1": {
        "web_app": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "data_science": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "algorithm": {"temperature": 0.1, "top_p": 0.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "default": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    },
    # GPT-4o parameters - Uses temperature
    "gpt-4o": {
        "web_app": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "data_science": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "algorithm": {"temperature": 0.1, "top_p": 0.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "default": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    },
    # GPT-4.1-mini parameters - Uses temperature
    "gpt-4.1-mini": {
        "web_app": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "data_science": {"temperature": 0.3, "top_p": 0.3, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "algorithm": {"temperature": 0.1, "top_p": 0.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "default": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    },
    # Default parameters for other OpenAI models
    "default": {
        "web_app": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "data_science": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "algorithm": {"temperature": 0.1, "top_p": 0.1, "frequency_penalty": 0.0, "presence_penalty": 0.0},
        "default": {"temperature": 0.2, "top_p": 0.2, "frequency_penalty": 0.0, "presence_penalty": 0.0}
    }
}

# Claude Parameters - Updated for Claude 3.7 Sonnet
CLAUDE_PARAMS = {
    # Claude 3.7 Sonnet parameters - Optimized with thinking budget
    "claude-3-7-sonnet-20250219": {
        "web_app": {"temperature": 0.2, "thinking_budget": 8192},
        "data_science": {"temperature": 0.1, "thinking_budget": 12000},
        "algorithm": {"temperature": 0.0, "thinking_budget": 16000},
        "default": {"temperature": 0.1, "thinking_budget": 8192}
    },
    # Default parameters for other Claude models
    "default": {
        "web_app": {"temperature": 0.2, "thinking_budget": 0},
        "data_science": {"temperature": 0.1, "thinking_budget": 0},
        "algorithm": {"temperature": 0.0, "thinking_budget": 0},
        "default": {"temperature": 0.1, "thinking_budget": 0}
    }
}

# Gemini Parameters - Updated for Gemini 2.5 Pro and 2.0 Flash
GEMINI_PARAMS = {
    # Gemini 2.5 Pro parameters - Enhanced with thinking budget
    "gemini-2.5-pro-exp-03-25": {
        "web_app": {"temperature": 0.2, "top_p": 0.9, "top_k": 30, "thinking_budget": 8192},
        "data_science": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "thinking_budget": 12288},
        "algorithm": {"temperature": 0.1, "top_p": 0.8, "top_k": 15, "thinking_budget": 16384},
        "default": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "thinking_budget": 8192}
    },
    # Gemini 2.0 Flash parameters
    "gemini-2.0-flash": {
        "web_app": {"temperature": 0.2, "top_p": 0.9, "top_k": 30, "thinking_budget": 4096},
        "data_science": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "thinking_budget": 6144},
        "algorithm": {"temperature": 0.1, "top_p": 0.8, "top_k": 15, "thinking_budget": 8192},
        "default": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "thinking_budget": 4096}
    },
    # Default parameters for other Gemini models
    "default": {
        "web_app": {"temperature": 0.2, "top_p": 0.9, "top_k": 30, "thinking_budget": 4096},
        "data_science": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "thinking_budget": 6144},
        "algorithm": {"temperature": 0.1, "top_p": 0.8, "top_k": 15, "thinking_budget": 8192},
        "default": {"temperature": 0.2, "top_p": 0.9, "top_k": 20, "thinking_budget": 4096}
    }
}

# DeepSeek Parameters - Updated with optimal temperature settings
DEEPSEEK_PARAMS = {
    # DeepSeek R1 parameters - Higher temperature (0.6) is optimal
    "deepseek-reasoner": {
        "web_app": {"temperature": 0.6, "top_p": 0.95},
        "data_science": {"temperature": 0.6, "top_p": 0.95},
        "algorithm": {"temperature": 0.6, "top_p": 0.95},
        "default": {"temperature": 0.6, "top_p": 0.95}
    },
    # Default parameters for other DeepSeek models
    "default": {
        "web_app": {"temperature": 0.6, "top_p": 0.95},
        "data_science": {"temperature": 0.6, "top_p": 0.95},
        "algorithm": {"temperature": 0.6, "top_p": 0.95},
        "default": {"temperature": 0.6, "top_p": 0.95}
    }
}

def load_dynamic_models():
    """Load previously discovered dynamic models from persistent storage."""
    try:
        dynamic_models_file = os.path.join(get_config_dir(), "dynamic_models.json")
        if os.path.exists(dynamic_models_file):
            with open(dynamic_models_file, 'r') as f:
                saved_models = json.load(f)
                DYNAMIC_MODEL_REGISTRY.update(saved_models)
                logger.debug(f"Loaded {len(saved_models)} dynamic models from storage")
    except Exception as e:
        logger.error(f"Error loading dynamic models: {e}")

def save_dynamic_models():
    """Save the current dynamic model registry to persistent storage."""
    try:
        config_dir = get_config_dir()
        os.makedirs(config_dir, exist_ok=True)
        dynamic_models_file = os.path.join(config_dir, "dynamic_models.json")
        
        with open(dynamic_models_file, 'w') as f:
            json.dump(DYNAMIC_MODEL_REGISTRY, f, indent=2)
        logger.debug(f"Saved {len(DYNAMIC_MODEL_REGISTRY)} dynamic models to storage")
    except Exception as e:
        logger.error(f"Error saving dynamic models: {e}")

def is_model_already_registered(display_name):
    """Check if a model is already registered in either LLM_MAP or DYNAMIC_MODEL_REGISTRY."""
    return display_name in LLM_MAP or display_name in DYNAMIC_MODEL_REGISTRY

def extract_date_from_model_id(model_id):
    """
    Extract date from model ID if possible.
    Returns date as YYYYMMDD integer for comparison, or None if no date found.
    """
    import re
    
    # Pattern for dates in various formats
    # Claude format: claude-3-7-sonnet-20250219
    # Look for 8-digit date pattern YYYYMMDD
    date_match = re.search(r'(\d{8})', model_id)
    if date_match:
        return int(date_match.group(1))
    
    # Alternative format YYYY-MM-DD
    date_match = re.search(r'(\d{4})-(\d{2})-(\d{2})', model_id)
    if date_match:
        year, month, day = date_match.groups()
        return int(f"{year}{month}{day}")
    
    return None

def get_newest_model_date_for_provider(provider_name):
    """
    Get the date of the newest model we have hardcoded for a provider.
    Returns date as YYYYMMDD integer or a default old date if none found.
    """
    newest_date = 0
    
    # Check all models in LLM_MAP for this provider
    for model_name, info in LLM_MAP.items():
        if info.get("family") == provider_name:
            model_id = info.get("model", "")
            date = extract_date_from_model_id(model_id)
            if date and date > newest_date:
                newest_date = date
    
    # If no date found, return a reasonable cutoff date
    # Using the actual dates from our newest models
    if newest_date == 0:
        if provider_name == "Claude":
            return 20250219  # Claude 3.7 Sonnet date
        elif provider_name == "OpenAI":
            return 20250416  # o3 date
        elif provider_name == "Gemini":
            return 20250325  # Gemini 2.5 Pro date
        elif provider_name == "DeepSeek":
            return 20250101  # Reasonable recent date
    
    return newest_date

def is_model_in_llm_map(display_name):
    """
    Check if a model display name is already in LLM_MAP.
    """
    return display_name in LLM_MAP

def is_model_newer_than_existing(model_id, provider_name, display_name=None):
    """
    Check if a model is newer than our existing models for a provider.
    Always returns True for models already in LLM_MAP.
    """
    # Always include models that are in LLM_MAP
    if display_name and is_model_in_llm_map(display_name):
        return True
    
    model_date = extract_date_from_model_id(model_id)
    if not model_date:
        # If we can't extract a date, assume it's an older model
        # This prevents old models like Claude 2.0, 2.1 from being included
        return False
    
    newest_existing_date = get_newest_model_date_for_provider(provider_name)
    return model_date > newest_existing_date

# Function to determine task type from description (basic heuristic)
def determine_task_type(description, language):
    """
    Determines the type of coding task based on the description.
    Returns one of: 'web_app', 'data_science', 'algorithm', or 'default'
    """
    description_lower = description.lower()
    
    # Check for web app keywords
    web_app_keywords = ['web', 'html', 'css', 'javascript', 'frontend', 'backend', 'api', 
                       'http', 'server', 'client', 'react', 'angular', 'vue', 'flask', 
                       'django', 'express', 'website', 'browser', 'webpage']
    
    # Check for data science keywords
    data_science_keywords = ['data', 'analysis', 'analytics', 'visualization', 'plot', 
                            'pandas', 'numpy', 'matplotlib', 'seaborn', 'scipy', 'statistics', 
                            'ml', 'machine learning', 'sklearn', 'tensorflow', 'keras', 
                            'regression', 'classification', 'clustering', 'prediction']
    
    # Check for algorithm keywords
    algorithm_keywords = ['algorithm', 'sort', 'search', 'graph', 'tree', 'optimize', 
                         'dynamic programming', 'recursion', 'efficiency', 'complexity', 
                         'data structure', 'binary', 'heap', 'queue', 'stack', 'linked list']
    
    # Count matches for each category
    web_app_count = sum(1 for keyword in web_app_keywords if keyword in description_lower)
    data_science_count = sum(1 for keyword in data_science_keywords if keyword in description_lower)
    algorithm_count = sum(1 for keyword in algorithm_keywords if keyword in description_lower)
    
    # Determine the category with the most matches
    max_count = max(web_app_count, data_science_count, algorithm_count)
    
    if max_count == 0:
        # No clear category, use some language-based heuristics
        if language.lower() in ['javascript', 'typescript', 'html', 'css', 'php']:
            return 'web_app'
        elif language.lower() in ['r', 'julia']:
            return 'data_science'
        elif language.lower() in ['c', 'c++', 'rust', 'go']:
            return 'algorithm'
        else:
            return 'default'
    
    # Return the category with the most matches
    if max_count == web_app_count:
        return 'web_app'
    elif max_count == data_science_count:
        return 'data_science'
    elif max_count == algorithm_count:
        return 'algorithm'
    else:
        return 'default'

def get_config_dir():
    """Get the configuration directory path - /data/ in current directory."""
    config_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(config_dir, exist_ok=True)
    return config_dir



#-----------------------------------------------------------------------------
# Custom Exceptions
#-----------------------------------------------------------------------------

class CodingAPIError(Exception):
    """Base exception for all CodingAPI errors."""
    pass

class ConfigurationError(CodingAPIError):
    """Raised when there's an issue with the configuration."""
    pass

class AuthenticationError(CodingAPIError):
    """Raised when authentication fails."""
    pass

class APIConnectionError(CodingAPIError):
    """Raised when connection to the API fails."""
    pass

class APIResponseError(CodingAPIError):
    """Raised when the API returns an error response."""
    def __init__(self, status_code, message, response_body=None):
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"API error (status {status_code}): {message}")

class RateLimitError(APIResponseError):
    """Raised when rate limit is exceeded."""
    def __init__(self, status_code, message, retry_after=None, response_body=None):
        self.retry_after = retry_after
        super().__init__(status_code, message, response_body)

class ValidationError(CodingAPIError):
    """Raised when input validation fails."""
    pass

#-----------------------------------------------------------------------------
# Response Models
#-----------------------------------------------------------------------------

@dataclass
class CodeGenerationResponse:
    """Standardized response format for code generation."""
    code: str                   # The generated code
    language: str               # Programming language of the code
    model: str                  # The LLM model used
    provider: str               # Provider name (openai, anthropic, etc.)
    token_usage: Dict[str, int] = None  # Token usage statistics if available
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def __str__(self) -> str:
        return self.code
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "code": self.code,
            "language": self.language,
            "model": self.model,
            "provider": self.provider,
            "token_usage": self.token_usage,
            "metadata": self.metadata
        }

#-----------------------------------------------------------------------------
# Utilities
#-----------------------------------------------------------------------------

class SecureConfig:
    """Secure configuration manager for API credentials."""
    
    def __init__(self, app_name: str = "CodingAPI"):
        """Initialize the secure configuration manager."""
        self.app_name = app_name
        self._encryption_key = self._get_or_create_encryption_key()
        self._cipher = None
        self._available_models = {}  # Cache for available models by provider
        if FERNET_AVAILABLE:
            try:
                self._cipher = Fernet(self._encryption_key)
            except Exception as e:
                logger.warning(f"Failed to create cipher: {str(e)}")
    
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key from secure storage."""
        # If keyring is available, try to use it
        if KEYRING_AVAILABLE:
            try:
                key = keyring.get_password(self.app_name, "encryption_key")
                if not key:
                    # Generate a new key if none exists
                    if FERNET_AVAILABLE:
                        key = Fernet.generate_key().decode('utf-8')
                    else:
                        # Fallback to a secure random key
                        import base64
                        import os
                        key = base64.urlsafe_b64encode(os.urandom(32)).decode('utf-8')
                    keyring.set_password(self.app_name, "encryption_key", key)
                return key.encode('utf-8')
            except Exception as e:
                logger.warning(f"Could not access secure storage: {str(e)}. Using environment-based fallback.")
        
        # Fallback to environment variable or default
        env_key = os.environ.get("CODINGAPI_ENCRYPTION_KEY")
        if env_key:
            return env_key.encode('utf-8')
        
        # Last resort: Use a default key (less secure)
        return b'hTXwB5kV2rsp3kkWvaCvXFy_B5HgU-03UdOtQdxpx9A='
    
    def get_api_key(self, provider_name: str) -> Optional[str]:
        """Get API key from secure storage or environment."""
        # First try environment variable (highest priority)
        # Special case for Claude provider which uses ANTHROPIC_API_KEY
        if provider_name == "Claude":
            env_key = os.environ.get("ANTHROPIC_API_KEY")
        else:
            env_key = os.environ.get(f"{provider_name.upper()}_API_KEY")
        
        if env_key:
            return env_key
            
        # Then try secure storage if available
        if KEYRING_AVAILABLE:
            try:
                key = keyring.get_password(self.app_name, f"{provider_name}_api_key")
                if key and self._cipher:
                    return self._cipher.decrypt(key.encode('utf-8')).decode('utf-8')
                elif key:
                    return key
            except Exception as e:
                logger.warning(f"Failed to retrieve API key from secure storage: {str(e)}")
        
        # Finally, try the config file
        config_file = os.path.join(get_config_dir(), "config.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    config = json.load(f)
                key = config.get("api_keys", {}).get(provider_name)
                if key and self._cipher:
                    try:
                        return self._cipher.decrypt(key.encode('utf-8')).decode('utf-8')
                    except:
                        # Maybe it's not encrypted
                        return key
                elif key:
                    return key
            except Exception as e:
                logger.warning(f"Failed to load config file: {str(e)}")
        
        # Finally, fall back to APIKeys file for backward compatibility
        try:
            api_keys = get_api_keys()
            return api_keys.get(provider_name, None)
        except:
            pass
            
        return None
    
    def set_api_key(self, provider_name: str, api_key: str) -> bool:
        """Store API key in secure storage."""
        # Try to encrypt the API key if possible
        encrypted_key = api_key
        if self._cipher:
            try:
                encrypted_key = self._cipher.encrypt(api_key.encode('utf-8')).decode('utf-8')
            except Exception as e:
                logger.warning(f"Failed to encrypt API key: {str(e)}")
        
        # Try to store in keyring if available
        if KEYRING_AVAILABLE:
            try:
                keyring.set_password(self.app_name, f"{provider_name}_api_key", encrypted_key)
                # Clear the cached available models since the API key changed
                if provider_name in self._available_models:
                    del self._available_models[provider_name]
                return True
            except Exception as e:
                logger.warning(f"Failed to store API key in secure storage: {str(e)}")
        
        # Fallback to config file
        try:
            config_dir = get_config_dir()
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "config.json")
            
            # Load existing config or create new
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    config = json.load(f)
            else:
                config = {"api_keys": {}}
            
            # Update config with encrypted key
            if "api_keys" not in config:
                config["api_keys"] = {}
            config["api_keys"][provider_name] = encrypted_key
            
            # Save config with restricted permissions
            with open(config_file, 'w') as f:
                json.dump(config, f)
            
            # Try to set restrictive permissions on Unix-like systems
            try:
                import stat
                os.chmod(config_file, stat.S_IRUSR | stat.S_IWUSR)
            except:
                pass
            
            # Clear the cached available models since the API key changed
            if provider_name in self._available_models:
                del self._available_models[provider_name]
                
            return True
        except Exception as e:
            logger.error(f"Failed to store API key in config file: {str(e)}")
            return False
        
    def get_available_models(self, provider_name: str) -> List[str]:
        """
        Get list of available LLM models for the given provider based on API key.
        
        Args:
            provider_name: The provider name (OpenAI, Claude, Gemini, DeepSeek)
            
        Returns:
            List of available model names for the provider
        """
        # Check if we have cached results
        if provider_name in self._available_models:
            return self._available_models[provider_name]
            
        # Get the API key
        api_key = self.get_api_key(provider_name)
        if not api_key:
            # No API key, no available models
            logger.warning(f"API key for {provider_name} is missing. Models from this provider will not be available.")
            self._available_models[provider_name] = []
            return []
            
        # Check model availability based on provider
        available_models = []
        try:
            if provider_name == "OpenAI":
                available_models = self._check_openai_models(api_key)
            elif provider_name == "Claude":
                available_models = self._check_claude_models(api_key)
            elif provider_name == "Gemini":
                available_models = self._check_gemini_models(api_key)
            elif provider_name == "DeepSeek":
                available_models = self._check_deepseek_models(api_key)
                
            # Print available models to terminal
            if available_models:
                print(f"Available {provider_name} models: {', '.join(available_models)}")
            else:
                print(f"No available models found for {provider_name}")
        except Exception as e:
            logger.error(f"Error checking available models for {provider_name}: {str(e)}")
            # If there's an error, assume no models are available
            self._available_models[provider_name] = []
            return []
            
        # Cache the results
        self._available_models[provider_name] = available_models
        return available_models
    
    def _check_openai_models(self, api_key: str) -> List[str]:
        """Check which OpenAI models are available with the given API key."""
        try:
            # Create a temporary client to check model access
            client = create_openai_client(api_key)
            
            # Try to list models to verify API key works
            try:
                client.models.list()
            except Exception as e:
                if "401" in str(e):  # Unauthorized
                    return []
                # For other errors, assume the API key is valid but with limited access
            
            # At this point, the API key is valid, but we need to check which specific models are available
            # We'll use a simple approach - try to do a tiny completion with the most expensive model
            # If it works, we assume all models are available, otherwise we assume only certain tiers
            
            # Try GPT-4.1 (most expensive) as a test
            try:
                # Small completion to check access
                response = client.chat.completions.create(
                    model="gpt-4.1",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                # If we get here, assume full access to all models
                return [model for model in OPENAI_MODELS]
            except Exception as e:
                error_msg = str(e).lower()
                # Check error message for clues about which tier this API key belongs to
                if "not have access" in error_msg or "permission" in error_msg or "unauthorized" in error_msg:
                    # Limited tier - let's try a less expensive model (GPT-4o)
                    try:
                        response = client.chat.completions.create(
                            model="gpt-4o",
                            messages=[{"role": "user", "content": "Hi"}],
                            max_tokens=1
                        )
                        # Mid-tier access - exclude the highest tier models
                        return ["OpenAI GPT4o", "OpenAI GPT-4.1-mini", "OpenAI o3", "OpenAI o4-mini"]
                    except:
                        # Try the lowest tier model
                        try:
                            response = client.chat.completions.create(
                                model="gpt-3.5-turbo",  # Fallback to check if basic access works
                                messages=[{"role": "user", "content": "Hi"}],
                                max_tokens=1
                            )
                            # Basic tier access
                            return ["OpenAI o3", "OpenAI o4-mini"]
                        except:
                            # No model access at all
                            return []
                else:
                    # Some other error - assume full access
                    return [model for model in OPENAI_MODELS]
        except Exception as e:
            logger.error(f"Error checking OpenAI models: {str(e)}")
            return []
        
    def _check_claude_models(self, api_key: str) -> List[str]:
        """Check which Claude models are available with the given API key."""
        try:
            # Create a temporary client to check model access
            client = create_claude_client(api_key)
            
            # Try to list models
            try:
                response = client.models.list(limit=20)
                
                # Extract model display names from the response
                available_models = []
                if hasattr(response, 'data') and response.data:
                    for model in response.data:
                        if hasattr(model, 'display_name') and model.display_name:
                            # Check if this model is in our known CLAUDE_MODELS list
                            if model.display_name in CLAUDE_MODELS:
                                available_models.append(model.display_name)
                
                return available_models
                
            except Exception as e:
                error_msg = str(e).lower()
                if "401" in error_msg or "unauthorized" in error_msg or "invalid" in error_msg:
                    # Invalid API key
                    return []
                else:
                    # Some other error, but API key might be valid
                    # Return all known Claude models as a fallback
                    return [model for model in CLAUDE_MODELS]
                    
        except Exception as e:
            logger.error(f"Error checking Claude models: {str(e)}")
            return []
    
    def _check_gemini_models(self, api_key: str) -> List[str]:
        """Check which Gemini models are available with the given API key."""
        try:
            # Create a temporary client to check model access
            client = create_gemini_client(api_key)
            
            # Try to generate with Gemini 2.5 Pro to see if it's available
            from google import genai
            from google.genai import types
            
            try:
                # Minimal generation to check access
                response = client.models.generate_content(
                    model="gemini-2.5-pro-exp-03-25",
                    config=types.GenerateContentConfig(temperature=0),
                    contents="Hi"
                )
                # If we get here, assume all Gemini models are available
                return [model for model in GEMINI_MODELS]
            except Exception as e:
                error_msg = str(e).lower()
                if "permission" in error_msg or "access" in error_msg or "unauthorized" in error_msg:
                    # Try Gemini 2.0 Flash (lower tier)
                    try:
                        response = client.models.generate_content(
                            model="gemini-2.0-flash",
                            config=types.GenerateContentConfig(temperature=0),
                            contents="Hi"
                        )
                        # Access to Flash only
                        return ["Gemini 2.0 Flash"]
                    except:
                        # No Gemini access
                        return []
                else:
                    # Some other error - assume full access
                    return [model for model in GEMINI_MODELS]
        except Exception as e:
            logger.error(f"Error checking Gemini models: {str(e)}")
            return []
    
    def _check_deepseek_models(self, api_key: str) -> List[str]:
        """Check which DeepSeek models are available with the given API key."""
        try:
            # Create a temporary client to check model access
            client = create_deepseek_client(api_key)
            
            # Try a simple completion to verify API key works
            try:
                response = client.chat.completions.create(
                    model="deepseek-reasoner",
                    messages=[{"role": "user", "content": "Hi"}],
                    max_tokens=1
                )
                # If we get here, assume all DeepSeek models are available
                return [model for model in DEEPSEEK_MODELS]
            except Exception as e:
                error_msg = str(e).lower()
                if "not have access" in error_msg or "permission" in error_msg or "unauthorized" in error_msg:
                    # No access to DeepSeek models
                    return []
                else:
                    # Some other error - assume access is valid
                    return [model for model in DEEPSEEK_MODELS]
        except Exception as e:
            logger.error(f"Error checking DeepSeek models: {str(e)}")
            return []

    def get_all_available_models(self) -> Dict[str, List[str]]:
        """
        Get all available models for all providers.
        
        Returns:
            Dictionary with provider names as keys and lists of available model names as values
        """
        providers = ["OpenAI", "Claude", "Gemini", "DeepSeek"]
        result = {}
        
        print("\n=== Available Models ===")
        for provider in providers:
            result[provider] = self.get_available_models(provider)
            
        print("========================\n")
        return result

    def get_all_available_models_from_api(self) -> Dict[str, List[str]]:
        """
        Get all available models directly from each provider's API.
        This discovers models that might not be in our predefined lists.
        
        Returns:
            Dictionary with provider names as keys and lists of available model names as values
        """
        providers = ["OpenAI", "Claude", "Gemini", "DeepSeek"]
        result = {}
        
        print("\n=== Discovering All Available Models ===")
        for provider in providers:
            print(f"Checking {provider}...")
            try:
                result[provider] = self._discover_models_for_provider(provider)
            except Exception as e:
                logger.error(f"Error discovering models for {provider}: {str(e)}")
                result[provider] = []
                
        print("========================================\n")
        return result

    def _discover_models_for_provider(self, provider_name: str) -> List[str]:
        """
        Discover all available models for a specific provider.
        
        Args:
            provider_name: The provider name (OpenAI, Claude, Gemini, DeepSeek)
            
        Returns:
            List of available model names for the provider
        """
        api_key = self.get_api_key(provider_name)
        if not api_key:
            logger.warning(f"API key for {provider_name} is missing. Cannot discover models.")
            return []
            
        try:
            if provider_name == "OpenAI":
                return self._discover_openai_models(api_key)
            elif provider_name == "Claude":
                return self._discover_claude_models(api_key)
            elif provider_name == "Gemini":
                return self._discover_gemini_models(api_key)
            elif provider_name == "DeepSeek":
                return self._discover_deepseek_models(api_key)
        except Exception as e:
            logger.error(f"Error discovering models for {provider_name}: {str(e)}")
            return []
        
        return []

    def _discover_openai_models(self, api_key: str) -> List[str]:
        """Discover all available OpenAI models."""
        try:
            client = create_openai_client(api_key)
            response = client.models.list()
            
            # Get the newest model date we have for OpenAI
            newest_existing_date = get_newest_model_date_for_provider("OpenAI")
            
            # Extract model IDs and filter for relevant models
            all_models = [model.id for model in response.data]
            
            # Filter to get only relevant models (GPT, O-series, etc.)
            relevant_models = []
            seen_display_names = set()  # Track display names to avoid duplicates
            
            # First, add all models from LLM_MAP for this provider
            for model_name, info in LLM_MAP.items():
                if info.get("family") == "OpenAI":
                    relevant_models.append(model_name)
                    seen_display_names.add(model_name)
            
            # Then check discovered models
            models_added = False
            for model_id in all_models:
                if any(pattern in model_id.lower() for pattern in ['gpt-4', 'o3', 'o4', 'gpt-3.5']):
                    # Convert model ID to display name
                    display_name = self._get_display_name_for_openai(model_id)
                    
                    # Check if this model is newer than our existing models
                    if is_model_newer_than_existing(model_id, "OpenAI", display_name):
                        # If we don't have a display name mapping, create one
                        if not display_name:
                            # Generate a display name from the model ID
                            display_name = f"OpenAI {model_id}"
                        
                        # Check if this model is already registered
                        if not is_model_already_registered(display_name):
                            # Add to dynamic registry
                            DYNAMIC_MODEL_REGISTRY[display_name] = {
                                "model": model_id,
                                "family": "OpenAI",
                                "temperature_allowed": True
                            }
                            logger.info(f"Dynamically registered new OpenAI model: {display_name} -> {model_id}")
                            models_added = True
                        
                        if display_name and display_name not in seen_display_names:
                            relevant_models.append(display_name)
                            seen_display_names.add(display_name)
                    else:
                        logger.debug(f"Skipping older OpenAI model: {model_id}")
            
            # Save dynamic models if any were added
            if models_added:
                save_dynamic_models()
            
            return relevant_models
        except Exception as e:
            logger.error(f"Error discovering OpenAI models: {str(e)}")
            return []

    def _discover_claude_models(self, api_key: str) -> List[str]:
        """Discover all available Claude models."""
        try:
            client = create_claude_client(api_key)
            
            # Get the newest model date we have for Claude
            newest_existing_date = get_newest_model_date_for_provider("Claude")
            
            # List all available models
            all_models = []
            has_more = True
            after_id = None
            
            # First, add all models from LLM_MAP for this provider
            for model_name, info in LLM_MAP.items():
                if info.get("family") == "Claude":
                    all_models.append(model_name)
            
            models_added = False
            while has_more:
                try:
                    # Make API call with pagination
                    if after_id:
                        response = client.models.list(limit=100, after_id=after_id)
                    else:
                        response = client.models.list(limit=100)
                    
                    # Extract models from response
                    if hasattr(response, 'data') and response.data:
                        for model in response.data:
                            if hasattr(model, 'display_name') and model.display_name and hasattr(model, 'id'):
                                # Skip if already in our list
                                if model.display_name in all_models:
                                    continue
                                    
                                # Check if this model is newer than our existing models
                                if is_model_newer_than_existing(model.id, "Claude", model.display_name):
                                    all_models.append(model.display_name)
                                    
                                    # If this model isn't already registered, add it to dynamic registry
                                    if not is_model_already_registered(model.display_name):
                                        # Special handling for Claude Opus 4.0
                                        if "opus-4" in model.id.lower():
                                            DYNAMIC_MODEL_REGISTRY[model.display_name] = {
                                                "model": model.id,
                                                "family": "Claude",
                                                "temperature_allowed": True,
                                                "max_tokens": 32000  # Special limit for Opus 4.0
                                            }
                                        else:
                                            DYNAMIC_MODEL_REGISTRY[model.display_name] = {
                                                "model": model.id,
                                                "family": "Claude",
                                                "temperature_allowed": True
                                            }
                                        logger.info(f"Dynamically registered new Claude model: {model.display_name} -> {model.id}")
                                        models_added = True
                                else:
                                    logger.debug(f"Skipping older Claude model: {model.display_name} ({model.id})")
                    
                    # Check if there are more pages
                    has_more = hasattr(response, 'has_more') and response.has_more
                    if has_more and hasattr(response, 'last_id'):
                        after_id = response.last_id
                    else:
                        has_more = False
                        
                except Exception as e:
                    logger.error(f"Error during Claude model pagination: {str(e)}")
                    break
            
            # Save dynamic models if any were added
            if models_added:
                save_dynamic_models()
            
            # Sort models for consistent display
            all_models.sort()
            
            return all_models
            
        except Exception as e:
            logger.error(f"Error discovering Claude models: {str(e)}")
            return []

    def _discover_gemini_models(self, api_key: str) -> List[str]:
        """Discover all available Gemini models."""
        try:
            client = create_gemini_client(api_key)
            
            # Get the newest model date we have for Gemini
            newest_existing_date = get_newest_model_date_for_provider("Gemini")
            
            # Gemini API to list models
            from google import genai
            
            # List all available models
            models = client.models.list()
            
            relevant_models = []
            seen_display_names = set()
            
            # First, add all models from LLM_MAP for this provider
            for model_name, info in LLM_MAP.items():
                if info.get("family") == "Gemini":
                    relevant_models.append(model_name)
                    seen_display_names.add(model_name)
            
            # Then check discovered models
            models_added = False
            for model in models:
                # Extract model name and convert to display name
                model_name = model.name.split('/')[-1]  # Get last part after slash
                if any(pattern in model_name.lower() for pattern in ['gemini-2', 'gemini-1.5']):
                    # Check if this model is newer than our existing models
                    if is_model_newer_than_existing(model_name, "Gemini"):
                        display_name = self._get_display_name_for_gemini(model_name)
                        
                        # If we don't have a display name mapping, create one
                        if not display_name:
                            # Generate a display name from the model ID
                            display_name = f"Gemini {model_name}"
                        
                        # Check if this model is already registered
                        if not is_model_already_registered(display_name):
                            # Add to dynamic registry
                            DYNAMIC_MODEL_REGISTRY[display_name] = {
                                "model": model_name,
                                "family": "Gemini",
                                "temperature_allowed": True
                            }
                            logger.info(f"Dynamically registered new Gemini model: {display_name} -> {model_name}")
                            models_added = True
                        
                        if display_name and display_name not in seen_display_names:
                            relevant_models.append(display_name)
                            seen_display_names.add(display_name)
                    else:
                        logger.debug(f"Skipping older Gemini model: {model_name}")
            
            # Save dynamic models if any were added
            if models_added:
                save_dynamic_models()
            
            return relevant_models
        except Exception as e:
            logger.error(f"Error discovering Gemini models: {str(e)}")
            return []

    def _discover_deepseek_models(self, api_key: str) -> List[str]:
        """Discover all available DeepSeek models."""
        try:
            client = create_deepseek_client(api_key)
            
            # Get the newest model date we have for DeepSeek
            newest_existing_date = get_newest_model_date_for_provider("DeepSeek")
            
            # List models endpoint
            response = client.models.list()
            
            # Extract model IDs and convert to display names
            relevant_models = []
            seen_display_names = set()
            
            # First, add all models from LLM_MAP for this provider
            for model_name, info in LLM_MAP.items():
                if info.get("family") == "DeepSeek":
                    relevant_models.append(model_name)
                    seen_display_names.add(model_name)
            
            # Then check discovered models
            models_added = False
            for model in response.data:
                if 'deepseek' in model.id.lower():
                    # Check if this model is newer than our existing models
                    if is_model_newer_than_existing(model.id, "DeepSeek"):
                        display_name = self._get_display_name_for_deepseek(model.id)
                        
                        # If we don't have a display name mapping, create one
                        if not display_name:
                            # Generate a display name from the model ID
                            display_name = f"DeepSeek {model.id}"
                        
                        # Check if this model is already registered
                        if not is_model_already_registered(display_name):
                            # Add to dynamic registry
                            DYNAMIC_MODEL_REGISTRY[display_name] = {
                                "model": model.id,
                                "family": "DeepSeek",
                                "temperature_allowed": True
                            }
                            logger.info(f"Dynamically registered new DeepSeek model: {display_name} -> {model.id}")
                            models_added = True
                        
                        if display_name and display_name not in seen_display_names:
                            relevant_models.append(display_name)
                            seen_display_names.add(display_name)
                    else:
                        logger.debug(f"Skipping older DeepSeek model: {model.id}")
            
            # Save dynamic models if any were added
            if models_added:
                save_dynamic_models()
            
            return relevant_models
        except Exception as e:
            logger.error(f"Error discovering DeepSeek models: {str(e)}")
            return []

    def _get_display_name_for_openai(self, model_id: str) -> Optional[str]:
        """Convert OpenAI model ID to display name."""
        mapping = {
            "o3-2025-04-16": "OpenAI o3",
            "o4-mini": "OpenAI o4-mini",
            "gpt-4o": "OpenAI GPT4o",
            "gpt-4.1": "OpenAI GPT-4.1",

            "gpt-4-turbo": "OpenAI GPT-4 Turbo",
            "gpt-3.5-turbo": "OpenAI GPT-3.5 Turbo",
        }
        return mapping.get(model_id)

    def _get_display_name_for_gemini(self, model_id: str) -> Optional[str]:
        """Convert Gemini model ID to display name."""
        mapping = {
            "gemini-2.5-pro-exp-03-25": "Gemini 2.5 Pro",
            "gemini-2.0-flash": "Gemini 2.0 Flash",
            "gemini-1.5-pro": "Gemini 1.5 Pro",
            "gemini-1.5-flash": "Gemini 1.5 Flash",
        }
        return mapping.get(model_id)

    def _get_display_name_for_deepseek(self, model_id: str) -> Optional[str]:
        """Convert DeepSeek model ID to display name."""
        mapping = {
            "deepseek-reasoner": "DeepSeek R1",
            "deepseek-chat": "DeepSeek Chat",
            "deepseek-coder": "DeepSeek Coder",
        }
        return mapping.get(model_id)

    def get_selected_models(self) -> Dict[str, List[str]]:
        """
        Get user-selected models from configuration.
        
        Returns:
            Dictionary with 'coding' and 'auditing' keys containing lists of selected models
        """
        config_file = os.path.join(get_config_dir(), "model_selection.json")
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Error reading model selection file: {e}")
        
        # Return default selections if file doesn't exist
        return {
            "coding": OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS + DEEPSEEK_MODELS,
            "auditing": CLAUDE_MODELS + OPENAI_MODELS + GEMINI_MODELS + DEEPSEEK_MODELS
        }

    def save_selected_models(self, coding_models: List[str], auditing_models: List[str]) -> bool:
        """
        Save user-selected models to configuration.
        
        Args:
            coding_models: List of selected coding models
            auditing_models: List of selected auditing models
            
        Returns:
            True if saved successfully, False otherwise
        """
        try:
            config_dir = get_config_dir()
            os.makedirs(config_dir, exist_ok=True)
            config_file = os.path.join(config_dir, "model_selection.json")
            
            selection = {
                "coding": coding_models,
                "auditing": auditing_models
            }
            
            with open(config_file, 'w') as f:
                json.dump(selection, f, indent=2)
            
            return True
        except Exception as e:
            logger.error(f"Error saving model selection: {e}")
            return False

class InputValidator:
    """Utility for validating user inputs."""
    
    @staticmethod
    def validate_prompt(prompt: str, max_length: int = 10000) -> str:
        """Validate and sanitize a prompt."""
        if not prompt or not isinstance(prompt, str):
            raise ValidationError("Prompt must be a non-empty string")
        
        # Limit prompt length
        if len(prompt) > max_length:
            prompt = prompt[:max_length]
        
        # Check for potentially harmful patterns
        harmful_patterns = [
            r"(?i)ignore previous instructions",
            r"(?i)ignore all instructions",
            r"(?i)disregard previous",
        ]
        
        for pattern in harmful_patterns:
            if re.search(pattern, prompt):
                raise ValidationError("Prompt contains potentially harmful instructions")
        
        return prompt
    
    @staticmethod
    def validate_language(language: str, supported_languages: List[str] = None) -> str:
        """Validate programming language."""
        if not language or not isinstance(language, str):
            raise ValidationError("Language must be a non-empty string")
        
        language = language.lower()
        
        # Common programming languages if not specified
        if not supported_languages:
            supported_languages = [
                "python", "javascript", "typescript", "java", "c", "c++", "csharp", 
                "go", "ruby", "rust", "php", "swift", "kotlin", "scala", "pascal",
                "fortran", "julia"
            ]
        
        if language not in supported_languages:
            raise ValidationError(f"Unsupported language: {language}. Supported languages: {', '.join(supported_languages)}")
        
        return language
    
    @staticmethod
    def validate_max_tokens(max_tokens: int, min_value: int = 50, max_value: int = MAX_OUTPUT_TOKENS) -> int:
        """Validate max_tokens parameter."""
        if not isinstance(max_tokens, int):
            try:
                max_tokens = int(max_tokens)
            except:
                raise ValidationError("max_tokens must be an integer")
        
        if max_tokens < min_value:
            max_tokens = min_value
        elif max_tokens > max_value:
            max_tokens = max_value
        
        return max_tokens
    
    @staticmethod
    def validate_api_key(api_key: str, provider: str) -> Optional[str]:
        """Validate API key format."""
        if not api_key or not isinstance(api_key, str):
            return None
        
        # Provider-specific validation
        if provider.lower() == "openai" and not api_key.startswith("sk-"):
            raise ValidationError("Invalid OpenAI API key format")
        elif provider.lower() == "anthropic" and len(api_key) < 20:
            raise ValidationError("Invalid Anthropic API key format")
        
        return api_key
    
    @staticmethod
    def validate_filename(filename: str) -> Tuple[bool, str]:
        """Validates filename for invalid characters."""
        # Check for empty filename
        if not filename:
            return False, "Filename cannot be empty"
        
        # Check for invalid characters
        invalid_chars = set(filename).intersection(set('\\/:*?"<>|'))
        if invalid_chars:
            return False, f"Filename contains invalid characters: {', '.join(invalid_chars)}"
        
        # Additional validations can be added here
        return True, ""

class LLMResponseCache:
    """Cache for LLM responses to avoid redundant API calls."""
    
    def __init__(self, cache_dir: str = None, ttl: int = 86400):
        """Initialize the cache with optional time-to-live (in seconds)."""
        self.enabled = DISKCACHE_AVAILABLE
        self.ttl = ttl  # Default: 1 day
        
        if cache_dir is None:
            cache_dir = os.path.join(get_config_dir(), "cache")
        
        if self.enabled:
            try:
                self.cache = diskcache.Cache(cache_dir)
            except Exception as e:
                logger.warning(f"Failed to initialize cache: {str(e)}")
                self.enabled = False
        else:
            logger.info("Caching disabled: diskcache package not available")
    
    def _get_cache_key(self, provider: str, model: str, prompt: str, language: str) -> str:
        """Generate a unique cache key."""
        cache_dict = {
            "provider": provider,
            "model": model,
            "prompt": prompt,
            "language": language
        }
        serialized = json.dumps(cache_dict, sort_keys=True)
        return hashlib.sha256(serialized.encode('utf-8')).hexdigest()
    
    def get(self, provider: str, model: str, prompt: str, language: str) -> Optional[Dict[str, Any]]:
        """Retrieve cached response if available and not expired."""
        if not self.enabled:
            return None
            
        try:
            key = self._get_cache_key(provider, model, prompt, language)
            cached_data = self.cache.get(key)
            
            if cached_data:
                # Check if cache entry is expired
                timestamp, data = cached_data
                if time.time() - timestamp <= self.ttl:
                    return data
                
                # Cache entry is expired, remove it
                self.cache.delete(key)
        except Exception as e:
            logger.warning(f"Cache retrieval error: {str(e)}")
            
        return None
    
    def set(self, provider: str, model: str, prompt: str, language: str, response_data: Dict[str, Any]) -> bool:
        """Store response in cache with timestamp."""
        if not self.enabled:
            return False
            
        try:
            key = self._get_cache_key(provider, model, prompt, language)
            self.cache.set(key, (time.time(), response_data))
            return True
        except Exception as e:
            logger.warning(f"Cache storage error: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """Clear all cached responses."""
        if not self.enabled:
            return False
            
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.warning(f"Cache clear error: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        if not self.enabled:
            return {"enabled": False}
            
        try:
            return {
                "enabled": True,
                "size": len(self.cache),
                "hit_count": self.cache.statistics()['hits'],
                "miss_count": self.cache.statistics()['misses']
            }
        except Exception as e:
            logger.warning(f"Cache stats error: {str(e)}")
            return {"enabled": True, "error": str(e)}

#-----------------------------------------------------------------------------
# Prompt Templates
#-----------------------------------------------------------------------------

class PromptLibrary:
    """Library of predefined prompt templates for code generation."""
    
    # Improved system prompts for different tasks
    SYSTEM_PROMPT_CODE_GEN = """You are a professional code generation assistant specialized in {language}.
Write clear, efficient, and correct code with no explanations outside the code.
Include helpful comments within the code to explain complex logic.
Focus on best practices, proper error handling, and efficient algorithms."""

    SYSTEM_PROMPT_CODE_AUDIT = """You are an expert code reviewer specialized in finding bugs and issues in {language} code.
Be thorough, precise, and constructive in your analysis.
Focus on critical issues first, then less severe problems.
Categorize issues by severity and provide clear explanations."""
    
    SYSTEM_PROMPT_CODE_CORRECTION = """You are a code correction assistant specialized in fixing {language} code.
Implement all suggested fixes methodically, starting with the most critical issues.
Make minimal changes necessary to fix each issue.
Return only the corrected code with no explanations outside of code comments.
Use delimiter-based format as specified in the instructions."""
    
    # Enhanced DeepSeek system prompts with explicit step-by-step instructions
    SYSTEM_PROMPT_DEEPSEEK_CODE_GEN = """You are a professional code generation assistant specialized in {language}.
Reason step by step through the problem carefully before writing any code.
Consider multiple approaches and their tradeoffs before deciding on the best implementation.
Write clear, efficient, and correct code with no explanations outside the code.
Include helpful comments within the code to explain complex logic.
Focus on best practices, proper error handling, and efficient algorithms."""

    SYSTEM_PROMPT_DEEPSEEK_CODE_AUDIT = """You are an expert code reviewer specialized in finding bugs and issues in {language} code.
Reason step by step through the code carefully before providing your analysis.
Be thorough, precise, and constructive in your analysis.
Focus on critical issues first, then less severe problems.
Categorize issues by severity and provide clear explanations."""
    
    SYSTEM_PROMPT_DEEPSEEK_CODE_CORRECTION = """You are a code correction assistant specialized in fixing {language} code.
Reason step by step through each issue and its potential fixes before implementing any changes.
Implement all suggested fixes methodically, starting with the most critical issues.
Make minimal changes necessary to fix each issue.
Return only the corrected code with no explanations outside of code comments.
Use delimiter-based format as specified in the instructions."""
    
    # Code generation prompt template
    @staticmethod
    def get_generation_prompt(description, language, file_content=None):
        """Returns a standardized code generation prompt"""
        if file_content:
            return (
                f"It is required to modify the program below in accordance with the task: {description}. "
                f"If the program is written in a language different from {language}, it should be translated (re-written) in {language}.\n\n"
                f"Program to modify:\n```\n{file_content}\n```\n\n"
                f"Return ONLY the complete code with no additional explanations. Include helpful comments within the code."
            )
        else:
            return (
                f"It is required to write a program in accordance with the task: {description}. "
                f"The program should be implemented in {language}.\n\n"
                f"Return ONLY the complete code with no additional explanations. Include helpful comments within the code."
            )
            
    # Enhanced DeepSeek code generation prompt template with step-by-step instructions
    @staticmethod
    def get_deepseek_generation_prompt(description, language, file_content=None):
        """Returns an enhanced code generation prompt for DeepSeek with step-by-step instructions"""
        if file_content:
            return (
                f"It is required to modify the program below in accordance with the task: {description}. "
                f"If the program is written in a language different from {language}, it should be translated (re-written) in {language}.\n\n"
                f"<think>\nPlease reason step by step through this task before writing any code. "
                f"Consider multiple approaches, their tradeoffs, and potential edge cases.\n</think>\n\n"
                f"Program to modify:\n```\n{file_content}\n```\n\n"
                f"Return ONLY the complete code with no additional explanations. Include helpful comments within the code."
            )
        else:
            return (
                f"It is required to write a program in accordance with the task: {description}. "
                f"The program should be implemented in {language}.\n\n"
                f"<think>\nPlease reason step by step through this task before writing any code. "
                f"Consider multiple approaches, their tradeoffs, and potential edge cases.\n</think>\n\n"
                f"Return ONLY the complete code with no additional explanations. Include helpful comments within the code."
            )

    # Code audit prompt template
    @staticmethod
    def get_audit_prompt(prompt_text, code_text):
        """Returns a standardized code audit prompt"""
        return f""" 
        Analyze the following code for bugs and issues.  
        Provide a prioritized list of bugs and corrections in the following categories: 
        1. Critical - Severe issues that will cause crashes, security vulnerabilities, incorrect behavior or if the code does not conform to the original prompt
        2. Serious - Important issues that may cause problems in some situations 
        3. Non-critical - Minor issues that should be fixed but don't significantly impact functionality 
        4. Recommendations - Suggestions for improvement that aren't bugs 
        
        For each category, number the items like: 
        1.1, 1.2, etc. for Critical issues 
        2.1, 2.2, etc. for Serious issues 
        3.1, 3.2, etc. for Non-critical issues 
        4.1, 4.2, etc. for Recommendations 
        
        Only provide the numbered list with brief explanations. Don't include explanations about your analysis process. 
        If there are no issues in some category just output "None", for example 
        2. Serious
        None

        Here is the code to analyze: 
        ```
        {code_text} 
        ```

        This code was created as a response to this prompt:
        ```
        {prompt_text} 
        ```
        """
        
    # Enhanced DeepSeek code audit prompt template with step-by-step instructions
    @staticmethod
    def get_deepseek_audit_prompt(prompt_text, code_text):
        """Returns an enhanced code audit prompt for DeepSeek with step-by-step instructions"""
        return f""" 
        Analyze the following code for bugs and issues.
        
        <think>
        Please reason step by step through the code carefully before providing your analysis.
        Consider edge cases, potential runtime issues, and whether the code fully satisfies the original prompt.
        </think>
        
        Provide a prioritized list of bugs and corrections in the following categories: 
        1. Critical - Severe issues that will cause crashes, security vulnerabilities, incorrect behavior or if the code does not conform to the original prompt
        2. Serious - Important issues that may cause problems in some situations 
        3. Non-critical - Minor issues that should be fixed but don't significantly impact functionality 
        4. Recommendations - Suggestions for improvement that aren't bugs 
        
        For each category, number the items like: 
        1.1, 1.2, etc. for Critical issues 
        2.1, 2.2, etc. for Serious issues 
        3.1, 3.2, etc. for Non-critical issues 
        4.1, 4.2, etc. for Recommendations 
        
        Only provide the numbered list with brief explanations. Don't include explanations about your analysis process. 
        If there are no issues in some category just output "None", for example 
        2. Serious
        None

        Here is the code to analyze: 
        ```
        {code_text} 
        ```

        This code was created as a response to this prompt:
        ```
        {prompt_text} 
        ```
        """

    # Code correction prompt template with delimiter-based format
    @staticmethod
    def get_correction_prompt(initial_prompt, program_code, code_analysis):
        """Returns a standardized code correction prompt using delimiter-based format"""
        return f"""This is a code of the program that was written as a response to the prompt {initial_prompt}.

Program code:
{program_code}
This is analysis of the code and suggestions for corrections: 
{code_analysis}

Audit the analysis and implement the corrections that you think are correct and will improve the code. Make the corrections one by one starting from critical errors, then serious, then non-critical, then suggestions.

Return your response in the following format:

###CODE###
[corrected code goes here]
###CORRECTIONS###
[corrections array goes here]

The corrections array should be in this format:
[[0,1,0],  // Critical fixes - 0 means fixed, 1 means not fixed
 [0,1],    // Serious fixes
 [1,0],    // Non-critical fixes
 [1,0]]    // Recommendations fixes

For example, if code_analysis contains items:
1.1., 1.2, 1.3
2.1, 2.2
3.1, 3.2., 3.3.
4.1, 4.2

and you corrected 1.1, 1.3, 2.1, 3.2, 3.3, 4.2, then the corrections array should be:
[[0,1,0], [0,1], [1,0,0], [1,0]]

That is, all the corrected issues are marked as 0, all the issues that haven't been corrected as 1.
If an error category contains None (for example "2. Serious\\nNone"), the corresponding array should be empty: [].

Make sure to include ONLY the corrected code between the ###CODE### and ###CORRECTIONS### delimiters, with no additional text, explanations, or backticks.
"""

    # Enhanced DeepSeek code correction prompt template with delimiter-based format
    @staticmethod
    def get_deepseek_correction_prompt(initial_prompt, program_code, code_analysis):
        """Returns an enhanced code correction prompt for DeepSeek with delimiter-based format"""
        return f"""This is a code of the program that was written as a response to the prompt {initial_prompt}.

Program code:
{program_code}
This is analysis of the code and suggestions for corrections: 
{code_analysis}

<think>
Reason step by step through each issue and its potential fixes before implementing any changes.
Consider potential impacts of each fix and whether they might introduce new issues.
</think>

Audit the analysis and implement the corrections that you think are correct and will improve the code. 
Make the corrections one by one starting from critical errors, then serious, then non-critical, then suggestions.

Return your response in the following format:

###CODE###
[corrected code goes here]
###CORRECTIONS###
[corrections array goes here]

The corrections array should be in this format:
[[0,1,0],  // Critical fixes - 0 means fixed, 1 means not fixed
 [0,1],    // Serious fixes
 [1,0],    // Non-critical fixes
 [1,0]]    // Recommendations fixes

For example, if code_analysis contains items:
1.1., 1.2, 1.3
2.1, 2.2
3.1, 3.2., 3.3.
4.1, 4.2

and you corrected 1.1, 1.3, 2.1, 3.2, 3.3, 4.2, then the corrections array should be:
[[0,1,0], [0,1], [1,0,0], [1,0]]

That is, all the corrected issues are marked as 0, all the issues that haven't been corrected as 1.
If an error category contains None (for example "2. Serious\\nNone"), the corresponding array should be empty: [].

Make sure to include ONLY the corrected code between the ###CODE### and ###CORRECTIONS### delimiters, with no additional text, explanations, or backticks.
"""

#-----------------------------------------------------------------------------
# Provider Base Classes
#-----------------------------------------------------------------------------

class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def generate_code(self, prompt: str, language: str, max_tokens: int = 1000) -> CodeGenerationResponse:
        """Generate code based on the given prompt."""
        pass
    
    @abstractmethod
    def audit_code(self, prompt_text: str, code_text: str, language: str, max_tokens: int = 2000) -> str:
        """Analyze code and return a structured analysis."""
        pass
    
    @abstractmethod
    def correct_code(self, initial_prompt: str, program_code: str, code_analysis: str, language: str, max_tokens: int = 2000) -> Tuple[str, str]:
        """Correct code based on audit analysis."""
        pass
    
    @abstractmethod
    def get_supported_languages(self) -> List[str]:
        """Get list of supported programming languages."""
        pass
    
    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get the provider name."""
        pass

#-----------------------------------------------------------------------------
# Legacy API Key Management
#-----------------------------------------------------------------------------

def get_api_keys():
    """
    Reads API keys from APIKeys file
    
    Returns:
        dict: Dictionary with OpenAI, Claude, Gemini, and DeepSeek API keys
    """
    keys = {}
    config = configparser.ConfigParser()
    
    try:
        config.read('APIKeys')
        if 'API Keys' in config:
            keys['OpenAI'] = config['API Keys'].get('OpenAI', '')
            keys['Claude'] = config['API Keys'].get('Claude', '')
            keys['Gemini'] = config['API Keys'].get('Gemini', '')
            keys['DeepSeek'] = config['API Keys'].get('DeepSeek', '')
    except Exception as e:
        logger.error(f"Error reading API keys: {e}")
    
    return keys

#-----------------------------------------------------------------------------
# Helper Functions
#-----------------------------------------------------------------------------

def extract_delimited_output(text):
    """
    Extracts corrected code and corrections list from delimiter-based format.
    
    Args:
        text: The raw text response containing delimited sections
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    # Default values in case parsing fails
    default_code = ""
    default_corrections = "[]"
    
    try:
        # Extract code section - between ###CODE### and ###CORRECTIONS###
        code_match = re.search(r'###CODE###\s*(.*?)(?=###CORRECTIONS###)', text, re.DOTALL)
        if code_match:
            corrected_code = code_match.group(1).strip()
        else:
            # Try to find any code block if the delimiter format wasn't followed
            code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
            if code_blocks:
                # Find the longest code block
                longest_block = max(code_blocks, key=len)
                # Strip the backticks
                corrected_code = re.sub(r'^```\w*\n|```$', '', longest_block).strip()
            else:
                corrected_code = default_code
        
        # Extract corrections section - after ###CORRECTIONS###
        corrections_match = re.search(r'###CORRECTIONS###\s*(.*?)(?=$|\Z)', text, re.DOTALL)
        if corrections_match:
            corrections_text = corrections_match.group(1).strip()
            # Try to extract the nested array part
            array_match = re.search(r'\[\s*\[.*?\]\s*\]', corrections_text, re.DOTALL)
            if array_match:
                corrections_list = array_match.group(0)
            else:
                corrections_list = default_corrections
        else:
            corrections_list = default_corrections
        
        return corrected_code, corrections_list
        
    except Exception as e:
        logger.error(f"Error parsing delimited response: {e}")
        logger.debug(f"Raw response: {text[:500]}...")  # Print first 500 chars for debugging
        
        # Fallback to looking for any code
        try:
            # Look for code blocks with triple backticks
            code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
            if code_blocks:
                # Find the longest code block
                longest_block = max(code_blocks, key=len)
                # Strip the backticks
                corrected_code = re.sub(r'^```\w*\n|```$', '', longest_block).strip()
                return corrected_code, default_corrections
        except:
            pass
            
        return default_code, default_corrections

def remove_surrounding_quotes(code_text):
    """
    Removes Markdown code block markers, triple quotes, letter markers like 'a)' or 'b)'
    and other surrounding quote characters from code returned by LLMs.
    """
    if not code_text:
        return ""
        
    # Remove letter markers like "a)" at the beginning
    code_text = re.sub(r'^[a-z]\)\s*', '', code_text)
        
    # Remove Markdown code block markers (backticks)
    if '```' in code_text:
        # Check if there are language specifiers like ```python
        lang_match = re.match(r'^```(\w+)?\n', code_text)
        if lang_match:
            # Remove opening line with language specification
            code_text = re.sub(r'^```(\w+)?\n', '', code_text)
        else:
            # Remove opening ``` if it's alone on the first line
            code_text = re.sub(r'^```\n', '', code_text)
            
        # Remove closing ```
        code_text = re.sub(r'\n```\s*$', '', code_text)
    
    # Remove triple quotes
    if code_text.startswith('"""') and code_text.endswith('"""'):
        return code_text[3:-3].strip()
    if code_text.startswith("'''") and code_text.endswith("'''"):
        return code_text[3:-3].strip()
    
    # Remove single quotes
    if code_text.startswith('"') and code_text.endswith('"'):
        return code_text[1:-1].strip()
    if code_text.startswith("'") and code_text.endswith("'"):
        return code_text[1:-1].strip()
    
    # Return original code if no quotes found
    return code_text.strip()

def parse_bug_counts(audit_text):
    """
    Counts the number of issues in each category of the code audit.
    
    Returns a tuple of (critical, serious, non_critical, suggestions) counts.
    """
    if not audit_text:
        return (0, 0, 0, 0)
        
    lines = audit_text.splitlines()
    
    # Counters for each category
    counts = {'critical': 0, 'serious': 0, 'non_critical': 0, 'suggestions': 0}
    
    # Current category
    current_category = None
    
    # Flag indicating if the category has "None"
    category_has_none = False
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Check if this is the start of a new category
        if re.search(r'1\.?\s+Critical', line, re.IGNORECASE):
            current_category = 'critical'
            category_has_none = False
        elif re.search(r'2\.?\s+Serious', line, re.IGNORECASE):
            current_category = 'serious'
            category_has_none = False
        elif re.search(r'3\.?\s+(Non-critical|N.?critical)', line, re.IGNORECASE):
            current_category = 'non_critical'
            category_has_none = False
        elif re.search(r'4\.?\s+(Recommendations|Suggestions)', line, re.IGNORECASE):
            current_category = 'suggestions'
            category_has_none = False
        
        # If we're in a category and encounter "None", mark the category as empty
        elif current_category and line.lower() == 'none':
            category_has_none = True
            counts[current_category] = 0
        
        # Fixed regex pattern to find issue numbers (1.1, 1.2, etc.)
        elif current_category and not category_has_none and re.match(r'\d+\.\d+\.?', line):
            counts[current_category] += 1
    
    return (counts['critical'], counts['serious'], 
            counts['non_critical'], counts['suggestions'])

def parse_corrections(clist_text, bug_counts=(float('inf'), float('inf'), float('inf'), float('inf'))):
    """
    Parses the correction list string in the form '[[0,1],[1,0],[0,1],[1,1]]'.
    Indices:
      0 -> Critical
      1 -> Serious
      2 -> N/crit
      3 -> Suggestions
      
    Returns a tuple with the count of fixed issues in each category,
    limited to the actual bug counts.
    
    Args:
        clist_text: String representation of the correction list
        bug_counts: Tuple of (critical, serious, non_critical, suggestions) counts
    """
    try:    
        # Validate input is a string that could represent a list
        if not isinstance(clist_text, str) or not clist_text.strip().startswith('['):
            return (0, 0, 0, 0)
            
        # Safe evaluation of the list
        import ast
        arr = ast.literal_eval(clist_text)
        
        # Check if result is a list
        if not isinstance(arr, list):
            return (0, 0, 0, 0)
            
        # Pad with empty lists if needed
        while len(arr) < 4:
            arr.append([])
            
        # Count zeros in each category, limiting to actual bug counts
        c_crit = min(sum(1 for x in arr[0] if x == 0) if arr[0] else 0, bug_counts[0])
        c_serious = min(sum(1 for x in arr[1] if x == 0) if arr[1] else 0, bug_counts[1])
        c_ncrit = min(sum(1 for x in arr[2] if x == 0) if arr[2] else 0, bug_counts[2])
        c_sugg = min(sum(1 for x in arr[3] if x == 0) if arr[3] else 0, bug_counts[3])
        
        return (c_crit, c_serious, c_ncrit, c_sugg)
        
    except (SyntaxError, ValueError) as e:
        logger.error(f"Error parsing corrections list: {e}")
        return (0, 0, 0, 0)
    except Exception as e:
        logger.error(f"Unexpected error parsing corrections: {e}")
        return (0, 0, 0, 0)

def extension_for_language(lang):
    """Returns the appropriate file extension for a given programming language"""
    mapping = {
        "python": ".py",
        "java": ".java",
        "javascript": ".js",
        "c": ".c",
        "c++": ".cpp",
        "pascal": ".pas",
        "julia": ".jl",
        "fortran": ".f90"
    }
    return mapping.get(lang.lower(), ".txt")

def get_output_directory(project_name):
    """
    Get the output directory based on settings.
    If a custom directory is set, use that path directly or with project name as subdirectory.
    Otherwise, use the project name in the current directory.
    """
    # Load settings
    config_file = os.path.join(get_config_dir(), "settings.json")
    settings = {}
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                settings = json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    
    # Get custom output directory from settings
    custom_dir = settings.get("output_directory", "").strip()
    
    if custom_dir:
        # Use custom directory directly - don't add project name as subdirectory
        # This fixes the issue with files being saved in the wrong location
        return custom_dir
    else:
        # Use project name in current directory (default behavior)
        return project_name

def clear_description():
    """Clear the Program Description text field."""
    text_description.delete("1.0", tk.END)

def open_model_selection_dialog(root, config, update_callback, all_available_models):
    """
    Open a dialog to select which models to use for coding and auditing.
    
    Args:
        root: Parent tkinter window
        config: SecureConfig instance
        update_callback: Function to call when models are updated
        all_available_models: Pre-fetched available models from all providers
    """
    dialog = tk.Toplevel(root)
    dialog.title("Select Active LLMs")
    dialog.geometry("800x600")
    dialog.minsize(800, 600)
    dialog.grab_set()  # Make window modal
    
    # Get current selections
    selected_models = config.get_selected_models()
    
    # Create main frame
    main_frame = ttk.Frame(dialog, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title
    title_label = ttk.Label(main_frame, text="Select Active LLMs", font=("Arial", 14, "bold"))
    title_label.pack(anchor="w", pady=(0, 10))
    
    # Create notebook for coding and auditing tabs
    notebook = ttk.Notebook(main_frame)
    notebook.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Variables to store checkbutton states
    coding_vars = {}
    auditing_vars = {}
    
    def create_model_tab(tab_name, models_dict, selected_list, var_dict):
        """Create a tab with checkboxes for model selection."""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text=tab_name)
        
        # Create scrollable frame
        canvas = tk.Canvas(frame)
        scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add models grouped by provider
        for provider, models in models_dict.items():
            if not models:
                continue
                
            # Sort models: selected ones first, then unselected
            selected_models_for_provider = [m for m in models if m in selected_list]
            unselected_models_for_provider = [m for m in models if m not in selected_list]
            sorted_models = selected_models_for_provider + unselected_models_for_provider
                
            # Provider header
            provider_frame = ttk.LabelFrame(scrollable_frame, text=provider, padding=10)
            provider_frame.pack(fill=tk.X, pady=5, padx=5)
            
            for model in sorted_models:
                var = tk.BooleanVar()
                var.set(model in selected_list)
                var_dict[model] = var
                
                cb = ttk.Checkbutton(provider_frame, text=model, variable=var)
                cb.pack(anchor="w", pady=2)
        
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        return frame
    
    # Create coding models tab
    create_model_tab("Coding Models", all_available_models, selected_models.get("coding", []), coding_vars)
    
    # Create auditing models tab
    create_model_tab("Auditing Models", all_available_models, selected_models.get("auditing", []), auditing_vars)
    
    # Buttons at the bottom
    btn_frame = ttk.Frame(main_frame)
    btn_frame.pack(fill=tk.X, pady=(10, 0))
    
    def save_selection():
        """Save the selected models and close dialog."""
        # Get selected models
        new_coding_models = [model for model, var in coding_vars.items() if var.get()]
        new_auditing_models = [model for model, var in auditing_vars.items() if var.get()]
        
        # Check if selections changed
        old_coding = set(selected_models.get("coding", []))
        old_auditing = set(selected_models.get("auditing", []))
        new_coding = set(new_coding_models)
        new_auditing = set(new_auditing_models)
        
        if new_coding != old_coding or new_auditing != old_auditing:
            # Save to configuration
            if config.save_selected_models(new_coding_models, new_auditing_models):
                # Update global variables and UI
                update_callback(new_coding_models, new_auditing_models)
                messagebox.showinfo("Success", "Model selection updated successfully!")
            else:
                messagebox.showerror("Error", "Failed to save model selection.")
        else:
            messagebox.showinfo("No Changes", "No changes were made to model selection.")
        
        dialog.destroy()
    
    def select_all_coding():
        """Select all coding models."""
        for var in coding_vars.values():
            var.set(True)
    
    def select_all_auditing():
        """Select all auditing models."""
        for var in auditing_vars.values():
            var.set(True)
    
    def deselect_all_coding():
        """Deselect all coding models."""
        for var in coding_vars.values():
            var.set(False)
    
    def deselect_all_auditing():
        """Deselect all auditing models."""
        for var in auditing_vars.values():
            var.set(False)
    
    # Selection buttons
    selection_frame = ttk.Frame(btn_frame)
    selection_frame.pack(side=tk.LEFT, padx=5)
    
    ttk.Button(selection_frame, text="Select All Coding", command=select_all_coding).pack(side=tk.LEFT, padx=2)
    ttk.Button(selection_frame, text="Deselect All Coding", command=deselect_all_coding).pack(side=tk.LEFT, padx=2)
    ttk.Button(selection_frame, text="Select All Auditing", command=select_all_auditing).pack(side=tk.LEFT, padx=2)
    ttk.Button(selection_frame, text="Deselect All Auditing", command=deselect_all_auditing).pack(side=tk.LEFT, padx=2)
    
    # Main buttons
    ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.RIGHT, padx=5)
    ttk.Button(btn_frame, text="OK", command=save_selection).pack(side=tk.RIGHT, padx=5)

#-----------------------------------------------------------------------------
# Client Helper Functions
#-----------------------------------------------------------------------------

def create_openai_client(api_key):
    """Creates an OpenAI client"""
    try:
        from openai import OpenAI
        if not api_key:
            raise ConfigurationError("OpenAI API key is missing.")
        
        logger.debug(f"Creating OpenAI client with key starting with: {api_key[:4]}...")
        
        client = OpenAI(api_key=api_key)
        
        # Test the client with a simple call
        try:
            # Just check if models attribute exists
            client.models.list
            logger.debug("OpenAI client successfully created")
            return client
        except Exception as e:
            logger.warning(f"OpenAI client created but test failed: {str(e)}")
            return client  # Still return client even if test fails
            
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    except Exception as e:
        logger.error(f"Error creating OpenAI client: {str(e)}")
        raise ConfigurationError(f"Failed to create OpenAI client: {str(e)}")

def create_claude_client(api_key):
    """Creates a Claude client"""
    try:
        from anthropic import Anthropic
        if not api_key:
            raise ConfigurationError("Claude API key is missing.")
        return Anthropic(api_key=api_key)
    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")

def create_gemini_client(api_key):
    """Creates a Gemini client"""
    try:
        from google import genai
        if not api_key:
            raise ConfigurationError("Gemini API key is missing.")
        client = genai.Client(api_key=api_key)
        return client
    except ImportError:
        raise ImportError("Google GenAI package not installed. Install with: pip install google-generativeai")

def create_deepseek_client(api_key):
    """Creates a DeepSeek client"""
    try:
        from openai import OpenAI
        if not api_key:
            raise ConfigurationError("DeepSeek API key is missing.")
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        return client
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")

def create_client(family):
    """
    Creates and returns appropriate client based on the LLM family
    
    Args:
        family: The LLM family name ('OpenAI', 'Claude', 'Gemini', 'DeepSeek', etc.)
    
    Returns:
        The initialized client or None if family not supported
    """
    config = SecureConfig()
    api_key = config.get_api_key(family)
    
    # Check if API key exists
    if not api_key:
        raise ConfigurationError(f"API key for {family} is missing. Please configure it first.")
    
    try:
        if family == "OpenAI":
            return create_openai_client(api_key)
        elif family == "Claude":
            return create_claude_client(api_key)
        elif family == "Gemini":
            return create_gemini_client(api_key)
        elif family == "DeepSeek":
            return create_deepseek_client(api_key)
        # Add support for other families here in the future
        else:
            raise ConfigurationError(f"Unsupported LLM family: {family}")
    except ImportError as e:
        raise
    except Exception as e:
        raise Exception(f"Failed to create client for {family}: {str(e)}")

#-----------------------------------------------------------------------------
# OpenAI Implementation
#-----------------------------------------------------------------------------

def generate_with_openai(client, model, description, language, file_content=None, temp_allowed=False):
    """
    Generates initial code using OpenAI models with optimized parameters
    
    Args:
        client: OpenAI client
        model: Model name
        description: Program description
        language: Programming language
        file_content: Optional content of an existing program file to modify
        temp_allowed: Whether the model allows temperature parameter
    
    Returns:
        Generated code
    """
    # Get standardized prompt
    prompt_text = PromptLibrary.get_generation_prompt(description, language, file_content)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_GEN.format(language=language)
    
    # Check if this is a reasoning model (o3, o4-mini)
    is_reasoning_model = model in ["o3-2025-04-16", "o4-mini"]
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(description, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in OPENAI_PARAMS else "default"
    params = OPENAI_PARAMS[model_key][task_type]
    
    # Set correct max tokens for the model
    # Limit to 32768 for GPT-4.1 and GPT-4.1-mini models
    if model in ["gpt-4.1", "gpt-4.1-mini"]:
        max_token_limit = 32768
    else:
        max_token_limit = MAX_OUTPUT_TOKENS
    
    try:
        # First try the current API format
        try:
            # Different setup for reasoning models vs standard models
            if is_reasoning_model:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_completion_tokens": max_token_limit  # Use max_completion_tokens for o-series models
                }
                
                # Removed reasoning parameter as it causes an error
                # Fallback to standard parameters
                if temp_allowed:
                    kwargs["temperature"] = 0.1  # Low temperature for reasoning models
                    kwargs["top_p"] = 0.95
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]
                }
                
                # Add temperature parameter if the model supports it
                if temp_allowed:
                    kwargs["temperature"] = params["temperature"]
                    kwargs["top_p"] = params["top_p"]
                    kwargs["frequency_penalty"] = params["frequency_penalty"]
                    kwargs["presence_penalty"] = params["presence_penalty"]
                
                # Add max_tokens for non-o-series models
                if not is_reasoning_model:
                    kwargs["max_tokens"] = max_token_limit
                
            response = client.chat.completions.create(**kwargs)
            
            # Extract content safely
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
            
            # If we get here, something went wrong with the extraction
            return f"Error: Failed to extract content from response: {str(response)}"
            
        except AttributeError:
            # Fallback for older API
            logger.warning("Using fallback OpenAI API approach due to AttributeError")
            
            # Different setup for reasoning models vs standard models
            if is_reasoning_model:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ],
                    "max_completion_tokens": max_token_limit  # Use max_completion_tokens for o-series models
                }
                
                # Removed reasoning parameter
                # Fallback to standard parameters
                if temp_allowed:
                    kwargs["temperature"] = 0.1  # Low temperature for reasoning models
                    kwargs["top_p"] = 0.95
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt_text}
                    ]
                }
                
                # Add temperature parameter if the model supports it
                if temp_allowed:
                    kwargs["temperature"] = params["temperature"]
                    kwargs["top_p"] = params["top_p"]
                    kwargs["frequency_penalty"] = params["frequency_penalty"]
                    kwargs["presence_penalty"] = params["presence_penalty"]
                
                # Add max_tokens for non-o-series models
                if not is_reasoning_model:
                    kwargs["max_tokens"] = max_token_limit
                
            # Use chat completions instead of responses
            response = client.chat.completions.create(**kwargs)
            
            # Extract text from the structure of the response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
            
            # Fallback to string representation
            return str(response)
    except Exception as e:
        error_msg = f"Error generating code with OpenAI: {str(e)}"
        logger.error(error_msg)
        return error_msg

def audit_with_openai(client, model, prompt_text, code_text, temp_allowed=False):
    """
    Analyzes code using OpenAI models with optimized parameters
    
    Args:
        client: OpenAI client
        model: Model name
        prompt_text: Original prompt description
        code_text: Code to analyze
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Structured analysis
    """
    # Get standardized audit prompt
    prompt = PromptLibrary.get_audit_prompt(prompt_text, code_text)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_AUDIT.format(language="")
    
    # Check if this is a reasoning model (o3, o4-mini)
    is_reasoning_model = model in ["o3-2025-04-16", "o4-mini"]
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(prompt_text, "")
    
    # Get optimal parameters based on model and task type
    model_key = model if model in OPENAI_PARAMS else "default"
    params = OPENAI_PARAMS[model_key][task_type]
    
    # Set correct max tokens for the model
    # Limit to 32768 for GPT-4.1 and GPT-4.1-mini models
    if model in ["gpt-4.1", "gpt-4.1-mini"]:
        max_token_limit = 32768
    else:
        max_token_limit = MAX_OUTPUT_TOKENS
    
    try:
        # Try newer API first
        try:
            # Different setup for reasoning models vs standard models
            if is_reasoning_model:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": max_token_limit  # Use max_completion_tokens for o-series models
                }
                
                # Removed reasoning parameter
                # Fallback to standard parameters
                if temp_allowed:
                    kwargs["temperature"] = 0.1  # Low temperature for reasoning models
                    kwargs["top_p"] = 0.95
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Add temperature parameter if the model supports it
                if temp_allowed:
                    kwargs["temperature"] = params["temperature"]
                    kwargs["top_p"] = params["top_p"]
                    kwargs["frequency_penalty"] = params["frequency_penalty"]
                    kwargs["presence_penalty"] = params["presence_penalty"]
                
                # Add max_tokens for non-o-series models
                if not is_reasoning_model:
                    kwargs["max_tokens"] = max_token_limit
                
            response = client.chat.completions.create(**kwargs)
            
            # Extract content safely
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
            
            # If we get here, something went wrong with the extraction
            return f"Error: Failed to extract content from response: {str(response)}"
            
        except AttributeError:
            # Fallback to older API
            logger.warning("Using fallback OpenAI API approach due to AttributeError")
            
            # Different setup for reasoning models vs standard models
            if is_reasoning_model:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    "max_completion_tokens": max_token_limit  # Use max_completion_tokens for o-series models
                }
                
                # Removed reasoning parameter
                # Fallback to standard parameters
                if temp_allowed:
                    kwargs["temperature"] = 0.1  # Low temperature for reasoning models
                    kwargs["top_p"] = 0.95
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ]
                }
                
                # Add temperature parameter if the model supports it
                if temp_allowed:
                    kwargs["temperature"] = params["temperature"]
                    kwargs["top_p"] = params["top_p"]
                    kwargs["frequency_penalty"] = params["frequency_penalty"]
                    kwargs["presence_penalty"] = params["presence_penalty"]
                
                # Add max_tokens for non-o-series models
                if not is_reasoning_model:
                    kwargs["max_tokens"] = max_token_limit
                
            # Use chat completions instead of responses
            response = client.chat.completions.create(**kwargs)
            
            # Extract content from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    return response.choices[0].message.content
            
            # Fallback to string representation
            return str(response)
    except Exception as e:
        error_msg = f"Error analyzing code with OpenAI: {str(e)}"
        logger.error(error_msg)
        return error_msg

def correct_with_openai(client, model, initial_prompt, program_code, code_analysis, temp_allowed=False):
    """
    Corrects code based on audit analysis using OpenAI models with optimized parameters
    
    Args:
        client: OpenAI client
        model: Model name
        initial_prompt: Original task description
        program_code: Current program code
        code_analysis: Audit results
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    # Get standardized correction prompt
    user_prompt = PromptLibrary.get_correction_prompt(initial_prompt, program_code, code_analysis)
    language = "python"  # Default language, can be improved by detecting from code
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_CORRECTION.format(language=language)

    # Check if this is a reasoning model (o3, o4-mini)
    is_reasoning_model = model in ["o3-2025-04-16", "o4-mini"]
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(initial_prompt, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in OPENAI_PARAMS else "default"
    params = OPENAI_PARAMS[model_key][task_type]
    
    # Set correct max tokens for the model
    # Limit to 32768 for GPT-4.1 and GPT-4.1-mini models
    if model in ["gpt-4.1", "gpt-4.1-mini"]:
        max_token_limit = 32768
    else:
        max_token_limit = MAX_OUTPUT_TOKENS
    
    try:
        # Try newer API first
        try:
            # Different setup for reasoning models vs standard models
            if is_reasoning_model:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_completion_tokens": max_token_limit  # Use max_completion_tokens for o-series models
                }
                
                # Removed reasoning parameter
                # Fallback to standard parameters
                if temp_allowed:
                    kwargs["temperature"] = 0.1  # Low temperature for reasoning models
                    kwargs["top_p"] = 0.95
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                # Add temperature parameter if the model supports it
                if temp_allowed:
                    kwargs["temperature"] = params["temperature"]
                    kwargs["top_p"] = params["top_p"]
                    kwargs["frequency_penalty"] = params["frequency_penalty"]
                    kwargs["presence_penalty"] = params["presence_penalty"]
                
                # Add max_tokens for non-o-series models
                if not is_reasoning_model:
                    kwargs["max_tokens"] = max_token_limit
                
            response = client.chat.completions.create(**kwargs)
            
            # Extract content safely
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    full_text = response.choices[0].message.content
                    # Parse response with delimiters
                    return extract_delimited_output(full_text)
            
            # If we get here, something went wrong with the extraction
            return f"Error: Failed to extract content from response: {str(response)}", "[]"
            
        except AttributeError:
            # Fallback to older API
            logger.warning("Using fallback OpenAI API approach due to AttributeError")
            
            # Different setup for reasoning models vs standard models
            if is_reasoning_model:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "max_completion_tokens": max_token_limit  # Use max_completion_tokens for o-series models
                }
                
                # Removed reasoning parameter
                # Fallback to standard parameters
                if temp_allowed:
                    kwargs["temperature"] = 0.1  # Low temperature for reasoning models
                    kwargs["top_p"] = 0.95
            else:
                kwargs = {
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                }
                
                # Add temperature parameter if the model supports it
                if temp_allowed:
                    kwargs["temperature"] = params["temperature"]
                    kwargs["top_p"] = params["top_p"]
                    kwargs["frequency_penalty"] = params["frequency_penalty"]
                    kwargs["presence_penalty"] = params["presence_penalty"]
                
                # Add max_tokens for non-o-series models
                if not is_reasoning_model:
                    kwargs["max_tokens"] = max_token_limit
                
            # Use chat completions instead of responses
            response = client.chat.completions.create(**kwargs)
            
            # Extract content from response
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    full_text = response.choices[0].message.content
                    # Parse response with delimiters
                    return extract_delimited_output(full_text)
            
            # Fallback to string representation
            full_text = str(response)
            return extract_delimited_output(full_text)
    except Exception as e:
        error_msg = f"Error correcting code with OpenAI: {str(e)}"
        logger.error(error_msg)
        return error_msg, "[]"

#-----------------------------------------------------------------------------
# Claude Implementation
#-----------------------------------------------------------------------------

def generate_with_claude(client, model, description, language, file_content=None, temp_allowed=True):
    """
    Generates initial code using Claude models with optimized parameters
    
    Args:
        client: Claude client
        model: Model name
        description: Program description
        language: Programming language
        file_content: Optional content of an existing program file to modify
        temp_allowed: Whether the model allows temperature parameter
    
    Returns:
        Generated code
    """
    # Get standardized prompt
    prompt_text = PromptLibrary.get_generation_prompt(description, language, file_content)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_GEN.format(language=language)
    
    # Check if this is Claude 3.7 Sonnet (the only one with thinking parameter)
    is_claude_3_7_sonnet = model == "claude-3-7-sonnet-20250219"
    
    # Check if this is Claude Opus 4.0 (has lower token limit)
    is_claude_opus_4 = model == "claude-opus-4-20250514"
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(description, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in CLAUDE_PARAMS else "default"
    params = CLAUDE_PARAMS[model_key][task_type]
    
    # Set max tokens based on model
    max_tokens = 32000 if is_claude_opus_4 else MAX_OUTPUT_TOKENS
    
    try:
        # Set up basic parameters
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt_text}
            ],
            "stream": True  # Keep streaming enabled for long requests as recommended
        }
        
        # Add thinking parameter for Claude 3.7 Sonnet with appropriate budget
        if is_claude_3_7_sonnet and params["thinking_budget"] > 0:
            # Reduce thinking budget to prevent overload
            reduced_budget = min(params["thinking_budget"], 4000)
            kwargs["thinking"] = {
                "type": "enabled", 
                "budget_tokens": reduced_budget
            }
            # When thinking is enabled, cannot set temperature
        elif temp_allowed:
            # Only add temperature if thinking is not enabled
            kwargs["temperature"] = params["temperature"]
        
        # Stream the response and collect the content
        try:
            response = client.messages.create(**kwargs)
            content_chunks = []
            
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                    content_chunks.append(chunk.delta.text)
            
            return ''.join(content_chunks)
        except Exception as stream_error:
            # If streaming fails, try again without streaming
            logger.warning(f"Streaming failed: {str(stream_error)}. Trying without streaming.")
            kwargs["stream"] = False
            response = client.messages.create(**kwargs)
            
            # Extract content from non-streaming response
            if hasattr(response, 'content') and isinstance(response.content, list):
                content_text = "".join(item.text for item in response.content if hasattr(item, 'text'))
                return content_text
            elif hasattr(response, 'content') and hasattr(response.content, 'text'):
                return response.content.text
            
            # Fallback to string representation
            return str(response)
            
    except Exception as e:
        error_msg = f"Error generating code with Claude: {str(e)}"
        logger.error(error_msg)
        return error_msg

def audit_with_claude(client, model, prompt_text, code_text, temp_allowed=True):
    """
    Analyzes code using Claude models with optimized parameters
    
    Args:
        client: Claude client
        model: Model name
        prompt_text: Original prompt description
        code_text: Code to analyze
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Structured analysis
    """
    # Get standardized audit prompt
    prompt = PromptLibrary.get_audit_prompt(prompt_text, code_text)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_AUDIT.format(language="")
    
    # Check if this is Claude 3.7 Sonnet (the only one with thinking parameter)
    is_claude_3_7_sonnet = model == "claude-3-7-sonnet-20250219"
    
    # Check if this is Claude Opus 4.0 (has lower token limit)
    is_claude_opus_4 = model == "claude-opus-4-20250514"
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(prompt_text, "")
    
    # Get optimal parameters based on model and task type
    model_key = model if model in CLAUDE_PARAMS else "default"
    params = CLAUDE_PARAMS[model_key][task_type]
    
    # Set max tokens based on model
    max_tokens = 32000 if is_claude_opus_4 else MAX_OUTPUT_TOKENS
    
    try:
        # Set up basic parameters
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": True  # Keep streaming enabled for long requests as recommended
        }
        
        # Add thinking parameter for Claude 3.7 Sonnet with appropriate budget
        if is_claude_3_7_sonnet and params["thinking_budget"] > 0:
            # Reduce thinking budget to prevent overload
            reduced_budget = min(params["thinking_budget"], 4000)
            kwargs["thinking"] = {
                "type": "enabled", 
                "budget_tokens": reduced_budget
            }
            # When thinking is enabled, cannot set temperature
        elif temp_allowed:
            # Only add temperature if thinking is not enabled
            kwargs["temperature"] = params["temperature"]
        
        # Stream the response and collect the content
        try:
            response = client.messages.create(**kwargs)
            content_chunks = []
            
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                    content_chunks.append(chunk.delta.text)
            
            return ''.join(content_chunks)
        except Exception as stream_error:
            # If streaming fails, try again without streaming
            logger.warning(f"Streaming failed: {str(stream_error)}. Trying without streaming.")
            kwargs["stream"] = False
            response = client.messages.create(**kwargs)
            
            # Extract content from non-streaming response
            if hasattr(response, 'content') and isinstance(response.content, list):
                content_text = "".join(item.text for item in response.content if hasattr(item, 'text'))
                return content_text
            elif hasattr(response, 'content') and hasattr(response.content, 'text'):
                return response.content.text
            
            # Fallback to string representation
            return str(response)
            
    except Exception as e:
        error_msg = f"Error analyzing code with Claude: {str(e)}"
        logger.error(error_msg)
        return error_msg

def correct_with_claude(client, model, initial_prompt, program_code, code_analysis, temp_allowed=True):
    """
    Corrects code based on audit analysis using Claude models with optimized parameters
    
    Args:
        client: Claude client
        model: Model name
        initial_prompt: Original task description
        program_code: Current program code
        code_analysis: Audit results
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    # Get standardized correction prompt
    user_prompt = PromptLibrary.get_correction_prompt(initial_prompt, program_code, code_analysis)
    language = "python"  # Default language, can be improved by detecting from code
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_CORRECTION.format(language=language)

    # Check if this is Claude 3.7 Sonnet (the only one with thinking parameter)
    is_claude_3_7_sonnet = model == "claude-3-7-sonnet-20250219"
    
    # Check if this is Claude Opus 4.0 (has lower token limit)
    is_claude_opus_4 = model == "claude-opus-4-20250514"
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(initial_prompt, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in CLAUDE_PARAMS else "default"
    params = CLAUDE_PARAMS[model_key][task_type]
    
    # Set max tokens based on model
    max_tokens = 32000 if is_claude_opus_4 else MAX_OUTPUT_TOKENS
    
    try:
        # Set up basic parameters
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ],
            "stream": True  # Keep streaming enabled for long requests as recommended
        }
        
        # Add thinking parameter for Claude 3.7 Sonnet with appropriate budget
        if is_claude_3_7_sonnet and params["thinking_budget"] > 0:
            # Reduce thinking budget to prevent overload
            reduced_budget = min(params["thinking_budget"], 4000)
            kwargs["thinking"] = {
                "type": "enabled", 
                "budget_tokens": reduced_budget
            }
            # When thinking is enabled, cannot set temperature
        elif temp_allowed:
            # Only add temperature if thinking is not enabled
            kwargs["temperature"] = params["temperature"]
        
        # Stream the response and collect the content
        try:
            response = client.messages.create(**kwargs)
            content_chunks = []
            
            for chunk in response:
                if hasattr(chunk, 'delta') and hasattr(chunk.delta, 'text') and chunk.delta.text:
                    content_chunks.append(chunk.delta.text)
            
            full_text = ''.join(content_chunks)
            return extract_delimited_output(full_text)
        except Exception as stream_error:
            # If streaming fails, try again without streaming
            logger.warning(f"Streaming failed: {str(stream_error)}. Trying without streaming.")
            kwargs["stream"] = False
            response = client.messages.create(**kwargs)
            
            # Extract content from non-streaming response
            full_text = ""
            if hasattr(response, 'content') and isinstance(response.content, list):
                full_text = "".join(item.text for item in response.content if hasattr(item, 'text'))
            elif hasattr(response, 'content') and hasattr(response.content, 'text'):
                full_text = response.content.text
            else:
                full_text = str(response)
            
            return extract_delimited_output(full_text)
            
    except Exception as e:
        error_msg = f"Error correcting code with Claude: {str(e)}"
        logger.error(error_msg)
        return error_msg, "[]"

#-----------------------------------------------------------------------------
# Gemini Implementation
#-----------------------------------------------------------------------------

def generate_with_gemini(client, model, description, language, file_content=None, temp_allowed=True):
    """
    Generates initial code using Gemini models with optimized parameters
    
    Args:
        client: Gemini client
        model: Model name
        description: Program description
        language: Programming language
        file_content: Optional content of an existing program file to modify
        temp_allowed: Whether the model allows temperature parameter
    
    Returns:
        Generated code
    """
    # Get standardized prompt
    prompt_text = PromptLibrary.get_generation_prompt(description, language, file_content)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_GEN.format(language=language)
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(description, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in GEMINI_PARAMS else "default"
    params = GEMINI_PARAMS[model_key][task_type]
    
    try:
        from google.genai import types
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            config_params["temperature"] = params["temperature"]
            config_params["top_p"] = params["top_p"]
            config_params["top_k"] = params["top_k"]
        
        # Add thinking_budget with appropriate value
        config_params["thinking_config"] = types.ThinkingConfig(
            thinking_budget=params["thinking_budget"]
        )
            
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(**config_params),
            contents=prompt_text
        )
        
        # Extract content
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                return response.candidates[0].content.parts[0].text
        
        # Fallback to string representation
        return str(response)
            
    except Exception as e:
        error_msg = f"Error generating code with Gemini: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def audit_with_gemini(client, model, prompt_text, code_text, temp_allowed=True):
    """
    Analyzes code using Gemini models with optimized parameters
    
    Args:
        client: Gemini client
        model: Model name
        prompt_text: Original prompt description
        code_text: Code to analyze
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Structured analysis
    """
    # Get standardized audit prompt
    prompt = PromptLibrary.get_audit_prompt(prompt_text, code_text)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_AUDIT.format(language="")
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(prompt_text, "")
    
    # Get optimal parameters based on model and task type
    model_key = model if model in GEMINI_PARAMS else "default"
    params = GEMINI_PARAMS[model_key][task_type]
    
    try:
        from google.genai import types
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            config_params["temperature"] = params["temperature"]
            config_params["top_p"] = params["top_p"]
            config_params["top_k"] = params["top_k"]
        
        # Add thinking_budget with appropriate value
        config_params["thinking_config"] = types.ThinkingConfig(
            thinking_budget=params["thinking_budget"]
        )
            
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(**config_params),
            contents=prompt
        )
        
        # Extract content
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                return response.candidates[0].content.parts[0].text
        
        # Fallback to string representation
        return str(response)
            
    except Exception as e:
        error_msg = f"Error analyzing code with Gemini: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def correct_with_gemini(client, model, initial_prompt, program_code, code_analysis, temp_allowed=True):
    """
    Corrects code based on audit analysis using Gemini models with optimized parameters
    
    Args:
        client: Gemini client
        model: Model name
        initial_prompt: Original task description
        program_code: Current program code
        code_analysis: Audit results
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    # Get standardized correction prompt
    user_prompt = PromptLibrary.get_correction_prompt(initial_prompt, program_code, code_analysis)
    language = "python"  # Default language, can be improved by detecting from code
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_CORRECTION.format(language=language)

    # Determine task type for parameter optimization
    task_type = determine_task_type(initial_prompt, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in GEMINI_PARAMS else "default"
    params = GEMINI_PARAMS[model_key][task_type]
    
    try:
        from google.genai import types
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            config_params["temperature"] = params["temperature"]
            config_params["top_p"] = params["top_p"]
            config_params["top_k"] = params["top_k"]
        
        # Add thinking_budget with appropriate value
        config_params["thinking_config"] = types.ThinkingConfig(
            thinking_budget=params["thinking_budget"]
        )
            
        response = client.models.generate_content(
            model=model,
            config=types.GenerateContentConfig(**config_params),
            contents=user_prompt
        )
        
        full_text = ""
        if hasattr(response, 'candidates') and len(response.candidates) > 0:
            if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                full_text = response.candidates[0].content.parts[0].text
        else:
            # Fallback to string representation
            full_text = str(response)
        
        # Parse response with delimiters
        return extract_delimited_output(full_text)
        
    except Exception as e:
        error_msg = f"Error correcting code with Gemini: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

#-----------------------------------------------------------------------------
# DeepSeek Implementation
#-----------------------------------------------------------------------------

def generate_with_deepseek(client, model, description, language, file_content=None, temp_allowed=True):
    """
    Generates initial code using DeepSeek models with optimized parameters
    
    Args:
        client: DeepSeek client (OpenAI client with custom base_url)
        model: Model name
        description: Program description
        language: Programming language
        file_content: Optional content of an existing program file to modify
        temp_allowed: Whether the model allows temperature parameter
    
    Returns:
        Generated code
    """
    # Get enhanced prompt with step-by-step instructions for DeepSeek
    prompt_text = PromptLibrary.get_deepseek_generation_prompt(description, language, file_content)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_DEEPSEEK_CODE_GEN.format(language=language)
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(description, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in DEEPSEEK_PARAMS else "default"
    params = DEEPSEEK_PARAMS[model_key][task_type]
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            "stream": False
        }
        
        # Add temperature and top_p parameters regardless of temp_allowed
        # DeepSeek R1 requires specific temperature settings (0.6)
        kwargs["temperature"] = params["temperature"]
        kwargs["top_p"] = params["top_p"]
            
        response = client.chat.completions.create(**kwargs)
        
        # Extract content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        
        # Fallback to string representation
        return str(response)
            
    except Exception as e:
        error_msg = f"Error generating code with DeepSeek: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def audit_with_deepseek(client, model, prompt_text, code_text, temp_allowed=True):
    """
    Analyzes code using DeepSeek models with optimized parameters
    
    Args:
        client: DeepSeek client (OpenAI client with custom base_url)
        model: Model name
        prompt_text: Original prompt description
        code_text: Code to analyze
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Structured analysis
    """
    # Get enhanced audit prompt with step-by-step instructions for DeepSeek
    prompt = PromptLibrary.get_deepseek_audit_prompt(prompt_text, code_text)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_DEEPSEEK_CODE_AUDIT.format(language="")
    
    # Determine task type for parameter optimization
    task_type = determine_task_type(prompt_text, "")
    
    # Get optimal parameters based on model and task type
    model_key = model if model in DEEPSEEK_PARAMS else "default"
    params = DEEPSEEK_PARAMS[model_key][task_type]
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        # Add temperature and top_p parameters
        # DeepSeek R1 requires specific temperature settings (0.6)
        kwargs["temperature"] = params["temperature"]
        kwargs["top_p"] = params["top_p"]
            
        response = client.chat.completions.create(**kwargs)
        
        # Extract content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        
        # Fallback to string representation
        return str(response)
            
    except Exception as e:
        error_msg = f"Error analyzing code with DeepSeek: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def correct_with_deepseek(client, model, initial_prompt, program_code, code_analysis, temp_allowed=True):
    """
    Corrects code based on audit analysis using DeepSeek models with optimized parameters
    
    Args:
        client: DeepSeek client (OpenAI client with custom base_url)
        model: Model name
        initial_prompt: Original task description
        program_code: Current program code
        code_analysis: Audit results
        temp_allowed: Whether the model allows temperature parameter
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    # Get enhanced correction prompt with step-by-step instructions for DeepSeek
    user_prompt = PromptLibrary.get_deepseek_correction_prompt(initial_prompt, program_code, code_analysis)
    language = "python"  # Default language, can be improved by detecting from code
    system_prompt = PromptLibrary.SYSTEM_PROMPT_DEEPSEEK_CODE_CORRECTION.format(language=language)

    # Determine task type for parameter optimization
    task_type = determine_task_type(initial_prompt, language)
    
    # Get optimal parameters based on model and task type
    model_key = model if model in DEEPSEEK_PARAMS else "default"
    params = DEEPSEEK_PARAMS[model_key][task_type]
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        # Add temperature and top_p parameters
        # DeepSeek R1 requires specific temperature settings (0.6)
        kwargs["temperature"] = params["temperature"]
        kwargs["top_p"] = params["top_p"]
            
        response = client.chat.completions.create(**kwargs)
        
        full_text = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                full_text = response.choices[0].message.content
        else:
            # Fallback to string representation
            full_text = str(response)
        
        # Parse response with delimiters
        return extract_delimited_output(full_text)
        
    except Exception as e:
        error_msg = f"Error correcting code with DeepSeek: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

#-----------------------------------------------------------------------------
# Generic Request Dispatcher Functions
#-----------------------------------------------------------------------------

def generate_code(llm_info, description, language, file_content=None):
    """
    Generates initial code using the specified LLM
    
    Args:
        llm_info: Dictionary with 'model' and 'family' keys
        description: Program description
        language: Programming language
        file_content: Optional content of an existing program file to modify
        
    Returns:
        Generated code
    """
    family = llm_info.get('family')
    model = llm_info.get('model')
    temp_allowed = llm_info.get('temperature_allowed', False)
    
    try:
        client = create_client(family)
        
        if family == "OpenAI":
            return generate_with_openai(client, model, description, language, file_content, temp_allowed)
        elif family == "Claude":
            return generate_with_claude(client, model, description, language, file_content, temp_allowed)
        elif family == "Gemini":
            return generate_with_gemini(client, model, description, language, file_content, temp_allowed)
        elif family == "DeepSeek":
            return generate_with_deepseek(client, model, description, language, file_content, temp_allowed)
        # Add support for other families here
        else:
            return f"Error: Unsupported LLM family {family}"
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_code(llm_info, prompt_text, code_text):
    """
    Analyzes code using the specified LLM
    
    Args:
        llm_info: Dictionary with 'model' and 'family' keys
        prompt_text: Original prompt description
        code_text: Code to analyze
        
    Returns:
        Structured analysis
    """
    family = llm_info.get('family')
    model = llm_info.get('model')
    temp_allowed = llm_info.get('temperature_allowed', False)
    
    try:
        client = create_client(family)
        
        if family == "OpenAI":
            return audit_with_openai(client, model, prompt_text, code_text, temp_allowed)
        elif family == "Claude":
            return audit_with_claude(client, model, prompt_text, code_text, temp_allowed)
        elif family == "Gemini":
            return audit_with_gemini(client, model, prompt_text, code_text, temp_allowed)
        elif family == "DeepSeek":
            return audit_with_deepseek(client, model, prompt_text, code_text, temp_allowed)
        # Add support for other families here
        else:
            return f"Error: Unsupported LLM family {family}"
    except Exception as e:
        return f"Error: {str(e)}"

def correct_code(llm_info, initial_prompt, program_code, code_analysis):
    """
    Corrects code based on audit analysis using the specified LLM
    
    Args:
        llm_info: Dictionary with 'model' and 'family' keys
        initial_prompt: Original task description
        program_code: Current program code
        code_analysis: Audit results
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    family = llm_info.get('family')
    model = llm_info.get('model')
    temp_allowed = llm_info.get('temperature_allowed', False)
    
    try:
        client = create_client(family)
        
        if family == "OpenAI":
            return correct_with_openai(client, model, initial_prompt, program_code, code_analysis, temp_allowed)
        elif family == "Claude":
            return correct_with_claude(client, model, initial_prompt, program_code, code_analysis, temp_allowed)
        elif family == "Gemini":
            return correct_with_gemini(client, model, initial_prompt, program_code, code_analysis, temp_allowed)
        elif family == "DeepSeek":
            return correct_with_deepseek(client, model, initial_prompt, program_code, code_analysis, temp_allowed)
        # Add support for other families here
        else:
            return f"Error: Unsupported LLM family {family}", "[]"
    except Exception as e:
        return f"Error: {str(e)}", "[]"

#-----------------------------------------------------------------------------
# Command-Line Interface
#-----------------------------------------------------------------------------

def setup_cli_parser():
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(description="CodingAPI - Generate, analyze, and correct code using LLMs")
    
    # Command group
    subparsers = parser.add_subparsers(dest="command", help="Command")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate code")
    generate_parser.add_argument("prompt", help="The coding task prompt")
    generate_parser.add_argument("--provider", "-p", default="OpenAI", 
                               help="LLM provider to use (default: OpenAI)")
    generate_parser.add_argument("--model", "-m", help="Model name (provider-specific)")
    generate_parser.add_argument("--language", "-l", default="python", 
                               help="Programming language (default: python)")
    generate_parser.add_argument("--max-tokens", "-t", type=int, default=1000, 
                               help="Maximum tokens in response (default: 1000)")
    generate_parser.add_argument("--output", "-o", help="Output file (default: print to console)")
    generate_parser.add_argument("--no-highlight", action="store_true", 
                               help="Disable syntax highlighting")
    
    # Config command
    config_parser = subparsers.add_parser("config", help="Configure API keys")
    config_parser.add_argument("provider", nargs="?", help="Provider to configure")
    config_parser.add_argument("--list", "-l", action="store_true", 
                              help="List supported providers")
    
    # Audit command
    audit_parser = subparsers.add_parser("audit", help="Audit existing code")
    audit_parser.add_argument("file", help="File containing code to audit")
    audit_parser.add_argument("--prompt", "-p", required=True, 
                            help="Description of what the code should do")
    audit_parser.add_argument("--provider", default="Claude", 
                            help="LLM provider to use (default: Claude)")
    audit_parser.add_argument("--model", "-m", help="Model name (provider-specific)")
    audit_parser.add_argument("--output", "-o", help="Output file (default: print to console)")
    
    # Run command for automated generation + audit + correction
    run_parser = subparsers.add_parser("run", help="Run automated generation, audit, correction pipeline")
    run_parser.add_argument("prompt", help="The coding task prompt")
    run_parser.add_argument("--language", "-l", default="python", 
                          help="Programming language (default: python)")
    run_parser.add_argument("--gen-provider", default="OpenAI", 
                          help="Provider for generation (default: OpenAI)")
    run_parser.add_argument("--audit-provider", default="Claude", 
                          help="Provider for auditing (default: Claude)")
    run_parser.add_argument("--iterations", "-i", type=int, default=3, 
                          help="Number of iterations (default: 3)")
    run_parser.add_argument("--output-dir", "-o", help="Output directory (default: ./output)")
    
    # Common arguments
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    return parser

def handle_generate_command(args):
    """Handle the 'generate' command."""
    # Find model name if not specified
    model = args.model
    if not model:
        for model_name, info in LLM_MAP.items():
            if info["family"] == args.provider:
                model = info["model"]
                break
    
    # Make sure we have a model
    if not model:
        logger.error(f"No model specified for provider {args.provider}")
        return False
    
    # Set up LLM info
    llm_info = {
        "family": args.provider,
        "model": model,
        "temperature_allowed": True
    }
    
    # Generate code
    code = generate_code(llm_info, args.prompt, args.language)
    
    if code.startswith("Error:"):
        logger.error(code)
        return False
    
    # Output the code
    if args.output:
        # Save to file
        with open(args.output, "w") as f:
            f.write(code)
        logger.info(f"Code saved to: {args.output}")
    else:
        # Print to console
        if RICH_AVAILABLE and not args.no_highlight:
            console = Console()
            syntax = Syntax(code, args.language, theme="monokai", line_numbers=True)
            console.print(Panel(syntax, title=f"{args.language} Code"))
        else:
            print(code)
    
    return True

def handle_config_command(args):
    """Handle the 'config' command."""
    config = SecureConfig()
    
    if args.list:
        # List supported providers
        providers = ["OpenAI", "Claude", "Gemini", "DeepSeek"]
        
        if RICH_AVAILABLE:
            console = Console()
            console.print("[bold]Supported LLM Providers:[/]")
            
            for provider in providers:
                api_key = config.get_api_key(provider)
                
                if api_key:
                    status = f"[green]Configured[/] (key: {api_key[:4]}...{api_key[-4:]})"
                else:
                    status = "[red]Not configured[/]"
                
                console.print(f"  - {provider}: {status}")
        else:
            print("Supported LLM Providers:")
            for provider in providers:
                api_key = config.get_api_key(provider)
                status = "Configured" if api_key else "Not configured"
                
                if api_key:
                    print(f"  - {provider}: {status} (key: {api_key[:4]}...{api_key[-4:]})")
                else:
                    print(f"  - {provider}: {status}")
    elif args.provider:
        # Configure specific provider
        provider = args.provider
        
        # Get current key if exists
        current_key = config.get_api_key(provider)
        if current_key:
            print(f"Current API key: {current_key[:4]}...{current_key[-4:]}")
        
        # Prompt for new key
        new_key = input(f"Enter {provider} API key (press Enter to keep current): ")
        
        if new_key:
            if config.set_api_key(provider, new_key):
                print(f"API key for {provider} stored successfully!")
            else:
                print(f"Failed to store API key for {provider}.")
        else:
            print("Keeping existing API key.")
    else:
        print("Please specify a provider to configure or use --list")
        return False
    
    return True

def handle_audit_command(args):
    """Handle the 'audit' command."""
    # Read code from file
    try:
        with open(args.file, "r") as f:
            code = f.read()
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return False
    
    # Find model name if not specified
    model = args.model
    if not model:
        for model_name, info in LLM_MAP.items():
            if info["family"] == args.provider:
                model = info["model"]
                break
    
    # Make sure we have a model
    if not model:
        logger.error(f"No model specified for provider {args.provider}")
        return False
    
    # Set up LLM info
    llm_info = {
        "family": args.provider,
        "model": model,
        "temperature_allowed": True
    }
    
    # Analyze code
    analysis = analyze_code(llm_info, args.prompt, code)
    
    if analysis.startswith("Error:"):
        logger.error(analysis)
        return False
    
    # Output the analysis
    if args.output:
        # Save to file
        with open(args.output, "w") as f:
            f.write(analysis)
        logger.info(f"Analysis saved to: {args.output}")
    else:
        # Print to console
        if RICH_AVAILABLE:
            console = Console()
            console.print(Panel(analysis, title="Code Analysis"))
        else:
            print(analysis)
    
    return True

def handle_run_command(args):
    """Handle the 'run' command."""
    # Create output directory
    output_dir = args.output_dir or f"output_{int(time.time())}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find generation model
    gen_model = None
    for model_name, info in LLM_MAP.items():
        if info["family"] == args.gen_provider:
            gen_model = info["model"]
            break
    
    if not gen_model:
        logger.error(f"No model found for provider {args.gen_provider}")
        return False
    
    # Find audit model
    audit_model = None
    for model_name, info in LLM_MAP.items():
        if info["family"] == args.audit_provider:
            audit_model = info["model"]
            break
    
    if not audit_model:
        logger.error(f"No model found for provider {args.audit_provider}")
        return False
    
    # Set up LLM info
    gen_llm_info = {
        "family": args.gen_provider,
        "model": gen_model,
        "temperature_allowed": True
    }
    
    audit_llm_info = {
        "family": args.audit_provider,
        "model": audit_model,
        "temperature_allowed": True
    }
    
    if RICH_AVAILABLE:
        console = Console()
        console.print(f"[bold]Running automated pipeline:[/]")
        console.print(f"Prompt: {args.prompt}")
        console.print(f"Language: {args.language}")
        console.print(f"Generation provider: {args.gen_provider} ({gen_model})")
        console.print(f"Audit provider: {args.audit_provider} ({audit_model})")
        console.print(f"Iterations: {args.iterations}")
        console.print(f"Output directory: {output_dir}")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}[/]"),
            TimeElapsedColumn(),
            console=console
        ) as progress:
            # Initial code generation
            task = progress.add_task("Generating initial code...", total=None)
            code = generate_code(gen_llm_info, args.prompt, args.language)
            progress.update(task, completed=True)
            
            if code.startswith("Error:"):
                console.print(f"[bold red]Error generating code:[/] {code}")
                return False
            
            # Remove surrounding quotes if present
            code = remove_surrounding_quotes(code)
            
            # Save initial code
            initial_code_file = os.path.join(output_dir, f"initial_code{extension_for_language(args.language)}")
            with open(initial_code_file, "w") as f:
                f.write(code)
            
            console.print(f"[green]Initial code saved to:[/] {initial_code_file}")
            
            # Iterate through audit and correction
            current_code = code
            for i in range(1, args.iterations + 1):
                console.print(f"[bold]Iteration {i}/{args.iterations}[/]")
                
                # Audit code
                task = progress.add_task(f"Auditing code (iteration {i})...", total=None)
                analysis = analyze_code(audit_llm_info, args.prompt, current_code)
                progress.update(task, completed=True)
                
                if analysis.startswith("Error:"):
                    console.print(f"[bold red]Error auditing code:[/] {analysis}")
                    break
                
                # Save audit
                audit_file = os.path.join(output_dir, f"audit_{i}.txt")
                with open(audit_file, "w") as f:
                    f.write(analysis)
                
                # Parse bug counts
                bug_counts = parse_bug_counts(analysis)
                console.print(f"Bugs found: Critical={bug_counts[0]}, Serious={bug_counts[1]}, Non-critical={bug_counts[2]}, Suggestions={bug_counts[3]}")
                
                # Check if we need to continue
                if bug_counts[0] == 0 and bug_counts[1] == 0:
                    console.print("[green]No critical or serious bugs found. Stopping iterations.[/]")
                    break
                
                # Correct code
                task = progress.add_task(f"Correcting code (iteration {i})...", total=None)
                corrected_code, corrections = correct_code(gen_llm_info, args.prompt, current_code, analysis)
                progress.update(task, completed=True)
                
                if isinstance(corrected_code, str) and corrected_code.startswith("Error:"):
                    console.print(f"[bold red]Error correcting code:[/] {corrected_code}")
                    break
                
                # Clean corrected code
                corrected_code = remove_surrounding_quotes(corrected_code)
                
                # Save corrected code
                corrected_file = os.path.join(output_dir, f"iteration_{i}{extension_for_language(args.language)}")
                with open(corrected_file, "w") as f:
                    f.write(corrected_code)
                
                # Parse corrections
                fixed_counts = parse_corrections(corrections, bug_counts)
                console.print(f"Fixed: Critical={fixed_counts[0]}/{bug_counts[0]}, Serious={fixed_counts[1]}/{bug_counts[1]}, Non-critical={fixed_counts[2]}/{bug_counts[2]}, Suggestions={fixed_counts[3]}/{bug_counts[3]}")
                
                # Update current code for next iteration
                current_code = corrected_code
            
            # Save final code
            final_file = os.path.join(output_dir, f"final_code{extension_for_language(args.language)}")
            with open(final_file, "w") as f:
                f.write(current_code)
            
            console.print(f"[bold green]Final code saved to:[/] {final_file}")
    else:
        # Non-rich version
        print(f"Running automated pipeline:")
        print(f"Prompt: {args.prompt}")
        print(f"Language: {args.language}")
        print(f"Generation provider: {args.gen_provider} ({gen_model})")
        print(f"Audit provider: {args.audit_provider} ({audit_model})")
        print(f"Iterations: {args.iterations}")
        print(f"Output directory: {output_dir}")
        
        # Initial code generation
        print("Generating initial code...")
        code = generate_code(gen_llm_info, args.prompt, args.language)
        
        if code.startswith("Error:"):
            print(f"Error generating code: {code}")
            return False
        
        # Remove surrounding quotes if present
        code = remove_surrounding_quotes(code)
        
        # Save initial code
        initial_code_file = os.path.join(output_dir, f"initial_code{extension_for_language(args.language)}")
        with open(initial_code_file, "w") as f:
            f.write(code)
        
        print(f"Initial code saved to: {initial_code_file}")
        
        # Iterate through audit and correction
        current_code = code
        for i in range(1, args.iterations + 1):
            print(f"Iteration {i}/{args.iterations}")
            
            # Audit code
            print(f"Auditing code (iteration {i})...")
            analysis = analyze_code(audit_llm_info, args.prompt, current_code)
            
            if analysis.startswith("Error:"):
                print(f"Error auditing code: {analysis}")
                break
            
            # Save audit
            audit_file = os.path.join(output_dir, f"audit_{i}.txt")
            with open(audit_file, "w") as f:
                f.write(analysis)
            
            # Parse bug counts
            bug_counts = parse_bug_counts(analysis)
            print(f"Bugs found: Critical={bug_counts[0]}, Serious={bug_counts[1]}, Non-critical={bug_counts[2]}, Suggestions={bug_counts[3]}")
            
            # Check if we need to continue
            if bug_counts[0] == 0 and bug_counts[1] == 0:
                print("No critical or serious bugs found. Stopping iterations.")
                break
            
            # Correct code
            print(f"Correcting code (iteration {i})...")
            corrected_code, corrections = correct_code(gen_llm_info, args.prompt, current_code, analysis)
            
            if isinstance(corrected_code, str) and corrected_code.startswith("Error:"):
                print(f"Error correcting code: {corrected_code}")
                break
            
            # Clean corrected code
            corrected_code = remove_surrounding_quotes(corrected_code)
            
            # Save corrected code
            corrected_file = os.path.join(output_dir, f"iteration_{i}{extension_for_language(args.language)}")
            with open(corrected_file, "w") as f:
                f.write(corrected_code)
            
            # Parse corrections
            fixed_counts = parse_corrections(corrections, bug_counts)
            print(f"Fixed: Critical={fixed_counts[0]}/{bug_counts[0]}, Serious={fixed_counts[1]}/{bug_counts[1]}, Non-critical={fixed_counts[2]}/{bug_counts[2]}, Suggestions={fixed_counts[3]}/{bug_counts[3]}")
            
            # Update current code for next iteration
            current_code = corrected_code
        
        # Save final code
        final_file = os.path.join(output_dir, f"final_code{extension_for_language(args.language)}")
        with open(final_file, "w") as f:
            f.write(current_code)
        
        print(f"Final code saved to: {final_file}")
    
    return True

def cli_main():
    """Main entry point for the command-line interface."""
    parser = setup_cli_parser()
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    
    # Process commands
    if args.command == "generate":
        return handle_generate_command(args)
    elif args.command == "config":
        return handle_config_command(args)
    elif args.command == "audit":
        return handle_audit_command(args)
    elif args.command == "run":
        return handle_run_command(args)
    else:
        # No command specified, show help
        parser.print_help()
        return False

def open_run_app_dialog(parent):
    """Open a dialog to select and run a Python application."""
    
    # Initialize working directory if not set
    if not hasattr(open_run_app_dialog, 'current_working_dir'):
        open_run_app_dialog.current_working_dir = os.getcwd()
    
    # Create dialog window
    run_dialog = tk.Toplevel(parent)
    run_dialog.title("Run Application")
    run_dialog.geometry("800x600")  # Increased size from 600x400 to 800x600
    run_dialog.minsize(800, 600)    # Set minimum size to ensure all elements fit
    run_dialog.grab_set()  # Make dialog modal
    
    # Main frame
    main_frame = ttk.Frame(run_dialog, padding=10)
    main_frame.pack(fill=tk.BOTH, expand=True)
    
    # Title label
    title_label = ttk.Label(main_frame, text="Select Python or Terminal Output File", font=("Arial", 14, "bold"))
    title_label.pack(anchor="w", pady=(0, 10))
    Tooltip(title_label, "Choose a Python program to run or a terminal output file to view")
    
    # Frame for file browser
    browser_frame = ttk.Frame(main_frame)
    browser_frame.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Current directory label
    dir_frame = ttk.Frame(browser_frame)
    dir_frame.pack(fill=tk.X, pady=(0, 5))
    
    dir_label = ttk.Label(dir_frame, text="Directory:")
    dir_label.pack(side=tk.LEFT, padx=(0, 5))
    Tooltip(dir_label, "Current directory being browsed")
    
    current_dir_var = tk.StringVar(value=open_run_app_dialog.current_working_dir)
    current_dir_entry = ttk.Entry(dir_frame, textvariable=current_dir_var, width=75)  # Slightly reduced width to make room for refresh button
    current_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    
    # Add refresh button with circular arrows symbol (↻)
    refresh_button = ttk.Button(dir_frame, text="↻", width=3)
    refresh_button.pack(side=tk.LEFT, padx=5)
    Tooltip(refresh_button, "Refresh directory contents")
    
    # File listbox with scrollbar
    list_frame = ttk.Frame(browser_frame)
    list_frame.pack(fill=tk.BOTH, expand=True)
    
    file_listbox = tk.Listbox(list_frame, width=100, height=20)  # Increased dimensions
    file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    Tooltip(file_listbox, "Double-click on directories to navigate, or select a file to run or view")
    
    scrollbar = ttk.Scrollbar(list_frame, orient="vertical", command=file_listbox.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    file_listbox.config(yscrollcommand=scrollbar.set)
    
    # Selected file variable and label
    selected_file_var = tk.StringVar()
    selected_file_frame = ttk.Frame(main_frame)
    selected_file_frame.pack(fill=tk.X, pady=5)
    
    selected_label = ttk.Label(selected_file_frame, text="Selected File:")
    selected_label.pack(side=tk.LEFT, padx=(0, 5))
    Tooltip(selected_label, "Currently selected file")
    
    selected_file_entry = ttk.Entry(selected_file_frame, textvariable=selected_file_var, width=80)
    selected_file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
    
    # Button frame
    button_frame = ttk.Frame(main_frame)
    button_frame.pack(fill=tk.X, pady=10)
    
    # Function to update file list based on current directory
    def update_file_list():
        file_listbox.delete(0, tk.END)
        
        # Add parent directory option
        file_listbox.insert(tk.END, "..")
        
        try:
            # List directories first
            entries = os.listdir(current_dir_var.get())
            directories = [d for d in entries if os.path.isdir(os.path.join(current_dir_var.get(), d))]
            for d in sorted(directories):
                file_listbox.insert(tk.END, f"[DIR] {d}")
            
            # Then list .py and .trm files
            files = [f for f in entries if f.endswith(('.py', '.trm')) and 
                     os.path.isfile(os.path.join(current_dir_var.get(), f))]
            for f in sorted(files):
                file_listbox.insert(tk.END, f)
        except Exception as e:
            messagebox.showerror("Error", f"Could not read directory: {str(e)}")
    
    # Function for refresh button
    def refresh_directory():
        # Re-read the current directory
        try:
            if os.path.isdir(current_dir_var.get()):
                # Just call update_file_list directly to refresh the directory contents
                update_file_list()
            else:
                messagebox.showerror("Error", "Current path is not a valid directory")
                # Reset to the last known valid directory
                current_dir_var.set(open_run_app_dialog.current_working_dir)
                update_file_list()
        except Exception as e:
            messagebox.showerror("Error", f"Could not refresh directory: {str(e)}")
    
    # Set command for refresh button
    refresh_button.config(command=refresh_directory)
    
    # Initial file list update
    update_file_list()
    
    # Handle directory or file selection
    def on_item_select(event):
        if not file_listbox.curselection():
            return
            
        selected_item = file_listbox.get(file_listbox.curselection()[0])
        
        # Handle parent directory
        if selected_item == "..":
            current_dir = current_dir_var.get()
            parent_dir = os.path.dirname(current_dir)
            current_dir_var.set(parent_dir)
            update_file_list()
            selected_file_var.set("")
            return
        
        # Handle directory
        if selected_item.startswith("[DIR] "):
            dir_name = selected_item[6:]  # Remove "[DIR] " prefix
            new_dir = os.path.join(current_dir_var.get(), dir_name)
            current_dir_var.set(new_dir)
            update_file_list()
            selected_file_var.set("")
            return
            
        # Handle file selection
        selected_file_var.set(selected_item)
    
    file_listbox.bind('<<ListboxSelect>>', on_item_select)
    
    # Double-click to enter directory or select file
    def on_item_double_click(event):
        on_item_select(event)
        
        if not file_listbox.curselection():
            return
            
        selected_item = file_listbox.get(file_listbox.curselection()[0])
        
        # If not a directory, consider it a file selection
        if not selected_item.startswith("[DIR] ") and selected_item != "..":
            selected_file_var.set(selected_item)
            if selected_item.endswith('.py'):
                on_run_app()
            else:
                on_view_file()
    
    file_listbox.bind('<Double-1>', on_item_double_click)
    
    # View file function
    def on_view_file():
        selected_file = selected_file_var.get()
        if not selected_file:
            messagebox.showerror("Error", "No file selected")
            return
            
        file_path = os.path.join(current_dir_var.get(), selected_file)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
                
            # Create view window
            view_window = tk.Toplevel(run_dialog)
            view_window.title(f"View: {selected_file}")
            view_window.geometry("900x700")  # Also increased this window size
            view_window.minsize(800, 600)    # Set minimum size
            
            # Create text widget with scrollbar
            text_frame = ttk.Frame(view_window, padding=10)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            text_widget = scrolledtext.ScrolledText(text_frame, width=100, height=40, wrap=tk.NONE)  # Increased dimensions
            text_widget.pack(fill=tk.BOTH, expand=True)
            
            # Insert content and make read-only
            text_widget.insert(tk.END, file_content)
            text_widget.config(state=tk.DISABLED)
            
            # Enable copy functionality but disable editing
            def handle_keypress(event):
                # Check if Ctrl key is pressed
                if event.state & 0x4:  # 0x4 is the mask for Ctrl key
                    if event.keycode == 67:  # C - copy
                        return None  # Allow default handling for copy
                    elif event.keycode == 65:  # A - select all
                        text_widget.tag_add(tk.SEL, "1.0", tk.END)
                        text_widget.mark_set(tk.INSERT, "1.0")
                        text_widget.see(tk.INSERT)
                        return "break"
                return "break"  # Block other keypresses
            
            text_widget.bind("<Key>", handle_keypress)
            
            # Close button
            close_button = ttk.Button(view_window, text="Close", command=view_window.destroy)
            close_button.pack(pady=10)
            Tooltip(close_button, "Close this window")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not open file: {str(e)}")
    
    # Run application function
    def on_run_app():
        selected_file = selected_file_var.get()
        if not selected_file:
            messagebox.showerror("Error", "No file selected")
            return
            
        if not selected_file.endswith('.py'):
            messagebox.showerror("Error", "Only Python (.py) files can be executed")
            return
            
        file_path = os.path.join(current_dir_var.get(), selected_file)
        
        # Save current working directory
        open_run_app_dialog.current_working_dir = current_dir_var.get()
        
        try:
            # Run the Python script using subprocess
            process = subprocess.Popen(
                [sys.executable, file_path],
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE,
                text=True,
                cwd=current_dir_var.get()
            )
            
            # Create output window for displaying process output in real-time
            output_window = tk.Toplevel(run_dialog)
            output_window.title(f"Running: {selected_file}")
            output_window.geometry("900x700")  # Increased size
            output_window.minsize(800, 600)    # Set minimum size
            
            # Create text widget with scrollbar for output
            text_frame = ttk.Frame(output_window, padding=10)
            text_frame.pack(fill=tk.BOTH, expand=True)
            
            output_text = scrolledtext.ScrolledText(text_frame, width=100, height=40, bg="black", fg="white")  # Increased dimensions
            output_text.pack(fill=tk.BOTH, expand=True)
            
            # Function to read process output and update the window
            def read_output():
                # Non-blocking read from stdout and stderr
                stdout_line = process.stdout.readline()
                stderr_line = process.stderr.readline()
                
                # Update text widget if there's output
                if stdout_line:
                    output_text.insert(tk.END, stdout_line)
                    output_text.see(tk.END)
                    
                if stderr_line:
                    output_text.insert(tk.END, stderr_line, "error")
                    output_text.tag_configure("error", foreground="red")
                    output_text.see(tk.END)
                
                # Check if process is still running
                if process.poll() is None:
                    # Continue reading output
                    output_window.after(100, read_output)
                else:
                    # Process completed, read any remaining output
                    remaining_stdout, remaining_stderr = process.communicate()
                    
                    if remaining_stdout:
                        output_text.insert(tk.END, remaining_stdout)
                    
                    if remaining_stderr:
                        output_text.insert(tk.END, remaining_stderr, "error")
                        
                    output_text.insert(tk.END, f"\n\nProcess completed with exit code: {process.returncode}\n")
                    output_text.see(tk.END)
                    
                    # Save terminal output to .trm file
                    try:
                        trm_filename = os.path.splitext(selected_file)[0] + ".trm"
                        trm_path = os.path.join(current_dir_var.get(), trm_filename)
                        
                        with open(trm_path, 'w', encoding='utf-8') as f:
                            f.write(output_text.get("1.0", tk.END))
                            
                        output_text.insert(tk.END, f"\nTerminal output saved to: {trm_filename}\n")
                        output_text.see(tk.END)
                    except Exception as e:
                        output_text.insert(tk.END, f"\nError saving terminal output: {str(e)}\n")
                        output_text.see(tk.END)
            
            # Start reading process output
            read_output()
            
            # Close button
            close_button = ttk.Button(output_window, text="Close", command=output_window.destroy)
            close_button.pack(pady=10)
            Tooltip(close_button, "Close this window")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not run application: {str(e)}")
    
    # Cancel button - increased width for all buttons
    cancel_button = ttk.Button(button_frame, text="Cancel", width=15)
    cancel_button.pack(side=tk.LEFT, padx=10)
    Tooltip(cancel_button, "Close this window without running anything")
    
    # View button
    view_button = ttk.Button(button_frame, text="View", width=15)
    view_button.pack(side=tk.LEFT, padx=10)
    Tooltip(view_button, "View the selected file content")
    
    # Run button
    run_button = ttk.Button(button_frame, text="Run app", width=15)
    run_button.pack(side=tk.LEFT, padx=10)
    Tooltip(run_button, "Run the selected Python application")
    
    # Set button commands
    cancel_button.config(command=run_dialog.destroy)
    view_button.config(command=on_view_file)
    run_button.config(command=on_run_app)


class Tooltip:
    """
    Creates a tooltip for a given widget.
    
    Args:
        widget: The widget to add the tooltip to
        text: The tooltip text
        delay: Delay in seconds before showing the tooltip (default: 0.5)
        wrap_length: Maximum line length before wrapping (default: 250 pixels)
    """
    def __init__(self, widget, text, delay=0.5, wrap_length=250):
        self.widget = widget
        self.text = text
        self.delay = delay  # seconds before tooltip appears
        self.wrap_length = wrap_length
        self.tip_window = None
        self.id = None
        
        self.widget.bind("<Enter>", self.schedule_tip)
        self.widget.bind("<Leave>", self.hide_tip)
        self.widget.bind("<ButtonPress>", self.hide_tip)
        
    def schedule_tip(self, event=None):
        """Schedule the tooltip to appear after delay."""
        self.hide_tip()  # Cancel any pending tooltips
        self.id = self.widget.after(int(self.delay * 1000), self.show_tip)
        
    def show_tip(self):
        """Display the tooltip."""
        if self.tip_window or not self.text:
            return
            
        # Get position for tooltip - safely handle different widget types
        x = y = 0
        
        # Get widget dimensions
        widget_width = self.widget.winfo_width()
        widget_height = self.widget.winfo_height()
        
        # Calculate a reasonable position - don't try to use bbox("insert")
        x = widget_width // 2
        y = widget_height // 2
        
        # Convert to absolute coordinates
        x_root = self.widget.winfo_rootx() + x
        y_root = self.widget.winfo_rooty() + y
        
        # Create tooltip window
        self.tip_window = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)  # Remove window decorations
        tw.wm_geometry(f"+{x_root+15}+{y_root+15}")  # Position near widget
        
        # Create tooltip content
        label = ttk.Label(tw, text=self.text, justify=tk.LEFT,
                         background="#ffffe0", relief=tk.SOLID, borderwidth=1,
                         wraplength=self.wrap_length)
        label.pack(padx=3, pady=3)
        
    def hide_tip(self, event=None):
        """Hide the tooltip."""
        if self.id:
            self.widget.after_cancel(self.id)
            self.id = None
        
        if self.tip_window:
            self.tip_window.destroy()
            self.tip_window = None



def get_model_sort_key(model_name):
    """
    Returns a sort key for models to order them by release date (newest first).
    Uses negative values so that newer models appear first when sorted.
    """
    # Define order based on known release dates (newer = lower number)
    model_order = {
        # OpenAI models (newest to oldest)
        "OpenAI o3": -50,  # April 2025
        "OpenAI o4-mini": -45,  # Early 2025
        "OpenAI GPT-4.1": -40,  # Late 2024
        "OpenAI GPT-4.1-mini": -35,  # Late 2024
        "OpenAI GPT4o": -30,  # Mid 2024
        "OpenAI GPT-4 Turbo": -25,
        "OpenAI GPT-3.5 Turbo": -20,
        
        # Claude models (newest to oldest)
        "Claude 3.7 Sonnet": -50,  # February 2025
        "Claude 3.5 Sonnet": -40,  # October 2024
        "Claude 3.5 Haiku": -35,  # October 2024
        "Claude 3 Opus": -30,  # February 2024
        
        # Gemini models (newest to oldest)
        "Gemini 2.5 Pro": -50,  # March 2025
        "Gemini 2.0 Flash": -40,  # December 2024
        "Gemini 1.5 Pro": -30,  # 2024
        "Gemini 1.5 Flash": -25,  # 2024
        
        # DeepSeek models (newest to oldest)
        "DeepSeek R1": -50,  # Recent
        "DeepSeek Chat": -40,
        "DeepSeek Coder": -35,
    }
    
    # Return the order value if found, otherwise return 0 (will appear after known models)
    return model_order.get(model_name, 0)

def get_model_info(model_name):
    """
    Get model information from either LLM_MAP or DYNAMIC_MODEL_REGISTRY.
    
    Args:
        model_name: Display name of the model
        
    Returns:
        Dict with model info or None if not found
    """
    # First check the static LLM_MAP
    if model_name in LLM_MAP:
        return LLM_MAP[model_name]
    
    # Then check the dynamic registry
    if model_name in DYNAMIC_MODEL_REGISTRY:
        return DYNAMIC_MODEL_REGISTRY[model_name]
    
    # Model not found
    return None

def open_settings_dialog(root, config, update_callback, all_available_models):
    """Open a settings dialog with tabs for Files and API Keys & Models."""
    settings_dialog = tk.Toplevel(root)
    settings_dialog.title("Settings")
    settings_dialog.geometry("800x600")
    settings_dialog.minsize(800, 600)
    settings_dialog.grab_set()  # Make window modal
    
    # Main layout - use a more reliable grid layout instead of pack
    settings_dialog.grid_columnconfigure(0, weight=1)
    settings_dialog.grid_rowconfigure(0, weight=1)
    settings_dialog.grid_rowconfigure(1, weight=0)  # Fixed height for buttons row
    
    # Main content frame (for notebook)
    content_frame = ttk.Frame(settings_dialog, padding=10)
    content_frame.grid(row=0, column=0, sticky="nsew")
    
    # Title
    title_label = ttk.Label(content_frame, text="Settings", font=("Arial", 14, "bold"))
    title_label.pack(anchor="w", pady=(0, 10))
    Tooltip(title_label, "Configure application settings")
    
    # Create notebook for tabs
    notebook = ttk.Notebook(content_frame)
    notebook.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Files tab
    files_frame = ttk.Frame(notebook, padding=10)
    notebook.add(files_frame, text="Files")
    
    # Output Directory setting
    output_dir_frame = ttk.Frame(files_frame)
    output_dir_frame.pack(fill=tk.X, pady=10)
    
    # Load current settings or use defaults
    config_file = os.path.join(get_config_dir(), "settings.json")
    settings = {}
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                settings = json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    
    # Get current output directory from settings
    current_output_dir = settings.get("output_directory", "")
    
    output_dir_label = ttk.Label(output_dir_frame, text="Output Directory:", style='Bold.TLabel')
    output_dir_label.pack(anchor="w")
    Tooltip(output_dir_label, "Directory where generated code and audit files will be saved\nLeave empty to use <Project name> in current directory")
    
    # Create StringVar with the loaded value
    output_dir_var = tk.StringVar(value=current_output_dir)
    
    output_dir_entry_frame = ttk.Frame(output_dir_frame)
    output_dir_entry_frame.pack(fill=tk.X, pady=5)
    
    output_dir_entry = ttk.Entry(output_dir_entry_frame, textvariable=output_dir_var, width=60)
    output_dir_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
    
    def browse_output_dir():
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            output_dir_var.set(directory)
    
    output_dir_browse = ttk.Button(output_dir_entry_frame, text="Browse...", command=browse_output_dir)
    output_dir_browse.pack(side=tk.RIGHT)
    Tooltip(output_dir_browse, "Browse for output directory")
    
    # Help text
    help_label = ttk.Label(output_dir_frame, text="Note: Leave empty to save files in a subdirectory with project name in the current directory.", style='TLabel')
    help_label.pack(anchor="w", pady=5)
    
    # API Keys & Models tab
    api_frame = ttk.Frame(notebook, padding=10)
    notebook.add(api_frame, text="API Keys & Models")
    
    # Create a notebook for provider tabs
    api_notebook = ttk.Notebook(api_frame)
    api_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
    
    # Variables to store model selections - IMPORTANT: Use provider-specific dictionaries
    coding_vars = {}
    auditing_vars = {}
    
    # Get current selections
    selected_models = config.get_selected_models()
    
    # Function to create a tab for each provider
    def create_provider_tab(provider_name):
        frame = ttk.Frame(api_notebook, padding=10)
        api_notebook.add(frame, text=provider_name)
        
        # Current API key
        current_key = config.get_api_key(provider_name) or ""
        masked_key = current_key[:4] + "*" * (len(current_key) - 8) + current_key[-4:] if current_key and len(current_key) > 8 else current_key
        
        lbl_current = ttk.Label(frame, text="Current API Key:", style='Bold.TLabel')
        lbl_current.pack(anchor="w", pady=(0, 5))
        Tooltip(lbl_current, f"Your current {provider_name} API key")
        
        lbl_mask = ttk.Label(frame, text=masked_key or "Not configured")
        lbl_mask.pack(anchor="w", pady=(0, 10))
        
        # Add information about setting API keys
        env_var_name = "ANTHROPIC_API_KEY" if provider_name == "Claude" else f"{provider_name.upper()}_API_KEY"
        lbl_instructions = ttk.Label(frame, text=f"Set API key through environment variable: {env_var_name}\nor in APIKeys file", wraplength=400)
        lbl_instructions.pack(anchor="w", pady=(10, 15))
        Tooltip(lbl_instructions, f"Instructions for setting {provider_name} API key")
        
        # Separator
        separator = ttk.Separator(frame, orient=tk.HORIZONTAL)
        separator.pack(fill=tk.X, pady=15)
        
        # Model selection section
        lbl_models_title = ttk.Label(frame, text="Model Selection:", style='Bold.TLabel')
        lbl_models_title.pack(anchor="w", pady=(5, 10))
        Tooltip(lbl_models_title, "Select which models to use for coding and auditing")
        
        # Get available models and sort them by release date (newest first)
        available_models = all_available_models.get(provider_name, [])
        available_models_sorted = sorted(available_models, key=get_model_sort_key)
        
        # Create a notebook for Coding Models and Audit Models tabs
        model_notebook = ttk.Notebook(frame)
        model_notebook.pack(fill=tk.BOTH, expand=True, pady=10)
        
        # Helper function to create model selection tab
        def create_model_selection_tab(tab_name, var_dict_key, selected_list):
            tab_frame = ttk.Frame(model_notebook)
            model_notebook.add(tab_frame, text=tab_name)
            
            # Create scrollable frame
            canvas = tk.Canvas(tab_frame, height=200)  # Set fixed height
            scrollbar = ttk.Scrollbar(tab_frame, orient="vertical", command=canvas.yview)
            scrollable_frame = ttk.Frame(canvas)
            
            scrollable_frame.bind(
                "<Configure>",
                lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
            )
            
            canvas_window = canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
            
            # Configure canvas to expand horizontally
            def configure_canvas(event):
                canvas.itemconfig(canvas_window, width=event.width)
            canvas.bind('<Configure>', configure_canvas)
            
            canvas.configure(yscrollcommand=scrollbar.set)
            
            # Create provider-specific key for variables
            provider_var_key = f"{provider_name}_{var_dict_key}"
            
            # Add checkboxes for models
            for model in available_models_sorted:
                var = tk.BooleanVar()
                var.set(model in selected_list)
                
                # Store in the appropriate dictionary with provider-specific key
                if var_dict_key == "coding":
                    coding_vars[f"{provider_name}:{model}"] = var
                else:
                    auditing_vars[f"{provider_name}:{model}"] = var
                
                cb = ttk.Checkbutton(scrollable_frame, text=model, variable=var)
                cb.pack(anchor="w", pady=2, padx=5)
            
            canvas.pack(side="left", fill="both", expand=True)
            scrollbar.pack(side="right", fill="y")
            
            # Add Select All / Deselect All buttons
            button_frame = ttk.Frame(tab_frame)
            button_frame.pack(fill=tk.X, pady=5)
            
            def select_all():
                for key, var in (coding_vars if var_dict_key == "coding" else auditing_vars).items():
                    if key.startswith(f"{provider_name}:"):
                        var.set(True)
            
            def deselect_all():
                for key, var in (coding_vars if var_dict_key == "coding" else auditing_vars).items():
                    if key.startswith(f"{provider_name}:"):
                        var.set(False)
            
            ttk.Button(button_frame, text="Select All", command=select_all, width=12).pack(side=tk.LEFT, padx=2)
            ttk.Button(button_frame, text="Deselect All", command=deselect_all, width=12).pack(side=tk.LEFT, padx=2)
        
        # Create Coding Models tab
        create_model_selection_tab("Coding Models", "coding", selected_models.get("coding", []))
        
        # Create Audit Models tab
        create_model_selection_tab("Audit Models", "auditing", selected_models.get("auditing", []))
        
        # Available models info at the bottom
        lbl_available = ttk.Label(frame, text="Available Models:", style='Bold.TLabel')
        lbl_available.pack(anchor="w", pady=(15, 5))
        Tooltip(lbl_available, f"Models available with your current {provider_name} API key")
        
        models_frame = ttk.Frame(frame)
        models_frame.pack(fill=tk.X, pady=(0, 10))
        
        models_text = tk.Text(models_frame, height=3, width=50, wrap=tk.WORD)
        models_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        models_scroll = ttk.Scrollbar(models_frame, command=models_text.yview)
        models_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        models_text.config(yscrollcommand=models_scroll.set)
        
        if available_models_sorted:
            models_text.insert(tk.END, ", ".join(available_models_sorted))
        else:
            models_text.insert(tk.END, "No models available with current API key")
        models_text.config(state=tk.DISABLED)  # Make read-only
        
        return frame
    
    # Create tabs for each provider
    providers = ["OpenAI", "Claude", "Gemini", "DeepSeek"]
    for provider in providers:
        create_provider_tab(provider)
    
    # Button frame at the bottom
    button_frame = ttk.Frame(settings_dialog, padding=(10, 5, 10, 10))
    button_frame.grid(row=1, column=0, sticky="ew")
    button_frame.columnconfigure(0, weight=1)
    
    # Add a separator above the buttons
    separator = ttk.Separator(button_frame, orient="horizontal")
    separator.grid(row=0, column=0, columnspan=2, sticky="ew", pady=5)
    
    def save_settings():
        """Save all settings from both tabs."""
        # Save output directory setting
        output_dir = output_dir_var.get().strip()
        
        # Create settings directory if needed
        config_dir = get_config_dir()
        os.makedirs(config_dir, exist_ok=True)
        
        # Save output directory setting
        settings["output_directory"] = output_dir
        
        try:
            with open(config_file, 'w') as f:
                json.dump(settings, f, indent=2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save settings: {e}")
            return False
        
        # Extract model names from the provider-specific keys
        new_coding_models = []
        new_auditing_models = []
        
        for key, var in coding_vars.items():
            if var.get() and ":" in key:
                model_name = key.split(":", 1)[1]
                if model_name not in new_coding_models:
                    new_coding_models.append(model_name)
        
        for key, var in auditing_vars.items():
            if var.get() and ":" in key:
                model_name = key.split(":", 1)[1]
                if model_name not in new_auditing_models:
                    new_auditing_models.append(model_name)
        
        # Save to configuration
        if config.save_selected_models(new_coding_models, new_auditing_models):
            # Update global variables and UI
            update_callback(new_coding_models, new_auditing_models)
        else:
            messagebox.showerror("Error", "Failed to save model selection.")
            return False
        
        messagebox.showinfo("Success", "Settings saved successfully!")
        settings_dialog.destroy()
        return True
    
    # Create Cancel button
    cancel_button = ttk.Button(
        button_frame, 
        text="Cancel", 
        command=settings_dialog.destroy, 
        width=15
    )
    cancel_button.grid(row=1, column=0, sticky="e", padx=(0, 5))
    Tooltip(cancel_button, "Close without saving changes")
    
    # Create Save button
    save_button = ttk.Button(
        button_frame, 
        text="Save", 
        command=save_settings, 
        width=15
    )
    save_button.grid(row=1, column=1, sticky="e")
    Tooltip(save_button, "Save all settings")


def init_database():
    """Initialize the database with required tables for storing coding projects."""
    import sqlite3
    import os
    
    # Get database path using the new function
    db_path = get_db_path()
    
    print(f"Initializing database at: {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create tables
    
    # Projects table for both modes
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS projects (
        project_id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_name TEXT NOT NULL,
        output_directory TEXT,
        description TEXT,
        has_upload BOOLEAN,
        upload_filename TEXT,
        upload_content TEXT,
        start_datetime TEXT,
        end_datetime TEXT,
        mode TEXT, -- "correction" or "creation"
        iterations_count INTEGER, -- for correction mode
        coding_llm TEXT, -- for correction mode
        auditing_llm TEXT, -- for correction mode
        coding_llms TEXT, -- comma-separated, for creation mode
        final_code TEXT -- final code result
    )
    ''')
    
    # Iterations table for multiple correction mode
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS iterations (
        iteration_id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        iteration_number INTEGER,
        code TEXT,
        audit TEXT,
        critical_count INTEGER,
        serious_count INTEGER,
        noncritical_count INTEGER,
        suggestions_count INTEGER,
        fixed_critical INTEGER,
        fixed_serious INTEGER,
        fixed_noncritical INTEGER,
        fixed_suggestions INTEGER,
        FOREIGN KEY (project_id) REFERENCES projects (project_id)
    )
    ''')
    
    # Model results table for multiple creation mode
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS model_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        project_id INTEGER,
        model_name TEXT,
        code TEXT,
        status TEXT, -- "completed" or "failed"
        FOREIGN KEY (project_id) REFERENCES projects (project_id)
    )
    ''')
    
    conn.commit()
    conn.close()
    
    print(f"Database initialized at {db_path}")
    return db_path

def get_db_connection():
    """Get a connection to the SQLite database."""
    try:
        import sqlite3
        
        db_path = get_db_path()
        if not os.path.exists(db_path):
            print(f"Database not found at {db_path}. Initializing...")
            init_database()  # Initialize if not exists
            
        return sqlite3.connect(db_path)
    except Exception as e:
        print(f"Error getting database connection: {e}")
        return None

def get_db_path():
    """
    Get the path to the database file in the data directory.
    """
    try:
        # Create data directory
        db_dir = get_config_dir()
        os.makedirs(db_dir, exist_ok=True)
        
        # Return path to database file
        return os.path.join(db_dir, "codingapi.db")
    except Exception as e:
        print(f"Error getting database path: {e}")
        # Fallback to temporary directory if there's an error
        import tempfile
        return os.path.join(tempfile.gettempdir(), "codingapi.db")

def save_project_start(project_name, output_dir, description, has_upload, upload_filename, 
                       upload_content, mode, iterations=None, coding_llm=None, 
                       auditing_llm=None, coding_llms=None):
    """
    Save the initial project information when starting a coding project.
    Returns the project_id for later updates, or None if operation fails.
    """
    try:
        import sqlite3
        import datetime
        
        now = datetime.datetime.now().isoformat()
        
        conn = get_db_connection()
        if conn is None:
            print("Failed to get database connection")
            return None
            
        cursor = conn.cursor()
        
        # Convert boolean to integer for SQLite
        has_upload_int = 1 if has_upload else 0
        
        # Format the coding_llms list for creation mode
        coding_llms_str = None
        if coding_llms:
            coding_llms_str = ','.join(coding_llms)
        
        try:
            cursor.execute('''
            INSERT INTO projects (
                project_name, output_directory, description, has_upload, 
                upload_filename, upload_content, start_datetime, mode, 
                iterations_count, coding_llm, auditing_llm, coding_llms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                project_name, output_dir, description, has_upload_int, 
                upload_filename, upload_content, now, mode, 
                iterations, coding_llm, auditing_llm, coding_llms_str
            ))
            
            project_id = cursor.lastrowid
            conn.commit()
            
            print(f"Project saved to database with ID: {project_id}")
            return project_id
            
        except sqlite3.Error as e:
            print(f"Database error in save_project_start: {e}")
            return None
        finally:
            if conn:
                conn.close()
    except Exception as e:
        print(f"Exception in save_project_start: {e}")
        return None

def save_iteration(project_id, iteration_number, code, audit, bug_counts, fix_counts):
    """
    Save information about an iteration in multiple correction mode.
    
    Args:
        project_id: The project ID from save_project_start
        iteration_number: Current iteration number
        code: Generated code for this iteration
        audit: Audit text for this iteration
        bug_counts: Tuple of (critical, serious, noncritical, suggestions) counts
        fix_counts: Tuple of (fixed_critical, fixed_serious, fixed_noncritical, fixed_suggestions) counts
    """
    import sqlite3
    
    critical, serious, noncritical, suggestions = bug_counts
    fixed_critical, fixed_serious, fixed_noncritical, fixed_suggestions = fix_counts
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO iterations (
            project_id, iteration_number, code, audit, 
            critical_count, serious_count, noncritical_count, suggestions_count,
            fixed_critical, fixed_serious, fixed_noncritical, fixed_suggestions
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            project_id, iteration_number, code, audit,
            critical, serious, noncritical, suggestions,
            fixed_critical, fixed_serious, fixed_noncritical, fixed_suggestions
        ))
        
        conn.commit()
        print(f"Iteration {iteration_number} saved to database for project {project_id}")
        
    except sqlite3.Error as e:
        print(f"Database error saving iteration: {e}")
    finally:
        conn.close()

def save_model_result(project_id, model_name, code, status):
    """
    Save a model result in multiple creation mode.
    
    Args:
        project_id: The project ID from save_project_start
        model_name: Name of the LLM model
        code: Generated code
        status: "completed" or "failed"
    """
    import sqlite3
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    try:
        cursor.execute('''
        INSERT INTO model_results (
            project_id, model_name, code, status
        ) VALUES (?, ?, ?, ?)
        ''', (
            project_id, model_name, code, status
        ))
        
        conn.commit()
        print(f"Model result for {model_name} saved to database for project {project_id}")
        
    except sqlite3.Error as e:
        print(f"Database error saving model result: {e}")
    finally:
        conn.close()

def get_base_output_directory():
    """
    Get just the base output directory from settings without project name.
    If no custom directory is set, return the current directory.
    """
    # Load settings
    config_file = os.path.join(get_config_dir(), "settings.json")
    settings = {}
    
    try:
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                settings = json.load(f)
    except Exception as e:
        print(f"Error loading settings: {e}")
    
    # Get custom output directory from settings
    custom_dir = settings.get("output_directory", "").strip()
    
    if custom_dir:
        # Use custom directory
        return custom_dir
    else:
        # Use current directory
        return os.getcwd()

def save_project_end(project_id, final_code):
    """
    Update project with end time and final code when completed.
    
    Args:
        project_id: The project ID from save_project_start
        final_code: The final code generated
    """
    try:
        import sqlite3
        import datetime
        
        # Skip if project_id is None
        if project_id is None:
            print("Cannot save project end: project_id is None")
            return
        
        now = datetime.datetime.now().isoformat()
        
        conn = get_db_connection()
        if conn is None:
            print("Cannot save project end: database connection failed")
            return
            
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
            UPDATE projects
            SET end_datetime = ?, final_code = ?
            WHERE project_id = ?
            ''', (now, final_code, project_id))
            
            conn.commit()
            print(f"Project {project_id} completed and saved to database")
            
        except sqlite3.Error as e:
            print(f"Database error completing project: {e}")
        finally:
            conn.close()
    except Exception as e:
        print(f"Error in save_project_end: {e}")

#-----------------------------------------------------------------------------
# GUI Implementation
#-----------------------------------------------------------------------------

def main():
    """Main entry point for the GUI application."""
    global user_filename, user_description, user_language
    global user_coding_llm, user_audit_llm, stop_flag, process_thread
    global model_status, current_displayed_model
    global text_description
    
    # Initialize these global variables
    user_filename = ""
    user_description = ""
    user_language = ""
    user_coding_llm = ""
    user_audit_llm = ""
    stop_flag = False
    process_thread = None
    # Dictionary to track model processing status for multiple creation mode
    model_status = {}
    # Variable to track which model's code is currently displayed
    current_displayed_model = None
    
    # Load previously discovered dynamic models
    load_dynamic_models()
    
    # Initialize the database
    init_database()
    
    root = tk.Tk()
    root.title("Code Generator with Audit (Responsive UI)")
    root.geometry("1200x800")
    root.minsize(1000, 600)
    root.configure(bg="#F5F5F5")

    # Initialize SecureConfig and discover all available models at startup
    config = SecureConfig()
    print("Discovering available models at startup...")
    all_available_models = config.get_all_available_models_from_api()  # Full discovery
    selected_models = config.get_selected_models()
    
    # Filter selected models to only include available ones
    available_coding_models = [model for model in selected_models.get("coding", []) 
                             if any(model in all_available_models.get(provider, []) 
                                   for provider in all_available_models)]
    available_auditing_models = [model for model in selected_models.get("auditing", []) 
                               if any(model in all_available_models.get(provider, []) 
                                     for provider in all_available_models)]
    
    # If no selected models are available, use defaults
    if not available_coding_models:
        available_coding_models = [model for provider_models in all_available_models.values() 
                                 for model in provider_models]
    if not available_auditing_models:
        available_auditing_models = [model for provider_models in all_available_models.values() 
                                   for model in provider_models]
    
    # If still no models, show error message
    if not available_coding_models:
        available_coding_models = ["No models available - configure API keys first"]
    if not available_auditing_models:
        available_auditing_models = ["No models available - configure API keys first"]

    # Styling configuration
    style = ttk.Style(root)
    
    try:
        style.theme_use("clam")
    except:
        pass
    style.configure('.', background="#F5F5F5", font=("Arial", 11))
    style.configure('TLabel', background="#F5F5F5", font=("Arial", 11))
    style.configure('TButton', font=("Arial", 11, "bold"))
    style.configure('TFrame', background="#F5F5F5")
    style.configure(
        'Status.TLabel',
        background="#E3F2FD",
        foreground="#1E88E5",
        font=("Arial", 12, "bold"),
        padding=5
    )
    # Style for headers (bold font)
    style.configure(
        'Bold.TLabel',
        background="#F5F5F5",
        font=("Arial", 11, "bold")
    )
    # Style for Start Coding button (white text on green background)
    style.configure(
        'Green.TButton',
        background="#4CAF50",
        foreground="white",
        font=("Arial", 12, "bold")
    )
    # Apply style to Start Coding button
    style.map('Green.TButton',
        background=[('active', '#45a049')],
        foreground=[('active', 'white')]
    )
    
    # Style for Run app button (white text on blue background)
    style.configure(
        'Blue.TButton',
        background="#2196F3",
        foreground="white",
        font=("Arial", 12, "bold")
    )
    # Apply style to Run app button
    style.map('Blue.TButton',
        background=[('active', '#1976D2')],
        foreground=[('active', 'white')]
    )
    
    # Style for progress indicators
    style.configure(
        'InProgress.TLabel',
        foreground="#FF9800", # Orange for in-progress
        background="#F5F5F5",
        font=("Arial", 11, "bold")
    )
    style.configure(
        'Complete.TLabel',
        foreground="#4CAF50", # Green for complete
        background="#F5F5F5",
        font=("Arial", 11, "bold")
    )
    style.configure(
        'Failed.TLabel',
        foreground="#F44336", # Red for failed
        background="#F5F5F5",
        font=("Arial", 11, "bold")
    )

    # ================== INPUT PARAMETERS SECTION ==================
    frame_input = ttk.Frame(root, padding=10)
    frame_input.pack(side=tk.TOP, fill=tk.X)

    lbl_title = ttk.Label(frame_input, text="Enter parameters for code generation:", font=("Arial", 14, "bold"))
    lbl_title.pack(anchor="w", pady=(0, 10))
    Tooltip(lbl_title, "Configure parameters for generating code with LLMs")

    # Program file name and file upload in the same row
    frm_filename_row = ttk.Frame(frame_input)
    frm_filename_row.pack(fill=tk.X, pady=5)
        
    # Program file name section (left)
    frm_filename = ttk.Frame(frm_filename_row)
    frm_filename.pack(side=tk.LEFT, fill=tk.X)
    label_filename = ttk.Label(frm_filename, text="Project name:", style='Bold.TLabel')
    label_filename.pack(anchor="w")
    Tooltip(label_filename, "Enter the project name for saving generated code (without extension)")
    entry_filename = ttk.Entry(frm_filename, width=30)
    entry_filename.pack(anchor="w", pady=3)
    
    # File upload section (right)
    frm_file_select = ttk.Frame(frm_filename_row)
    frm_file_select.pack_forget()  # Initially hidden
    
    label_file = ttk.Label(frm_file_select, text="Program File", style='Bold.TLabel')
    label_file.pack(anchor="w")
    Tooltip(label_file, "Select an existing file to modify or use as a starting point")
    
    # File path and browse button in a horizontal layout
    frm_file_path = ttk.Frame(frm_file_select)
    frm_file_path.pack(fill=tk.X, pady=3)
    
    entry_file_path = ttk.Entry(frm_file_path, width=30)
    entry_file_path.pack(side=tk.LEFT, padx=(0, 5))
    
    btn_browse = ttk.Button(frm_file_path, text="Browse...")
    btn_browse.pack(side=tk.LEFT)
    Tooltip(btn_browse, "Browse your file system to select a program file")

    # Add Checkbox for uploading existing file
    frm_upload = ttk.Frame(frame_input)
    frm_upload.pack(fill=tk.X, pady=5)
    var_upload_file = tk.BooleanVar()
    var_upload_file.set(False)
    check_upload = ttk.Checkbutton(frm_upload, text="Upload existing program to modify", variable=var_upload_file)
    check_upload.pack(anchor="w")
    Tooltip(check_upload, "Check this to modify an existing program instead of creating a new one")

    # Add mode toggle: Multiple Correction / Multiple Creation
    frm_mode = ttk.Frame(frame_input)
    frm_mode.pack(fill=tk.X, pady=5)
    
    mode_var = tk.StringVar(value="correction")  # Default to Multiple Correction
    
    # Create a container frame for the toggle
    toggle_frame = ttk.Frame(frm_mode)
    toggle_frame.pack(anchor="w")
    
    # Multiple Correction radio button (left)
    rb_correction = ttk.Radiobutton(
        toggle_frame, 
        text="Multiple Correction", 
        value="correction", 
        variable=mode_var
    )
    rb_correction.pack(side=tk.LEFT, padx=(0, 10))
    Tooltip(rb_correction, "Generate code once and improve it through multiple rounds of audit and correction")
    
    # Multiple Creation radio button (right)
    rb_creation = ttk.Radiobutton(
        toggle_frame, 
        text="Multiple Creation", 
        value="creation", 
        variable=mode_var
    )
    rb_creation.pack(side=tk.LEFT)
    Tooltip(rb_creation, "Generate code using multiple LLMs simultaneously and select the best result")
    
    # Program Description
    frm_description = ttk.Frame(frame_input)
    frm_description.pack(fill=tk.X, pady=5)
    
    # Header with label and clear button
    frm_desc_header = ttk.Frame(frm_description)
    frm_desc_header.pack(fill=tk.X)
    
    label_description = ttk.Label(frm_desc_header, text="Program Description/Comment", style='Bold.TLabel')
    label_description.pack(side=tk.LEFT, anchor="w")
    Tooltip(label_description, "Enter a detailed description of the program you want to generate")
    
    # Clear button (positioned at the top right)
    btn_clear_desc = ttk.Button(frm_desc_header, text="Clear", width=8)
    btn_clear_desc.pack(side=tk.RIGHT, anchor="ne")
    Tooltip(btn_clear_desc, "Clear the program description field")

    # Create text field
    text_description = scrolledtext.ScrolledText(frm_description, width=80, height=5)
    text_description.pack(fill="x", pady=3)
    
    # Key press handler based on physical key code, regardless of keyboard layout
    def handle_keypress(event):
        # Check if Ctrl key is pressed
        if event.state & 0x4:  # 0x4 is the mask for Ctrl key
            # Key code for V = 86 (in ASCII/Unicode)
            if event.keycode == 86:  # V - paste
                event.widget.event_generate("<<Paste>>")
                return "break"
            elif event.keycode == 67:  # C - copy
                event.widget.event_generate("<<Copy>>")
                return "break"
            elif event.keycode == 88:  # X - cut
                event.widget.event_generate("<<Cut>>")
                return "break"
            elif event.keycode == 65:  # A - select all
                event.widget.tag_add(tk.SEL, "1.0", tk.END)
                event.widget.mark_set(tk.INSERT, "1.0")
                event.widget.see(tk.INSERT)
                return "break"
            elif event.keycode == 90:  # Z - undo
                try:
                    event.widget.edit_undo()
                except:
                    pass
                return "break"
            elif event.keycode == 89:  # Y - redo
                try:
                    event.widget.edit_redo()
                except:
                    pass
                return "break"
        return None
    
    # Bind handler to any key press event
    text_description.bind("<Key>", handle_keypress)

    # Horizontal frame for Language, LLMs and Iterations
    frm_options = ttk.Frame(frame_input)
    frm_options.pack(fill=tk.X, pady=5)
    
    # Programming Language - first column
    frm_lang = ttk.Frame(frm_options)
    frm_lang.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    label_language = ttk.Label(frm_lang, text="Programming Language", style='Bold.TLabel')
    label_language.pack(anchor="w")
    Tooltip(label_language, "Select the programming language for your code")
    languages = ["Python","Java","JavaScript","C","C++","Pascal","Julia","FORTRAN"]
    combo_language = ttk.Combobox(frm_lang, values=languages, state="readonly")
    combo_language.current(0)
    combo_language.pack(anchor="w", pady=3, fill=tk.X)
    
    # Coding LLM - second column
    frm_coding_llm = ttk.Frame(frm_options)
    frm_coding_llm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    label_coding_llm = ttk.Label(frm_coding_llm, text="Coding LLM", style='Bold.TLabel')
    label_coding_llm.pack(anchor="w")
    Tooltip(label_coding_llm, "Select the LLM to use for code generation")
    
    # Two different widgets for Coding LLM based on mode
    # 1. Standard Combobox for Multiple Correction mode
    combo_coding_llm = ttk.Combobox(frm_coding_llm, values=available_coding_models, state="readonly")
    if available_coding_models:
        combo_coding_llm.current(0)
    combo_coding_llm.pack(anchor="w", pady=3, fill=tk.X)
    
    # 2. Listbox with multiple selection for Multiple Creation mode
    frame_listbox = ttk.Frame(frm_coding_llm)
    lb_coding_llm = tk.Listbox(frame_listbox, selectmode=tk.MULTIPLE, height=6)
    for model in available_coding_models:
        lb_coding_llm.insert(tk.END, model)
    
    scrollbar = ttk.Scrollbar(frame_listbox, orient="vertical", command=lb_coding_llm.yview)
    lb_coding_llm.configure(yscrollcommand=scrollbar.set)
    
    lb_coding_llm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    Tooltip(frame_listbox, "Select multiple LLMs to generate code with")
    
    # Initially hide the listbox
    frame_listbox.pack_forget()
    
    # Auditing LLM - third column
    frm_audit_llm = ttk.Frame(frm_options)
    frm_audit_llm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    label_audit_llm = ttk.Label(frm_audit_llm, text="Auditing LLM", style='Bold.TLabel')
    label_audit_llm.pack(anchor="w")
    Tooltip(label_audit_llm, "Select the LLM to use for code auditing and analysis")
    combo_audit_llm = ttk.Combobox(frm_audit_llm, values=available_auditing_models, state="readonly")
    if available_auditing_models:
        combo_audit_llm.current(0)  # Default to first available model for auditing
    combo_audit_llm.pack(anchor="w", pady=3, fill=tk.X)
    
    # Iterations - fourth column
    frm_iterations = ttk.Frame(frm_options)
    frm_iterations.pack(side=tk.LEFT, fill=tk.X, expand=True)
    label_iterations = ttk.Label(frm_iterations, text="Iterations", style='Bold.TLabel')
    label_iterations.pack(anchor="w")
    Tooltip(label_iterations, "Number of audit-correction cycles to perform (1-20)")
    
    # Spinbox for selecting number of iterations (1-20)
    spinbox_iterations = ttk.Spinbox(frm_iterations, from_=1, to=20, width=5)
    spinbox_iterations.set(5)  # Default is 5
    spinbox_iterations.pack(anchor="w", pady=3)

    # Buttons: Cancel / Start
    frm_buttons_top = ttk.Frame(frame_input)
    frm_buttons_top.pack(fill=tk.X, pady=10)

    # Modified button order as requested
    btn_cancel = ttk.Button(frm_buttons_top, text="Cancel")
    btn_start = ttk.Button(frm_buttons_top, text="Start Coding", style='Green.TButton', width=15)
    btn_stop = ttk.Button(frm_buttons_top, text="Stop Process", width=15)
    btn_run_app = ttk.Button(frm_buttons_top, text="Run app", style='Blue.TButton', width=15)
    btn_settings = ttk.Button(frm_buttons_top, text="⚙", width=3)  # Cog icon for settings
    btn_exit = ttk.Button(frm_buttons_top, text="Exit", width=10)

    # Pack buttons in the modified order
    btn_cancel.pack(side=tk.LEFT, padx=5)
    btn_start.pack(side=tk.LEFT, padx=5)
    btn_stop.pack(side=tk.LEFT, padx=5)
    btn_run_app.pack(side=tk.LEFT, padx=5)  # Run app next to Stop Process

    # Right side buttons
    btn_exit.pack(side=tk.RIGHT, padx=5)
    btn_settings.pack(side=tk.RIGHT, padx=5)  # Settings button replacing API Keys and Check Models

    # Add tooltips to all buttons
    Tooltip(btn_cancel, "Close the current window without saving")
    Tooltip(btn_start, "Start the code generation process with selected parameters")
    Tooltip(btn_stop, "Stop the current code generation process")
    Tooltip(btn_run_app, "Run a Python application from your file system")
    Tooltip(btn_settings, "Open settings dialog")
    Tooltip(btn_exit, "Exit the application")

    btn_stop.config(state=tk.DISABLED)  # Initially disabled

    def on_cancel():
        root.destroy()
        sys.exit(0)
    btn_cancel.config(command=on_cancel)

    # Configure Exit button to use the same function as Cancel
    btn_exit.config(command=on_cancel)
    
    # Set command for Run app button
    btn_run_app.config(command=lambda: open_run_app_dialog(root))
    
    # Set command for Clear Description button
    btn_clear_desc.config(command=clear_description)

    sep = ttk.Separator(root, orient="horizontal")
    sep.pack(fill="x", padx=10, pady=(5, 5))

    # ================== LOWER SECTION: PROCESS ==================
    frame_process = ttk.Frame(root, padding=10)
    frame_process.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    lbl_title_process = ttk.Label(frame_process, text="Audit & Correction Process:", font=("Arial", 14, "bold"))
    lbl_title_process.pack(anchor="w", pady=(0, 10))
    Tooltip(lbl_title_process, "View and monitor the code generation and correction process")

    lbl_status = ttk.Label(frame_process, text="Status: Idle", style='Status.TLabel')
    lbl_status.pack(anchor="w", fill=tk.X, pady=(0, 10))
    Tooltip(lbl_status, "Current status of the code generation process")

    lbl_iter = ttk.Label(frame_process, text="Iteration: 1/1", style='Bold.TLabel')
    lbl_iter.pack(anchor="w", pady=(0, 10))
    Tooltip(lbl_iter, "Current iteration number in the audit-correction cycle")

    # Panel: left (Current Code) and right (Audit Result)
    pane = ttk.Panedwindow(frame_process, orient=tk.HORIZONTAL)
    pane.pack(fill="both", expand=True)

    frame_left = ttk.Frame(pane)
    pane.add(frame_left, weight=1)
    frame_left.grid_columnconfigure(0, weight=1)

    # Frame for header and copy button on the left
    frm_left_header = ttk.Frame(frame_left)
    frm_left_header.grid(row=0, column=0, sticky="ew", padx=5, pady=(0,5))
    lbl_code_left = ttk.Label(frm_left_header, text="Current Code", style='Bold.TLabel')
    lbl_code_left.pack(side=tk.LEFT)
    Tooltip(lbl_code_left, "Current version of the generated code")
    
    # Copy button for Current Code
    btn_copy_code = ttk.Button(frm_left_header, text="📋", width=3)
    btn_copy_code.pack(side=tk.RIGHT)
    Tooltip(btn_copy_code, "Copy code to clipboard")

    txt_current_code = scrolledtext.ScrolledText(frame_left, width=60, height=10)
    txt_current_code.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    frame_left.grid_rowconfigure(1, weight=1)

    # Corrected Bugs -- with full names
    lbl_corrected_bugs = ttk.Label(frame_left, text="Corrected: Critical=0, Serious=0, N/critical=0, Suggestions=0", style='Bold.TLabel')
    lbl_corrected_bugs.grid(row=2, column=0, sticky="nw", padx=5, pady=(0,5))
    Tooltip(lbl_corrected_bugs, "Number of issues successfully corrected in each category")

    frame_right = ttk.Frame(pane)
    pane.add(frame_right, weight=1)
    frame_right.grid_columnconfigure(0, weight=1)

    # Frame for header and copy button on the right
    frm_right_header = ttk.Frame(frame_right)
    frm_right_header.grid(row=0, column=0, sticky="ew", padx=5, pady=(0,5))
    lbl_code_right = ttk.Label(frm_right_header, text="Audit Result", style='Bold.TLabel')
    lbl_code_right.pack(side=tk.LEFT)
    Tooltip(lbl_code_right, "Results of code analysis and audit")
    
    # Copy button for Audit Result
    btn_copy_audit = ttk.Button(frm_right_header, text="📋", width=3)
    btn_copy_audit.pack(side=tk.RIGHT)
    Tooltip(btn_copy_audit, "Copy audit results to clipboard")

    txt_audit_result = scrolledtext.ScrolledText(frame_right, width=60, height=10)
    txt_audit_result.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    frame_right.grid_rowconfigure(1, weight=1)

    # Bugs/Corrections Count -- with full names
    lbl_bugs_count = ttk.Label(frame_right, text="Bugs: Critical=0, Serious=0, N/critical=0, Suggestions=0", style='Bold.TLabel')
    lbl_bugs_count.grid(row=2, column=0, sticky="nw", padx=5, pady=(0,5))
    Tooltip(lbl_bugs_count, "Number of issues found in each category during audit")

    # Add function to update model lists and UI components after model selection
    def update_model_lists(new_coding_models, new_auditing_models):
        """Update the model lists and UI components after model selection."""
        nonlocal available_coding_models, available_auditing_models
        
        # Update the lists
        available_coding_models = new_coding_models
        available_auditing_models = new_auditing_models
        
        # Update coding combobox
        combo_coding_llm['values'] = available_coding_models
        if available_coding_models and available_coding_models[0] != "No models available - configure API keys first":
            combo_coding_llm.current(0)
        
        # Update auditing combobox
        combo_audit_llm['values'] = available_auditing_models
        if available_auditing_models and available_auditing_models[0] != "No models available - configure API keys first":
            combo_audit_llm.current(0)
        
        # Update listbox for multiple creation mode
        lb_coding_llm.delete(0, tk.END)
        for model in available_coding_models:
            lb_coding_llm.insert(tk.END, model)

    # Add file browsing functionality
    def browse_file():
        file_path = filedialog.askopenfilename(
            title="Select Program File",
            filetypes=(
                ("Python files", "*.py"),
                ("Java files", "*.java"),
                ("JavaScript files", "*.js"),
                ("C files", "*.c"),
                ("C++ files", "*.cpp"),
                ("Pascal files", "*.pas"),
                ("Julia files", "*.jl"),
                ("FORTRAN files", "*.f90"),
                ("All files", "*.*")
            )
        )
        if file_path:
            entry_file_path.delete(0, tk.END)
            entry_file_path.insert(0, file_path)

    btn_browse.config(command=browse_file)

    # Function to show/hide file selection based on checkbox
    def toggle_file_selection():
        if var_upload_file.get():
            frm_file_select.pack(side=tk.LEFT, padx=(20, 0))
        else:
            frm_file_select.pack_forget()

    # Bind the checkbox to the toggle function
    check_upload.config(command=toggle_file_selection)
    
    # Function to toggle UI based on selected mode
    def toggle_mode():
        mode = mode_var.get()
        if mode == "correction":
            # Show single selection for Coding LLM
            frame_listbox.pack_forget()
            combo_coding_llm.pack(anchor="w", pady=3, fill=tk.X)
            
            # Show Auditing LLM and Iterations
            label_audit_llm.pack(anchor="w")
            combo_audit_llm.pack(anchor="w", pady=3, fill=tk.X)
            frm_audit_llm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
            
            label_iterations.pack(anchor="w")
            spinbox_iterations.pack(anchor="w", pady=3)
            frm_iterations.pack(side=tk.LEFT, fill=tk.X, expand=True)
            
            lbl_title_process.config(text="Audit & Correction Process:")
            lbl_code_right.config(text="Audit Result")
            lbl_code_left.config(text="Current Code")
            
            # Show bugs count and corrected bugs labels
            lbl_bugs_count.grid(row=2, column=0, sticky="nw", padx=5, pady=(0,5))
            lbl_corrected_bugs.grid(row=2, column=0, sticky="nw", padx=5, pady=(0,5))
            
            # Show iteration label
            lbl_iter.pack(anchor="w", pady=(0, 10))
            
        else:  # "creation" mode
            # Show multiple selection for Coding LLM
            combo_coding_llm.pack_forget()
            frame_listbox.pack(anchor="w", pady=3, fill=tk.X)
            
            # Hide Auditing LLM and Iterations
            label_audit_llm.pack_forget()
            combo_audit_llm.pack_forget()
            frm_audit_llm.pack_forget()
            
            label_iterations.pack_forget()
            spinbox_iterations.pack_forget()
            frm_iterations.pack_forget()
            
            lbl_title_process.config(text="Multiple Code Generation:")
            lbl_code_right.config(text="Code Creation Process")
            lbl_code_left.config(text="Current Code")
            
            # Hide bugs count and corrected bugs labels
            lbl_bugs_count.grid_forget()
            lbl_corrected_bugs.grid_forget()
            
            # Hide iteration label in creation mode
            lbl_iter.pack_forget()
    
    # Bind the radio buttons to the toggle function
    rb_correction.config(command=toggle_mode)
    rb_creation.config(command=toggle_mode)

    # Function to read file content
    def read_file_content(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()
        except Exception as e:
            messagebox.showerror("File Error", f"Could not read file: {str(e)}")
            return None

    # UI update functions
    def ui_set_status(text):
        lbl_status.config(text=text)

    def ui_set_iteration(num, max_iter=None):
        """Update the iteration label with current/max format."""
        if max_iter is None:
            # Try to get max iterations from spinbox if not specified
            try:
                max_iter = int(spinbox_iterations.get())
            except:
                max_iter = 1
        lbl_iter.config(text=f"Iteration: {num}/{max_iter}")

    def ui_set_current_code(code, model_name=None):
        txt_current_code.delete("1.0", tk.END)
        txt_current_code.insert("1.0", code)
        
        # Update the title if a model name is provided
        if model_name and mode_var.get() == "creation":
            lbl_code_left.config(text=f"Current Code: {model_name}")
            global current_displayed_model
            current_displayed_model = model_name

    def ui_set_audit_result(audit):
        txt_audit_result.delete("1.0", tk.END)
        txt_audit_result.insert("1.0", audit)

    def ui_set_corrected_bugs(c_crit, c_serious, c_ncrit, c_sugg):
        lbl_corrected_bugs.config(
            text=(
                f"Corrected: Critical={c_crit}, Serious={c_serious}, N/critical={c_ncrit}, Suggestions={c_sugg}"
            )
        )

    def ui_set_bugs_count(crit, serious, ncrit, sugg):
        lbl_bugs_count.config(
            text=(
                f"Bugs: Critical={crit}, Serious={serious}, N/critical={ncrit}, Suggestions={sugg}"
            )
        )
        
    # Animation functions for creating progress indicators
    def update_model_status_ui():
        """Updates the audit result area with the current status of all models"""
        global model_status
        
        if mode_var.get() != "creation" or not model_status:
            return
                
        status_text = "Generation Results:\n\n"
        status_text += "Models:\n"
        
        # Show all models with appropriate status indicators
        for model_name in model_status.keys():
            status = model_status[model_name]
            if status == "completed":
                indicator = "✓"  # Green tick for completed
            elif status == "in_progress":
                # Rotating indicator
                frame_count = int(time.time() * 2) % 8  # 8 frames of animation
                indicators = ["⟳", "⟲", "⟳", "⟲", "⟳", "⟲", "⟳", "⟲"]
                indicator = indicators[frame_count]
            elif status == "failed":
                indicator = "✗"  # Red X for failed
            else:  # pending
                indicator = " "  # Space for pending
                    
            status_text += f"{indicator} {model_name}\n"
                    
        txt_audit_result.delete("1.0", tk.END)
        txt_audit_result.insert("1.0", status_text)
    
    # Create a recurring animation function for the progress indicators
    def animate_progress():
        global stop_flag
        
        if mode_var.get() == "creation" and not stop_flag:
            update_model_status_ui()
            root.after(250, animate_progress)  # Update every 250ms
        
    # Clipboard copy functions
    def copy_current_code_to_clipboard():
        code_text = txt_current_code.get("1.0", tk.END)
        root.clipboard_clear()
        root.clipboard_append(code_text)
        messagebox.showinfo("Copied", "Code copied to clipboard")
        
    def copy_audit_result_to_clipboard():
        audit_text = txt_audit_result.get("1.0", tk.END)
        root.clipboard_clear()
        root.clipboard_append(audit_text)
        messagebox.showinfo("Copied", "Audit result copied to clipboard")
    
    # Binding functions to copy buttons
    btn_copy_code.config(command=copy_current_code_to_clipboard)
    btn_copy_audit.config(command=copy_audit_result_to_clipboard)
        
    # Function to stop processing
    def stop_processing():
        global stop_flag
        stop_flag = True
        ui_set_status("Status: Process stopped by user")
        btn_stop.config(state=tk.DISABLED)
        btn_start.config(state=tk.NORMAL)
    
    btn_stop.config(command=stop_processing)
            
    def run_iteration_loop(output_dir, base_filename, description, init_code, language, coding_llm, audit_llm, max_iterations, project_id=None):
        """
        Runs the iterative code audit and correction process.
        
        Args:
            output_dir: Directory where files should be saved
            base_filename: Base filename for generated files (without extension)
            description: Program description
            init_code: Initial code to start with
            language: Programming language
            coding_llm: LLM for code generation
            audit_llm: LLM for auditing
            max_iterations: Maximum number of iterations
            project_id: Database project ID for storing results (can be None)
        """
        global stop_flag
        current_code = init_code
        coding_llm_info = get_model_info(coding_llm) or {"model": "o3-2025-04-16", "family": "OpenAI"}
        audit_llm_info = get_model_info(audit_llm) or {"model": "claude-3-7-sonnet-20250219", "family": "Claude"}

        print(f"run_iteration_loop: project_id={project_id}, max_iterations={max_iterations}")

        # Ensure subdirectory exists
        try:
            if output_dir:
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            else:
                root.after(0, lambda: messagebox.showerror("Path Error", "Output directory cannot be empty"))
                return
        except OSError as e:
            error_msg = f"Error creating directory {output_dir}: {str(e)}"
            root.after(0, lambda: messagebox.showerror("Directory Error", error_msg))
            return

        for i in range(1, max_iterations + 1):
            if stop_flag:
                # Save the project as stopped in the database if project_id is available
                if project_id is not None:
                    try:
                        save_project_end(project_id, current_code)
                    except Exception as e:
                        print(f"Error saving project as stopped: {e}")
                break

            # 1) Analyze code using the selected auditing LLM
            root.after(0, lambda i=i, m=max_iterations: ui_set_iteration(i, m))
            root.after(0, ui_set_status, f"Status: Auditing Code with {audit_llm}")

            try:
                audit_text = analyze_code(audit_llm_info, description, current_code)
                
                # FIX: Store the parsed bug counts in local variables to avoid them being lost
                crit, serious, ncrit, sugg = parse_bug_counts(audit_text)
                
                print(f"Iteration {i} - Parsed bug counts: Critical={crit}, Serious={serious}, N/Critical={ncrit}, Suggestions={sugg}")

                # Update UI with bug counts - Ensure we're passing the local variables
                def update_ui_audit(audit_result, critical, serious, non_critical, suggestions):
                    ui_set_audit_result(audit_result)
                    ui_set_bugs_count(critical, serious, non_critical, suggestions)
                    
                # Pass the local variables to prevent them from being lost
                root.after(0, lambda at=audit_text, c=crit, s=serious, nc=ncrit, sg=sugg: 
                          update_ui_audit(at, c, s, nc, sg))

                # Save audit result to file - For the last iteration or when no critical/serious bugs, use -Final suffix
                try:
                    if output_dir:
                        # Determine if this is the final iteration - either last iteration number or no critical/serious bugs
                        is_final = (i == max_iterations) or (crit == 0 and serious == 0)
                        
                        if is_final:
                            audit_file = os.path.join(output_dir, f"{base_filename}_audit-Final.txt")
                        else:
                            audit_file = os.path.join(output_dir, f"{base_filename}_audit_{i}.txt")
                        
                        with open(audit_file, "w", encoding="utf-8") as f:
                            f.write(audit_text)
                            print(f"Saved audit to: {audit_file}")
                except Exception as e:
                    print(f"Error saving audit file: {e}")
                    # Continue processing even if file save fails
            except Exception as e:
                error_msg = f"Error during code audit: {str(e)}"
                root.after(0, lambda: messagebox.showerror("Audit Error", error_msg))
                root.after(0, ui_set_status, f"Status: Error in auditing - {str(e)}")
                
                # Enable the Start button and disable the Stop button
                def reenable_button():
                    btn_start.config(state=tk.NORMAL)
                    btn_stop.config(state=tk.DISABLED)
                root.after(0, reenable_button)
                
                # Save project failure to database if project_id is available
                if project_id is not None:
                    try:
                        save_project_end(project_id, current_code)
                    except Exception as db_e:
                        print(f"Error saving project end after audit error: {db_e}")
                return

            if stop_flag:
                # Save the project as stopped in the database if project_id is available
                if project_id is not None:
                    try:
                        save_project_end(project_id, current_code)
                    except Exception as e:
                        print(f"Error saving project as stopped: {e}")
                break

            # 2) Correct code using the selected coding LLM
            root.after(0, ui_set_status, f"Status: Correcting Code with {coding_llm}")
            
            try:
                corrected_code, correction_list = correct_code(
                    coding_llm_info,
                    initial_prompt=description,
                    program_code=current_code,
                    code_analysis=audit_text
                )
                
                # Clean code of surrounding quotes after correction
                corrected_code = remove_surrounding_quotes(corrected_code)
                
                # Update the current code
                current_code = corrected_code

                # Count of fixed issues - UPDATED to pass bug counts as a limit
                # Make sure we're using the local variables here as well
                c_crit, c_serious, c_ncrit, c_sugg = parse_corrections(correction_list, (crit, serious, ncrit, sugg))

                # Update UI with corrected code and fixed bug counts - Use local variables
                def update_ui_corrected(code, cc, cs, cnc, css):
                    ui_set_current_code(code)
                    ui_set_corrected_bugs(cc, cs, cnc, css)
                    
                # Pass the local variables to prevent them from being lost
                root.after(0, lambda code=current_code, cc=c_crit, cs=c_serious, cnc=c_ncrit, css=c_sugg: 
                          update_ui_corrected(code, cc, cs, cnc, css))

                # Save - For the last iteration or when no critical/serious bugs, use -Final suffix
                try:
                    if output_dir:
                        ext = extension_for_language(language)
                        
                        # Determine if this is the final iteration - either last iteration number or no critical/serious bugs
                        is_final = (i == max_iterations) or (crit == 0 and serious == 0)
                        
                        if is_final:
                            iteration_file = os.path.join(output_dir, f"{base_filename}-Final{ext}")
                        else:
                            iteration_file = os.path.join(output_dir, f"{base_filename}_{i}{ext}")
                        
                        if current_code.strip():
                            with open(iteration_file, "w", encoding="utf-8") as f:
                                f.write(current_code)
                                print(f"Saved code to: {iteration_file}")
                except Exception as e:
                    print(f"Error saving code file: {e}")
                    # Continue processing even if file save fails
                    
                # Save iteration to database if project_id is available
                if project_id is not None:
                    try:
                        save_iteration(
                            project_id=project_id,
                            iteration_number=i,
                            code=current_code,
                            audit=audit_text,
                            bug_counts=(crit, serious, ncrit, sugg),
                            fix_counts=(c_crit, c_serious, c_ncrit, c_sugg)
                        )
                    except Exception as e:
                        print(f"Error saving iteration: {e}")
                
            except Exception as e:
                error_msg = f"Error during code correction: {str(e)}"
                root.after(0, lambda: messagebox.showerror("Correction Error", error_msg))
                root.after(0, ui_set_status, f"Status: Error in correction - {str(e)}")
                
                # Enable the Start button and disable the Stop button
                def reenable_button():
                    btn_start.config(state=tk.NORMAL)
                    btn_stop.config(state=tk.DISABLED)
                root.after(0, reenable_button)
                
                # Save project failure to database if project_id is available
                if project_id is not None:
                    try:
                        save_project_end(project_id, current_code)
                    except Exception as db_e:
                        print(f"Error saving project end after correction error: {db_e}")
                return

            # Exit condition - stop if no critical or serious bugs remain
            # Check the local variables, not potentially updated global state
            if crit == 0 and serious == 0:
                root.after(0, ui_set_status, f"Finished / No Critical and Serious Bugs (Iteration {i})")
                
                # Enable the Start button and disable the Stop button
                def reenable_button():
                    btn_start.config(state=tk.NORMAL)
                    btn_stop.config(state=tk.DISABLED)
                root.after(0, reenable_button)
                
                # Save project completion to database if project_id is available
                if project_id is not None:
                    try:
                        save_project_end(project_id, current_code)
                    except Exception as e:
                        print(f"Error saving project completion: {e}")
                return
                
            # Check if this was the last iteration
            if i >= max_iterations:
                root.after(0, ui_set_status, f"Finished / Max Iterations ({max_iterations}) Reached")
                
                # Enable the Start button and disable the Stop button
                def reenable_button():
                    btn_start.config(state=tk.NORMAL)
                    btn_stop.config(state=tk.DISABLED)
                root.after(0, reenable_button)
                
                # Save project completion to database if project_id is available
                if project_id is not None:
                    try:
                        save_project_end(project_id, current_code)
                    except Exception as e:
                        print(f"Error saving project completion: {e}")
                return

        # This code should never be reached due to the checks above, but as a safeguard:
        root.after(0, ui_set_status, "Finished / Process Complete")
        
        # Enable the Start button and disable the Stop button
        def reenable_button():
            btn_start.config(state=tk.NORMAL)
            btn_stop.config(state=tk.DISABLED)
        root.after(0, reenable_button)
        
        # Make sure project is marked as complete in the database if project_id is available
        if project_id is not None:
            try:
                save_project_end(project_id, current_code)
            except Exception as e:
                print(f"Error saving project completion: {e}")

    # Function to process a single model for generation
    def process_single_model(model_name, description, language, file_content, output_dir, result_queue, project_id=None):
        """
        Processes a single model for code generation and returns the result.
        
        Args:
            model_name: Name of the LLM model
            description: Program description
            language: Programming language
            file_content: Optional content of an existing program file to modify
            output_dir: Output directory for saving files
            result_queue: Queue to put results in
            project_id: Database project ID for storing results (can be None)
        """
        global model_status, current_displayed_model
        
        # Get the base filename (last part of the path)
        base_filename = os.path.basename(output_dir)
        
        try:
            # Update status in the model_status dictionary
            def update_status(status):
                model_status[model_name] = status
                # Force an immediate update of the UI
                root.after(0, update_model_status_ui)
            
            # Set status to in_progress
            root.after(0, lambda: update_status("in_progress"))
            
            # Get the LLM info for the current model
            llm_info = get_model_info(model_name)
            if not llm_info:
                root.after(0, lambda: update_status("failed"))
                result_queue.put((model_name, f"Error: Unknown model {model_name}", False, 0))
                
                # Save failed result to database if project_id is available
                if project_id is not None:
                    try:
                        save_model_result(project_id, model_name, f"Error: Unknown model {model_name}", "failed")
                    except Exception as db_error:
                        print(f"Database error saving model result: {db_error}")
                return
                    
            # Generate code
            generated_code = generate_code(llm_info, description, language, file_content)
            
            if generated_code.startswith("Error:"):
                root.after(0, lambda: update_status("failed"))
                result_queue.put((model_name, generated_code, False, 0))
                
                # Save failed result to database if project_id is available
                if project_id is not None:
                    try:
                        save_model_result(project_id, model_name, generated_code, "failed")
                    except Exception as db_error:
                        print(f"Database error saving model result: {db_error}")
                return
            
            # Clean code of surrounding quotes
            generated_code = remove_surrounding_quotes(generated_code)
            
            # Save to file
            saved = False
            try:
                if output_dir:
                    ext = extension_for_language(language)
                    # Create safe filename from model name (remove any special characters)
                    safe_model_name = re.sub(r'[^\w\s-]', '', model_name).strip().replace(' ', '_')
                    model_file = os.path.join(output_dir, f"{base_filename}_{safe_model_name}{ext}")
                    
                    with open(model_file, "w", encoding="utf-8") as f:
                        f.write(generated_code)
                    saved = True
                    print(f"Saved code for {model_name} to: {model_file}")
                
            except Exception as e:
                print(f"Error saving code for {model_name}: {e}")
                saved = False
                
            # Get timestamp for completion ordering
            completion_time = time.time()
            
            # Update status to completed
            root.after(0, lambda: update_status("completed"))
            
            # Always display the latest completed model's code
            def update_display():
                ui_set_current_code(generated_code, model_name)
            root.after(0, update_display)
            
            # Save successful result to database if project_id is available
            if project_id is not None:
                try:
                    save_model_result(project_id, model_name, generated_code, "completed")
                except Exception as db_error:
                    print(f"Database error saving model result: {db_error}")
                
            # Return the results using the Queue
            result_queue.put((model_name, generated_code, saved, completion_time))
                
        except Exception as e:
            error_msg = f"Error with {model_name}: {str(e)}"
            print(error_msg)
            root.after(0, lambda: update_status("failed"))
            result_queue.put((model_name, error_msg, False, 0))
            
            # Save error to database if project_id is available
            if project_id is not None:
                try:
                    save_model_result(project_id, model_name, error_msg, "failed")
                except Exception as db_error:
                    print(f"Database error saving model result: {db_error}")              
                    
    def run_multiple_creation(output_dir, description, language, selected_models, file_content=None, project_id=None):
        """
        Generates code with multiple LLM models asynchronously
        
        Args:
            output_dir: Output directory for saving results
            description: Program description
            language: Programming language
            selected_models: List of selected LLM model names
            file_content: Optional content of an existing program file to modify
            project_id: Database project ID for storing results (can be None)
        """
        global stop_flag, model_status, current_displayed_model
        
        # Get the base filename (last part of the path)
        base_filename = os.path.basename(output_dir)
        
        # FIX: Reset global tracking variables to avoid phantom LLMs
        # Clear the model_status dictionary completely
        model_status.clear()
        
        # Initialize status only for the currently selected models
        model_status.update({model: "pending" for model in selected_models})
        current_displayed_model = None
        
        # Initialize the status display immediately
        root.after(0, update_model_status_ui)
        
        # Start the animation for progress indicators
        root.after(0, animate_progress)
        
        # Ensure subdirectory exists
        try:
            if output_dir:  # Check that output_dir is not empty
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
            else:
                root.after(0, lambda: messagebox.showerror("Path Error", "Output directory cannot be empty"))
                return
        except OSError as e:
            error_msg = f"Error creating directory {output_dir}: {str(e)}"
            root.after(0, lambda: messagebox.showerror("Directory Error", error_msg))
            return
        
        # Keep track of successful generations for summary
        successful_models = []
        failed_models = []
        completed_models_with_time = []  # Track completion time for models
        
        # Create a thread-safe queue for result communication
        import queue
        result_queue = queue.Queue()
        
        # Update status
        root.after(0, ui_set_status, f"Status: Generating Code with All Selected Models Simultaneously")
        
        # Use ThreadPoolExecutor to run generation tasks concurrently
        with ThreadPoolExecutor(max_workers=min(10, len(selected_models))) as executor:
            # Submit all tasks and store the futures
            futures = []
            for model_name in selected_models:
                if stop_flag:
                    break
                    
                # Submit the task - Pass project_id to process_single_model
                future = executor.submit(
                    process_single_model, 
                    model_name, 
                    description, 
                    language, 
                    file_content, 
                    output_dir,
                    result_queue,
                    project_id  # This can be None, but process_single_model handles that
                )
                futures.append(future)
            
            # Wait for all futures to complete
            if not stop_flag:
                try:
                    # Wait for completion with a progress message
                    for future in futures:
                        future.result()  # This will block until the task is done
                        
                except Exception as e:
                    error_msg = f"Error waiting for tasks: {str(e)}"
                    print(error_msg)
        
        # Process results from the queue
        results = {}
        try:
            # Get all results from the queue
            while not result_queue.empty():
                model_name, code, saved, completion_time = result_queue.get_nowait()
                results[model_name] = (code, saved, completion_time)
                if saved and not code.startswith("Error:"):
                    completed_models_with_time.append((model_name, completion_time))
        except queue.Empty:
            pass
        
        # Process results
        for model_name, (code, saved, _) in results.items():
            if code.startswith("Error:"):
                failed_models.append(f"{model_name} ({code})")
            else:
                if saved:
                    successful_models.append(model_name)
                else:
                    failed_models.append(f"{model_name} (Failed to save)")
        
        # Show completion status
        summary = f"Code Generation Complete: {len(successful_models)}/{len(selected_models)} models successful"
        root.after(0, ui_set_status, summary)
        
        # Find the last completed model (by timestamp)
        final_code = ""
        if completed_models_with_time:
            # Sort by completion time to find the last completed model
            completed_models_with_time.sort(key=lambda x: x[1], reverse=True)
            latest_model = completed_models_with_time[0][0]
            latest_code = results[latest_model][0]
            final_code = latest_code
            
            # Update UI to show the latest completed model
            root.after(0, ui_set_current_code, latest_code, latest_model)
            
            # Save the latest completed model code with -Final suffix
            try:
                if output_dir:
                    ext = extension_for_language(language)
                    final_file = os.path.join(output_dir, f"{base_filename}-Final{ext}")
                    
                    with open(final_file, "w", encoding="utf-8") as f:
                        f.write(latest_code)
                        
                    print(f"Saved final code to: {final_file}")
            except Exception as e:
                print(f"Error saving final code: {e}")
                
        # Final update of the status display
        root.after(0, update_model_status_ui)
        
        # Show detailed summary in the audit result area
        detail = "Generation Results:\n\n"
        if successful_models:
            detail += "Successful Models:\n"
            for model in successful_models:
                detail += f"✓ {model}\n"
            detail += "\n"
            
        if failed_models:
            detail += "Failed Models:\n"
            for model in failed_models:
                detail += f"✗ {model}\n"
                    
        root.after(0, ui_set_audit_result, detail)
        
        # Mark project as complete in the database if project_id is available
        if project_id is not None:
            try:
                save_project_end(project_id, final_code)
            except Exception as db_error:
                print(f"Database error completing project: {db_error}")

    # Background thread function with parameters
    def background_process_with_params(filename, description, language, coding_llm, audit_llm):
        """
        Main processing function with explicit parameters to avoid global variables.
        """
        global stop_flag, process_thread, model_status
        
        try:
            # Get output directory based on settings
            output_dir = get_output_directory(filename)
            print(f"Using output directory: {output_dir}")
            
            # Get upload information if enabled
            has_upload = var_upload_file.get()
            upload_filename = ""
            upload_content = ""
            
            if has_upload:
                upload_filename = entry_file_path.get().strip()
                if upload_filename:
                    upload_content = read_file_content(upload_filename) or ""
            
            # Initialize project_id to None in case database operations fail
            project_id = None
            
            # Check if we're in multiple creation mode
            if mode_var.get() == "creation":
                # Get selected models from the listbox
                selected_indices = lb_coding_llm.curselection()
                if not selected_indices:
                    root.after(0, lambda: messagebox.showerror("Selection Error", "Please select at least one LLM model"))
                    def reenable_button():
                        btn_start.config(state=tk.NORMAL)
                        btn_stop.config(state=tk.DISABLED)
                    root.after(0, reenable_button)
                    return
                    
                selected_models = [lb_coding_llm.get(idx) for idx in selected_indices]
                
                # Check if we need to use an existing file
                file_content = None
                if has_upload:
                    if not upload_filename:
                        root.after(0, lambda: messagebox.showerror("File Error", "Please select a file to upload"))
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                        
                    file_content = upload_content
                    if not file_content:
                        # Error message already shown in read_file_content
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                
                # Save project start to database - handle possible None result
                try:
                    project_id = save_project_start(
                        project_name=filename,
                        output_dir=output_dir,
                        description=description,
                        has_upload=has_upload,
                        upload_filename=upload_filename,
                        upload_content=upload_content,
                        mode="creation",
                        coding_llms=selected_models
                    )
                    print(f"Creation mode: Project ID is {project_id}")
                except Exception as e:
                    print(f"Error saving project start: {e}")
                    project_id = None
                
                # Pass project_id to run_multiple_creation (it handles None properly)
                run_multiple_creation(output_dir, description, language, selected_models, file_content, project_id)
                
                # Ensure buttons are re-enabled after completion
                def reenable_button():
                    btn_start.config(state=tk.NORMAL)
                    btn_stop.config(state=tk.DISABLED)
                root.after(0, reenable_button)
                
            else:  # Multiple Correction mode (default)
                # Get number of iterations from Spinbox
                max_iterations = int(spinbox_iterations.get())
                
                # Check if we need to use an existing file
                file_content = None
                if has_upload:
                    if not upload_filename:
                        root.after(0, lambda: messagebox.showerror("File Error", "Please select a file to upload"))
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                        
                    file_content = upload_content
                    if not file_content:
                        # Error message already shown in read_file_content
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                
                # Save project start to database - handle possible None result
                try:
                    project_id = save_project_start(
                        project_name=filename,
                        output_dir=output_dir,
                        description=description,
                        has_upload=has_upload,
                        upload_filename=upload_filename,
                        upload_content=upload_content,
                        mode="correction",
                        iterations=max_iterations,
                        coding_llm=coding_llm,
                        auditing_llm=audit_llm
                    )
                    print(f"Correction mode: Project ID is {project_id}")
                except Exception as e:
                    print(f"Error saving project start: {e}")
                    project_id = None
                
                root.after(0, ui_set_status, f"Status: Generating Initial Code with {coding_llm}")
                
                # Get the LLM info for the selected coding LLM
                # Get the LLM info for the selected coding LLM
                coding_llm_info = get_model_info(coding_llm) or {"model": "o3-2025-04-16", "family": "OpenAI"}
                
                # Generate initial code with optional file content
                init_code = generate_code(coding_llm_info, description, language, file_content)
                
                if init_code.startswith("Error:"):
                    root.after(0, lambda: messagebox.showerror("Code Generation Error", init_code))
                    root.after(0, ui_set_status, "Status: Failed to generate initial code")
                    def reenable_button():
                        btn_start.config(state=tk.NORMAL)
                        btn_stop.config(state=tk.DISABLED)
                    root.after(0, reenable_button)
                    # Save project failure to database
                    if project_id is not None:
                        try:
                            save_project_end(project_id, f"ERROR: {init_code}")
                        except Exception as e:
                            print(f"Error saving project end: {e}")
                    return
                
                # Remove surrounding quotes if present
                init_code = remove_surrounding_quotes(init_code)
                
                # Save initial code to file
                try:
                    if output_dir:
                        ext = extension_for_language(language)
                        if init_code.strip():
                            # Create the output directory if it doesn't exist
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                                
                            # Save to the output directory
                            file_path = os.path.join(output_dir, filename + ext)
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(init_code)
                                print(f"Saved initial code to: {file_path}")
                except Exception as e:
                    print(f"Error saving initial code file: {e}")
                    # Continue even if file save fails
                
                def after_gen():
                    ui_set_current_code(init_code)
                    ui_set_iteration(1, max_iterations)  # Pass max_iterations to show progress
                root.after(0, after_gen)

                if stop_flag:
                    def reenable_button():
                        btn_start.config(state=tk.NORMAL)
                        btn_stop.config(state=tk.DISABLED)
                    root.after(0, reenable_button)
                    if project_id is not None:
                        try:
                            save_project_end(project_id, "STOPPED")
                        except Exception as e:
                            print(f"Error saving project end: {e}")
                    return
                
                # Make sure project_id is defined before passing to run_iteration_loop
                if project_id is None:
                    print("WARNING: project_id is None, database operations will be skipped")
                    
                # Pass all parameters including project_id (which might be None)
                run_iteration_loop(
                    output_dir,        # output_dir
                    filename,          # base_filename
                    description,       # description
                    init_code,         # init_code
                    language,          # language
                    coding_llm,        # coding_llm
                    audit_llm,         # audit_llm
                    max_iterations,    # max_iterations
                    project_id         # project_id - might be None but function handles that
                )

        except Exception as e:
            error_msg = f"Unexpected error in process: {str(e)}"
            print(f"EXCEPTION in background_process_with_params: {error_msg}")
            import traceback
            traceback.print_exc()
            root.after(0, lambda: messagebox.showerror("Process Error", error_msg))
            root.after(0, ui_set_status, f"Status: Process error - {str(e)}")
        finally:
            # Ensure the buttons are always re-enabled, even if there was an exception
            def reenable_button():
                btn_start.config(state=tk.NORMAL)
                btn_stop.config(state=tk.DISABLED)
            root.after(0, reenable_button)


    # Start Coding button handler
    def on_start_coding():
        """
        Handler for the Start Coding button.
        Validates inputs, initializes variables, and starts the background thread.
        """
        global user_filename, user_description, user_language
        global user_coding_llm, user_audit_llm, stop_flag, process_thread

        # Check for filename
        user_filename = entry_filename.get().strip()
        if not user_filename:
            messagebox.showerror("Error", "Please enter a program file name")
            return
            
        # Validate filename
        is_valid, error_msg = InputValidator.validate_filename(user_filename)
        if not is_valid:
            messagebox.showerror("Invalid Filename", error_msg)
            return

        # Get user description
        user_description = text_description.get("1.0", tk.END).strip()
        if not user_description:
            messagebox.showerror("Error", "Please enter a program description")
            return
            
        # Get programming language
        user_language = combo_language.get()
        
        # Handle different modes
        if mode_var.get() == "correction":
            # Validate iterations input for correction mode
            try:
                iterations = int(spinbox_iterations.get())
                if iterations < 1 or iterations > 20:
                    messagebox.showerror("Invalid Input", "Iterations must be between 1 and 20")
                    return
            except ValueError:
                messagebox.showerror("Invalid Input", "Iterations must be a valid number")
                return
            
            # Get selected LLMs for correction mode
            user_coding_llm = combo_coding_llm.get()
            user_audit_llm = combo_audit_llm.get()
            
            # Check if any models are available
            if user_coding_llm == "No models available - configure API keys first" or user_audit_llm == "No models available - configure API keys first":
                messagebox.showerror("Error", "No LLM models available. Please configure API keys first.")
                return
                        
            # Validate LLM selections - check if we support these families
            coding_info = get_model_info(user_coding_llm)
            audit_info = get_model_info(user_audit_llm)

            coding_family = coding_info.get('family') if coding_info else None
            audit_family = audit_info.get('family') if audit_info else None
            
            supported_families = ["OpenAI", "Claude", "Gemini", "DeepSeek"]
            
            if coding_family not in supported_families:
                messagebox.showerror("Unsupported Model", 
                                     f"The selected coding LLM family '{coding_family}' is not currently supported. "
                                     f"Please choose a model from one of these families: {', '.join(supported_families)}")
                return
                
            if audit_family not in supported_families:
                messagebox.showerror("Unsupported Model", 
                                     f"The selected auditing LLM family '{audit_family}' is not currently supported. "
                                     f"Please choose a model from one of these families: {', '.join(supported_families)}")
                return
        else:  # "creation" mode
            # Check if any models are selected in the listbox
            if not lb_coding_llm.curselection():
                messagebox.showerror("Error", "Please select at least one model for code generation")
                return
                
            # Check if any models are available
            if lb_coding_llm.get(0) == "No models available - configure API keys first":
                messagebox.showerror("Error", "No LLM models available. Please configure API keys first.")
                return
            
            # These are used in the thread but not actually needed for multiple creation
            user_coding_llm = ""
            user_audit_llm = ""

        # Reset UI
        ui_set_iteration(1)  # Start with iteration 1 instead of 0
        ui_set_current_code("")
        ui_set_audit_result("")
        ui_set_corrected_bugs(0, 0, 0, 0)
        ui_set_bugs_count(0, 0, 0, 0)
        ui_set_status("Status: Starting process...")

        stop_flag = False
        btn_start.config(state=tk.DISABLED)
        btn_stop.config(state=tk.NORMAL)

        # Start processing in background thread with explicitly passed parameters
        process_thread = threading.Thread(
            target=lambda: background_process_with_params(
                user_filename, 
                user_description, 
                user_language, 
                user_coding_llm, 
                user_audit_llm
            ), 
            daemon=True
        )
        process_thread.start()

    btn_start.config(command=on_start_coding)

    # Set command for Configure API Keys button
    def open_settings():
        open_settings_dialog(root, config, update_model_lists, all_available_models)

    btn_settings.config(command=open_settings)

    def on_close():
        global stop_flag
        stop_flag = True  # Signal threads to stop
        root.destroy()
        sys.exit(0)

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

if __name__ == "__main__":
    # Check if command-line arguments are provided
    if len(sys.argv) > 1:
        cli_main()
    else:
        main()