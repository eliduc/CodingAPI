#!/usr/bin/env python3
"""
CodingAPI - Enhanced version 6.0

A comprehensive library for generating code using multiple LLM providers with
improved error handling, security, performance, and user experience.

Key features:
- Modular provider architecture with wrapper pattern
- Enhanced error handling with retries and fallbacks
- Secure credential management
- Input validation and sanitization
- Asynchronous processing with caching
- Improved UI with progress indicators
- Enhanced prompting templates

Author: Claude
Date: May 3, 2025
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

# LLM Provider Constants
LLM_MAP = {
    # OpenAI models
    "OpenAI o4-mini":       {"model": "o4-mini", "family": "OpenAI", "temperature_allowed": False},
    "OpenAI o1-pro":        {"model": "o1-pro", "family": "OpenAI", "temperature_allowed": False},
    "OpenAI o3":            {"model": "o3-2025-04-16", "family": "OpenAI", "temperature_allowed": False},
    "OpenAI o3-mini":       {"model": "o3-mini", "family": "OpenAI", "temperature_allowed": False},
    "OpenAI GPT4o":         {"model": "gpt-4o", "family": "OpenAI", "temperature_allowed": True},
    "OpenAI GPT-4.1":       {"model": "gpt-4.1", "family": "OpenAI", "temperature_allowed": True},
    "OpenAI GPT-4.1-mini":  {"model": "gpt-4.1-mini", "family": "OpenAI", "temperature_allowed": True},
    
    # Claude models
    "Claude 3.7 Sonnet":    {"model": "claude-3-7-sonnet-20250219", "family": "Claude", "temperature_allowed": True},
    "Claude-3-Opus":        {"model": "claude-3-opus-latest", "family": "Claude", "temperature_allowed": True},
    "Claude-3-Haiku":       {"model": "claude-3-haiku-20240307", "family": "Claude", "temperature_allowed": True},
    
    # Gemini models
    "Gemini 2.5 Pro":       {"model": "gemini-2.5-pro-exp-03-25", "family": "Gemini", "temperature_allowed": True},
    "Gemini 2.0 Flash":     {"model": "gemini-2.0-flash", "family": "Gemini", "temperature_allowed": True},
    
    # DeepSeek models
    "DeepSeek R1":          {"model": "deepseek-reasoner", "family": "DeepSeek", "temperature_allowed": True},
}

# Model grouping by family
OPENAI_MODELS = [
    "OpenAI o4-mini",
    "OpenAI o1-pro",
    "OpenAI o3",
    "OpenAI o3-mini",
    "OpenAI GPT4o",
    "OpenAI GPT-4.1",
    "OpenAI GPT-4.1-mini"
]

CLAUDE_MODELS = [
    "Claude 3.7 Sonnet",
    "Claude-3-Opus",
    "Claude-3-Haiku"
]

GEMINI_MODELS = [
    "Gemini 2.5 Pro",
    "Gemini 2.0 Flash"
]

DEEPSEEK_MODELS = [
    "DeepSeek R1"
]

# Grouped models for coding dropdown (OpenAI first)
CODING_LLM_OPTIONS = OPENAI_MODELS + CLAUDE_MODELS + GEMINI_MODELS + DEEPSEEK_MODELS

# Grouped models for auditing dropdown (Claude first)
AUDITING_LLM_OPTIONS = CLAUDE_MODELS + OPENAI_MODELS + GEMINI_MODELS + DEEPSEEK_MODELS

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
        config_file = os.path.expanduser("~/.codingapi/config.json")
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
                return True
            except Exception as e:
                logger.warning(f"Failed to store API key in secure storage: {str(e)}")
        
        # Fallback to config file
        try:
            config_dir = os.path.expanduser("~/.codingapi")
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
                
            return True
        except Exception as e:
            logger.error(f"Failed to store API key in config file: {str(e)}")
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
    def validate_max_tokens(max_tokens: int, min_value: int = 50, max_value: int = 4096) -> int:
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
    
    def __init__(self, cache_dir: str = "~/.codingapi/cache", ttl: int = 86400):
        """Initialize the cache with optional time-to-live (in seconds)."""
        self.enabled = DISKCACHE_AVAILABLE
        self.ttl = ttl  # Default: 1 day
        
        if self.enabled:
            try:
                self.cache = diskcache.Cache(os.path.expanduser(cache_dir))
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
Return only the corrected code with no explanations outside of code comments."""
    
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

    # Code correction prompt template
    @staticmethod
    def get_correction_prompt(initial_prompt, program_code, code_analysis):
        """Returns a standardized code correction prompt"""
        return f"""This is a code of the program that was written as a response to the prompt {initial_prompt}.

Program code:
```
{program_code}
```

This is analysis of the code and suggestions for corrections: 
{code_analysis}

Audit the analysis and implement the corrections that you think are correct and will improve the code. Make the corrections one by one starting from critical errors, then serious, then non-critical, then suggestions.

Return your response in JSON format with the following structure:
{{
    "corrected_code": "full corrected code here",
    "corrections": [
        [0,1,0],  // Critical fixes - 0 means fixed, 1 means not fixed
        [0,1],    // Serious fixes
        [0,1,0],  // Non-critical fixes
        [1,0]     // Recommendations fixes
    ]
}}

For example, if code_analysis contains items:
1.1., 1.2, 1.3
2.1, 2.2
3.1, 3.2., 3.3.
4.1, 4.2

and you corrected 1.1, 1.3, 2.1, 3.2, 3.3, 4.2, then the "corrections" array should be:
[[0,1,0], [0,1], [1,0,0], [1,0]]

That is, all the corrected issues are marked as 0, all the issues that haven't been corrected as 1.
If an error category contains None (for example "2. Serious\\nNone"), the corresponding array should be empty: [].

Ensure your entire response can be parsed as valid JSON.
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

def extract_json_output(text):
    """
    Extracts corrected code and corrections list from JSON response.
    
    Args:
        text: The raw text response that should contain JSON
        
    Returns:
        Tuple of (corrected_code, corrections_list)
    """
    # Default values in case parsing fails
    default_code = ""
    default_corrections = "[]"
    
    try:
        # Try to find JSON content in the response
        # First, try to find content between triple backticks if it's formatted that way
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find content between curly braces
            json_match = re.search(r'(\{.*\})', text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # If no JSON pattern found, try parsing the entire text as JSON
                json_str = text
        
        # Parse the JSON
        result = json.loads(json_str)
        
        # Extract code and corrections
        if isinstance(result, dict):
            code = result.get('corrected_code', '')
            
            # Get the corrections array and convert to string representation
            corrections = result.get('corrections', [])
            corrections_str = str(corrections)
            
            return code, corrections_str
        else:
            # If the result is not a dictionary, return default values
            logger.error("Error: JSON response is not a dictionary")
            return default_code, default_corrections
            
    except Exception as e:
        logger.error(f"Error parsing JSON response: {e}")
        logger.debug(f"Raw response: {text[:500]}...")  # Print first 500 chars for debugging
        
        # If JSON parsing fails, try to extract code in a more traditional way
        # by finding the largest chunk of code-looking text
        code_blocks = re.findall(r'```.*?```', text, re.DOTALL)
        if code_blocks:
            # Find the longest code block
            longest_block = max(code_blocks, key=len)
            # Strip the backticks
            code = re.sub(r'^```\w*\n|```$', '', longest_block).strip()
            return code, default_corrections
            
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

#-----------------------------------------------------------------------------
# Client Helper Functions
#-----------------------------------------------------------------------------

def create_openai_client(api_key):
    """Creates an OpenAI client"""
    try:
        from openai import OpenAI
        if not api_key:
            raise ConfigurationError("OpenAI API key is missing.")
        
        # Add debug log
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
    Generates initial code using OpenAI models
    
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
    
    try:
        # First try the current API format
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ]
            }
            
            # Add temperature parameter if the model supports it
            if temp_allowed:
                kwargs["temperature"] = 0
                
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
            kwargs = {
                "model": model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ]
            }
            
            # Add temperature parameter if the model supports it
            if temp_allowed:
                kwargs["temperature"] = 0
                
            response = client.responses.create(**kwargs)
            
            # Extract text from the structure of the response
            code = ""
            for item in response.output:
                if item.type == "message":
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            code = content_item.text
                            break
            
            if not code:
                # Fallback extraction if the structure is different
                code = str(response)
                
            return code
    except Exception as e:
        error_msg = f"Error generating code with OpenAI: {str(e)}"
        logger.error(error_msg)
        return error_msg

def audit_with_openai(client, model, prompt_text, code_text, temp_allowed=False):
    """
    Analyzes code using OpenAI models
    
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
    
    try:
        # Try newer API first
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add temperature parameter if the model supports it
            if temp_allowed:
                kwargs["temperature"] = 0
                
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
            kwargs = {
                "model": model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Add temperature parameter if the model supports it
            if temp_allowed:
                kwargs["temperature"] = 0
                
            response = client.responses.create(**kwargs)
            
            # Extract text from the structure of the response
            result = ""
            for item in response.output:
                if item.type == "message":
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            result = content_item.text
                            break
            
            if not result:
                # Fallback extraction if the structure is different
                result = str(response)
                
            return result
    except Exception as e:
        error_msg = f"Error analyzing code with OpenAI: {str(e)}"
        logger.error(error_msg)
        return error_msg

def correct_with_openai(client, model, initial_prompt, program_code, code_analysis, temp_allowed=False):
    """
    Corrects code based on audit analysis using OpenAI models
    
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

    try:
        # Try newer API first
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Add temperature parameter if the model supports it
            if temp_allowed:
                kwargs["temperature"] = 0
                
            response = client.chat.completions.create(**kwargs)
            
            # Extract content safely
            if hasattr(response, 'choices') and len(response.choices) > 0:
                if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                    full_text = response.choices[0].message.content
                    # Parse JSON response
                    return extract_json_output(full_text)
            
            # If we get here, something went wrong with the extraction
            return f"Error: Failed to extract content from response: {str(response)}", "[]"
            
        except AttributeError:
            # Fallback to older API
            logger.warning("Using fallback OpenAI API approach due to AttributeError")
            kwargs = {
                "model": model,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            }
            
            # Add temperature parameter if the model supports it
            if temp_allowed:
                kwargs["temperature"] = 0
                
            response = client.responses.create(**kwargs)
            
            # Extract text from the structure of the response
            full_text = ""
            for item in response.output:
                if item.type == "message":
                    for content_item in item.content:
                        if content_item.type == "output_text":
                            full_text = content_item.text
                            break
            
            if not full_text:
                # Fallback extraction if the structure is different
                full_text = str(response)
            
            # Parse JSON response
            return extract_json_output(full_text)
    except Exception as e:
        error_msg = f"Error correcting code with OpenAI: {str(e)}"
        logger.error(error_msg)
        return error_msg, "[]"

#-----------------------------------------------------------------------------
# Claude Implementation
#-----------------------------------------------------------------------------

def generate_with_claude(client, model, description, language, file_content=None, temp_allowed=True):
    """
    Generates initial code using Claude models
    
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
    
    try:
        kwargs = {
            "model": model,
            "max_tokens": 20000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt_text}
            ]
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.messages.create(**kwargs)
        
        # Extract content safely
        if hasattr(response, 'content') and len(response.content) > 0:
            return response.content[0].text
        else:
            # Fallback extraction if the structure is different
            return str(response)
            
    except Exception as e:
        error_msg = f"Error generating code with Claude: {str(e)}"
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

def audit_with_claude(client, model, prompt_text, code_text, temp_allowed=True):
    """
    Analyzes code using Claude models
    
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
    
    try:
        kwargs = {
            "model": model,
            "max_tokens": 20000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.messages.create(**kwargs)
        
        # Extract content safely
        if hasattr(response, 'content') and len(response.content) > 0:
            return response.content[0].text
        else:
            # Fallback extraction if the structure is different
            return str(response)
            
    except Exception as e:
        error_msg = f"Error analyzing code with Claude: {str(e)}"
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

def correct_with_claude(client, model, initial_prompt, program_code, code_analysis, temp_allowed=True):
    """
    Corrects code based on audit analysis using Claude models
    
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

    try:
        kwargs = {
            "model": model,
            "max_tokens": 20000,
            "system": system_prompt,
            "messages": [
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.messages.create(**kwargs)
        
        full_text = ""
        if hasattr(response, 'content') and len(response.content) > 0:
            full_text = response.content[0].text
        else:
            # Fallback extraction if the structure is different
            full_text = str(response)
        
        # Parse JSON response
        return extract_json_output(full_text)
        
    except Exception as e:
        error_msg = f"Error correcting code with Claude: {str(e)}"
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

#-----------------------------------------------------------------------------
# Gemini Implementation
#-----------------------------------------------------------------------------

def generate_with_gemini(client, model, description, language, file_content=None, temp_allowed=True):
    """
    Generates initial code using Gemini models
    
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
    
    try:
        from google.genai import types
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            config_params["temperature"] = 0
            
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
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

def audit_with_gemini(client, model, prompt_text, code_text, temp_allowed=True):
    """
    Analyzes code using Gemini models
    
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
    
    try:
        from google.genai import types
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            config_params["temperature"] = 0
            
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
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

def correct_with_gemini(client, model, initial_prompt, program_code, code_analysis, temp_allowed=True):
    """
    Corrects code based on audit analysis using Gemini models
    
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

    try:
        from google.genai import types
        
        config_params = {
            "system_instruction": system_prompt
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            config_params["temperature"] = 0
            
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
        
        # Parse JSON response
        return extract_json_output(full_text)
        
    except Exception as e:
        error_msg = f"Error correcting code with Gemini: {str(e)}"
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

#-----------------------------------------------------------------------------
# DeepSeek Implementation
#-----------------------------------------------------------------------------

def generate_with_deepseek(client, model, description, language, file_content=None, temp_allowed=True):
    """
    Generates initial code using DeepSeek models
    
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
    # Get standardized prompt
    prompt_text = PromptLibrary.get_generation_prompt(description, language, file_content)
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_GEN.format(language=language)
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ],
            "stream": False
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.chat.completions.create(**kwargs)
        
        # Extract content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        
        # Fallback to string representation
        return str(response)
            
    except Exception as e:
        error_msg = f"Error generating code with DeepSeek: {str(e)}"
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

def audit_with_deepseek(client, model, prompt_text, code_text, temp_allowed=True):
    """
    Analyzes code using DeepSeek models
    
    Args:
        client: DeepSeek client (OpenAI client with custom base_url)
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
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": False
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.chat.completions.create(**kwargs)
        
        # Extract content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                return response.choices[0].message.content
        
        # Fallback to string representation
        return str(response)
            
    except Exception as e:
        error_msg = f"Error analyzing code with DeepSeek: {str(e)}"
        # Log the error for debugging
        logger.error(error_msg)
        raise Exception(error_msg)

def correct_with_deepseek(client, model, initial_prompt, program_code, code_analysis, temp_allowed=True):
    """
    Corrects code based on audit analysis using DeepSeek models
    
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
    # Get standardized correction prompt
    user_prompt = PromptLibrary.get_correction_prompt(initial_prompt, program_code, code_analysis)
    language = "python"  # Default language, can be improved by detecting from code
    system_prompt = PromptLibrary.SYSTEM_PROMPT_CODE_CORRECTION.format(language=language)

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "stream": False
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.chat.completions.create(**kwargs)
        
        full_text = ""
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                full_text = response.choices[0].message.content
        else:
            # Fallback to string representation
            full_text = str(response)
        
        # Parse JSON response
        return extract_json_output(full_text)
        
    except Exception as e:
        error_msg = f"Error correcting code with DeepSeek: {str(e)}"
        # Log the error for debugging
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

#-----------------------------------------------------------------------------
# GUI Implementation
#-----------------------------------------------------------------------------

def main():
    """Main entry point for the GUI application."""
    root = tk.Tk()
    root.title("Code Generator with Audit (Responsive UI)")
    root.geometry("1200x800")
    root.minsize(1000, 600)
    root.configure(bg="#F5F5F5")

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

    # Program file name and file upload in the same row
    frm_filename_row = ttk.Frame(frame_input)
    frm_filename_row.pack(fill=tk.X, pady=5)
    
    # Program file name section (left)
    frm_filename = ttk.Frame(frm_filename_row)
    frm_filename.pack(side=tk.LEFT, fill=tk.X)
    label_filename = ttk.Label(frm_filename, text="Program file name (no extension)", style='Bold.TLabel')
    label_filename.pack(anchor="w")
    entry_filename = ttk.Entry(frm_filename, width=30)
    entry_filename.pack(anchor="w", pady=3)
    
    # File upload section (right)
    frm_file_select = ttk.Frame(frm_filename_row)
    frm_file_select.pack_forget()  # Initially hidden
    
    label_file = ttk.Label(frm_file_select, text="Program File", style='Bold.TLabel')
    label_file.pack(anchor="w")
    
    # File path and browse button in a horizontal layout
    frm_file_path = ttk.Frame(frm_file_select)
    frm_file_path.pack(fill=tk.X, pady=3)
    
    entry_file_path = ttk.Entry(frm_file_path, width=30)
    entry_file_path.pack(side=tk.LEFT, padx=(0, 5))
    
    btn_browse = ttk.Button(frm_file_path, text="Browse...")
    btn_browse.pack(side=tk.LEFT)

    # Add Checkbox for uploading existing file
    frm_upload = ttk.Frame(frame_input)
    frm_upload.pack(fill=tk.X, pady=5)
    var_upload_file = tk.BooleanVar()
    var_upload_file.set(False)
    check_upload = ttk.Checkbutton(frm_upload, text="Upload existing program to modify", variable=var_upload_file)
    check_upload.pack(anchor="w")

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
    
    # Multiple Creation radio button (right)
    rb_creation = ttk.Radiobutton(
        toggle_frame, 
        text="Multiple Creation", 
        value="creation", 
        variable=mode_var
    )
    rb_creation.pack(side=tk.LEFT)
    
    # Program Description
    frm_description = ttk.Frame(frame_input)
    frm_description.pack(fill=tk.X, pady=5)
    label_description = ttk.Label(frm_description, text="Program Description", style='Bold.TLabel')
    label_description.pack(anchor="w")

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
    languages = ["Python","Java","JavaScript","C","C++","Pascal","Julia","FORTRAN"]
    combo_language = ttk.Combobox(frm_lang, values=languages, state="readonly")
    combo_language.current(0)
    combo_language.pack(anchor="w", pady=3, fill=tk.X)
    
    # Coding LLM - second column
    frm_coding_llm = ttk.Frame(frm_options)
    frm_coding_llm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    label_coding_llm = ttk.Label(frm_coding_llm, text="Coding LLM", style='Bold.TLabel')
    label_coding_llm.pack(anchor="w")
    
    # Two different widgets for Coding LLM based on mode
    # 1. Standard Combobox for Multiple Correction mode
    combo_coding_llm = ttk.Combobox(frm_coding_llm, values=CODING_LLM_OPTIONS, state="readonly")
    combo_coding_llm.current(0)
    combo_coding_llm.pack(anchor="w", pady=3, fill=tk.X)
    
    # 2. Listbox with multiple selection for Multiple Creation mode
    frame_listbox = ttk.Frame(frm_coding_llm)
    lb_coding_llm = tk.Listbox(frame_listbox, selectmode=tk.MULTIPLE, height=6)
    for model in CODING_LLM_OPTIONS:
        lb_coding_llm.insert(tk.END, model)
    
    scrollbar = ttk.Scrollbar(frame_listbox, orient="vertical", command=lb_coding_llm.yview)
    lb_coding_llm.configure(yscrollcommand=scrollbar.set)
    
    lb_coding_llm.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    # Initially hide the listbox
    frame_listbox.pack_forget()
    
    # Auditing LLM - third column
    frm_audit_llm = ttk.Frame(frm_options)
    frm_audit_llm.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 10))
    label_audit_llm = ttk.Label(frm_audit_llm, text="Auditing LLM", style='Bold.TLabel')
    label_audit_llm.pack(anchor="w")
    combo_audit_llm = ttk.Combobox(frm_audit_llm, values=AUDITING_LLM_OPTIONS, state="readonly")
    combo_audit_llm.current(0)  # Default to first Claude model for auditing
    combo_audit_llm.pack(anchor="w", pady=3, fill=tk.X)
    
    # Iterations - fourth column
    frm_iterations = ttk.Frame(frm_options)
    frm_iterations.pack(side=tk.LEFT, fill=tk.X, expand=True)
    label_iterations = ttk.Label(frm_iterations, text="Iterations", style='Bold.TLabel')
    label_iterations.pack(anchor="w")
    
    # Spinbox for selecting number of iterations (1-20)
    spinbox_iterations = ttk.Spinbox(frm_iterations, from_=1, to=20, width=5)
    spinbox_iterations.set(5)  # Default is 5
    spinbox_iterations.pack(anchor="w", pady=3)

    # Buttons: Cancel / Start
    frm_buttons_top = ttk.Frame(frame_input)
    frm_buttons_top.pack(fill=tk.X, pady=10)
    btn_cancel = ttk.Button(frm_buttons_top, text="Cancel")
    btn_start = ttk.Button(frm_buttons_top, text="Start Coding", style='Green.TButton', width=15)
    btn_stop = ttk.Button(frm_buttons_top, text="Stop Process", width=15)
    btn_cancel.pack(side=tk.LEFT, padx=5)
    btn_start.pack(side=tk.LEFT, padx=5)
    btn_stop.pack(side=tk.LEFT, padx=5)
    btn_stop.config(state=tk.DISABLED)  # Initially disabled

    def on_cancel():
        root.destroy()
        sys.exit(0)
    btn_cancel.config(command=on_cancel)

    sep = ttk.Separator(root, orient="horizontal")
    sep.pack(fill="x", padx=10, pady=(5, 5))

    # ================== LOWER SECTION: PROCESS ==================
    frame_process = ttk.Frame(root, padding=10)
    frame_process.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    lbl_title_process = ttk.Label(frame_process, text="Audit & Correction Process:", font=("Arial", 14, "bold"))
    lbl_title_process.pack(anchor="w", pady=(0, 10))

    lbl_status = ttk.Label(frame_process, text="Status: Idle", style='Status.TLabel')
    lbl_status.pack(anchor="w", fill=tk.X, pady=(0, 10))

    lbl_iter = ttk.Label(frame_process, text="Iteration: 1")
    lbl_iter.pack(anchor="w", pady=(0, 10))

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
    
    # Copy button for Current Code
    btn_copy_code = ttk.Button(frm_left_header, text="📋", width=3)
    btn_copy_code.pack(side=tk.RIGHT)

    txt_current_code = scrolledtext.ScrolledText(frame_left, width=60, height=10)
    txt_current_code.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    frame_left.grid_rowconfigure(1, weight=1)

    # Corrected Bugs -- with full names
    lbl_corrected_bugs = ttk.Label(frame_left, text="Corrected: Critical=0, Serious=0, N/critical=0, Suggestions=0", style='Bold.TLabel')
    lbl_corrected_bugs.grid(row=2, column=0, sticky="nw", padx=5, pady=(0,5))

    frame_right = ttk.Frame(pane)
    pane.add(frame_right, weight=1)
    frame_right.grid_columnconfigure(0, weight=1)

    # Frame for header and copy button on the right
    frm_right_header = ttk.Frame(frame_right)
    frm_right_header.grid(row=0, column=0, sticky="ew", padx=5, pady=(0,5))
    lbl_code_right = ttk.Label(frm_right_header, text="Audit Result", style='Bold.TLabel')
    lbl_code_right.pack(side=tk.LEFT)
    
    # Copy button for Audit Result
    btn_copy_audit = ttk.Button(frm_right_header, text="📋", width=3)
    btn_copy_audit.pack(side=tk.RIGHT)

    txt_audit_result = scrolledtext.ScrolledText(frame_right, width=60, height=10)
    txt_audit_result.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
    frame_right.grid_rowconfigure(1, weight=1)

    # Bugs/Corrections Count -- with full names
    lbl_bugs_count = ttk.Label(frame_right, text="Bugs: Critical=0, Serious=0, N/critical=0, Suggestions=0", style='Bold.TLabel')
    lbl_bugs_count.grid(row=2, column=0, sticky="nw", padx=5, pady=(0,5))

    # ------------------------------------
    # LOGIC
    # ------------------------------------
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

    def ui_set_iteration(num):
        lbl_iter.config(text=f"Iteration: {num}")

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
            
    # Iteration loop (analysis + correction)
    def run_iteration_loop(filename, description, init_code, language, coding_llm, audit_llm, max_iterations):
        """
        Runs the iterative code audit and correction process.
        
        Each iteration:
        1. Analyzes code using the audit LLM 
        2. Applies corrections using the coding LLM
        3. Updates the UI and saves files
        4. Continues until max iterations or no critical/serious bugs remain
        """
        global stop_flag
        current_code = init_code
        coding_llm_info = LLM_MAP.get(coding_llm, {"model": "o1-pro", "family": "OpenAI"})
        audit_llm_info = LLM_MAP.get(audit_llm, {"model": "claude-3-7-sonnet-20250219", "family": "Claude"})

        # Ensure subdirectory exists
        try:
            if filename:
                if not os.path.exists(filename):
                    os.makedirs(filename)
            else:
                root.after(0, lambda: messagebox.showerror("Path Error", "Filename cannot be empty"))
                return
        except OSError as e:
            error_msg = f"Error creating directory {filename}: {str(e)}"
            root.after(0, lambda: messagebox.showerror("Directory Error", error_msg))
            return

        for i in range(1, max_iterations + 1):
            if stop_flag:
                break

            # 1) Analyze code using the selected auditing LLM
            root.after(0, ui_set_iteration, i)
            root.after(0, ui_set_status, f"Status: Auditing Code with {audit_llm}")

            try:
                audit_text = analyze_code(audit_llm_info, description, current_code)
                crit, serious, ncrit, sugg = parse_bug_counts(audit_text)

                def update_ui_audit():
                    ui_set_audit_result(audit_text)
                    ui_set_bugs_count(crit, serious, ncrit, sugg)
                root.after(0, update_ui_audit)

                # Save audit result to file
                try:
                    if filename:
                        audit_file = os.path.join(filename, f"{filename}_audit_{i}.txt")
                        with open(audit_file, "w", encoding="utf-8") as f:
                            f.write(audit_text)
                except Exception as e:
                    print(f"Error saving audit file: {e}")
                    # Continue processing even if file save fails
            except Exception as e:
                error_msg = f"Error during code audit: {str(e)}"
                root.after(0, lambda: messagebox.showerror("Audit Error", error_msg))
                root.after(0, ui_set_status, f"Status: Error in auditing - {str(e)}")
                return

            if stop_flag:
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
                
                current_code = corrected_code

                # Count of fixed issues - UPDATED to pass bug counts as a limit
                c_crit, c_serious, c_ncrit, c_sugg = parse_corrections(correction_list, (crit, serious, ncrit, sugg))

                def update_ui_corrected():
                    ui_set_current_code(current_code)
                    ui_set_corrected_bugs(c_crit, c_serious, c_ncrit, c_sugg)
                root.after(0, update_ui_corrected)

                # Save
                try:
                    if filename:
                        ext = extension_for_language(language)
                        iteration_file = os.path.join(filename, f"{filename}_{i}{ext}")
                        if current_code.strip():
                            with open(iteration_file, "w", encoding="utf-8") as f:
                                f.write(current_code)
                except Exception as e:
                    print(f"Error saving code file: {e}")
                    # Continue processing even if file save fails
            except Exception as e:
                error_msg = f"Error during code correction: {str(e)}"
                root.after(0, lambda: messagebox.showerror("Correction Error", error_msg))
                root.after(0, ui_set_status, f"Status: Error in correction - {str(e)}")
                return

            # Exit condition - stop if no critical or serious bugs remain
            if crit == 0 and serious == 0:
                root.after(0, ui_set_status, "Finished / No Critical and Serious Bugs")
                return

        root.after(0, ui_set_status, "Finished / Iteration Number Expired")

    # Function to process a single model for generation
    def process_single_model(model_name, description, language, file_content, filename, result_queue):
        """
        Processes a single model for code generation and returns the result.
        
        Args:
            model_name: Name of the LLM model
            description: Program description
            language: Programming language
            file_content: Optional content of an existing program file to modify
            filename: Base filename for saving
            result_queue: Queue to put results in
        """
        try:
            # Update status in the model_status dictionary
            def update_status(status):
                model_status[model_name] = status
                # Force an immediate update of the UI
                root.after(0, update_model_status_ui)
            
            # Set status to in_progress
            root.after(0, lambda: update_status("in_progress"))
            
            # Get the LLM info for the current model
            llm_info = LLM_MAP.get(model_name, None)
            if not llm_info:
                root.after(0, lambda: update_status("failed"))
                result_queue.put((model_name, f"Error: Unknown model {model_name}", False, 0))
                return
                
            # Generate code
            generated_code = generate_code(llm_info, description, language, file_content)
            
            if generated_code.startswith("Error:"):
                root.after(0, lambda: update_status("failed"))
                result_queue.put((model_name, generated_code, False, 0))
                return
            
            # Clean code of surrounding quotes
            generated_code = remove_surrounding_quotes(generated_code)
            
            # Save to file
            saved = False
            try:
                if filename:
                    ext = extension_for_language(language)
                    # Create safe filename from model name (remove any special characters)
                    safe_model_name = re.sub(r'[^\w\s-]', '', model_name).strip().replace(' ', '_')
                    model_file = os.path.join(filename, f"{filename}_{safe_model_name}{ext}")
                    
                    with open(model_file, "w", encoding="utf-8") as f:
                        f.write(generated_code)
                    saved = True
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
                
            # Return the results using the Queue
            result_queue.put((model_name, generated_code, saved, completion_time))
                
        except Exception as e:
            error_msg = f"Error with {model_name}: {str(e)}"
            print(error_msg)
            root.after(0, lambda: update_status("failed"))
            result_queue.put((model_name, error_msg, False, 0))

    # Asynchronous version of run_multiple_creation
    def run_multiple_creation(filename, description, language, selected_models, file_content=None):
        """
        Generates code with multiple LLM models asynchronously
        
        Args:
            filename: Base filename for saving results
            description: Program description
            language: Programming language
            selected_models: List of selected LLM model names
            file_content: Optional content of an existing program file to modify
        """
        global stop_flag, model_status, current_displayed_model
        
        # Reset global tracking variables
        model_status = {model: "pending" for model in selected_models}
        current_displayed_model = None
        
        # Initialize the status display immediately
        root.after(0, update_model_status_ui)
        
        # Start the animation for progress indicators
        root.after(0, animate_progress)
        
        # Ensure subdirectory exists
        try:
            if filename:  # Check that filename is not empty
                if not os.path.exists(filename):
                    os.makedirs(filename)
            else:
                root.after(0, lambda: messagebox.showerror("Path Error", "Filename cannot be empty"))
                return
        except OSError as e:
            error_msg = f"Error creating directory {filename}: {str(e)}"
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
                    
                # Submit the task
                future = executor.submit(
                    process_single_model, 
                    model_name, 
                    description, 
                    language, 
                    file_content, 
                    filename,
                    result_queue
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
        
        # Explicitly display the last completed model (by timestamp) at the end of the process
        if completed_models_with_time:
            # Sort by completion time to find the last completed model
            completed_models_with_time.sort(key=lambda x: x[1], reverse=True)
            latest_model = completed_models_with_time[0][0]
            latest_code = results[latest_model][0]
            
            # Update UI to show the latest completed model
            root.after(0, ui_set_current_code, latest_code, latest_model)
            
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

    # Background thread function with parameters
    def background_process_with_params(filename, description, language, coding_llm, audit_llm):
        """
        Main processing function with explicit parameters to avoid global variables.
        """
        global stop_flag, model_status
        
        try:
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
                if var_upload_file.get():
                    file_path = entry_file_path.get().strip()
                    if not file_path:
                        root.after(0, lambda: messagebox.showerror("File Error", "Please select a file to upload"))
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                        
                    file_content = read_file_content(file_path)
                    if not file_content:
                        # Error message already shown in read_file_content
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                
                # Run multiple code generation asynchronously
                run_multiple_creation(filename, description, language, selected_models, file_content)
                
            else:  # Multiple Correction mode (default)
                # Check if we need to use an existing file
                file_content = None
                if var_upload_file.get():
                    file_path = entry_file_path.get().strip()
                    if not file_path:
                        root.after(0, lambda: messagebox.showerror("File Error", "Please select a file to upload"))
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                        
                    file_content = read_file_content(file_path)
                    if not file_content:
                        # Error message already shown in read_file_content
                        def reenable_button():
                            btn_start.config(state=tk.NORMAL)
                            btn_stop.config(state=tk.DISABLED)
                        root.after(0, reenable_button)
                        return
                
                root.after(0, ui_set_status, f"Status: Generating Initial Code with {coding_llm}")
                
                # Get the LLM info for the selected coding LLM
                coding_llm_info = LLM_MAP.get(coding_llm, {"model": "o1-pro", "family": "OpenAI"})
                
                # Generate initial code with optional file content
                init_code = generate_code(coding_llm_info, description, language, file_content)
                
                if init_code.startswith("Error:"):
                    root.after(0, lambda: messagebox.showerror("Code Generation Error", init_code))
                    root.after(0, ui_set_status, "Status: Failed to generate initial code")
                    def reenable_button():
                        btn_start.config(state=tk.NORMAL)
                        btn_stop.config(state=tk.DISABLED)
                    root.after(0, reenable_button)
                    return
                
                # Remove surrounding quotes if present
                init_code = remove_surrounding_quotes(init_code)
                
                # Save initial code to file
                try:
                    if filename:
                        ext = extension_for_language(language)
                        if init_code.strip():
                            # Create a subdirectory if it doesn't exist
                            if not os.path.exists(filename):
                                os.makedirs(filename)
                                
                            # Save to the subdirectory
                            file_path = os.path.join(filename, filename + ext)
                            with open(file_path, "w", encoding="utf-8") as f:
                                f.write(init_code)
                except Exception as e:
                    print(f"Error saving initial code file: {e}")
                    # Continue even if file save fails

                def after_gen():
                    ui_set_current_code(init_code)
                    ui_set_iteration(1)  # Start with iteration 1 instead of 0
                root.after(0, after_gen)

                if stop_flag:
                    def reenable_button():
                        btn_start.config(state=tk.NORMAL)
                        btn_stop.config(state=tk.DISABLED)
                    root.after(0, reenable_button)
                    return

                # Get number of iterations from Spinbox
                max_iterations = int(spinbox_iterations.get())
                
                run_iteration_loop(
                    filename, description, init_code,
                    language, coding_llm, audit_llm, max_iterations
                )

        except Exception as e:
            error_msg = f"Unexpected error in process: {str(e)}"
            root.after(0, lambda: messagebox.showerror("Process Error", error_msg))
            root.after(0, ui_set_status, f"Status: Process error - {str(e)}")
        finally:
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
            
            # Validate LLM selections - check if we support these families
            coding_family = LLM_MAP.get(user_coding_llm, {}).get('family')
            audit_family = LLM_MAP.get(user_audit_llm, {}).get('family')
            
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