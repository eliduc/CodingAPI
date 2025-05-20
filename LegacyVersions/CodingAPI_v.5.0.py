import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import sys
import os
import ast
import re
import json
import threading
import configparser
import string
import concurrent.futures
import queue
import time

from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types

# Constants for models and delimiters
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
    
    # Only implement OpenAI, Claude, Gemini, and DeepSeek for now
    # Other models will be commented out until implemented
    # "DeepSeek":             {"model": "deepseek", "family": "DeepSeek"},
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

# ===== Standardized prompts for code generation, audit, and correction =====
# These prompts are used across different LLM families to ensure consistency

# System prompts for different tasks
SYSTEM_PROMPT_CODE_GEN = "You are a professional code generation assistant. Write clear, efficient code with no explanations."
SYSTEM_PROMPT_CODE_AUDIT = "You are an expert code reviewer specialized in finding bugs and issues. Be thorough and precise."
SYSTEM_PROMPT_CODE_CORRECTION = "You are a code generation assistant. Write clear, efficient code with no explanations."

# Code generation prompt template
def get_generation_prompt(description, language, file_content=None):
    """Returns a standardized code generation prompt"""
    if file_content:
        return (
            f"It is required to modify the program below in accordance with the task: {description}. "
            f"If the program is written in a language different from {language}, it should be translated (re-written) in {language}.\n\n"
            f"Program to modify:\n```\n{file_content}\n```"
        )
    else:
        return (
            f"It is required to write a program or adjust the program in accordance with the task: {description}. "
            f"If the task does not specify an algorithmic language, it should be done in {language}."
        )

# Code audit prompt template
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
def get_correction_prompt(initial_prompt, program_code, code_analysis):
    """Returns a standardized code correction prompt"""
    return f"""This is a code of the program that was written as a response to the prompt {initial_prompt}.

Program code:
{program_code}

This is analysis of the code and suggestions for corrections: {code_analysis}
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
            print("Error: JSON response is not a dictionary")
            return default_code, default_corrections
            
    except Exception as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {text[:500]}...")  # Print first 500 chars for debugging
        
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
        print(f"Error reading API keys: {e}")
    
    return keys

def validate_filename(filename):
    """
    Validates filename for invalid characters
    
    Args:
        filename: The filename to validate
        
    Returns:
        tuple: (is_valid, error_message)
    """
    # Check for empty filename
    if not filename:
        return False, "Filename cannot be empty"
    
    # Check for invalid characters
    invalid_chars = set(filename).intersection(set('\\/:*?"<>|'))
    if invalid_chars:
        return False, f"Filename contains invalid characters: {', '.join(invalid_chars)}"
    
    # Additional validations can be added here
    return True, ""

# ===== Client creation functions for different LLM families =====

def create_openai_client(api_key):
    """Creates an OpenAI client"""
    if not api_key:
        raise ValueError("OpenAI API key is missing. Please add it to the APIKeys file.")
    client = OpenAI(api_key=api_key)
    return client

def create_claude_client(api_key):
    """Creates a Claude client"""
    if not api_key:
        raise ValueError("Claude API key is missing. Please add it to the APIKeys file.")
    client = Anthropic(api_key=api_key)
    return client

def create_gemini_client(api_key):
    """Creates a Gemini client"""
    if not api_key:
        raise ValueError("Gemini API key is missing. Please add it to the APIKeys file.")
    client = genai.Client(api_key=api_key)
    return client

def create_deepseek_client(api_key):
    """Creates a DeepSeek client"""
    if not api_key:
        raise ValueError("DeepSeek API key is missing. Please add it to the APIKeys file.")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    return client

def create_client(family):
    """
    Creates and returns appropriate client based on the LLM family
    
    Args:
        family: The LLM family name ('OpenAI', 'Claude', 'Gemini', 'DeepSeek', etc.)
    
    Returns:
        The initialized client or None if family not supported
    """
    api_keys = get_api_keys()
    
    try:
        if family == "OpenAI":
            return create_openai_client(api_keys.get('OpenAI', ''))
        elif family == "Claude":
            return create_claude_client(api_keys.get('Claude', ''))
        elif family == "Gemini":
            return create_gemini_client(api_keys.get('Gemini', ''))
        elif family == "DeepSeek":
            return create_deepseek_client(api_keys.get('DeepSeek', ''))
        # Add support for other families here in the future
        else:
            raise ValueError(f"Unsupported LLM family: {family}")
    except ValueError as e:
        raise
    except Exception as e:
        raise Exception(f"Failed to create client for {family}: {str(e)}")

# ===== Generation request functions (initial code generation) =====

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
    prompt_text = get_generation_prompt(description, language, file_content)
    
    try:
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_CODE_GEN},
                {"role": "user", "content": prompt_text}
            ]
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.responses.create(**kwargs)
        
        # Извлекаем текст из структуры ответа
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
        print(error_msg)
        return error_msg

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
    prompt_text = get_generation_prompt(description, language, file_content)
    
    try:
        kwargs = {
            "model": model,
            "max_tokens": 20000,
            "system": SYSTEM_PROMPT_CODE_GEN,
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
        print(error_msg)
        raise Exception(error_msg)

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
    prompt_text = get_generation_prompt(description, language, file_content)
    
    try:
        config_params = {
            "system_instruction": SYSTEM_PROMPT_CODE_GEN
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
        print(error_msg)
        raise Exception(error_msg)

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
    prompt_text = get_generation_prompt(description, language, file_content)
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CODE_GEN},
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
        print(error_msg)
        raise Exception(error_msg)

# ===== Audit request functions (code analysis) =====

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
    prompt = get_audit_prompt(prompt_text, code_text)
    
    try:
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_CODE_AUDIT},
                {"role": "user", "content": prompt}
            ]
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.responses.create(**kwargs)
        
        # Извлекаем текст из структуры ответа
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
        print(error_msg)
        return error_msg

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
    prompt = get_audit_prompt(prompt_text, code_text)
    
    try:
        kwargs = {
            "model": model,
            "max_tokens": 20000,
            "system": SYSTEM_PROMPT_CODE_AUDIT,
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
        print(error_msg)
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
    prompt = get_audit_prompt(prompt_text, code_text)
    
    try:
        config_params = {
            "system_instruction": SYSTEM_PROMPT_CODE_AUDIT
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
        print(error_msg)
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
    prompt = get_audit_prompt(prompt_text, code_text)
    
    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CODE_AUDIT},
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
        print(error_msg)
        raise Exception(error_msg)

# ===== Correction request functions (fixing code based on audit) =====

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
    user_prompt = get_correction_prompt(initial_prompt, program_code, code_analysis)

    try:
        kwargs = {
            "model": model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT_CODE_CORRECTION},
                {"role": "user", "content": user_prompt}
            ]
        }
        
        # Add temperature parameter if the model supports it
        if temp_allowed:
            kwargs["temperature"] = 0
            
        response = client.responses.create(**kwargs)
        
        # Извлекаем текст из структуры ответа
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
        print(error_msg)
        return error_msg, "[]"

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
    user_prompt = get_correction_prompt(initial_prompt, program_code, code_analysis)

    try:
        kwargs = {
            "model": model,
            "max_tokens": 20000,
            "system": SYSTEM_PROMPT_CODE_CORRECTION,
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
        print(error_msg)
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
    user_prompt = get_correction_prompt(initial_prompt, program_code, code_analysis)

    try:
        config_params = {
            "system_instruction": SYSTEM_PROMPT_CODE_CORRECTION
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
        print(error_msg)
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
    user_prompt = get_correction_prompt(initial_prompt, program_code, code_analysis)

    try:
        kwargs = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_CODE_CORRECTION},
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
        print(error_msg)
        raise Exception(error_msg)

# ===== Generic request dispatcher functions =====

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

def main():
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
        from tkinter import filedialog
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

    def extension_for_language(lang):
        """Returns the appropriate file extension for a given programming language"""
        mapping = {
            "Python": ".py",
            "Java": ".java",
            "JavaScript": ".js",
            "C": ".c",
            "C++": ".cpp",
            "Pascal": ".pas",
            "Julia": ".jl",
            "FORTRAN": ".f90"
        }
        return mapping.get(lang, ".txt")

    # --- parse_bug_counts ---
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

    # --- parse_corrections --- (FIXED)
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
            print(f"Error parsing corrections list: {e}")
            return (0, 0, 0, 0)
        except Exception as e:
            print(f"Unexpected error parsing corrections: {e}")
            return (0, 0, 0, 0)

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
        
    # Function to remove surrounding quotes from code
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
        return code_text
        
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
        result_queue = queue.Queue()
        
        # Update status
        root.after(0, ui_set_status, f"Status: Generating Code with All Selected Models Simultaneously")
        
        # Use ThreadPoolExecutor to run generation tasks concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(10, len(selected_models))) as executor:
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
                    pending = list(futures)
                    completed = 0
                    total = len(pending)
                    
                    while pending and not stop_flag:
                        # Check for completed futures
                        done, pending = concurrent.futures.wait(
                            pending, 
                            timeout=0.5,
                            return_when=concurrent.futures.FIRST_COMPLETED
                        )
                        
                        # Update completed count
                        completed += len(done)
                        
                        # Update progress status
                        progress_msg = f"Status: Generated {completed}/{total} models..."
                        root.after(0, ui_set_status, progress_msg)
                        
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
        is_valid, error_msg = validate_filename(user_filename)
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
    main()