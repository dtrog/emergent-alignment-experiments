"""
LLM client for unified API access across different providers.

This module provides a unified interface for interacting with various LLM providers
including OpenAI, DeepSeek, Grok, Gemini, and Anthropic.
"""

import os
import requests
from typing import Dict, List, Any
try:
    from utils import retry_on_overload, get_safe_config_for_logging
except ImportError:
    # Fallback implementations if utils module is not available
    def retry_on_overload(func):
        """Simple fallback decorator that doesn't retry."""
        return func
    
    def get_safe_config_for_logging(config):
        """Simple fallback that returns a safe representation."""
        return str(config)[:100] + "..." if len(str(config)) > 100 else str(config)

import logging


class LLMClientService:
    """
    Universal LLM client service that supports multiple providers (OpenAI, Anthropic, Google, Grok, DeepSeek).
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize LLM client with provider configuration.
        
        Args:
            config: Configuration dictionary containing provider, model, and API details
        """
        self.config = config
        self.provider = config.get('provider', '').lower()
        self.model = config.get('model', '')
        self.api_key = self._get_api_key()
        self.base_url = config.get('base_url', '')
        
        # Provider-specific configurations
        self.headers = self._build_headers()
        self.api_endpoint = self._get_api_endpoint()
        
        logging.info(f"Initialized LLMClient for {self.provider} with model {self.model}")
    
    def _get_api_key(self) -> str:
        """Get API key from config or environment variables."""
        # Try config first
        if 'api_key' in self.config:
            return self.config['api_key']
        
        # Try environment variables based on provider
        env_key_map = {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY', 
            'google': 'GOOGLE_API_KEY',
            'gemini': 'GOOGLE_API_KEY',
            'grok': 'GROK_API_KEY',
            'deepseek': 'DEEPSEEK_API_KEY',
            'xai': 'XAI_API_KEY'
        }
        
        env_key = env_key_map.get(self.provider)
        if env_key and env_key in os.environ:
            return os.environ[env_key]
        
        raise ValueError(f"No API key found for provider {self.provider}")
    
    def _build_headers(self) -> Dict[str, str]:
        """Build HTTP headers for API requests."""
        headers = {'Content-Type': 'application/json'}
        
        if self.provider in ['openai', 'grok', 'xai']:
            headers['Authorization'] = f'Bearer {self.api_key}'
        elif self.provider == 'anthropic':
            headers['x-api-key'] = self.api_key
            headers['anthropic-version'] = '2023-06-01'
        elif self.provider in ['google', 'gemini']:
            # Google uses API key in URL parameters
            pass
        elif self.provider == 'deepseek':
            headers['Authorization'] = f'Bearer {self.api_key}'
        
        return headers
    
    def _get_api_endpoint(self) -> str:
        """Get the API endpoint URL for the provider."""
        if self.base_url:
            return self.base_url
        
        endpoint_map = {
            'openai': 'https://api.openai.com/v1/chat/completions',
            'anthropic': 'https://api.anthropic.com/v1/messages',
            'google': f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
            'gemini': f'https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent',
            'grok': 'https://api.x.ai/v1/chat/completions',
            'xai': 'https://api.x.ai/v1/chat/completions',
            'deepseek': 'https://api.deepseek.com/v1/chat/completions'
        }
        
        return endpoint_map.get(self.provider, '')
    
    def _build_request_payload(self, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Build the request payload for the specific provider."""
        if self.provider in ['openai', 'grok', 'xai', 'deepseek']:
            return {
                'model': self.model,
                'messages': messages,
                'temperature': kwargs.get('temperature', 0.7),
                'max_tokens': kwargs.get('max_tokens', 1000)
            }
        
        elif self.provider == 'anthropic':
            # Anthropic requires system message separate from other messages
            system_message = ""
            user_messages = []
            
            for msg in messages:
                if msg['role'] == 'system':
                    system_message = msg['content']
                else:
                    user_messages.append(msg)
            
            payload = {
                'model': self.model,
                'messages': user_messages,
                'max_tokens': kwargs.get('max_tokens', 1000),
                'temperature': kwargs.get('temperature', 0.7)
            }
            
            if system_message:
                payload['system'] = system_message
                
            return payload
        
        elif self.provider in ['google', 'gemini']:
            # Convert messages to Google's format
            parts = []
            for msg in messages:
                if msg['role'] != 'system':  # Google handles system messages differently
                    parts.append({'text': msg['content']})
            
            return {
                'contents': [{'parts': parts}],
                'generationConfig': {
                    'temperature': kwargs.get('temperature', 0.7),
                    'maxOutputTokens': kwargs.get('max_tokens', 1000)
                }
            }
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _extract_response_content(self, response_data: Dict[str, Any]) -> str:
        """Extract the content from the provider-specific response format."""
        try:
            if self.provider in ['openai', 'grok', 'xai', 'deepseek']:
                return response_data['choices'][0]['message']['content']
            
            elif self.provider == 'anthropic':
                return response_data['content'][0]['text']
            
            elif self.provider in ['google', 'gemini']:
                return response_data['candidates'][0]['content']['parts'][0]['text']
            
            else:
                raise ValueError(f"Unsupported provider: {self.provider}")
                
        except (KeyError, IndexError, TypeError) as e:
            raise ValueError(f"Failed to extract content from response: {e}")
    
    @retry_on_overload()
    def generate(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional parameters like temperature, max_tokens
            
        Returns:
            Generated response content
        """
        payload = self._build_request_payload(messages, **kwargs)
        
        # Add API key to URL for Google
        url = self.api_endpoint
        if self.provider in ['google', 'gemini']:
            url += f'?key={self.api_key}'
        
        try:
            logging.debug(f"Making request to {self.provider} with payload: {get_safe_config_for_logging(payload)}")
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            
            response.raise_for_status()
            response_data = response.json()
            
            content = self._extract_response_content(response_data)
            
            logging.debug(f"Received response from {self.provider}: {content[:100]}...")
            return content
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed for {self.provider}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error generating response from {self.provider}: {e}")
            raise
