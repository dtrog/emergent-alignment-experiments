#!/usr/bin/env python3
"""Test script to verify icon functionality."""

import sys
import os
sys.path.append('.')

from gui.gui_monitor import IconManager

def test_icon_mapping():
    """Test the icon mapping functionality."""
    icon_manager = IconManager()
    
    test_names = [
        "OpenAI GPT-4 Full Memory",
        "Anthropic Claude 3.5 Sonnet", 
        "DeepSeek V3",
        "DeepSeek",
        "Claude",
        "GPT-4",
        "Gemini",
        "Social Engineer",
        "Authority Figure"
    ]
    
    print("Testing icon mappings:")
    print("="*50)
    
    for name in test_names:
        emoji = icon_manager.get_provider_emoji(name)
        print(f"{name:30} -> {emoji}")
    
    print("="*50)

if __name__ == "__main__":
    test_icon_mapping()
