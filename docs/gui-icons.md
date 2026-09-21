# GUI Monitor Icon Support

The GUI monitor now includes provider icons next to ARM names, judges, and manipulators for better visual identification.

## Icon Display

The system automatically displays appropriate icons/emojis for:

- **ARM providers**: OpenAI/GPT (🤖), Anthropic/Claude (🧠), DeepSeek (🔍), Gemini (💎), Grok (⚡)
- **Judge providers**: Same as ARM providers based on judge names
- **Manipulator types**: Social Engineer (🎭), Authority Figure (👔)

## Icon Sources

Icons are loaded from the `assets/icons/` directory using SVG files named `logo-<provider>.svg`:

- `logo-copilot.svg` - Used for OpenAI/GPT models
- `logo-claude.svg` - Used for Anthropic/Claude models  
- `logo-deepseek.svg` - Used for DeepSeek models
- `logo-gemini.svg` - Used for Google/Gemini models
- `logo-grok.svg` - Used for Grok models

## Enhanced Icon Support (Optional)

For better SVG icon rendering, install additional packages:

```bash
python install_icon_support.py
```

This installs:
- **Pillow (PIL)**: Advanced image processing
- **cairosvg**: SVG to PNG conversion

Without these packages, the system falls back to emoji-based icons, which still provide good visual distinction.

## Icon Mapping

The system intelligently maps provider names to icons:

- Partial matching (e.g., "OpenAI GPT-4" → OpenAI icon)
- Case-insensitive matching
- Fallback to generic icons for unknown providers

## Visual Features

Icons appear in:

1. **ARM Titles**: Next to experiment arm names in the left panel
2. **Committee Analysis**: Next to judge names and scores  
3. **Flash Messages**: In live updates referencing specific ARMs
4. **Manipulator Display**: Next to manipulator tactic names

The icons are sized appropriately (16-20px) to provide visual cues without cluttering the interface.
