"""Theme definitions for MIRA TUI."""

from rich.console import Console

# Theme definitions with color palettes
# Colors use direct hex codes for Rich markup: [#ff00ff]text[/#ff00ff]
THEMES = {
    "default": {
        "name": "Default",
        "description": "Classic MIRA cyberpunk theme",
        "primary": "#ff00ff",  # Magenta
        "secondary": "#00ffff",  # Cyan
        "accent": "#ff0066",  # Pink
        "success": "#00ff00",  # Green
        "warning": "#ffff00",  # Yellow
        "error": "#ff3366",  # Red
        "dim": "#666666",  # Gray
        "banner": "#ff00ff",
    },
    "nord": {
        "name": "Nord",
        "description": "Arctic cold theme based on Nord palette",
        "primary": "#88c0d0",  # Nord Blue
        "secondary": "#81a1c1",  # Nord Blue 2
        "accent": "#5e81ac",  # Nord Blue 3
        "success": "#a3be8c",  # Nord Green
        "warning": "#ebcb8b",  # Nord Yellow
        "error": "#bf616a",  # Nord Red
        "dim": "#4c566a",  # Nord Gray
        "banner": "#88c0d0",
    },
    "dracula": {
        "name": "Dracula",
        "description": "Dark vampire theme",
        "primary": "#bd93f9",  # Dracula Purple
        "secondary": "#50fa7b",  # Dracula Green
        "accent": "#ff79c6",  # Dracula Pink
        "success": "#50fa7b",  # Green
        "warning": "#f1fa8c",  # Yellow
        "error": "#ff5555",  # Red
        "dim": "#6272a4",  # Dracula Dim
        "banner": "#bd93f9",
    },
    "gruvbox": {
        "name": "Gruvbox",
        "description": "Retro warmth theme",
        "primary": "#fabd2f",  # Gruvbox Yellow
        "secondary": "#83a598",  # Gruvbox Blue
        "accent": "#d3869b",  # Gruvbox Purple
        "success": "#b8bb26",  # Gruvbox Green
        "warning": "#fabd2f",  # Yellow
        "error": "#fb4934",  # Red
        "dim": "#928374",  # Gruvbox Gray
        "banner": "#fabd2f",
    },
    "monokai": {
        "name": "Monokai",
        "description": "Classic code editor theme",
        "primary": "#f92672",  # Monokai Pink
        "secondary": "#66d9e8",  # Monokai Blue
        "accent": "#a6e22e",  # Monokai Green
        "success": "#a6e22e",  # Green
        "warning": "#e6db74",  # Yellow
        "error": "#f92672",  # Red
        "dim": "#75715e",  # Monokai Gray
        "banner": "#f92672",
    },
    "solarized": {
        "name": "Solarized",
        "description": "Zen nature theme",
        "primary": "#268bd2",  # Solarized Blue
        "secondary": "#2aa198",  # Solarized Cyan
        "accent": "#859900",  # Solarized Green
        "success": "#859900",  # Green
        "warning": "#b58900",  # Yellow
        "error": "#dc322f",  # Red
        "dim": "#586e75",  # Solarized Gray
        "banner": "#268bd2",
    },
}

# Current theme
_current_theme = "default"


def get_theme(name: str = None) -> dict:
    """Get theme by name or current theme if None."""
    global _current_theme
    if name is None:
        name = _current_theme
    return THEMES.get(name, THEMES["default"])


def set_theme(name: str) -> bool:
    """Set current theme. Returns True if successful."""
    global _current_theme
    if name not in THEMES:
        return False
    _current_theme = name
    return True


def get_current_theme_name() -> str:
    """Get current theme name."""
    return _current_theme


def list_themes() -> list:
    """List all available themes."""
    return [{"name": name, **info} for name, info in THEMES.items()]


def theme_help_text() -> str:
    """Generate help text for /theme command."""
    theme = get_theme()
    lines = ["[bold]Available Themes:[/bold]"]
    for name, info in THEMES.items():
        marker = " *" if name == _current_theme else ""
        lines.append(f"  [{theme['primary']}]{name}[/{theme['primary']}] - {info['description']}{marker}")
    lines.append("")
    lines.append(f"[dim]Use /theme <name> to switch themes[/dim]")
    return "\n".join(lines)


def get_markup(color: str, text: str, closing: str = None) -> str:
    """Get Rich markup string with color.

    Args:
        color: Hex color code like "#ff00ff"
        text: Text to colorize
        closing: Optional explicit closing tag

    Returns:
        Rich markup string like "[#ff00ff]text[/#ff00ff]"
    """
    if closing is None:
        closing = f"[/{color}]"
    return f"[{color}]{text}{closing}"
