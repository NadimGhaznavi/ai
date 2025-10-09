"""
snake/AISim.py

    Database 4 Everything
    Author: Nadim-Daniel Ghaznavi 
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai
    License: GPL 3.0

"""


from textual.app import App, ComposeResult
from textual.widgets import Label


class AISim(App):
    """A Textual app that has an AI Agent playing the Snake Game."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Label("Hello AI Sim")


if __name__ == "__main__":
    app = AISim()
    app.run()