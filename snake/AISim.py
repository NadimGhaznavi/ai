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
from textual.scroll_view import ScrollView
from textual.geometry import Offset, Region, Size
from textual.strip import Strip
from textual.reactive import var

from rich.segment import Segment
from rich.style import Style

class Board(ScrollView):
    COMPONENT_CLASSES = {
        "board--emtpy-square",
        "board--food-square",
        "board--snake-square",
        "board--snake-head-square",
    }

    DEFAULT_CSS = """
    Board > .board--empty-square {
        background: #aaaaaa;
    }
    Board > .board--food-square {
        background: #ff0000;
    }
    Board > .board--snake-square {
        background: #00aa00;
    }
    Board > .board--snake-head-square {
        background: #00ff00;
    }
    """

    cursor_square = var(Offset(0, 0))

    def __init__(self, board_size: int) -> None:
        super().__init__()
        self.board_size = board_size
        self.virtual_size = Size(board_size * 8, board_size * 4)

    
    def render_line(self, y: int) -> Strip:
        scroll_x, scroll_y = self.scroll_offset
        y += scroll_y
        row_index = y // 4

        empty = self.get_component_rich_style("board--empty-square")
        food = self.get_component_rich_style("board--food-square")
        snake = self.get_component_rich_style("board--snake-square")
        snake_head = self.get_component_rich_style("board--snake-head-square")

        def get_square_style(column: int, row: int) -> Style:
            square_style = empty
            return square_style
        
        segments = [
            Segment(" " * 8, get_square_style(column, row_index))
            for column in range(self.board_size)
        ]
        strip = Strip(segments, self.board_size * 8)
        # Crop the strip so that is covers the visible area
        strip = strip.crop(scroll_x, scroll_x + self.size.width)
        return strip


class AISim(App):
    """A Textual app that has an AI Agent playing the Snake Game."""

    BINDINGS = [("d", "toggle_dark", "Toggle dark mode")]

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Label("Hello AI Sim")


if __name__ == "__main__":
    app = AISim()
    app.run()