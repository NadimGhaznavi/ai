"""
snake/AISim.py

    Database 4 Everything
    Author: Nadim-Daniel Ghaznavi
    Copyright: (c) 2024-2025 Nadim-Daniel Ghaznavi
    GitHub: https://github.com/NadimGhaznavi/ai
    License: GPL 3.0
"""

import threading

from textual.app import App, ComposeResult
from textual.widgets import Label
from textual.geometry import Offset

from GameBoard import GameBoard
from SnakeGame import SnakeGame
from AIAgent import AIAgent


class AISim(App):
    """A Textual app that has an AI Agent playing the Snake Game."""

    def __init__(self) -> None:
        super().__init__()
        self.game_board = GameBoard(20)
        self.snake_game = SnakeGame(game_board=self.game_board)
        self.agent = AIAgent()
        self.score_widget = Label("Highscore: 0, Score: 0")
        self.score = 0
        self.highscore = 0
        self.epoch = 1

        self.score = Label("Game: 0, Highscore: 0, Score: 0")

        # Launch the simulator in a background thread
        self.stop_event = threading.Event()
        self.simulator_thread = threading.Thread(target=self.start_sim)
        self.simulator_thread.start()

    async def action_quit(self) -> None:
        """Quit the application."""
        self.stop_event.set()
        if self.simulator_thread.is_alive():
            self.simulator_thread.join(timeout=2)
        await super().action_quit()

    def compose(self) -> ComposeResult:
        """Create child widgets for the app."""
        yield Label("AI Snake Game Simulator")
        yield self.score_widget
        yield self.game_board

    def on_quit(self):
        print("Quiting....")
        self.simulator_thread.join()
        super().on_quit()

    def start_sim(self):
        self.snake_game.reset()
        game_board = self.game_board
        agent = self.agent
        snake_game = self.snake_game
        score = self.score
        highscore = self.highscore

        while not self.stop_event.is_set():
            # The actual training loop...
            old_state = game_board.get_state()
            move = agent.get_move(old_state)
            reward, game_over, score = snake_game.play_step(move)
            if score > highscore:
                highscore = score
            self.score_widget.update(
                f"Game: {self.epoch}, Highscore: {highscore}, Score: {score}"
            )
            if not game_over:
                ## Keep playing
                new_state = game_board.get_state()
                agent.train_short_memory(old_state, move, reward, new_state, game_over)
                agent.remember(old_state, move, reward, new_state, game_over)
            else:
                ## Game over
                self.epoch += 1
                # Remember the last move
                agent.remember(old_state, move, reward, new_state, game_over)
                # Train long memory
                agent.train_long_memory()
                # Reset the game
                snake_game.reset()
                # Let the agent know we've finished a game
                agent.played_game(score)


if __name__ == "__main__":
    app = AISim()
    app.run()
