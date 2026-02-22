import time
import math
import copy
import random
import tkinter as tk

from main import Game, get_valid_moves

# -------------------- ESTADO LÓGICO --------------------

class GameState:
    def __init__(self, board, players, current_idx):
        self.board = copy.deepcopy(board)
        self.players = copy.deepcopy(players)
        self.current_player_idx = current_idx

    def clone(self):
        return copy.deepcopy(self)

    def get_current_player(self):
        return self.players[self.current_player_idx]

    def get_valid_moves(self):
        player = self.get_current_player()
        prev_value = self.board.get_value(player.position)
        steps = 4 if player.first_turn else prev_value
        return get_valid_moves(self.board, player.position, steps, player.first_turn)

    def apply_move(self, move):
        player = self.get_current_player()
        self.board.mark_used(player.position)
        player.position = move
        player.first_turn = False
        self.current_player_idx = 1 - self.current_player_idx

    def is_terminal(self):
        return len(self.get_valid_moves()) == 0

    def evaluate(self):
        if self.is_terminal():
            if self.current_player_idx == 1:
                return -1000
            else:
                return 1000

        my_moves = len(self.get_valid_moves())

        opponent_state = self.clone()
        opponent_state.current_player_idx = 1 - self.current_player_idx
        opponent_moves = len(opponent_state.get_valid_moves())

        mobility = my_moves - opponent_moves

        center = self.board.grid_size // 2
        r, c = self.players[1].position
        centrality = -(abs(r-center) + abs(c-center))

        pressure = -opponent_moves

        corner_penalty = 0
        if r in [0, self.board.grid_size-1] and c in [0, self.board.grid_size-1]:
            corner_penalty = -3

        value_diff = self.board.get_value(self.players[1].position) - \
                     self.board.get_value(self.players[0].position)

        return 3*mobility + 2*pressure + centrality + value_diff + corner_penalty


# -------------------- MINIMAX --------------------

class MinimaxAI:
    def __init__(self, max_time=3):
        self.max_time = max_time
        self.start_time = None
        self.nodes_expanded = 0

    def get_best_move(self, state):
        self.start_time = time.time()
        depth = 1
        best_move = None

        while time.time() - self.start_time < self.max_time:
            try:
                move = self.minimax_root(state, depth)
                if move:
                    best_move = move
                depth += 1
            except:
                break

        return best_move

    def minimax_root(self, state, depth):
        best_value = -math.inf
        best_move = None

        for move in state.get_valid_moves():
            child = state.clone()
            child.apply_move(move)
            value = self.minimax(child, depth-1, -math.inf, math.inf, False)

            if value > best_value:
                best_value = value
                best_move = move

        return best_move

    def minimax(self, state, depth, alpha, beta, maximizing):

        if time.time() - self.start_time > self.max_time:
            raise Exception("Timeout")

        self.nodes_expanded += 1

        if depth == 0 or state.is_terminal():
            return state.evaluate()

        if maximizing:
            value = -math.inf
            for move in state.get_valid_moves():
                child = state.clone()
                child.apply_move(move)
                value = max(value, self.minimax(child, depth-1, alpha, beta, False))
                alpha = max(alpha, value)
                if beta <= alpha:
                    break
            return value
        else:
            value = math.inf
            for move in state.get_valid_moves():
                child = state.clone()
                child.apply_move(move)
                value = min(value, self.minimax(child, depth-1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break
            return value


# -------------------- EXTENDER GAME ORIGINAL --------------------

class GameWithAI(Game):
    def __init__(self, root):
        super().__init__(root)
        self.ai = MinimaxAI(max_time=3)
        # Variable para saber si está activo modo IA
        self.vs_ai = False

        # Botón para cambiar modo de juego
        self.ai_mode_btn = tk.Button(root, text="Modo vs IA", command=self.toggle_ai_mode)
        self.ai_mode_btn.pack(pady=5)

    def toggle_ai_mode(self):
        self.vs_ai = not self.vs_ai

        if self.vs_ai:
            self.ai_mode_btn.config(text="Modo vs Humano")
        else:
            self.ai_mode_btn.config(text="Modo vs IA")

    def select_tile(self, event):
        super().select_tile(event)

        if self.vs_ai and self.current_player_idx == 1:
            self.root.after(200, self.ai_move)

    def ai_move(self):
        state = GameState(self.board, self.players, self.current_player_idx)
        move = self.ai.get_best_move(state)

        if move is None:
            # IA no puede moverse == gana humano
            winner = self.players[0]
            self.labels[0].config(text=f"Jugador {winner.name} ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} ganó!")
            self.canvas.unbind("<Button-1>")
            return

        player = self.players[self.current_player_idx]

        # Mover IA
        self.board.mark_used(player.position)
        player.position = move
        player.first_turn = False

        self.draw_board()
        self.update_labels()

        # Verificar si el humano puede moverse
        next_player = self.players[0]
        prev_value = self.board.get_value(next_player.position)
        steps = 4 if next_player.first_turn else prev_value
        valid_moves = get_valid_moves(self.board, next_player.position, steps, next_player.first_turn)

        if not valid_moves:
            winner = self.players[1]
            self.labels[0].config(text=f"Jugador {winner.name} ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} ganó!")
            self.canvas.unbind("<Button-1>")
            return

        # Cambiar turno al humano
        self.current_player_idx = 0
        self.update_labels()


# -------------------- EJECUTAR --------------------

root = tk.Tk()
root.title("Collapsi Game IA")
game = GameWithAI(root)
root.mainloop()