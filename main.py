import tkinter as tk
import random
import copy
import time
import numpy as np

# -------------------- CLASES --------------------

class Player:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.position = [0,0]
        self.first_turn = True

class Board:
    def __init__(self, grid_size, values, cell_size):
        self.grid_size = grid_size
        self.values = values
        self.cell_size = cell_size
        self.used = [[False]*grid_size for _ in range(grid_size)]
        self.board = self.create_board()

    def create_board(self):
        shuffled = self.values.copy()
        random.shuffle(shuffled)
        b = []
        for i in range(self.grid_size):
            row = []
            for j in range(self.grid_size):
                row.append(shuffled.pop())
            b.append(row)
        return b

        def get_value(self, pos):
            return self.board[pos[0]][pos[1]]

        def mark_used(self, pos):
            self.used[pos[0]][pos[1]] = True


# -------------------- FUNCIONES AUXILIARES --------------------

def place_players_on_zeros(board, players):
    zeros = []
    for i in range(board.grid_size):
        for j in range(board.grid_size):
            if board.board[i][j] == 0:
                zeros.append([i,j])
    players[0].position = zeros[0]
    players[1].position = zeros[1]

def get_valid_moves(board, pos, steps, first_turn):
    moves = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    visited = set()

    def dfs(r,c,remaining):
        if r<0 or r>=board.grid_size or c<0 or c>=board.grid_size:
            return
        if board.used[r][c]:
            return
        if (r,c) in visited:
            return

        visited.add((r,c))

        if first_turn:
            if remaining >= 0 and [r,c] != pos:
                moves.append([r,c])
        else:
            if remaining == 0 and [r,c] != pos:
                moves.append([r,c])

        if remaining > 0:
            for dr,dc in directions:
                dfs(r+dr,c+dc,remaining-1)

        visited.remove((r,c))

    dfs(pos[0], pos[1], steps)

    unique = []
    for m in moves:
        if m not in unique:
            unique.append(m)

    return unique


# -------------------- GAME STATE --------------------

class GameState:
    def __init__(self, game):
        self.grid_size = game.grid_size
        self.board = copy.deepcopy(game.board.board)
        self.used = copy.deepcopy(game.board.used)

        self.positions = {
            0: game.players[0].position.copy(),
            1: game.players[1].position.copy()
        }

        self.first_turns = {
            0: game.players[0].first_turn,
            1: game.players[1].first_turn
        }

        self.current_player = game.current_player_idx

    def get_valid_moves(self, player):
        pos = self.positions[player]
        first_turn = self.first_turns[player]
        steps = 4 if first_turn else self.board[pos[0]][pos[1]]
        return get_valid_moves_dummy(self, pos, steps, first_turn)

    def children(self):
        options = self.get_valid_moves(self.current_player)
        children = []

        for move in options:
            child = copy.deepcopy(self)

            old_pos = child.positions[self.current_player]
            child.used[old_pos[0]][old_pos[1]] = True

            child.positions[self.current_player] = list(move)
            child.first_turns[self.current_player] = False
            child.current_player = 1 - self.current_player

            children.append((move, child))

        return children

    def is_terminal(self):
        return len(self.get_valid_moves(self.current_player)) == 0

    # --- Heurísticas ---
    def heuristic_mobility(self):
        return (len(self.get_valid_moves(self.current_player)) -
                len(self.get_valid_moves(1 - self.current_player)))

    def heuristic_center(self):
        center = self.grid_size // 2
        r, c = self.positions[self.current_player]
        return -(abs(r-center) + abs(c-center))

    def heuristic_block(self):
        return -len(self.get_valid_moves(1 - self.current_player))

    def heuristic_tiles(self):
        return sum(row.count(False) for row in self.used)

    def combined_heuristic(self, weights):
        return (weights[0]*self.heuristic_mobility() +
                weights[1]*self.heuristic_center() +
                weights[2]*self.heuristic_block() +
                weights[3]*self.heuristic_tiles())


def get_valid_moves_dummy(state, pos, steps, first_turn):
    dummy_board = type("Dummy", (), {})()
    dummy_board.grid_size = state.grid_size
    dummy_board.used = state.used
    return get_valid_moves(dummy_board, pos, steps, first_turn)


# -------------------- MINIMAX --------------------

class MinimaxSolver:
    def __init__(self, player, weights, time_limit):
        self.player = player
        self.weights = weights
        self.time_limit = time_limit

    def solve(self, state):
        start = time.time()
        best_move = None
        depth = 1

        while time.time() - start < self.time_limit:
            move, _ = self.maximize(state, -np.inf, np.inf, depth)
            if time.time() - start < self.time_limit:
                best_move = move
            depth += 1

        return best_move

    def maximize(self, state, alpha, beta, depth):
        if depth == 0 or state.is_terminal():
            return None, state.combined_heuristic(self.weights)

        max_value = -np.inf
        best_move = None

        for move, child in state.children():
            _, value = self.minimize(child, alpha, beta, depth-1)

            if value > max_value:
                max_value = value
                best_move = move

            if max_value >= beta:
                break

            alpha = max(alpha, max_value)

        return best_move, max_value

    def minimize(self, state, alpha, beta, depth):
        if depth == 0 or state.is_terminal():
            return None, state.combined_heuristic(self.weights)

        min_value = np.inf
        best_move = None

        for move, child in state.children():
            _, value = self.maximize(child, alpha, beta, depth-1)

            if value < min_value:
                min_value = value
                best_move = move

            if min_value <= alpha:
                break

            beta = min(beta, min_value)

        return best_move, min_value


# -------------------- WORST PLAYER --------------------

def worst_player(state, weights):
    worst = float('inf')
    worst_move = None

    for move, child in state.children():
        score = child.combined_heuristic(weights)
        if score < worst:
            worst = score
            worst_move = move

    return worst_move


# -------------------- GAME --------------------

class Game:
    def __init__(self, root, player_types):

        self.root = root
        self.player_types = player_types

        self.grid_size = 4
        self.cell_size = 110
        self.values = [0,0,4,4,1,1,1,1,2,2,2,2,3,3,3,3]

        self.weights = [2.0, 0.3, 1.5, 0.1]
        self.time_limit = 2

        self.players = [Player("Rojo","red"), Player("Verde","green")]
        self.current_player_idx = 0

        self.game_over = False

        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)

            self.canvas = tk.Canvas(root, width=self.cell_size*self.grid_size,
                                    height=self.cell_size*self.grid_size)
            self.canvas.pack()

            self.labels = [tk.Label(root,font=("Arial",12)),
                        tk.Label(root,font=("Arial",12))]
            self.labels[0].pack()
            self.labels[1].pack()

            self.draw_board()
            self.update_labels()

        self.canvas.bind("<Button-1>", self.select_tile)

        if self.player_types[self.current_player_idx] != "human":
            self.root.after(500, self.play_ai)

    def select_tile(self, event):
        if self.player_types[self.current_player_idx] != "human":
            return

        col = event.x // self.cell_size
        row = event.y // self.cell_size
        self.make_move(row,col)

    def make_move(self, row, col):
        
        if self.game_over:
            return
        
        player = self.players[self.current_player_idx]

        steps = 4 if player.first_turn else self.board.get_value(player.position)
        valid = get_valid_moves(self.board, player.position, steps, player.first_turn)

        if [row,col] not in valid:
            return

        # Evitar superposición con el otro jugador
        for idx, p in enumerate(self.players):
            if idx != self.current_player_idx and p.position == [row, col]:
                return

        self.board.mark_used(player.position)
        player.position = [row,col]
            player.first_turn = False

            next_idx = 1 - self.current_player_idx
            next_player = self.players[next_idx]

        next_steps = 4 if next_player.first_turn else self.board.get_value(next_player.position)
        if not get_valid_moves(self.board, next_player.position, next_steps, next_player.first_turn):

            winner = player

            print(f"Ganador: {winner.name}")

            self.game_over = True
            self.canvas.unbind("<Button-1>")

            self.labels[0].config(text=f"Ganador: {winner.name}")
            self.labels[1].config(text="")

            return

            self.current_player_idx = next_idx
            self.draw_board()
            self.update_labels()

        if self.player_types[self.current_player_idx] != "human":
            self.root.after(500, self.play_ai)

    def play_ai(self):

        state = GameState(self)
        kind = self.player_types[self.current_player_idx]

        if kind == "minimax":
            solver = MinimaxSolver(self.current_player_idx,
                                   self.weights,
                                   self.time_limit)
            move = solver.solve(state)

        elif kind == "worst":
            move = worst_player(state, self.weights)

        else:
            return

        if move:
            self.make_move(move[0], move[1])

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x=j*self.cell_size
                y=i*self.cell_size
                color="lightgray" if self.board.used[i][j] else "lightblue"
                self.canvas.create_rectangle(x,y,x+self.cell_size,y+self.cell_size,fill=color)
                if not self.board.used[i][j]:
                    self.canvas.create_text(x+55,y+55,text=str(self.board.board[i][j]))

        for p in self.players:
            x=p.position[1]*self.cell_size+55
            y=p.position[0]*self.cell_size+55
            self.canvas.create_oval(x-15,y-15,x+15,y+15,fill=p.color)

    def update_labels(self):
        for idx,p in enumerate(self.players):
            turn=" ← Turno" if idx==self.current_player_idx else ""
            self.labels[idx].config(text=f"{p.name}{turn}")


# -------------------- SELECCIÓN POR CONSOLA --------------------

def choose_player(player_id):
    while True:
        print(f"\nJugador {player_id}:")
        print("1 - Humano")
        print("2 - Minimax")
        print("3 - Worst")
        choice=input("Opción: ")

        if choice=="1": return "human"
        if choice=="2": return "minimax"
        if choice=="3": return "worst"


p0=choose_player(0)
p1=choose_player(1)

    root=tk.Tk()
    root.title("Collapsi Game")
    game=Game(root,{0:p0,1:p1})
    root.mainloop()