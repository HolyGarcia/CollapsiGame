import tkinter as tk
import random
import tkinter.font as tkFont
import copy
import time

# -------------------- CLASES --------------------

class Player:
    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.position = [0,0]
        self.first_turn = True
        self.ai_type = None
        self.steps_taken = 0  # Contador de pasos

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

    def reset_used(self):
        self.used = [[False]*self.grid_size for _ in range(self.grid_size)]

    def get_value(self, pos):
        return self.board[pos[0]][pos[1]]

    def mark_used(self, pos):
        self.used[pos[0]][pos[1]] = True

# -------------------- MÉTODOS --------------------

def place_players_on_zeros(board, players):
    zeros = []
    for i in range(board.grid_size):
        for j in range(board.grid_size):
            if board.board[i][j] == 0:
                zeros.append([i,j])
    players[0].position = zeros[0]
    players[1].position = zeros[1]

def get_valid_moves(board, pos, steps, first_turn, other_pos):
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
        is_final = (remaining ==0 or first_turn)
        if is_final and [r, c] != pos and [r, c] != other_pos:
            moves.append([r,c])
        if remaining > 0:
            for dr,dc in directions:
                dfs(r+dr,c+dc,remaining-1)
        visited.remove((r,c))

    dfs(pos[0], pos[1], steps)
    unique_moves = []
    for m in moves:
        if m not in unique_moves:
            unique_moves.append(m)
    return unique_moves

# -------------------- LÓGICA DEL JUEGO --------------------

class Game:
    def __init__(self, root):
        self.root = root
        self.grid_size = 4
        self.cell_size = 110
        self.values = [0,0,4,4,1,1,1,1,2,2,2,2,3,3,3,3]

        # Configuración heurística y pesos
        self.heuristic_type = 4
        self.weights = (3, -4, 2)
        self.max_depth = 3
        self.nodes_expanded = 0
        self.max_depth_reached = 0
        self.start_time = None

        # Jugadores
        self.players = [Player("", "red"), Player("", "green")]
        self.current_player_idx = 0
        self.game_over = False
        self.ai_players = [False, False]

        # Tablero
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)

        # Canvas
        self.canvas = tk.Canvas(root, width=self.cell_size*self.grid_size,
                                height=self.cell_size*self.grid_size)
        self.canvas.pack()

        # Fuentes y etiquetas
        self.normal_font = tkFont.Font(size=12)
        self.bold_font = tkFont.Font(size=12, weight="bold")
        self.labels = [tk.Label(root, text="", font=("Arial",12)),
                       tk.Label(root, text="", font=("Arial",12))]
        self.labels[0].pack()
        self.labels[1].pack()

        # Botones de modo
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=5)
        tk.Button(self.btn_frame, text="Humano", command=lambda: self.start_game_type("human")).pack(side="left", padx=5)
        tk.Button(self.btn_frame, text="Random", command=lambda: self.start_game_type("random")).pack(side="left", padx=5)
        tk.Button(self.btn_frame, text="Greedy", command=lambda: self.start_game_type("greedy")).pack(side="left", padx=5)
        tk.Button(self.btn_frame, text="Worst", command=lambda: self.start_game_type("worst")).pack(side="left", padx=5)
        tk.Button(self.btn_frame, text="IA vs IA", command=lambda: self.start_game_type("ai")).pack(side="left", padx=5)

        self.draw_board()
        self.update_labels()

    # -------------------- LABELS / BOARD --------------------
    def update_labels(self):
        for idx, player in enumerate(self.players):
            turn_text = " ← Turno" if idx == self.current_player_idx else ""
            turn_font = self.bold_font if idx == self.current_player_idx else self.normal_font
            self.labels[idx].config(
                text=f"{player.name} en carta: {self.board.get_value(player.position)}{turn_text}",
                font=turn_font
            )

    def draw_board(self):
        self.canvas.delete("all")
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                x = j*self.cell_size
                y = i*self.cell_size
                color = "lightgray" if self.board.used[i][j] else "lightblue"
                self.canvas.create_rectangle(x, y, x+self.cell_size, y+self.cell_size, fill=color)
                if not self.board.used[i][j] and [i,j] not in [p.position for p in self.players]:
                    self.canvas.create_text(x+self.cell_size//2, y+self.cell_size//2,
                                            text=str(self.board.board[i][j]), font=("Arial",20))
        for player in self.players:
            x = player.position[1]*self.cell_size + self.cell_size//2
            y = player.position[0]*self.cell_size + self.cell_size//2
            self.canvas.create_oval(x-15, y-15, x+15, y+15, fill=player.color)

    # -------------------- START GAME TYPE --------------------
    def start_game_type(self, game_type):
        self.new_game()
        self.game_over = False

        # Configurar nombres y tipos de jugador según modo
        if game_type == "human":
            self.ai_players = [True, False]
            self.players[0].ai_type = "minimax"
            self.players[0].name = "IA"
            self.players[1].ai_type = None
            self.players[1].name = "Humano"
        elif game_type == "random":
            self.ai_players = [True, True]
            self.players[0].ai_type = "random"
            self.players[0].name = "Random"
            self.players[1].ai_type = "minimax"
            self.players[1].name = "IA"
        elif game_type == "greedy":
            self.ai_players = [True, True]
            self.players[0].ai_type = "greedy"
            self.players[0].name = "Greedy"
            self.players[1].ai_type = "minimax"
            self.players[1].name = "IA"
        elif game_type == "worst":
            self.ai_players = [True, True]
            self.players[0].ai_type = "worst"
            self.players[0].name = "Worst"
            self.players[1].ai_type = "minimax"
            self.players[1].name = "IA"
        elif game_type == "ai":
            self.ai_players = [True, True]
            self.players[0].ai_type = "minimax"
            self.players[0].name = "IA1"
            self.players[1].ai_type = "minimax"
            self.players[1].name = "IA2"

        if not any(self.ai_players[i] for i in range(2) if self.players[i].ai_type is None):
            self.canvas.bind("<Button-1>", self.select_tile)
        else:
            self.canvas.unbind("<Button-1>")

        self.update_labels()

        current = self.players[self.current_player_idx]
        other = self.players[1 - self.current_player_idx]
        steps = 4 if current.first_turn else self.board.get_value(current.position)
        moves = get_valid_moves(self.board,
                                current.position,
                                steps,
                                current.first_turn,
                                other.position)
        if self.ai_players[self.current_player_idx] and moves:
            self.root.after(500, self.ai_move)

        elif not moves:
            winner = other
            self.end_game_summary(winner)

    # -------------------- MOVIMIENTOS IA --------------------
    def ai_move(self):
        if self.game_over:
            return

        current_ai = self.players[self.current_player_idx]
        other = self.players[1 - self.current_player_idx]

        self.nodes_expanded = 0
        self.max_depth_reached = 0
        self.start_time = time.time()

        steps = 4 if current_ai.first_turn else self.board.get_value(current_ai.position)
        valid_moves = get_valid_moves(self.board, current_ai.position, steps, current_ai.first_turn, other.position)
        move = None

        if not valid_moves:
            move = None
        else:
            if current_ai.ai_type == "random":
                move = random.choice(valid_moves)
            elif current_ai.ai_type == "greedy":
                move = max(valid_moves, key=lambda m: self.board.get_value(m))
            elif current_ai.ai_type == "worst":
                move = self.worst_move(current_ai, other)
            else:
                move = self.get_best_move_ai(current_ai, other, depth=self.max_depth)

        if move is None:
            winner = other
            self.end_game_summary(winner)
            return

        self.board.mark_used(current_ai.position)
        current_ai.position = move
        current_ai.first_turn = False
        current_ai.steps_taken += 1

        self.draw_board()
        self.update_labels()

        if self.check_win():
            return

        self.current_player_idx = 1 - self.current_player_idx
        self.update_labels()

        if self.ai_players[self.current_player_idx]:
            self.root.after(800, self.ai_move)

    # -------------------- JUGADOR HUMANO --------------------
    def select_tile(self, event):
        if self.game_over:
            return
        x_click = event.x // self.cell_size
        y_click = event.y // self.cell_size
        if self.board.used[y_click][x_click]:
            return
        player = self.players[self.current_player_idx]
        other_player = self.players[1 - self.current_player_idx]
        prev_value = self.board.get_value(player.position)
        steps_allowed = 4 if player.first_turn else prev_value
        valid_moves = get_valid_moves(self.board, player.position, steps_allowed, player.first_turn, other_player.position)
        if [y_click, x_click] not in valid_moves:
            return
        self.board.mark_used(player.position)
        player.position = [y_click, x_click]
        player.first_turn = False
        player.steps_taken += 1

        self.draw_board()
        self.update_labels()

        self.current_player_idx = 1 - self.current_player_idx
        self.update_labels()
        if self.check_win():
            return

        if self.ai_players[self.current_player_idx]:
            self.root.after(500, self.ai_move)

    # -------------------- CHECK WINNER --------------------
    def check_win(self):
        current = self.players[self.current_player_idx]
        other = self.players[1 - self.current_player_idx]
        steps = 4 if current.first_turn else self.board.get_value(current.position)
        moves = get_valid_moves(self.board, current.position, steps, current.first_turn, other.position)
        if not moves:
            winner = other
            self.end_game_summary(winner)
            return True
        return False

    # -------------------- END GAME SUMMARY --------------------
    def end_game_summary(self, winner):
        self.game_over = True
        self.canvas.unbind("<Button-1>")
        print(f"\n===== PARTIDA TERMINADA =====")
        print(f"Ganador: {winner.name}")
        print(f"Cantidad de pasos: {winner.steps_taken}")
        print(f"Nodos expandidos: {self.nodes_expanded}")
        print(f"Tiempo de ejecución: {time.time() - self.start_time:.3f} s")
        print(f"Pesos usados: {self.weights}")
        print(f"Profundidad máxima alcanzada: {self.max_depth_reached}")
        self.labels[0].config(text=f"{winner.name} Ganó!")
        self.labels[1].config(text=f"{winner.name} Ganó!")

    # -------------------- NEW GAME --------------------
    def new_game(self):
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)
        for p in self.players:
            p.first_turn = True
            p.steps_taken = 0
            p.ai_type = None
            p.name = ""
        self.current_player_idx = 0
        self.game_over = False
        self.draw_board()
        self.update_labels()

    # -------------------- MINIMAX --------------------
    def evaluate(self, board, ai_player, other_player):
        ai_steps = 4 if ai_player.first_turn else board.get_value(ai_player.position)
        other_steps = 4 if other_player.first_turn else board.get_value(other_player.position)
        ai_moves = get_valid_moves(board, ai_player.position, ai_steps, ai_player.first_turn, other_player.position)
        other_moves = get_valid_moves(board, other_player.position, other_steps, other_player.first_turn, ai_player.position)
        return len(ai_moves) - len(other_moves)

    def minimax(self, board, ai_player, other_player, depth, maximizing):
        self.nodes_expanded += 1
        self.max_depth_reached = max(self.max_depth_reached, self.max_depth - depth + 1)
        if depth == 0:
            return self.evaluate(board, ai_player, other_player)
        steps = 4 if ai_player.first_turn else board.get_value(ai_player.position)
        moves = get_valid_moves(board, ai_player.position, steps, ai_player.first_turn, other_player.position)
        if not moves:
            return self.evaluate(board, ai_player, other_player)
        if maximizing:
            max_eval = -float("inf")
            for move in moves:
                new_board = copy.deepcopy(board)
                new_ai = copy.deepcopy(ai_player)
                new_other = copy.deepcopy(other_player)
                new_board.mark_used(new_ai.position)
                new_ai.position = move
                new_ai.first_turn = False
                eval_score = self.minimax(new_board, new_ai, new_other, depth-1, False)
                max_eval = max(max_eval, eval_score)
            return max_eval
        else:
            min_eval = float("inf")
            for move in moves:
                new_board = copy.deepcopy(board)
                new_ai = copy.deepcopy(ai_player)
                new_other = copy.deepcopy(other_player)
                new_board.mark_used(new_other.position)
                new_other.position = move
                new_other.first_turn = False
                eval_score = self.minimax(new_board, new_ai, new_other, depth-1, True)
                min_eval = min(min_eval, eval_score)
            return min_eval

    def get_best_move_ai(self, ai_player, other_ai_player, depth=2):
        best_score = -float("inf")
        best_move = None
        steps = 4 if ai_player.first_turn else self.board.get_value(ai_player.position)
        moves = get_valid_moves(self.board, ai_player.position, steps, ai_player.first_turn, other_ai_player.position)
        for move in moves:
            new_board = copy.deepcopy(self.board)
            new_ai = copy.deepcopy(ai_player)
            new_other = copy.deepcopy(other_ai_player)
            new_board.mark_used(new_ai.position)
            new_ai.position = move
            new_ai.first_turn = False
            score = self.minimax(new_board, new_ai, new_other, depth-1, False)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    # ------------------------- PEOR JUGADOR ----------------------------------
    def worst_move(self, player, opponent):
        steps = 4 if player.first_turn else self.board.get_value(player.position)
        moves = get_valid_moves(self.board, player.position, steps, player.first_turn, opponent.position)
        if not moves:
            return None

        worst_score = -float("inf")  # buscamos maximizar el beneficio del oponente
        worst_move = None
        for move in moves:
            temp_board = copy.deepcopy(self.board)
            temp_player = copy.deepcopy(player)
            temp_opponent = copy.deepcopy(opponent)

            temp_board.mark_used(temp_player.position)
            temp_player.position = move
            temp_player.first_turn = False

            score = self.evaluate(temp_board, temp_opponent, temp_player)  # invertir roles
            if score > worst_score:
                worst_score = score
                worst_move = move

        return worst_move

# -------------------- MAIN --------------------
if __name__ == "__main__":
    root = tk.Tk()
    root.title("Collapsi Game")
    game = Game(root)
    root.mainloop()
