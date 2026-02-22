import tkinter as tk
import random
import time
import math
import copy

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

    def reset_used(self):
        self.used = [[False]*self.grid_size for _ in range(self.grid_size)]

    def get_value(self, pos):
        return self.board[pos[0]][pos[1]]

    def mark_used(self, pos):
        self.used[pos[0]][pos[1]] = True


# -------------------- ESTADO PARA IA --------------------

class GameState:
    def __init__(self, board, players, current_player_idx):
        self.board = copy.deepcopy(board)
        self.players = copy.deepcopy(players)
        self.current_player_idx = current_player_idx

    def clone(self):
        return GameState(self.board, self.players, self.current_player_idx)

# -------------------- IA MINIMAX --------------------

NODES_EXPANDED = 0

def is_terminal(state):
    player = state.players[state.current_player_idx]
    steps = 4 if player.first_turn else state.board.get_value(player.position)
    moves = get_valid_moves(state.board, player.position, steps, player.first_turn)
    return len(moves) == 0

def get_actions(state):
    player = state.players[state.current_player_idx]
    steps = 4 if player.first_turn else state.board.get_value(player.position)
    return get_valid_moves(state.board, player.position, steps, player.first_turn)

def result(state, action):
    new_state = state.clone()
    player = new_state.players[new_state.current_player_idx]

    new_state.board.mark_used(player.position)
    player.position = action
    player.first_turn = False

    new_state.current_player_idx = 1 - new_state.current_player_idx
    return new_state

# -------------------- HEURISTICAS ----------------
def evaluate(state, maximizing_player_idx):
    player = state.players[maximizing_player_idx]
    opponent = state.players[1 - maximizing_player_idx]

    # H1 Movilidad propia
    steps = 4 if player.first_turn else state.board.get_value(player.position)
    h1 = len(get_valid_moves(state.board, player.position, steps, player.first_turn))

    # H2 Movilidad oponente
    steps_op = 4 if opponent.first_turn else state.board.get_value(opponent.position)
    h2 = len(get_valid_moves(state.board, opponent.position, steps_op, opponent.first_turn))

    # H3 Valor de carta actual
    h3 = state.board.get_value(player.position)

    # H4 Control centro
    center = state.board.grid_size // 2
    h4 = - (abs(player.position[0] - center) + abs(player.position[1] - center))

    # H5 Diferencia movilidad
    h5 = h1 - h2

    # Pesos
    w1, w2, w3, w4, w5 = 4, -5, 2, 1, 6

    return w1*h1 + w2*h2 + w3*h3 + w4*h4 + w5*h5

# ------------- MINIMAX + ALPHA BETA --------------
def minimax(state, depth, alpha, beta, maximizing_player, maximizing_idx):
    global NODES_EXPANDED
    NODES_EXPANDED += 1

    if is_terminal(state):
        if state.current_player_idx != maximizing_idx:
            return math.inf, None
        else:
            return -math.inf, None

    if depth == 0:
        return evaluate(state, maximizing_idx), None

    best_move = None

    if maximizing_player:
        max_eval = -math.inf
        for action in get_actions(state):
            eval, _ = minimax(result(state, action), depth-1,
                              alpha, beta, False, maximizing_idx)
            if eval > max_eval:
                max_eval = eval
                best_move = action
            alpha = max(alpha, eval)
            if beta <= alpha:
                break
        return max_eval, best_move
    else:
        min_eval = math.inf
        for action in get_actions(state):
            eval, _ = minimax(result(state, action), depth-1,
                              alpha, beta, True, maximizing_idx)
            if eval < min_eval:
                min_eval = eval
                best_move = action
            beta = min(beta, eval)
            if beta <= alpha:
                break
        return min_eval, best_move

# ------- PROFUNDIZACIÓN ITERATIVA (IDS) ----------
def iterative_deepening(state, max_time):
    start_time = time.time()
    depth = 1
    best_move = None
    maximizing_idx = state.current_player_idx

    while time.time() - start_time < max_time:
        value, move = minimax(state, depth, -math.inf, math.inf, True, maximizing_idx)
        if move is not None:
            best_move = move
        depth += 1

    return best_move

# ----------------- JUGADOR GREEDY ----------------
def greedy_move(state):
    best_score = -math.inf
    best_action = None
    idx = state.current_player_idx

    for action in get_actions(state):
        new_state = result(state, action)
        score = evaluate(new_state, idx)
        if score > best_score:
            best_score = score
            best_action = action

    return best_action


# -------------------- MÉTODOS --------------------
def place_players_on_zeros(board, players):
    zeros = [] #----creo un arreglo
    for i in range(board.grid_size): #----crea las filas
        for j in range(board.grid_size): #----crea los columnas
            if board.board[i][j] == 0:
                zeros.append([i,j])
    players[0].position = zeros[0]  #----jugador rojo
    players[1].position = zeros[1]  #----jugador verde

def get_valid_moves(board,
                    pos,
                    steps,
                    first_turn,
                    other_pos): #-- NUEVO!!! Se agrego para jugador no ocupar misma posicion
    moves = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    visited = set()

    def dfs(r,c,remaining):

        #--- Limite del tablero
        if r<0 or r>=board.grid_size or c<0 or c>=board.grid_size:
            return
        if board.used[r][c]:
            return
        # evita ciclos
        if (r,c) in visited:
            return
        visited.add((r,c))

        #--- Bloquea la posicion final del otro jugador
        is_final = (remaining ==0 or first_turn)
        if is_final and [r, c] != pos and [r, c] != other_pos:
            moves.append([r,c])

        if remaining > 0:
            for dr,dc in directions:
                dfs(r+dr,c+dc,remaining-1)
        visited.remove((r,c))

    dfs(pos[0], pos[1], steps)

    # Quitar duplicados
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
        self.ai_players = []
        self.ai_enabled = True
        self.ai_time_limit = 3
        self.ai_mode = "minimax"

        # Jugadores
        self.players = [Player("Rojo", "red"), Player("Verde (IA)", "green")]
        self.current_player_idx = 0
        self.game_over = False
        self.ai_players = [False, False] # por defecto humano contra humano

        # Tablero
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)

        # Canvas
        self.canvas = tk.Canvas(root, width=self.cell_size*self.grid_size,
                                height=self.cell_size*self.grid_size)
        self.canvas.pack()

        # Fuentes y Etiquetas
        self.normal_font = tkFont.Font(size=12) #--Fuente normal etiquetas ---NUEVO!!!
        self.bold_font = tkFont.Font(size=12, weight="bold") #--Fuente negrita etiquetas---NUEVO!!!

        self.labels = [tk.Label(root, text="", font=("Arial",12)),
                       tk.Label(root, text="", font=("Arial",12))]
        self.labels[0].pack()
        self.labels[1].pack()

        # Botones para tipo de juego
        self.btn_frame = tk.Frame(root)
        self.btn_frame.pack(pady=5)
        tk.Button(self.btn_frame, text="Hum vs IA", command=lambda: self.start_game([False, True])).pack(side="left", padx=5)
        tk.Button(self.btn_frame, text="IA vs IA", command=lambda: self.start_game([True, True])).pack(side="left", padx=5)
        tk.Button(self.btn_frame, text="New Game", command= self.new_game).pack(side="left", padx=5)

        # Dibujar y actualizar tablero
        self.draw_board()
        self.update_labels()
        self.canvas.bind("<Button-1>", self.select_tile)
        
        if self.current_player_idx in self.ai_players:
            self.root.after(500, self.make_ai_move)

    def update_labels(self):
        for idx, player in enumerate(self.players):
            turn_text = " ← Turno" if idx == self.current_player_idx else ""
            turn_font = self.bold_font if idx == self.current_player_idx else self.normal_font #--Fuente etiquetas---NUEVO!!!
            self.labels[idx].config(
                text=f"Jugador {player.name} en carta: {self.board.get_value(player.position)}{turn_text}",
                font=turn_font #--Fuente negrita etiquetas---NUEVO!!!
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

# ------------------------------ JUGADOR HUMANO ----------------------------

    def select_tile(self, event):

        if self.game_over:
            return

        if not self.ai_players[self.current_player_idx]: #Solo mueve si es humano
           x_click = event.x // self.cell_size
           y_click = event.y // self.cell_size
           if self.board.used[y_click][x_click]:
               return

        player = self.players[self.current_player_idx]
        other_player = self.players[1 - self.current_player_idx]
        prev_value = self.board.get_value(player.position)
        steps_allowed = 4 if player.first_turn else prev_value
        valid_moves = get_valid_moves(self.board,
                                      player.position,
                                      steps_allowed,
                                      player.first_turn,
                                      other_player.position) #-- NUEVO!!! Se agrego para jugador no ocupar misma posicion

        if [y_click, x_click] not in valid_moves:
            return

        # Marcar carta usada y mover jugador
        self.board.mark_used(player.position)
        player.position = [y_click, x_click]
        player.first_turn = False
        self.draw_board()
        self.update_labels()


        # Verificar si el siguiente jugador puede moverse
        next_idx = 1 - self.current_player_idx
        next_player = self.players[next_idx]
        next_steps = 4 if next_player.first_turn else self.board.get_value(next_player.position)
        next_moves = get_valid_moves(self.board,
                                     next_player.position,
                                     next_steps,
                                     next_player.first_turn,
                                     player.position,
                                     )
        if not next_moves:
            winner = self.players[self.current_player_idx]
            self.labels[0].config(text=f"Jugador {winner.name} ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} ganó!")
            self.canvas.unbind("<Button-1>")
            return

        # Cambiar turno
        self.current_player_idx = 1 - self.current_player_idx
        self.update_labels()

        if self.check_win():
            return

        # si es turno de IA (Jugador Verde), mover automaticamente
        if self.ai_players [self.current_player_idx]:
            self.root.after(300, self.ai_move)

#-------------------------------- MINIMAX y HEURISTICAS ------------------------------------

    def evaluate(self, board, ai_player, other_player):

        ai_steps = 4 if ai_player.first_turn else board.get_value(ai_player.position)
        other_player_steps = 4 if other_player.first_turn else board.get_value(other_player.position)

        ai_moves = get_valid_moves(
            board,
            ai_player.position,
            ai_steps,
            ai_player.first_turn,
            other_player.position,
        )

        other_player_moves = get_valid_moves(
            board,
            other_player.position,
            other_player_steps,
            other_player.first_turn,
            ai_player.position,
        )

        # Anterior heuristica
        #return len(ai_moves) - len(other_player_moves)

        # Heuristica mejorada
        score = 3*len(ai_moves) - 4*len(other_player_moves) + board.get_value(ai_player.position)
        return score


# ------------------------ MINIMAX -------------------------
    def minimax(self, board, ai_player, other_player, depth, maximizing):

        steps = 4 if ai_player.first_turn else board.get_value(ai_player.position)

        moves = get_valid_moves(
            board,
            ai_player.position,
            steps,
            ai_player.first_turn,
            other_player.position,
        )

    #------ caso terminal
        if depth == 0 or not moves:
            return self.evaluate(board, ai_player, other_player)

        if maximizing:
            max_eval = -float("inf")

            for move in moves:
                new_board = copy.deepcopy(board)
                new_ai = copy.deepcopy(ai_player)
                new_other_player = copy.deepcopy(other_player)

                # simular movimiento AI
                new_board.mark_used(new_ai.position)
                new_ai.position = move
                new_ai.first_turn = False

                eval_score = self.minimax(new_board, new_ai, new_other_player, depth - 1, False)
                max_eval = max(max_eval, eval_score)

            return max_eval

        else:
            min_eval = float("inf")

            for move in moves:
                new_board = copy.deepcopy(board)
                new_ai = copy.deepcopy(ai_player)
                new_other_player = copy.deepcopy(other_player)

                new_board.mark_used(new_other_player.position)
                new_other_player.position = move
                new_other_player.first_turn = False

                eval_score = self.minimax(new_board, new_ai, new_other_player, depth - 1, True)
                min_eval = min(min_eval, eval_score)

            return min_eval

#--------------------- MEJOR MOVIMIENTO --------------------

    def get_best_move_ai(self, ai_player, other_ai_player, depth = 2):

        #ai = self.players[1] #Verde = IA
        #human = self.players[0] # Rojo = humano

        best_score = -float("inf")
        best_move = None

        steps = 4 if ai_player.first_turn else self.board.get_value(ai_player.position)

        moves = get_valid_moves(
            self.board,
            ai_player.position,
            steps,
            ai_player.first_turn,
            other_ai_player.position,
        )

        for move in moves:
            new_board = copy.deepcopy(self.board)
            new_ai = copy.deepcopy(ai_player)
            new_other_player = copy.deepcopy(other_ai_player)

            new_board.mark_used(new_ai.position)
            new_ai.position = move
            new_ai.first_turn = False

            score = self.minimax(new_board, new_ai, new_other_player, depth - 1, False)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


#-------------------------MOVIMIENTOS IA ----------------------

    def ai_move(self):
        if self.game_over:
            return

        current_ai = self.players[self.current_player_idx]
        other = self.players[1 - self.current_player_idx]

        move = self.get_best_move_ai(current_ai, other)

        if move is None:
            # Other gana
            winner = other
            self.labels[0].config(text=f"Jugador {winner.name} Ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} Ganó!")
            self.game_over = True
            return

        # Aplicar movimiento IA
        self.board.mark_used(current_ai.position)
        current_ai.position = move
        current_ai.first_turn = False

        self.draw_board()
        self.update_labels()

        #cambiar turno
        self.current_player_idx = 1 - self.current_player_idx
        self.update_labels()

        if self.current_player_idx in self.ai_players:
            self.root.after(500, self.make_ai_move)

    def new_game(self):
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)

        for p in self.players:
            p.first_turn = True

        self.current_player_idx = 0
        self.game_over = False
        self.draw_board()
        self.update_labels()

        if 0 in self.ai_players and 1 in self.ai_players:
            self.canvas.unbind("<Button-1>")
        else:
            self.canvas.bind("<Button-1>", self.select_tile)

        if self.current_player_idx in self.ai_players:
            self.root.after(300, self.make_ai_move)
        
    
    def make_ai_move(self):
        state = GameState(self.board, self.players, self.current_player_idx)
        if self.ai_mode == "minimax":
            move = iterative_deepening(state, self.ai_time_limit)

        elif self.ai_mode == "ai_vs_greedy":
            if self.current_player_idx == 0:
                move = iterative_deepening(state, self.ai_time_limit)
            else:
                move = greedy_move(state)

        if move is None:
            return

        player = self.players[self.current_player_idx]

        self.board.mark_used(player.position)
        player.position = move
        player.first_turn = False

        self.draw_board()
        self.update_labels()

        next_idx = 1 - self.current_player_idx
        next_player = self.players[next_idx]
        next_steps = 4 if next_player.first_turn else self.board.get_value(next_player.position)
        next_moves = get_valid_moves(self.board, next_player.position, next_steps, next_player.first_turn)

        if not next_moves:
            winner = self.players[self.current_player_idx]
            self.labels[0].config(text=f"Jugador {winner.name} ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} ganó!")
            self.canvas.unbind("<Button-1>")
            return

        self.current_player_idx = next_idx
        self.update_labels()    
        
        if self.current_player_idx in self.ai_players:
            self.root.after(300, self.make_ai_move)

# -------------------- CANVAS DEL JUEGO --------------------

def start_game(mode):
    menu.destroy()

    root = tk.Tk()
    root.title("Collapsi Game")
    
    game = Game(root)

    if mode == "human_vs_ai":
        game.players[0].name = "Rojo (Humano)"
        game.players[1].name = "Verde (Minimax)"
        game.ai_players = [1]
        game.ai_mode = "minimax"

    elif mode == "ai_vs_greedy":
        game.players[0].name = "Rojo (Minimax)"
        game.players[1].name = "Verde (Greedy)"
        game.ai_players = [0, 1]
        game.ai_mode = "ai_vs_greedy"
    game.update_labels()
    
    if mode == "ai_vs_greedy":
        game.canvas.unbind("<Button-1>")

    if game.current_player_idx in game.ai_players:
        root.after(300, game.make_ai_move)

    root.mainloop()



menu = tk.Tk()
menu.title("Seleccionar modo de juego")
menu.geometry("300x200")

label = tk.Label(menu, text="Selecciona modo de juego", font=("Arial",12))
label.pack(pady=20)

btn1 = tk.Button(menu, text="Jugar contra IA (Minimax)",
                 command=lambda: start_game("human_vs_ai"))
btn1.pack(pady=10)

btn2 = tk.Button(menu, text="IA vs Greedy",
                 command=lambda: start_game("ai_vs_greedy"))
btn2.pack(pady=10)

menu.mainloop()
