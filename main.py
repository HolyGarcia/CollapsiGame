import tkinter as tk
import random
import tkinter.font as tkFont
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

        # Jugadores
        self.players = [Player("Rojo", "red"), Player("Verde", "green")]
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

        #----------INTERFAZ DEL TABLERO -------------------------

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

        if self.check_win():
            return

        # si el siguiente es IA, continuar automaticamente
        if self.ai_players[self.current_player_idx]:
            self.root.after(1500, self.ai_move)


    #--------------------- IA contra IA ---------------------------

    def auto_play(self, delay = 500):
        if self.game_over:
            return

        self.ai_players = [True, True]
        self.canvas.unbind("<Button-1>")
        self.root.after(delay, self.ai_move)

#---------------------- CHECK WINNER / GAME OVER --------------------

    def check_win(self):
        current = self.players[self.current_player_idx]
        other = self.players[1 - self.current_player_idx]

        steps = 4 if current.first_turn else self.board.get_value(current.position)

        moves = get_valid_moves(
            self.board,
            current.position,
            steps,
            current.first_turn,
            other.position,
        )

        if not moves:
            winner = other
            self.labels[0].config(text=f"Jugador {winner.name} Ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} Ganó!")
            self.game_over = True
            self.canvas.unbind("<Button-1>")
            return True

        return False

#------------------------ NEW GAME -------------------------

    def new_game(self):
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)
        for p in self.players:
            p.first_turn = True
        self.current_player_idx = 0
        self.game_over = False
        self.draw_board()
        self.update_labels()

# -------------------- Iniciar partida segun Modo -------------------

    def start_game(self, ai_players):
        self.ai_players = ai_players
        self.new_game()
        self.update_labels()

        if self.check_win():
            return

        if self.ai_players[0] and self.ai_players[1]:

            #IA vs IA
            self.canvas.unbind("<Button-1>")
            self.root.after(500, self.auto_play)
        elif self.ai_players[0] == False and self.ai_players[1] == True:

            # Humano vs IA
            self.canvas.bind("<Button-1>", self.select_tile)
        else:

            # Otros modos
            self.canvas.bind("<Button-1>", self.select_tile)


# -------------------- CANVAS DEL JUEGO --------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Collapsi Game")

    game = Game(root)
    root.mainloop()



