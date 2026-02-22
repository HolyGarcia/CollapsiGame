"""Collapsi Game
    -------------
    Juego de tablero por turnos implementado con Tkinter.
    Permite jugar Humano vs IA.

    La IA utiliza el algoritmo Minimax con límite de tiempo configurable
    para seleccionar el mejor movimiento posible en cada turno."""

import tkinter as tk
import random
import time
from ai.state import GameState
from ai.minimax import minimax


# -------------------- CLASES --------------------

class Player:
    """Representa a un jugador del juego.

        Atributos:
        - name: nombre del jugador
        - color: color usado para dibujarlo en el tablero
        - position: posición actual [fila, columna]
        - first_turn: indica si es el primer turno del jugador"""

    def __init__(self, name, color):
        self.name = name
        self.color = color
        self.position = [0,0]
        self.first_turn = True

class Board:
    """Representa el tablero del juego.

        Contiene:
        - La matriz de valores de las cartas
        - El control de cartas ya utilizadas"""

    def __init__(self, grid_size, values, cell_size):
        self.grid_size = grid_size
        self.values = values
        self.cell_size = cell_size
        self.used = [[False]*grid_size for _ in range(grid_size)]
        self.board = self.create_board()

    def create_board(self):
        """Crea el tablero mezclando aleatoriamente los valores disponibles."""
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
        """Marca una carta como usada para que no pueda volver a pisarse."""
        self.used[pos[0]][pos[1]] = True

# -------------------- MÉTODOS --------------------

def place_players_on_zeros(board, players):
    """Coloca a los jugadores inicialmente en las casillas con valor 0.
        Se asume que existen al menos dos ceros en el tablero."""
    zeros = [] #----creo un arreglo
    for i in range(board.grid_size): #----crea las filas
        for j in range(board.grid_size): #----crea los columnas
            if board.board[i][j] == 0:
                zeros.append([i,j])
    players[0].position = zeros[0]  #----jugador rojo
    players[1].position = zeros[1]  #----jugador verde

def get_valid_moves(board, pos, steps, first_turn, players):
    """Calcula todos los movimientos válidos para un jugador.

        Parámetros:
        - board: tablero del juego
        - pos: posición actual del jugador
        - steps: número de pasos permitidos
        - first_turn: indica si es el primer turno
        - players: lista de jugadores (para evitar casillas ocupadas)

        Retorna:
        - Lista de posiciones válidas [fila, columna]"""

    moves = []
    directions = [(-1,0),(1,0),(0,-1),(0,1)]
    visited = set()

    def dfs(r,c,remaining):
        # Evita salir del tablero
        if r<0 or r>=board.grid_size or c<0 or c>=board.grid_size:
            return
        if board.used[r][c]:
            return

        # Evitar casilla ocupada por otro jugador
        for p in players:
            if [r, c] == p.position and [r, c] != pos:
                return

        # Evita cartas ya usadas
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
    # Quitar duplicados
    unique_moves = []
    for m in moves:
        if m not in unique_moves:
            unique_moves.append(m)
    return unique_moves

# -------------------- LÓGICA DEL JUEGO --------------------

class Game:
    """Controla la lógica principal del juego, los turnos
        y la interacción con la interfaz gráfica."""
    def __init__(self, root):
        self.root = root
        self.grid_size = 4
        self.cell_size = 110
        self.values = [0,0,4,4,1,1,1,1,2,2,2,2,3,3,3,3]

        # Jugadores
        self.ai_player_name = "Verde" # Configuración de la IA
        self.ai_time_limit = 1.0  # ⏱ # Tiempo máximo de búsqueda del Minimax (segundos)
        self.players = [Player("Rojo", "red"), Player("Verde", "green")]
        self.current_player_idx = 0

        # Tablero
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)

        # Canvas
        self.canvas = tk.Canvas(root, width=self.cell_size*self.grid_size,
                                height=self.cell_size*self.grid_size)
        self.canvas.pack()

        # Etiquetas
        self.labels = [tk.Label(root, text="", font=("Arial",12)),
                       tk.Label(root, text="", font=("Arial",12))]
        self.labels[0].pack()
        self.labels[1].pack()

        # Botón nuevo juego
        self.new_game_btn = tk.Button(root, text="Juego Nuevo", command=self.new_game)
        self.new_game_btn.pack(pady=10)

        self.draw_board()
        self.update_labels()
        self.canvas.bind("<Button-1>", self.select_tile)

    def ai_move(self):
        """Ejecuta el turno de la IA.
            Obtiene el estado actual del juego y utiliza Minimax
            para decidir el mejor movimiento posible."""

        state = GameState(self.board, self.players, self.current_player_idx)

        if not self.root.winfo_exists():
            return

        start_time = time.time()
        _, best_move = minimax(
            state,
            depth=4,
            maximizing=True,
            start_time=start_time,
            time_limit=self.ai_time_limit,
            get_valid_moves=get_valid_moves
        )

        if best_move:
            y, x = best_move
            fake_event = type(
                "Event", (),
                {"x": x * self.cell_size + 1, "y": y * self.cell_size + 1}
            )
            self.select_tile(fake_event)

    def update_labels(self):
        for idx, player in enumerate(self.players):
            turn_text = " ← Turno" if idx == self.current_player_idx else ""
            self.labels[idx].config(
                text=f"Jugador {player.name} en carta: {self.board.get_value(player.position)}{turn_text}"
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

    def select_tile(self, event):
        """Maneja la selección de una casilla por parte de un jugador.
            Valida el movimiento, actualiza el estado del juego
            y gestiona el cambio de turno."""
        x_click = event.x // self.cell_size
        y_click = event.y // self.cell_size
        if self.board.used[y_click][x_click]:
            return

        player = self.players[self.current_player_idx]
        prev_value = self.board.get_value(player.position)
        steps_allowed = 4 if player.first_turn else prev_value
        # Obtiene los movimientos válidos según las reglas del juego
        valid_moves = get_valid_moves(
            self.board,
            player.position,
            steps_allowed,
            player.first_turn,
            self.players
        )

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
        next_moves = get_valid_moves(self.board, next_player.position, next_steps, next_player.first_turn, self.players)
        if not next_moves:
            winner = self.players[self.current_player_idx]
            self.labels[0].config(text=f"Jugador {winner.name} ganó!")
            self.labels[1].config(text=f"Jugador {winner.name} ganó!")
            self.canvas.unbind("<Button-1>")
            return

        # Cambiar turno
        self.current_player_idx = next_idx
        self.update_labels()

        # Si es turno de la IA
        current_player = self.players[self.current_player_idx]
        if current_player.name == self.ai_player_name:
            self.root.after(500, self.ai_move)

    def new_game(self):
        self.board = Board(self.grid_size, self.values, self.cell_size)
        place_players_on_zeros(self.board, self.players)
        for p in self.players:
            p.first_turn = True
        self.current_player_idx = 0
        self.draw_board()
        self.update_labels()
        self.canvas.bind("<Button-1>", self.select_tile)

# -------------------- CANVAS DEL JUEGO --------------------

root = tk.Tk()
root.title("Collapsi Game")
game = Game(root)
root.mainloop()
