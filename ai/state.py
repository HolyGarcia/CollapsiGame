class GameState:
    """Representa un estado del juego para el algoritmo Minimax.

       Contiene:
       - board: tablero
       - players: jugadores
       - current_player_idx: índice del jugador actual"""
    def __init__(self, board, players, current_player_idx):
        self.board = board
        self.players = players
        self.current_player_idx = current_player_idx