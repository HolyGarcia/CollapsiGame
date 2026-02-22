"""Implementación del algoritmo Minimax con límite de tiempo.

    Este módulo permite a la IA evaluar estados del juego y seleccionar
    el mejor movimiento posible considerando al oponente."""
import time
import copy

def get_possible_moves(state, get_valid_moves):
    """Obtiene todos los movimientos válidos del jugador actual
        a partir del estado del juego."""

    player = state.players[state.current_player_idx]
    steps = 4 if player.first_turn else state.board.get_value(player.position)

    return get_valid_moves(
        state.board,
        player.position,
        steps,
        player.first_turn,
        state.players
    )

# Heurística: ventaja basada en cantidad de movimientos disponibles
def evaluate_state(state, get_valid_moves):
    current = state.players[state.current_player_idx]
    opponent = state.players[1 - state.current_player_idx]

    def count_moves(player):
        steps = 4 if player.first_turn else state.board.get_value(player.position)
        return len(
            get_valid_moves(
                state.board,
                player.position,
                steps,
                player.first_turn,
                state.players
            )
        )

    return count_moves(current) - count_moves(opponent)

def simulate_move(state, move):
    """Simula un movimiento sin modificar el juego real.
       Se utiliza para explorar estados dentro del Minimax."""

    new_state = copy.deepcopy(state)
    player = new_state.players[new_state.current_player_idx]

    new_state.board.mark_used(player.position)
    player.position = move
    player.first_turn = False

    new_state.current_player_idx = 1 - new_state.current_player_idx
    return new_state

def minimax(state, depth, maximizing, start_time, time_limit, get_valid_moves):
    """Algoritmo Minimax con poda por tiempo.

        - state: estado actual del juego
        - depth: profundidad máxima de búsqueda
        - maximizing: indica si el jugador actual maximiza o minimiza
        - start_time: tiempo inicial de búsqueda
        - time_limit: tiempo máximo permitido"""

    if depth == 0 or time.time() - start_time > time_limit:
        return evaluate_state(state, get_valid_moves), None

    moves = get_possible_moves(state, get_valid_moves)

    if not moves:
        return (-999 if maximizing else 999), None

    best_move = None

    if maximizing:
        best_value = -float("inf")
        for move in moves:
            new_state = simulate_move(state, move)
            value, _ = minimax(
                new_state, depth - 1, False,
                start_time, time_limit, get_valid_moves
            )
            if value > best_value:
                best_value = value
                best_move = move
        return best_value, best_move

    else:
        best_value = float("inf")
        for move in moves:
            new_state = simulate_move(state, move)
            value, _ = minimax(
                new_state, depth - 1, True,
                start_time, time_limit, get_valid_moves
            )
            if value < best_value:
                best_value = value
                best_move = move
        return best_value, best_move