"""Finish all TODO items in this file to complete the isolation project, then
test your agent's strength against a set of known agents using tournament.py
and include the results in your report.
"""
import random
import operator
import math


class SearchTimeout(Exception):
    """Subclass base exception for code clarity. """
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    This should be the best heuristic function for your project submission.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # If the board is empty, or the player has won or lost, use the const score.
    res = const_score(game, player)
    if res:
        return res

    # Forward to the selected heuristic
    return float(look_ahead_improved_score_v3(game, player))


def custom_score_2(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # If the board is empty, or the player has won or lost, use the const score.
    res = const_score(game, player)
    if res:
        return res

    # Forward to the selected heuristic
    return float(look_ahead_improved_score_v2(game, player))


def custom_score_3(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # If the board is empty, or the player has won or lost, use the const score.
    res = const_score(game, player)
    if res:
        return res

    # Forward to the selected heuristic
    return float(moves_ratio_score(game, player))


class IsolationPlayer:
    """Base class for minimax and alphabeta agents -- this class is never
    constructed or tested directly.

    ********************  DO NOT MODIFY THIS CLASS  ********************

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """
    def __init__(self, search_depth=3, score_fn=custom_score, timeout=10.):
        self.search_depth = search_depth
        self.score = score_fn
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


class MinimaxPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using depth-limited minimax
    search. You must finish and test this player to make sure it properly uses
    minimax to return a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **************  YOU DO NOT NEED TO MODIFY THIS FUNCTION  *************

        For fixed-depth search, this function simply wraps the call to the
        minimax method, but this method provides a common interface for all
        Isolation agents, and you will replace it in the AlphaBetaPlayer with
        iterative deepening search.

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        self.time_left = time_left

        # Initialize the best move so that this function returns something
        # in case the search fails due to timeout
        best_move = get_init_move(game, self)

        if game.move_count > 1:
            try:
                # The try/except block will automatically catch the exception
                # raised when the timer is about to expire.
                best_move = self.minimax(game, self.search_depth)

            except SearchTimeout:
                pass  # Handle any actions required after timeout as needed

        # Return the best move from the last completed search iteration
        return best_move

    def minimax(self, game, depth):
        """Implement depth-limited minimax search algorithm as described in
        the lectures.

        This should be a modified version of MINIMAX-DECISION in the AIMA text.
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """

        score, move = self.__minimax(game, depth)
        return move

    def __minimax(self, game, depth, max_p = True):
        """
        Minimax implementation.
        
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        max_p: bool
            Indicates whether this a maximization node or not.
            
        :return: score, move
            The move is a tuple (x,y) represeting a position in the game board.
        """

        # Check if we still have time
        ensure_not_timeout(self)

        # If we have reach the target depth, compute the score of the board
        if depth == 0:
            return self.score(game, self), (-1, -1)

        # Get available moves, If we do not have any movements left, return the utility
        available_moves = game.get_legal_moves()
        if not available_moves:
            return game.utility(self), (-1, -1)

        # Get cmp operator and best score ceil depending if we are maximizing the node or not.
        cmp, best_score = (operator.gt, float("-inf")) if max_p else (operator.lt, float("inf"))
        best_move = (-1, -1)

        for move in available_moves:
            # Forecast move and compute score
            forecast = game.forecast_move(move)
            score, _ = self.__minimax(forecast, depth - 1, max_p=not max_p)

            # If the new_score is better, update the best score and best move
            if cmp(score, best_score):
                best_score = score
                best_move = move

        return best_score, best_move


class AlphaBetaPlayer(IsolationPlayer):
    """Game-playing agent that chooses a move using iterative deepening minimax
    search with alpha-beta pruning. You must finish and test this player to
    make sure it returns a good move before the search time limit expires.
    """

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        Modify the get_move() method from the MinimaxPlayer class to implement
        iterative deepening search instead of fixed-depth search.

        **********************************************************************
        NOTE: If time_left() < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Initialize the best move so that this function returns something in case the search fails due to timeout
        best_move = get_init_move(game, self)

        if game.move_count > 1:

            try:
                # The try/except block will automatically catch the exception,
                # raised when the timer is about to expire.
                best_move = self.alphabeta(game, 1)

                # Iterative deepening
                for depth in range(2, len(game.get_blank_spaces())):
                    best_move = self.alphabeta(game, depth)

            except SearchTimeout:
                pass

        # Return the best move from the last completed search iteration
        return best_move

    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf")):
        """Implement depth-limited minimax search with alpha-beta pruning as
        described in the lectures.

        This should be a modified version of ALPHA-BETA-SEARCH in the AIMA text
        https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        **********************************************************************
            You MAY add additional methods to this class, or define helper
                 functions to implement the required functionality.
        **********************************************************************

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        Returns
        -------
        (int, int)
            The board coordinates of the best move found in the current search;
            (-1, -1) if there are no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project tests; you cannot call any other evaluation
                function directly.

            (2) If you use any helper functions (e.g., as shown in the AIMA
                pseudocode) then you must copy the timer check into the top of
                each helper function or else your agent will timeout during
                testing.
        """
        score, move = self.__alphabeta(game, depth, alpha, beta)
        return move

    def __alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), max_p = True):
        """
        
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers
        
        max_p: bool
            Indicates whether this a maximization node or not.
         
        :return: score, move
            The move is a tuple (x,y) representing a position in the game board.
        """

        # Check if we still have time
        ensure_not_timeout(self)

        # If we have reach the target depth, compute the score of the board
        if depth == 0:
            return self.score(game, self), (-1, -1)

        # Get available moves, If we do not have any movements left, return the utility
        available_moves = game.get_legal_moves()
        if not available_moves:
            return game.utility(self), (-1, -1)

        # Get cmp operator and best score ceil depending if we are maximizing the node or not.
        cmp, best_score = (operator.gt, float("-inf")) if max_p else (operator.lt, float("inf"))
        best_move = (-1, -1)

        for move in available_moves:

            # Forecast move and compute score
            forecast = game.forecast_move(move)
            score, _ = self.__alphabeta(forecast, depth - 1, alpha=alpha, beta=beta, max_p=not max_p)

            # If the new_score is better, update the best score and best move
            if cmp(score, best_score):
                best_score = score
                best_move = move

            # Check if we should the search
            if self.should_prune(alpha, beta, best_score, max_p):
                break

            # Update the alpha-beta values
            alpha, beta = self.update_alphabeta(alpha, beta, best_score, max_p)

        return best_score, best_move

    @staticmethod
    def should_prune(alpha, beta, best_score, max_p):
        """
        Check if the branch should be pruned
        
        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers
            
        best_score: float
            The current best score
            
        max_p: bool
            Indicates whether this a maximization node or not.
            
        :return: 
            bool indicating whether the branch should be prune or not.
        """
        if max_p and best_score >= beta:
            return True
        elif not max_p and best_score <= alpha:
            return True
        return False

    @staticmethod
    def update_alphabeta(alpha, beta, score,  max_p):
        """
        Update the value of alpha (maximization node) or beta(minimization node)
        
        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers
            
        best_score: float
            The current best score
            
        max_p: bool
            Indicates whether this a maximization node or not.
            
        :return: tuple (alpha, beta)
        """
        if max_p:
            alpha = max(alpha, score)
        else:
            beta = min(beta, score)

        return alpha, beta


def ensure_not_timeout(player):
    """
    Helper function to ensure that the player is in time. A SearchTimeout is thrown if the player is out of time
    
    player : object
        A player instance in the current game
        
    """
    if isinstance(player, IsolationPlayer):
        if player.time_left() < player.TIMER_THRESHOLD:
            raise SearchTimeout()


def score_next_move(game, player, score_move_f, merge_scores_f, default_val=0):
    """
    Score the next move (look ahead) using score_move_f and then select a score using merge_scores_f
    among the possible scores.
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    score_move_f: function
        Compute the score of a given move
        
    merge_scores_f: function
        Select a score from a group of scores.
        
    default_val: object
        Value to return if there are no moves.

    return: float
        result returned by merge_scores_f or default_val if the sequence is empty
    """
    scores = [score_move_f(game._Board__get_moves(move)) for move in game.get_legal_moves(player)]

    if scores:
        return merge_scores_f(scores)
    return default_val


def expand_moves_cycles(game, player, max_depth=-1):
    """
    Expand moves from the player position.
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    max_depth: int
        Max depth of expansion (BFS depth). (If the depth is -1, then expand while possible)
        
    return: dict
        Return frequencies of the nodes
        
    """
    moves = game.get_legal_moves(player)
    if moves:
        visited = dict()

        queue = list(map(lambda x: (x, 1), moves))

        while queue:
            ensure_not_timeout(player)
            move, depth = queue.pop()
            visited[move] = visited.get(move, 0) + 1

            if max_depth == -1 or depth + 1 < max_depth:
                queue.extend(
                    [(next_move, depth + 1) for next_move in game._Board__get_moves(move) if next_move not in visited])

        return visited
    else:
        return dict()


def euclidean_distance(pos1, pos2):
    """
    Computes euclidean distance between pos1 y pos2
    
    :param pos1: pair (x, y)
    :param pos2: pair (x, y)
    
    :return: return euclidean distance between pos1 y pos2
    """
    return math.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)


def get_init_move(game, player):
    """
    Get the move which is closest to the center of the board.
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
    
    return: (x, y)
        A tuple (x,y) representing a position in the game board. (-1, -1) if there are no legal moves
    """
    legal_moves = game.get_legal_moves(player)
    if legal_moves:
        return min(legal_moves,
                   key=lambda x: euclidean_distance(x, (int(math.ceil(game.height / 2)), int(math.ceil(game.width / 2)))))
    return (-1, -1)


def is_game_finish(game, player):
    """
    Check whether the game has finished or not
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        inf, -inf if the game is finished, None otherwise.
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    return None


def empty_board_score(game):
    """
    Empty board score
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    return : float
        0 if the board is empty, None otherwise
        
    """
    if game.move_count == 0:
        return float(0)
    return None


def const_score(game, player):
    """
    Compute constant score (Game finished or empty)
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        inf, -inf, 0 if the board is empty, None otherwise
    """
    return is_game_finish(game, player) or empty_board_score(game)


def look_ahead_improved_score(game, player):
    """
    Look ahead improved score is a heuristic based on improved score, 
    but instead of taking in account the current position, we use the best next move for each player.
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        board score
    """
    return score_next_move(game, player, len, max) - score_next_move(game, game.get_opponent(player), len, max)


def look_ahead_improved_score_v2(game, player):
    """
    This heuristic encourages those moves which results in more moves, in other words, it enforces the player to go
    from closed areas to open areas.
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        board score
    """
    own_cycles_depth = expand_moves_cycles(game, player, max_depth=3)
    own_cycles_depth = own_cycles_depth.values()

    opp_cycles_depth = expand_moves_cycles(game, game.get_opponent(player), max_depth=3)
    opp_cycles_depth = opp_cycles_depth.values()

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    func = sum

    return (func(own_cycles_depth) - own_moves) - (func(opp_cycles_depth) - opp_moves)


def look_ahead_improved_score_v3(game, player):
    """
    This heuristic is similar to the look ahead improved score, but encourages the positions in which there are loops. 
    Basically, the positions in which there are loops are open areas, so it is similar to count the blank spaces 
    around the player.
    
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        board score
    """
    own_cycles = expand_moves_cycles(game, player, max_depth=3)
    own_cycles = own_cycles.values()

    opp_cycles = expand_moves_cycles(game, game.get_opponent(player), max_depth=3)
    opp_cycles = opp_cycles.values()

    func = sum
    return func(own_cycles) - func(opp_cycles) + (min(own_cycles) if own_cycles else 0 / max(opp_cycles) if opp_cycles else 1)


def moves_ratio_score(game, player):
    """
    Ratio score is another heuristic function based on improved score. 
    The heuristic is calculated as follows "own_moves \ opp_moves".

    The difference with the improved score is how the scores are scaled. For example, in the improved score, 
    the following tuples are equivalent.

    (own_moves = 1, opp_moves = 3), (own_moves = 2, opp_moves = 4) which results in -2

    But in the ratio score, those tuples returns different values and the second is better than the first one.
    Basically, the scores get worse as we approach to 0.
     
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        board score
    """
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return own_moves / (opp_moves if opp_moves else 1)


def lock_opponent_score(game, player):
    """
    Lock opponent score is a variation of the improved score, but the heuristic enforces to minimize the number
     of moves of the opponent by weighting the opp_moves values (in our case, we multiply it by 2)
      
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).
        
    player : object
        A player instance in the current game
        
    return : float
        board score
    """
    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))
    return float(own_moves - 2 * opp_moves)
