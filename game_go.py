# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

class Board(object):  # maybe we'll need to inherit Array
    def __init__(self):
	self.height = 4
	self.width = 4
	self.states = {}  # {int: int} i.e. {move: player} no chesspieces means it doesn't exist in this dict
	self.players = [1, 2]
	self.states_history = []   # x*height*width dimension

    def init_board(self, start_player=np.random.randint(0, 2)):
	self.current_player = self.players[start_player]
	self.availables = set(range(self.height*self.width))  # locations that are blank, without considering rules restriction
	self.last_move = -1

    def move_to_location(self, move):
	"""
	4*3 board's moves like:
	    0 1 2
	    3 4 5
	    6 7 8
	    9 10 11
	and move 5's location is (1,2)
	"""
	h = move // self.width
	w = move % self.width
	return [h, w]

    def location_to_move(self, location):
	h = location[0]
	w = location[1]
	move = h * self.width + w
	return move

    def add_states_history(self):
	"""add the current locations of all chesspieces the current player put
	   shape: width*height
	   get something like:
	   [[1., 0., 0.]
	    [0., 1., 0.]
	    [0., 0., 0.]]
	    where 1. indicates the loc of current player's pieces
	"""

	state = np.zeros((self.width, self.height))
	moves, players = np.array(zip(*self.states.items()))
	move_curr = moves[players == self.current_player]
	state[move_curr // self.width,
              move_curr % self.height] = 1.0
	self.states_history.append(state)
	return state

    def do_move(self, move):
	# Check if coordinates are occupied(this is for human player's case)
	if move not in self.availables:
	    raise Exception("Move unavailable.")

	# Make move
	self.states[move] = self.current_player

	# Check if any pieces have been taken
	h, w = self.move_to_location(move)
	num_taken = self._take_pieces(h, w)

	# Check if move is suicidal.  A suicidal move is a move that takes no
	# pieces and is played on a coordinate which has no liberties.
	suicide = 0
	if num_taken == 0:
	    suicide = self._check_for_suicide(h, w)

        # renew availablities. also killing groups renews avai too
	if not suicide:
	    self.availables.discard(move)

	# Store history for the opposite player's use
	self.add_states_history()
	self.last_move = move

        # change player
	self.current_player = (
	    self.players[1] if self.current_player == self.players[0]
	    else self.players[0])

    def _check_for_suicide(self, h, w):
        """
        Checks if move is suicidal.
        If it is, delete it by the way
        """
	move = self.location_to_move([h, w])
        if self.count_liberties(h, w) == 0:
            self.states.pop(move)
	    return True
	else:
	    return False

    def reach_border(self, move):
	"""
	get the corresponding blank group and its border
	border: a set documenting players on the border, can be {1}, {2} or {1, 2}
	blank_area: [move0, move1, move2, ...]
	"""
	border = set()
	blank_area = []
	h, w = self.move_to_location(move)
	blank_group = self._get_group(h, w, set())
	for h1, w1 in blank_group:
	    blank_area.append(self.location_to_move([h1, w1]))
	    for p, (a, b) in self._get_surrounding(h, w):
		border.add(p)
	return border, blank_area

    def count_go_number(self):  # needs to be examined
	"""
	fill the board blank with go pieces
	if a blank doesn't belong to any player, then fill it with None
	count how much area each player has gained at the end of the game
	"""
	# change set to list because list can change while in for loop condition
	self.availables = list(self.availables)
	for move in self.availables:
	    border, blank_area = self.reach_border(move)

	    if list(border) == [self.players[0]]:
		for move_bd in blank_area:
		    self.states[move_bd] = self.players[0]
		    self.availables.remove(move_bd)

	    elif list(border) == [self.players[1]]:
		for move_bd in blank_area:
		    self.states[move_bd] = self.players[1]
		    self.availables.remove(move_bd)
	    elif list(border) == []:
		# empty board
		for move_bd in blank_area:
		    self.states[move_bd] = None
		    self.availables.remove(move_bd)
	    else:
		# if this blank area has both two players' pieces in the border
		# then this area belongs to no one
		for move_bd in blank_area:
		    self.states[move_bd] = None
		    self.availables.remove(move_bd)

	# count the number
	state = self.states.values()
	num_player1 = state.count(self.players[0])
	num_player2 = state.count(self.players[1])
	return num_player1, num_player2

    def get_current_player(self):
	return self.current_player

    def _take_pieces(self, h, w):
        """
        Find out the opposite groups around this piece and remove the dead groups
        """
	num_taken = 0
        for p, (h1, w1) in self._get_surrounding(h, w):
            # If location is opponent's color and has no liberties, kill it
            if p is not self.current_player and self.count_liberties(h1, w1) == 0:
                num_taken += self._kill_group(h1, w1)
	return num_taken

    def _kill_group(self, h, w):
        """
        Kills a group of black or white pieces
        Kills the group which contains [h, w]
        """
        group = self._get_group(h, w, set())
	num_taken = 0

        for h1, w1 in group:
            move = self.location_to_move([h1, w1])
	    self.states.pop(move)
	    self.availables.add(move)
	    num_taken += 1
	return num_taken

    def _within_boundary(self, h, w):
	if h >= 0 and h < self.height and w >= 0 and w < self.width:
	    return True
	else:
	    return False

    def _get_none(self, h, w):
	"""
	Same thing as Array.__getitem__, but returns None if coordinates are
	not within array dimensions.
	"""
	move = self.location_to_move([h, w])
	if not self._within_boundary(h, w):
	    return False
	elif not self.states.has_key(move):
	    return None
	else:
	    return self.states[move]

    def _get_surrounding(self, h, w):
        """
        Gets information about the surrounding locations for a specified
        coordinate.  Returns a tuple of the locations clockwise starting from
        the top.
        """
        coords = (
            (h, w - 1),
            (h + 1, w),
            (h, w + 1),
            (h - 1, w),
        )
        # find the surrounding for a location, and discard those out of the board locs
	# blank areas are labelled as None
        # an example output: [([1], (1, 1)), ([2], (2, 2)), ([1], (1, 3)), ([None], (0, 2))]
        return filter(lambda i: bool(i[0]!=False), [(self._get_none(h, w), (h, w))
					     for h, w in coords])

    def _get_group(self, h, w, traversed):
	"""
	Recursively traverses adjacent locations of the same color to find all
	locations which are members of the same group.
	"""
	# get the player of (h, w), i.e. black/white/blank
	move = self.location_to_move([h, w])
	if self.states.has_key(move):
	    player = self.states[move]
	else:
	    player = None

	# Get surrounding locations which have the same color and whose
	# coordinates have not already been traversed
	locations = [
	    (p, (a, b))
	    for p, (a, b) in self._get_surrounding(h, w)
	    if p is player and (a, b) not in traversed
	    ]

	# Add current coordinates to traversed coordinates
	traversed.add((h, w))
	print(traversed)

	# Find coordinates of similar neighbors
	if locations:
	    return traversed.union(*[self._get_group(a, b, traversed) for _, (a, b) in locations])
	else:
	    return traversed

    def _get_liberties(self, h, w, traversed):
        """
        Recursively traverses adjacent locations of the same color to find all
        surrounding liberties for the group at the given coordinates.
        Give a location -> if it's blank, then it's a liberty
        Give a location -> if it's a color, then find its surroundings
        -> if surrounding is the same color then find its surroundings, if it's blank then add it to liberty
        -> finally we get a set of liberties
        """
	# get the player of (h, w), i.e. black/white/blank
	move = self.location_to_move([h, w])
	if self.states.has_key(move):
	    player = self.states[move]
	else:
	    player = None

        if player is None:
            # Return coords of empty location (this counts as a liberty)
            return set([(h, w)])
        else:
            # Get surrounding locations which are empty or have the same color
            # and whose coordinates have not already been traversed
            locations = [
                (p, (a, b))
                for p, (a, b) in self._get_surrounding(h, w)
                if (p is player or p is None) and (a, b) not in traversed
            ]

            # Mark current coordinates as having been traversed
            traversed.add((h, w))

            # Collect unique coordinates of surrounding liberties
            if locations:
                return set.union(*[
                    self._get_liberties(a, b, traversed)
                    for _, (a, b) in locations
                ])
            else:
                return set()

    def count_liberties(self, h, w):
	"""
	Gets the number of liberties surrounding the group at the given
	coordinates.
	"""
	return len(self._get_liberties(h, w, set()))

    def game_end(self, is_shown):
	"""End the game and print the competition info"""
	num_player1, num_player2 = self.count_go_number()
	if num_player1 > num_player2:
	    winner = self.players[0]
	elif num_player2 > num_player1:
	    winner = self.players[1]
	else:
	    winner = None

	if is_shown:
	    if winner != None:
		print("Game end. The winner is", winner)
	    else:
		print("Game end. Tie")
	return winner

class Game(object):
    """game server"""

    def __init__(self, board):
	self.board = board

    def graphic(self, board, player1, player2):
	"""
	Draw the board and show game info
	player1, player2 are both ints, i.e. 1 or 2
	"""
	width = board.width
	height = board.height

	print("Player %s with ●" % player1)
	print("Player %s with ○" % player2)
	print("\n")

	print(' ', end='')
	for x in range(width):
	    print("{0:2}".format(x), end='')
	print('')

	for i in range(height):
	    print("{0:<2d}".format(i), end='')
	    for j in range(width):
		loc = i * width + j
		p = board.states.get(loc, -1)  # ?
		if p == player1:
		    print('●'.center(4), end='')
		elif p == player2:
		    print('○'.center(4), end='')
		else:
		    print(' '.center(2), end='')
	    print('')

    def start_play(self, player1, player2, is_shown=1): # what is  inside the player1?
	"""start a game between two players"""
	self.board.init_board()

	# steps limits settings
	height = self.board.height
	width = self.board.width
	steps_limit = 0.75*height*width
	current_step = 0

	# player settings
	p1, p2 = self.board.players  # remember board.players is only a list
	player1.set_player_ind(p1)   # player1 and player2 are objects
	player2.set_player_ind(p2)
	players = {p1: player1, p2: player2}

	# running the game
	if is_shown:
	    self.graphic(self.board, player1.player, player2.player)  # player1.player is an int, i.e. 1 or 2
	while True:
	    current_step += 1
	    current_player = self.board.get_current_player()
	    player_in_turn = players[current_player]
	    move = player_in_turn.get_action(self.board)
	    if move == "pass":
		winner = self.board.game_end(is_shown)
	    elif current_step > steps_limit:
		winner = self.board.game_end(is_shown)
	    else:
	        self.board.do_move(move)
	    if is_shown:
		self.graphic(self.board, player1.player, player2.player)

	return winner

    #####################################################################
    def start_self_play(self, player, is_shown=0, temp=1e-3): # ?
	""" start a self-play game using a MCTS player, reuse the search tree,
	and store the self-play data: (state, mcts_probs, z) for training
	"""
	self.board.init_board()
	p1, p2 = self.board.players
	states, mcts_probs, current_players = [], [], []
	while True:
	    move, move_probs = player.get_action(self.board,   # move: an int
						 temp=temp,
						 return_prob=1)
	    # store the data
	    states.append(self.board.current_state())
	    mcts_probs.append(move_probs)
	    current_players.append(self.board.current_player)
	    # perform a move
	    self.board.do_move(move)
	    if is_shown:
		self.graphic(self.board, p1, p2)
	    end, winner = self.board.game_end()

	    if end:
		winner_z = np.zeros(len(current_players))
		if winner != -1:
		    winner_z[np.array(current_players) == winner] = 1.0
		    winner_z[np.array(current_players) != winner] = -1.0
		#reset MCTS tree node
		player.reset_player()
		if is_shown:
		    if winner != -1:
			print("Game end. Winner is player:", winner)
		    else:
			print("Game end. Tie")
		return winner, zip(states, mcts_probs, winner_z)






















