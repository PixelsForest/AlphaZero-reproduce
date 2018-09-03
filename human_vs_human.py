# -*- coding: utf-8 -*-
from __future__ import print_function
import os
from game_go import Board, Game

os.environ["CUDA_VISIBLE_DEVICES"]=""

class Human(object):
    """human player"""

    def __init__(self):
	self.player = None

    def set_player_ind(self, p):
	self.player = p

    def get_action(self, board):
	"""
	the acceptable input form: [1, 1], (1, 1), "pass", 'pass'
	"""
	location = input("Your move:")
	if not isinstance(location, str):
	    move = board.location_to_move(location)
	    return move
	else:
	    # "pass"
	    return location

    def __str__(self):
	return "Human {}".format(self.player)

def run():
    board = Board()
    game = Game(board)

    human_player1 = Human()
    human_player2 = Human()

    game.start_play(human_player1, human_player2, is_shown=1)

if __name__ == '__main__':
    run()

