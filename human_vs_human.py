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
        the acceptable input form:
        """
        x1 = int(input("Your move(line):"))
        x2 = int(input("Your move(column):"))
        move = board.location_to_move([x1, x2])
        return move

    def __str__(self):
        return "Human {}".format(self.player)

def run():
    board = Board(width=4, height=4)
    game = Game(board)

    human_player1 = Human()
    human_player2 = Human()

    game.start_play(human_player1, human_player2, is_shown=1)

if __name__ == '__main__':
    run()

