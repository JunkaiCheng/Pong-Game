'''
x = 0, y = 0, y = 1 are the boarder of the game
paddle move along x = 1
paddle_height = 0.2, paddle_y = top of the paddle
ball bounce off the wall/paddle when it hits

initial state:
board = Board(0.5, 0.5, 0.03, 0.01, 0.5 - paddle_height / 2)

action:
{stay, paddle_y += 0.4, paddle_y -= 0.4}
if paddle go top off the screen : paddle_y = 0
if paddle go bottom off the screen : paddle_y = 1 - paddle_height

reward:
+1 if rebounding the ball off the paddle
-1 if ball pass x = 1
0 else

terminate:
ball pass x = 1
'''

import numpy as np
import math


class Board():
    def __init__(self, ball_x=0.5, ball_y=0.5, velocity_x=0.03, velocity_y=0.01, paddle_y=0.5, paddle2_y=0.5):
        self.paddle_height = 0.2
        self.ball_x = ball_x
        self.ball_y = ball_y
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.paddle_y = paddle_y - self.paddle_height / 2
        self.paddle2_y = paddle2_y - self.paddle_height / 2
        self.action_list = [0, +0.04, -0.04]

    '''
    Parameter:
    action - 0 if stay, 1 if move down 0.04, 2 if move up 0.04

    The return value is the reward of the current step
    '''

    def update_board(self, action1, action2):
        # update the paddle coordinate
        self.paddle_y += self.action_list[action1]
        if self.paddle_y >= 1 - self.paddle_height:
            self.paddle_y = 1 - self.paddle_height
        elif self.paddle_y <= 0:
            self.paddle_y = 0
        self.paddle2_y += self.action_list[action2]
        if self.paddle2_y >= 1 - self.paddle_height:
            self.paddle2_y = 1 - self.paddle_height
        elif self.paddle2_y <= 0:
            self.paddle2_y = 0

        self.ball_x += self.velocity_x
        self.ball_y += self.velocity_y
        # off top
        if self.ball_y < 0:
            self.ball_y = -self.ball_y
            self.velocity_y = -self.velocity_y
        # off bottom
        if self.ball_y > 1:
            self.ball_y = 2 - self.ball_y
            self.velocity_y = -self.velocity_y
        # off left edge of the game
        if self.ball_x < 0:
            # the ball misses the player's paddle, agent wins
            if self.ball_y <= self.paddle2_y or self.ball_y >= self.paddle2_y + self.paddle_height:
                return -1
            else:
                u = np.random.uniform(-0.015, 0.015)
                v = np.random.uniform(-0.03, 0.03)
                # making sure |velocity_x| > 0.03 and |velocity_x| < 1
                while abs(-self.velocity_x + u) <= 0.03 or abs(-self.velocity_x + u) >= 1:
                    u = np.random.uniform(-0.015, 0.015)
                # making sure |velocity_y| < 1
                while abs(self.velocity_y + v) >= 1:
                    u = np.random.uniform(-0.015, 0.015)
                self.ball_x = -self.ball_x
                self.velocity_x = -self.velocity_x + u
                self.velocity_y = self.velocity_y + v
                return 1
        # the ball is still within the boundary
        if self.ball_x < 1:
            return 0
        # the ball misses the paddle, player wins
        if self.ball_y <= self.paddle_y or self.ball_y >= self.paddle_y + self.paddle_height:
            return -1
        # the ball hits the paddle
        else:
            self.ball_x = 2 - self.ball_x
            u = np.random.uniform(-0.015, 0.015)
            v = np.random.uniform(-0.03, 0.03)
            # making sure |velocity_x| > 0.03 and |velocity_x| < 1
            while abs(-self.velocity_x + u) <= 0.03 or abs(-self.velocity_x + u) >= 1:
                u = np.random.uniform(-0.015, 0.015)
            # making sure |velocity_y| < 1
            while abs(self.velocity_y + v) >= 1:
                u = np.random.uniform(-0.015, 0.015)
            self.velocity_x = -self.velocity_x + u
            self.velocity_y = self.velocity_y + v
            return 1

    def convert(self):
        # The special state where the ball is out of boundary
        if (self.ball_x > 1):
            return (12, 12, 12, 12, 12)

        # The new ball_x, ball_y, paddle_y should be in the range of [0, 11]
        # The new velocity_x should be 1 or -1, new velocity_y should be 1, 0 or -1
        new_ball_x = int(math.floor(12 * self.ball_x))
        if new_ball_x == 12:
            new_ball_x = 11
        new_ball_y = int(math.floor(12 * self.ball_y))
        if new_ball_y == 12:
            new_ball_y = 11
        new_velocity_x = int(abs(self.velocity_x) / self.velocity_x)
        new_velocity_y = int(abs(self.velocity_y) / self.velocity_y)
        if abs(self.velocity_y) < 0.015:
            new_velocity_y = 0
        new_paddle_y = int(math.floor(12 * self.paddle_y / (1 - self.paddle_height)))
        if (self.paddle_y == 1 - self.paddle_height):
            new_paddle_y = 11

        return (new_ball_x, new_ball_y, new_velocity_x, new_velocity_y, new_paddle_y)
