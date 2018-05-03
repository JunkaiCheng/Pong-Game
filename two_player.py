from tkinter import *
import tkinter.messagebox
import time

import newBoard

infinity = 999
c = 500


class QLearning():
    def __init__(self, episodes=100000, n_threshold=50, lr_constant=50, discount_factor=0.8):
        self.board = newBoard.Board()
        self.q_table = dict()
        self.n_table = dict()
        self.episodes = episodes
        self.n_threshold = n_threshold
        self.lr_constant = lr_constant
        self.discount_factor = discount_factor

    def get_qvalue(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0, 0, 0]
        return self.q_table[state][action]

    def set_qvalue(self, state, action, value):
        self.q_table[state][action] = value

    def get_nvalue(self, state, action):
        if state not in self.n_table:
            self.n_table[state] = [0, 0, 0]
        return self.n_table[state][action]

    def update_ntable(self, state, action):
        self.n_table[state][action] += 1

    def exploration_function(self, state, action):
        if self.get_nvalue(state, action) <= self.n_threshold:
            return infinity
        else:
            return self.get_qvalue(state, action)

    def get_action(self, state):
        max_qvalue = -infinity
        max_action = -1
        for action in range(3):
            current_value = self.exploration_function(state, action)
            if current_value > max_qvalue:
                max_qvalue = current_value
                max_action = action
        return max_action

    def get_maxQ_action(self, state):
        max_qvalue = -infinity
        max_action = -1
        for action in range(3):
            current_value = self.get_qvalue(state, action)
            if current_value > max_qvalue:
                max_qvalue = current_value
                max_action = action
        return max_action


class Game:
    def __init__(self):
        self.qlearning = QLearning()

        q_table = dict()
        with open("Q-table.txt", "r") as f:
            for line in f:
                line = line.rstrip('\n')
                entry = line.split(":")
                q_table[eval(entry[0])] = eval(entry[1])
        self.qlearning.q_table = q_table
        n_table = dict()
        with open("N-table.txt", "r") as f:
            for line in f:
                line = line.rstrip('\n')
                entry = line.split(":")
                n_table[eval(entry[0])] = eval(entry[1])
        self.qlearning.n_table = n_table
        self.act2 = 0

        # self.qlearning.train()
        self.qlearning.board = newBoard.Board()
        self.root = Tk()
        self.root.title('Pong')
        self.table = Canvas(self.root, width=600, height=600, bg='white')
        self.table.create_rectangle(48, 48, 52 + c, 52 + c)

    def click(self, event):
        if event.y > 300:
            self.act2 = 1
        else:
            self.act2 = 2

    def run(self):
        while True:
            oldx = self.qlearning.board.ball_x
            oldy = self.qlearning.board.ball_y
            oldp = self.qlearning.board.paddle_y
            oldp2 = self.qlearning.board.paddle2_y

            current_state = self.qlearning.board.convert()
            current_action = self.qlearning.get_action(current_state)
            current_qvalue = self.qlearning.get_qvalue(current_state, current_action)
            self.qlearning.update_ntable(current_state, current_action)
            reward = self.qlearning.board.update_board(current_action, self.act2)

            next_state = self.qlearning.board.convert()
            next_action = self.qlearning.get_maxQ_action(next_state)
            next_qvalue = self.qlearning.get_qvalue(next_state, next_action)
            print("current state: ", current_state)
            print("next state: ", next_state, ", next action: ", next_action)

            lr = self.qlearning.lr_constant / (
                        self.qlearning.lr_constant + self.qlearning.get_nvalue(current_state, current_action))
            new_qvalue = current_qvalue + lr * (reward + self.qlearning.discount_factor * next_qvalue - current_qvalue)
            self.qlearning.set_qvalue(current_state, current_action, new_qvalue)

            self.draw(oldx, oldy, oldp, oldp2)

            if reward == -1:
                return

    def draw(self, oldx, oldy, oldp, oldp2):
        sleep_time = 0.1
        x = oldx
        y = oldy
        p = oldp
        p2 = oldp2
        p_ary = self.paddle(p)
        p_ary2 = self.paddle2(p2)
        ball_ary = self.ball(x, y)
        self.table.create_rectangle(p_ary[0], p_ary[1], p_ary[2], p_ary[3], fill="black", tag="pad")
        self.table.create_rectangle(p_ary2[0], p_ary2[1], p_ary2[2], p_ary2[3], fill="black", tag="pad2")
        self.table.create_oval(ball_ary[0], ball_ary[1], ball_ary[2], ball_ary[3], fill="red", tag="ball")
        self.table.update()
        time.sleep(sleep_time)
        self.table.delete("pad")
        self.table.delete("pad2")
        self.table.delete("ball")

    def ball(self, x, y):
        return self.getx(x) - 2, self.gety(y) - 2, self.getx(x) + 2, self.gety(y) + 2

    def paddle(self, y):
        return 48 + c, 50 + y * c, 52 + c, 50 + (y + 0.2) * c

    def paddle2(self, y):
        return 48, 50 + y * c, 52, 50 + (y + 0.2) * c

    def getx(self, x):
        return 50 + x * c

    def gety(self, y):
        return 50 + y * c

    def start(self):
        self.table.pack()
        self.table.bind("<Button-1>", self.click)
        self.run()
        # self.chess_board.bind("<Button-1>", self.click)
        # Button(self.root, text="next", command=self.next).place(x=200, y=410)
        self.root.mainloop()


if __name__ == '__main__':
    game = Game()
    game.start()
