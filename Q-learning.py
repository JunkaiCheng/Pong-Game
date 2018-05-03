import Board

infinity = 999

class QLearning():
    def __init__(self, episodes = 100000, n_threshold = 50, lr_constant = 50, discount_factor = 0.8):
        self.board = Board.Board()
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

    def run(self):
        bounces = 0
        while True:
            current_state = self.board.convert()
            current_action = self.get_action(current_state)
            current_qvalue = self.get_qvalue(current_state, current_action)
            self.update_ntable(current_state, current_action)
            reward = self.board.update_board(current_action)

            next_state = self.board.convert()
            next_action = self.get_maxQ_action(next_state)
            next_qvalue = self.get_qvalue(next_state, next_action)

            lr = self.lr_constant / (self.lr_constant + self.get_nvalue(current_state, current_action))
            new_qvalue = current_qvalue + lr * (reward + self.discount_factor * next_qvalue - current_qvalue)
            self.set_qvalue(current_state, current_action, new_qvalue)

            if reward == -1:
                return bounces

            bounces += reward

    def train(self):
        for episode in range(self.episodes):
            self.board = Board.Board()
            bounces = self.run()
            if (episode + 1) % 1000 == 0:
                print ("For episode", episode + 1, "ball bounces", bounces, "times")

if __name__ == "__main__":
    qlearning = QLearning()
    print ("=======Training=======")
    qlearning.train()
    print ("=======Testing=======")
    average_bounce = 0
    for i in range(200):
        qlearning.board = Board.Board()
        average_bounce += qlearning.run()/200
    print ("The average bounce is", average_bounce)
    print ("Saving Q-table now...")
    with open("Q-table.txt", "w") as f:
        for item in qlearning.q_table:
            f.write(str(item) + ":" + str(qlearning.q_table[item]) + "\n")
    # q_table = dict()
    # with open("Q-table.txt", "r") as f:
    #     for line in f:
    #         line = line.rstrip('\n')
    #         entry = line.split(":")
    #         q_table[eval(entry[0])] = eval(entry[1])
