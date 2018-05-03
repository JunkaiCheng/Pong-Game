'''
Each row in expert policy:
ball_x, ball_y, velocity_x, velocity_y, paddle_y, action

Action:
0 - move up
1 - stay
2 - down

This program will build a deep network that classidies states into action classes
Minibatch Gradient-descent is used to train this network

Four Layer neural network with 256 units per layer, with the last layer of 3 units
'''

from random import shuffle
import numpy as np
import math


def readData(inputFilePath):
    file = open(inputFilePath, "r")
    lines = file.readlines()
    boardData = list()
    choiceData = list()
    for line in lines:
        cache = line.strip('\n')
        cache = line.split(' ')
        board_temp = list()
        for index, item in enumerate(cache):
            if index != 5:
                board_temp.append(float(item))
            else:
                choiceData.append(float(item))
        boardData.append(board_temp)
    return boardData, choiceData


class Cloning():
    def __init__(self, data, expert_choice, batch_num=1):
        self.data = data
        self.expert_choice = expert_choice
        self.batch_num = batch_num
        self.batch, self.batch_target = list(), list()
        self.weight, self.bias = list(), list()
        self.scale_constant = 1 / 256
        self.lr = 0.1
        self.feature = 5
        self.hiddenunit = 256
        self.outputunit = 3
        self.layer = 4

    def weightInitialize(self):
        output_weight = list()
        for count in range(self.layer):
            if count == 0:
                output_weight.append(
                    [[np.random.uniform(0, 1) * self.scale_constant for i in range(self.hiddenunit)] for j in
                     range(self.feature)])
            elif count == self.layer - 1:
                output_weight.append(
                    [[np.random.uniform(0, 1) * self.scale_constant for i in range(self.outputunit)] for j in
                     range(self.hiddenunit)])
            else:
                output_weight.append(
                    [[np.random.uniform(0, 1) * self.scale_constant for i in range(self.hiddenunit)] for j in
                     range(self.hiddenunit)])
        return output_weight

    def biasInitialize(self):
        output_bias = list()
        for count in range(self.layer):
            if count == self.layer - 1:
                output_bias.append([0 for i in range(self.outputunit)])
            else:
                output_bias.append([0 for i in range(self.hiddenunit)])
        return output_bias

    def splitData(self):
        self.batch, self.batch_target = list(), list()
        combination = list(zip(self.data, self.expert_choice))
        shuffle(combination)
        self.data, self.expert_choice = zip(*combination)
        self.data = np.asarray(self.data)
        self.expert_choice = np.asarray(self.expert_choice)
        total = len(self.data)
        batch_size = int(math.ceil(total / self.batch_num))
        for i in range(0, total, batch_size):
            if i + batch_size >= total:
                batch = self.data[i: total]
                choice = self.expert_choice[i: total]
            else:
                batch = self.data[i: i + batch_size]
                choice = self.expert_choice[i: i + batch_size]
            self.batch.append(np.asarray(batch))
            self.batch_target.append(np.asarray(choice))

    def affine_forward(self, A, W, b):
        acache = (A, W, b)
        new_A = np.asarray(A)
        new_W = np.asarray(W)
        new_B = np.tile(np.asarray(b), (len(A), 1))
        Z = np.add(np.matmul(new_A, new_W), new_B)
        return (Z.tolist(), acache)

    def affine_backward(self, dZ, acache):
        A, W, b = acache
        new_A = np.asarray(A)
        new_W = np.asarray(W)
        new_dZ = np.asarray(dZ)
        dA = np.matmul(new_dZ, np.transpose(new_W))
        dW = np.matmul(np.transpose(new_A), new_dZ)
        db = np.matmul(np.transpose(new_dZ), np.ones(new_A.shape[0]))

        return (dA.tolist(), dW.tolist(), db.tolist())

    def relu_forward(self, Z):
        new_Z = np.asarray(Z)
        output = np.maximum(new_Z, 0)
        return (output.tolist(), Z)

    def relu_backward(self, dA, rcache):
        cache = np.asarray(rcache)
        dZ = np.array(dA, copy=True)
        dZ[cache < 0] = 0
        return dZ.tolist()

    def cross_entropy(self, F, y):
        new_F = np.asarray(F)
        # new_F -= np.max(new_F)
        L = 0
        for i in range(len(F)):
            L -= new_F[i][int(y[i])] - math.log(np.sum(np.exp(new_F[i])))
        L /= len(F)
        dF = []
        for i in range(len(F)):
            row = []
            exp_sum = np.sum(np.exp(new_F[i]))
            for j in range(len(F[0])):
                value = int(j == y[i]) - math.exp(new_F[i][j]) / exp_sum
                value /= -len(F)
                row.append(value)
            dF.append(row)
        return (L, dF)

    def four_network(self, x, y, test):
        # print (self.weight[0])
        z1, acache1 = self.affine_forward(x, self.weight[0], self.bias[0])
        # print (z1[1] == z1[-2])
        a1, rcache1 = self.relu_forward(z1)
        # print (a1[1] == a1[-1])
        z2, acache2 = self.affine_forward(a1, self.weight[1], self.bias[1])
        # print (z2[1] == z2[-2])
        a2, rcache2 = self.relu_forward(z2)
        z3, acache3 = self.affine_forward(a2, self.weight[2], self.bias[2])
        a3, rcache3 = self.relu_forward(z3)
        f, acache4 = self.affine_forward(a3, self.weight[3], self.bias[3])
        print (f)
        if test:
            classification = list()
            # print (f)
            for item in f:
                classification.append(item.index(max(item)))
            return classification

        loss, df = self.cross_entropy(f, y)

        da3, dw4, db4 = self.affine_backward(df, acache4)
        dz3 = self.relu_backward(da3, rcache3)
        da2, dw3, db3 = self.affine_backward(dz3, acache3)
        dz2 = self.relu_backward(da2, rcache2)
        da1, dw2, db2 = self.affine_backward(dz2, acache2)
        dz1 = self.relu_backward(da1, rcache1)
        dx, dw1, db1 = self.affine_backward(dz1, acache1)

        dw = [dw1, dw2, dw3, dw4]
        # print ("dw is:",dw[0])
        db = [db1, db2, db3, db4]
        # print ("db is:",db)

        # use gradient descent to update parameter
        for i in range(self.layer):
            conversion_w = np.asarray(self.weight[i])
            conversion_dw = np.asarray(self.lr * np.asarray(dw[i]))
            result = np.subtract(conversion_w, conversion_dw)
            self.weight[i] = result.tolist()

            conversion_b = np.asarray(self.bias[i])
            conversion_db = np.asarray(self.lr * np.asarray(db[i]))
            result = np.subtract(conversion_b, conversion_db)
            self.bias[i] = result.tolist()

        return loss

    def MinibatchGD(self, epochs):
        # initialize weight and bias
        self.weight = self.weightInitialize()
        self.bias = self.biasInitialize()

        for epoch in range(epochs):
            self.splitData()
            loss = list()
            for count in range(self.batch_num):
                x = self.batch[count]
                y = self.batch_target[count]
                loss.append(self.four_network(x, y, False))
            print ("Finished epoch", epoch + 1)
            print ("The loss is:", sum(loss) / float(len(loss)))

        result = self.four_network(self.data, self.expert_choice, True)
        # print (result)
        error = 0
        for i in range(len(self.expert_choice)):
            if result[i] != self.expert_choice[i]:
                error += 1
        print ("The accuracy for epoch is", 1 - (error / 10000))


if __name__ == "__main__":
    batchData, batchTarget = readData("expert_policy.txt")
    cloning = Cloning(np.asarray(batchData), np.asarray(batchTarget))
    cloning.MinibatchGD(20)
