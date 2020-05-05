"""
Invokes genderPredict as module
"""
from man import Man
from neural_net import NeuralNet
from neuron import Neuron


def file_parse(path):
    men = []
    with open(path, "r") as file:
        data = file.readlines()

        for i in range(1, 8000):
            split_data = data[i].split(",")
            men.append(Man(split_data[2], split_data[1],
                           split_data[0]))

    return men


def main():
    data = []
    men = file_parse("../data/task1.csv")
    nn = NeuralNet(2, 3, 1)

    predict = 0
    for m in men:
        res = 1 if nn.ff([m.height, m.weight])[0] > 0.5 else 0
        print(nn.ff([m.height, m.weight])[0])
        predict += 1 if res == m.gender else 0

    print("result ::", predict)


if __name__ == "__main__":
    main()
