import sys
import hmm2
import Constants
import numpy as np
import copy

tmp_a = []
tmp_b = []
tmp_pi = []


def initialization(initial_path):
    with open(initial_path, 'r') as file:
        while line := file.readline():
            if 'initial' in line:
                line = list(map(str.strip, line.split(":")))
                Constants.setN(int(line[1]))
                line = file.readline()
                Constants.setPi(list(map(float, line.strip().split())))

            elif 'transition' in line:
                A = []
                line = list(map(str.strip, line.split(":")))
                for _ in range(int(line[1])):
                    line = file.readline()
                    A.append(list(map(float, line.strip().split())))
                Constants.setA(A)

            elif 'observation' in line:
                B = []
                line = list(map(str.strip, line.split(":")))
                Constants.setM(int(line[1]))
                for _ in range(Constants.M):
                    line = file.readline()
                    B.append(list(map(float, line.strip().split())))
                Constants.setB(B)
    return


def extractingData(dataPath):
    tmp = []
    with open(dataPath, 'r') as file:
        while line := file.readline():
            lst = list(line.strip())
            tmp.append(lst)
    return tmp



def update_model(length_of_data):
    Constants.setPi(tmp_pi)
    Constants.setA(tmp_a / length_of_data)
    Constants.setB(tmp_b / length_of_data)

    return

def make_new_model(Alpha, Beta, Gamma, Xi, obs):
    global tmp_a
    global tmp_b
    global tmp_pi


    # Pi
    pi = Gamma[:, 0]
    # Constants.setPi(copy.deepcopy(pi))
    tmp_pi = copy.deepcopy(pi)

    # A
    denominator = Gamma.sum(axis=1)
    # a = np.zeros(Constants.A.shape)
    for j in range(Constants.N):
        nominator = Xi[:, :, j].sum(axis=0)
        # a[:, j] = nominator / denominator
        tmp_a[:, j] += (nominator / denominator)
    # Constants.setA(copy.deepcopy(a))


    # B
    denominator = Gamma.sum(axis=1)
    # b = np.zeros(Constants.B.shape)
    for k in range(Constants.M):
        mask = []
        for t in range(len(obs)):
            if ord(obs[t]) - 65 == k:
                mask.append([t])

        nominator = np.zeros((Constants.N, 1))
        for t in mask:
            nominator += Gamma[:, t]
        nominator = nominator.squeeze()
        # b[k, :] = nominator / denominator
        tmp_b[k, :] += (nominator / denominator)
    # Constants.setB(b)
    return


def prepare_lines():
    newline = '\n'
    lines = []
    initial = 'initial: 6'
    transition = 'transition: 6'
    observation = 'observation: 6'
    lines.append(initial)
    lines.append(newline)
    res = " ".join([str(i) for i in Constants.Pi])
    lines.append(res)
    lines.append(newline)
    lines.append(newline)
    lines.append(transition)
    lines.append(newline)
    for i in range(Constants.A.shape[0]):
        lst = Constants.A[i, :]
        res = " ".join([str(i) for i in lst])
        lines.append(res)
        lines.append(newline)
    lines.append(newline)
    lines.append(observation)
    lines.append(newline)
    for i in range(Constants.B.shape[0]):
        lst = Constants.B[i, :]
        res = " ".join([str(i) for i in lst])
        lines.append(res)
        lines.append(newline)
    return lines


def saving_model(output):
    with open(output, 'w') as f:
        lines = prepare_lines()
        f.writelines(lines)
    return


def initialize_tmp_data():
    global tmp_a
    global tmp_b
    global tmp_pi

    tmp_a = np.zeros(Constants.A.shape)
    tmp_b = np.zeros(Constants.B.shape)
    tmp_pi = np.zeros(Constants.Pi.shape)
    return


def train(no_iterations, data, batch_size=50):
    for iter in range(no_iterations):
        print("ok1")
        initialize_tmp_data()
        for i in range(len(data)):
            obs = data[i]
            Alpha = hmm2.alpha(obs)
            Beta = hmm2.beta(obs)
            Gamma = hmm2.gamma(obs, Alpha, Beta)
            Xi = hmm2.xi(obs, Alpha, Beta)
            # print("ok2")
            make_new_model(Alpha, Beta, Gamma, Xi, obs)
            if i % batch_size == 0:
                update_model(batch_size)
                initialize_tmp_data()
            # print("ok3")
        # update_model(len(data))
    return


if __name__ == "__main__":
    assert len(sys.argv) == 5, "Number of Args must be 4!"
    args = sys.argv
    no_iterations = int(args[1])
    model_init = args[2]
    train_data = args[3]
    output_model = args[4]
    initialization(model_init)
    data = extractingData(train_data)
    # print(len(data))
    # exit()
    train(no_iterations, data)
    saving_model(output_model)

    # hmm2.tmp(data)
    # hmm.tmp(data)




