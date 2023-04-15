import sys
import TestingVars
import Constants
import hmm2
import numpy as np


def extractingData(dataPath):
    tmp = []
    with open(dataPath, 'r') as file:
        while line := file.readline():
            lst = list(line.strip())
            tmp.append(lst)
    return tmp


def initialization(initial_path, idx):
    initial_path = 'output/' + initial_path
    with open(initial_path, 'r') as file:
        while line := file.readline():
            if 'initial' in line:
                line = list(map(str.strip, line.split(":")))
                TestingVars.setN(int(line[1]), idx)
                line = file.readline()
                TestingVars.setPi(list(map(float, line.strip().split())), idx)

            elif 'transition' in line:
                A = []
                line = list(map(str.strip, line.split(":")))
                for _ in range(int(line[1])):
                    line = file.readline()
                    A.append(list(map(float, line.strip().split())))
                TestingVars.setA(A, idx)

            elif 'observation' in line:
                B = []
                line = list(map(str.strip, line.split(":")))
                TestingVars.setM(int(line[1]), idx)
                for _ in range(TestingVars.M[idx]):
                    line = file.readline()
                    B.append(list(map(float, line.strip().split())))
                TestingVars.setB(B, idx)
    return


def extract_models(path):
    models = []
    with open(path, 'r') as f:
        while line := f.readline():
            models.append(line[:-1])
    return models


def setting_Constants(idx):
    Constants.setN(TestingVars.N[idx])
    Constants.setM(TestingVars.M[idx])
    Constants.setA(TestingVars.A[idx])
    Constants.setB(TestingVars.B[idx])
    Constants.setPi(TestingVars.Pi[idx])
    return


def finding_ans(data, models):
    result = []
    for i in range(len(data)):
        lst = []
        test = data[i]
        for j in range(len(models)):
            setting_Constants(j)
            prob, _ = hmm2.viterbi(test)
            lst.append(prob)
        lst = np.array(lst)
        result.append([np.amax(lst), np.argmax(lst)])
    return result


def save_ans(output, result):
    newline = '\n'
    with open(output, 'w') as f:
        for x in result:
            prob = x[0]
            index = x[1] + 1
            if index == 3:
                index = 5
            string = 'model_0' + str(index) + '.txt '
            f.write(string)
            f.write(str(prob))
            f.write(newline)
    return


def find_accuracy():
    y_path = 'hmm_data/testing_answer.txt'
    y_hat_path = 'output/result1.txt'
    indices = set()
    y = []
    y_hat = []
    i = -1
    with open(y_path, 'r') as f1:
        while line := f1.readline():
            i += 1
            if '04' in line or '03' in line:
                continue
            else:
                y.append(line[:-1])
                indices.add(i)
    j = -1
    with open(y_hat_path, 'r') as f2:
        while line := f2.readline():
            j += 1
            if j in indices:
                y_hat.append(line[:-1].split()[0])

    true = 0
    for k in range(len(y)):
        if y[k] == y_hat[k]:
            true += 1
    print(i)
    print('accuracy =', true/len(y))
    print('ovarall accuracy =', true / i)
    return

if __name__ == "__main__":
    assert len(sys.argv) == 4, "Number of Args must be 4!"
    args = sys.argv
    name_of_models = args[1]
    data_path = args[2]
    output_model = args[3]
    test_data = extractingData(data_path)
    models = extract_models(name_of_models)
    models = [models[0], models[1], models[-1]]
    for i in range(len(models)):
        initialization(models[i], i)
    res = finding_ans(test_data, models)
    save_ans(output_model, res)
    find_accuracy()
    # print(res)