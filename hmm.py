import Constants
import numpy as np

res = []


def getObservationIndex(obs):
    if Constants.IndexOfWords.get(obs):
        return Constants.IndexOfWords[obs]
    else:
        return Constants.IndexOfWords['oov']


def beta(obs):
    Beta = np.zeros((Constants.N, len(obs)))
    Beta[:, len(obs) - 1] = 1
    for t in range(Beta.shape[1] - 2, -1, -1):
        for i in range(Beta.shape[0]):
            tmp = np.multiply(Beta[:, t+1], Constants.A[i, :])
            Beta[i, t] = np.sum(np.multiply(tmp, Constants.B[getObservationIndex(obs[t+1]), :]))
    return Beta


def alpha(obs):
    Alpha = np.zeros((Constants.N, len(obs)))
    Alpha[:, 0] = Constants.Pi
    for t in range(1, Alpha.shape[1]):
        for i in range(Alpha.shape[0]):
            x = np.multiply(Alpha[:, t - 1], Constants.A[:, i])
            Alpha[i, t] = np.sum(np.multiply(Alpha[:, t - 1], Constants.A[:, i]))
            Alpha[i, t] = np.multiply(Alpha[i, t], Constants.B[getObservationIndex(obs[t]), i])
    return Alpha


def delta_psi(obs):
    Delta = np.zeros((Constants.N, len(obs)))
    # Psi = np.zeros((Constants.N, len(obs)))
    Psi = np.full((Constants.N, len(obs)), -1)
    Delta[:, 0] = np.multiply(Constants.Pi, Constants.B[getObservationIndex(obs[0]), :])
    for t in range(1, Delta.shape[1]):
        for i in range(Delta.shape[0]):
            tmp = np.multiply(Delta[:, t - 1], Constants.A[:, i])
            Delta[i, t] = np.amax(tmp)
            Delta[i, t] = np.multiply(Delta[i, t], Constants.B[getObservationIndex(obs[t]), i])
            Psi[i, t] = np.argmax(tmp)
    return Delta, Psi


def forward(obs):
    Alpha = alpha(obs)
    return np.sum(Alpha[:, len(obs) - 1])


def backward(obs):
    Beta = beta(obs)
    return np.sum(np.multiply(Beta[:, 0], Constants.Pi))


def find_path(Delta, Psi):
    index = np.argmax(Delta[:, Delta.shape[1] - 1])
    path = [Constants.ListOfTags[index]]
    for t in range(Delta.shape[1] - 1, 0, -1):
        index = Psi[index, t]
        state = Constants.ListOfTags[index]
        path.insert(0, state)
    return path


def viterbi(obs):
    Delta, Psi = delta_psi(obs)
    path = find_path(Delta, Psi)
    return np.amax(Delta[:, Delta.shape[1] - 1]), path


def gamma(obs, Alpha, Beta):
    Gamma = np.zeros((Constants.N, len(obs)))
    for t in range(Gamma.shape[1]):
        Gamma[:, t] = np.multiply(Alpha[:, t], Beta[:, t])
        Gamma[:, t] /= np.sum(Gamma[:, t])
    return Gamma


def xi(obs, Alpha, Beta):
    T = len(obs)
    Xi = np.zeros((T - 1, Constants.N, Constants.N))
    for t in range(Xi.shape[0]):
        for j in range(Xi.shape[2]):
            x = np.multiply(Alpha[:, t], Constants.A[:, j])
            x = x * Constants.B[getObservationIndex(obs[t+1]), j] * Beta[j, t+1]
            Xi[t, :, j] = x
        s = np.sum(Xi[t, :, :])
        Xi[t, :, :] /= s
    return Xi


def tmp(Obs):
    for i in range(len(Obs)):
        print(Obs[i])
        obs = Obs[i]
        Alpah = alpha(obs)
        Beta = beta(obs)
        Gamma = gamma(obs, Alpah, Beta)
        Xi = xi(obs, Alpah, Beta)
        for t in range(0, len(obs) - 3):
            for i in range(Constants.N):
                print("i=", i, "   t=", t)
                print(Gamma[i, t])
                print(np.sum(Xi[t, i, :]))
        break
