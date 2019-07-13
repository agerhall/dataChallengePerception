import rlcompleter, readline
readline.parse_and_bind("tab: complete")

"""
important: change the first row in standard.loglikelihood and standard.filter to 'Z = np.atleast_2d(X)'.
"""

import pandas as pd
import numpy as np
import pykalman as pk
import itertools as it

dataroot = "../../../dataChallenge"

def load_data(foldername):
    filename = "groundTruth.txt"
    data = pd.read_csv(dataroot + '/' + foldername + '/' + filename, sep=" ", header=None)
    data = data.values  # numpy array
#    data = data.astype(float)  # ask for non-rounding arithmetic
    imax, jmax = data.shape
    mask = [[1 if data[i][j] == -1 else 0 for j in range(jmax)] for i in range(imax)]
    data = np.ma.masked_array(data, mask=mask)  # explicitly mask the missing values
    return data

def by_frame(data):
    maxframe = data[:,0].max()
    return [data[data[:,0] == i+1] for i in range(maxframe)]

def by_person(data):
    maxid = data[:,1].max()
    return [data[data[:,1] == i+1] for i in range(maxid)]

data = load_data('Scenario03-01')
xmax, ymax = 800.0, 600.0

class KalmanJoint:
    def __init__(self, var):
        self.kf = pk.KalmanFilter(
            transition_matrices=np.array([[1.0,0.0],[0.0,1.0]]),
            transition_offsets=np.array([0.0,0.0]),
            transition_covariance=np.array([[var[0],0.0],[0.0,var[1]]]),
            observation_matrices=np.array([[1.0,0.0],[0.0,1.0]]),
            observation_offsets=np.array([0.0,0.0]),
            observation_covariance=np.array([[0.0,0.0],[0.0,0.0]]),  #np.array([[var[0],0.0],[0.0,var[1]]]), #np.array([[0.0,0.0],[0.0,0.0]]),
            initial_state_mean=np.array([xmax/2, ymax/2]),
            initial_state_covariance=np.array([[xmax**2/4,0.0],[0.0,ymax**2/4]]))

    def likelihood(self, observation):
        """ just compute the likelihood of a given next observation """
        x, y = observation
        if x == np.ma.masked or y == np.ma.masked:
            return 0
        return self.kf.loglikelihood([observation])

    def commit(self, observation):
        """ mutate the state based on the observation """
        x, y = observation
        if x == np.ma.masked or y == np.ma.masked:
            mean, var = pk.standard._filter_predict(
                self.transition_matrix, self.transition_covariance, self.transition_offset,
                self.initial_state_mean, self.initial_state_covariance)
        else:
            [mean], [var] = self.kf.filter([observation])
        self.kf.initial_state_mean = mean
        self.kf.initial_state_covariance = var
        return mean

    def mean(self):
        return self.kf.initial_state_mean

def view_joint(observation, i):
    return observation[3+2*i:5+2*i]

class KalmanBody:
    def __init__(self, var):
        self.joints = [KalmanJoint(view_joint(var, i)) for i in range(18)]

    def likelihood(self, observation):
        #return sum(self.joints[i].likelihood(view_joint(observation, i)) for i in range(18))
        terms = [self.joints[i].likelihood(view_joint(observation, i)) for i in range(18)]
        return sum(sorted(terms)[10:])


    def commit(self, observation):
        for i in range(18):
            self.joints[i].commit(view_joint(observation, i))

    def means(self):
        return [i.mean() for i in self.joints]

def train(data):
    return np.mean([np.diff(i, axis=0).var(axis=0) for i in by_person(data)], axis=0)

class KalmanRoom:
    def __init__(self, n_people, trained):
        self.frame = 0
        self.people = [KalmanBody(trained) for _ in range(n_people)]

    def filter(self, frame):
        matching = lambda inj: zip(self.people, frame[inj,:])
        score = lambda inj: sum(body.likelihood(obs) for body, obs in matching(inj))
        inj = max(it.permutations(range(len(frame))), key=score)
        print([np.log(-score(i)) for i in it.permutations(range(len(frame)))])
        for body, observation in matching(inj):
            body.commit(observation)
        return inj

    def means(self):
        return [i.means() for i in self.people]

frames = by_frame(data)

"""
kr = KalmanRoom(3, train(data))
kb = kr.people[0]
kj = kb.joints[0]

frame = frames[0]
obs = frame[0]
jnt = view_joint(obs, 0)
"""

kr = KalmanRoom(3, train(data))
injs = []
for i, frame in enumerate(frames):
    inj = kr.filter(frame)
    injs.append(inj)
    print(i, inj)


def writeResults(perms, foldername, name):
    out = []
    for frame, perm in zip(by_frame(np.array(data)), perms):
        tmp = frame.copy()
        tmp[:,1] = perm
        tmp[:,2] = [1, 1, 1]
        out += list(tmp)
    f = open(dataroot + '/' + foldername + '/' + 'potentialTruth' + name + '.txt', 'w')
    f.write('\n'.join(' '.join(str(i) for i in j) for j in out))
    f.close()

#perms = eval(open('tmp2.txt').read())
#writeResults(perms, 'Scenario03-01', 'Kalman0')