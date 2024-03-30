import gzip
from collections import defaultdict
import math
import scipy.optimize
from sklearn import svm
import numpy as np
import string
import random
import string
from sklearn import linear_model
import tensorflow as tf
def assertFloat(x):
    assert type(float(x)) == float

def assertFloatList(items, N):
    assert len(items) == N
    assert [type(float(x)) for x in items] == [float]*N
def readGz(path):
    for l in gzip.open(path, 'rt'):
        yield eval(l)
def readJSON(path):
    f = gzip.open(path, 'rt')
    f.readline()
    for l in f:
        d = eval(l)
        u = d['userID']
        g = d['gameID']
        yield u,g,d

## Setting up variables and reading file into train and validation sets
allHours = []
for l in readJSON("train.json.gz"):
    allHours.append(l)
hoursTrain = allHours[:165000]
hoursValid = allHours[165000:]
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
for u,g,d in hoursTrain:
    r = d['hours_transformed']
    hoursPerUser[u].append((g,r))
    hoursPerItem[g].append((u,r))
##################################################
# Play prediction                                #
##################################################
# Generate a negative set

validation_users = [d[0] for d in hoursValid]
neg_valid = []
gamesPerUser = defaultdict(list)

for i in allHours:
    u, g = i[0], i[1]
    gamesPerUser[u].append(g)

allgames = set([d[1] for d in allHours])
for u in validation_users:
    notplayed = list(allgames - set(gamesPerUser[u]))
    neg_game = random.choice(notplayed)
    while (u, neg_game, 0) in neg_valid:
        neg_game = random.choice(notplayed)
    neg_valid.append((u, neg_game, 0))


playedValid = set()
for u,g,r in hoursValid:
    playedValid.add((u,g))
def Jaccard(s1, s2):
    numer = len(s1.intersection(s2))
    denom = len(s1.union(s2))
    if denom > 0:
        return numer/denom
    return 0

########################################
### TASK 1 Would Play classification ###
########################################

userIDs,itemIDs = {},{}

for d in hoursTrain:
    u,i = d[0],d[1]
    if not u in userIDs: userIDs[u] = len(userIDs)
    if not i in itemIDs: itemIDs[i] = len(itemIDs)

nUsers,nItems = len(userIDs),len(itemIDs)

items = list(itemIDs.keys())

class BPRbatch(tf.keras.Model):
    def __init__(self, K, lamb):
        super(BPRbatch, self).__init__()
        # Initialize variables
        self.betaI = tf.Variable(tf.random.normal([len(itemIDs)],stddev=0.001))
        self.gammaU = tf.Variable(tf.random.normal([len(userIDs),K],stddev=0.001))
        self.gammaI = tf.Variable(tf.random.normal([len(itemIDs),K],stddev=0.001))
        # Regularization coefficient
        self.lamb = lamb

    # Prediction for a single instance
    def predict(self, u, i):
        p = self.betaI[i] + tf.tensordot(self.gammaU[u], self.gammaI[i], 1)
        return p

    # Regularizer
    def reg(self):
        return self.lamb * (tf.nn.l2_loss(self.betaI) +\
                            tf.nn.l2_loss(self.gammaU) +\
                            tf.nn.l2_loss(self.gammaI))
    
    def score(self, sampleU, sampleI):
        u = tf.convert_to_tensor(sampleU, dtype=tf.int32)
        i = tf.convert_to_tensor(sampleI, dtype=tf.int32)
        beta_i = tf.nn.embedding_lookup(self.betaI, i)
        gamma_u = tf.nn.embedding_lookup(self.gammaU, u)
        gamma_i = tf.nn.embedding_lookup(self.gammaI, i)
        x_ui = beta_i + tf.reduce_sum(tf.multiply(gamma_u, gamma_i), 1)
        return x_ui

    def call(self, sampleU, sampleI, sampleJ):
        x_ui = self.score(sampleU, sampleI)
        x_uj = self.score(sampleU, sampleJ)
        return -tf.reduce_mean(tf.math.log(tf.math.sigmoid(x_ui - x_uj)))

optimizer = tf.keras.optimizers.Adam(0.1)
modelBPR = BPRbatch(1, 0.00001)

itemsPerUser = defaultdict(list)
usersPerItem = defaultdict(list)
for u,i,r in hoursTrain:
    itemsPerUser[u].append(i)
    usersPerItem[i].append(u)

def trainingStepBPR(model, interactions):
    Nsamples = 50000
    with tf.GradientTape() as tape:
        sampleU, sampleI, sampleJ = [], [], []
        for _ in range(Nsamples):
            u,i,_ = random.choice(interactions) # positive sample
            j = random.choice(items) # negative sample
            while j in itemsPerUser[u]:
                j = random.choice(items)
            sampleU.append(userIDs[u])
            sampleI.append(itemIDs[i])
            sampleJ.append(itemIDs[j])

        loss = model(sampleU,sampleI,sampleJ)
        loss += model.reg()
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients((grad, var) for
                              (grad, var) in zip(gradients, model.trainable_variables)
                              if grad is not None)
    return loss.numpy()

#Run 100 batches of gradient descent

for i in range(100):
    obj = trainingStepBPR(modelBPR, hoursTrain)
    if (i % 10 == 9): print("iteration " + str(i+1) + ", objective = " + str(obj))

### Test accuracy on Validation Set
correct = 0
scores = []
user_score = defaultdict(list)
for (label,sample) in [(1, hoursValid), (0, neg_valid)]:
    for (u,g,_) in sample:
        score = modelBPR.predict(userIDs[u], itemIDs[g]).numpy()
        user_score[u].append((score, g))
        

for (label,sample) in [(1, hoursValid), (0, neg_valid)]:
    for (u,g,_) in sample:
        all_score = sorted(user_score[u])
        games = [i for i in range(len(all_score)) if all_score[i][1] == g][0]
        pred = 0
        if games >= len(all_score)//2:
            pred = 1
        if label == pred:
            correct += 1
correct / (len(hoursValid) + len(neg_valid))

## Write prediction in csv file (would play)
user_score = defaultdict(list)
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        continue
    u,g = l.strip().split(',')
    if u in userIDs and g in itemIDs:
        score = modelBPR.predict(userIDs[u], itemIDs[g]).numpy()
        user_score[u].append((score, g))

    
predictions = open("predictions_Played.csv", 'w')
for l in open("pairs_Played.csv"):
    if l.startswith("userID"):
        predictions.write(l)
        continue
    u,g = l.strip().split(',')
    
    # Logic...
    u,g = l.strip().split(',')
    if u not in userIDs:
        pred = int(g in return1)
        _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')
    elif g not in itemIDs:
        pred = int(len(itemsPerUser[u]) >=19)
        _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')
    else:
        all_score = sorted(user_score[u])
        games = [i for i in range(len(all_score)) if all_score[i][1] == g][0]
        pred = 0
        if games >= len(all_score)//2:
            pred = 1
    
        _ = predictions.write(u + ',' + g + ',' + str(pred) + '\n')

predictions.close()





####################################
### TASK 2: Time Play Prediction ###
####################################
## setting up variables
trainHours = [r[2]['hours_transformed'] for r in hoursTrain]
globalAverage = sum(trainHours) * 1.0 / len(trainHours)
hoursPerUser = defaultdict(list)
hoursPerItem = defaultdict(list)
gamesPerUser = defaultdict(list)
usersPerGame = defaultdict(list)
playedHour = {}
for d in hoursTrain:
    u, g, r = d[0], d[1], d[2]['hours_transformed']
    gamesPerUser[u].append(g)
    usersPerGame[g].append(u)
    hoursPerUser[u].append((g, r))
    hoursPerItem[g].append((u, r))
    playedHour[(u, g)] = d[2]['hours_transformed']


def iterate(lamb1, lamb2):
    # initialize beta terms to 0, and alpha to global average
    betaU = {u: 0 for u in hoursPerUser}
    betaI = {g: 0 for g in hoursPerItem}
    alpha = globalAverage
    # set convergence status for each offset terms to False
    alphaConv = False
    Uconv = False
    Iconv = False
    # store the mse on validation set, initialize to None, will update later
    oldmse = None
    # keep iterating until triggers early stop or convergence (break will be included under these situations)
    while True:
        # if alpha is not converged yet, keep updating, same thing for betaU and betaI in the next 2 if statements
        if alphaConv is False:
            # solve objective function in regard to alpha and set to 0, update alpha, same thing for betaU and betaI in the other two if statements
            new_alpha = sum([t[2]['hours_transformed'] - (betaU[t[0]] + betaI[t[1]]) for t in hoursTrain]) / len(hoursTrain)
            # check for convergence on offset alpha
            if alpha != globalAverage and abs(new_alpha - alpha) < 0.001:
                alphaConv = True
            # update alpha
            alpha = new_alpha
        if Uconv is False:
            diff = 0
            # update beta u for all users by looping through all users
            for d in betaU:
                gs = gamesPerUser[d] # all games u played
                new_betaU = sum([playedHour[(d, game)] - (alpha + betaI[game]) for game in gs]) / (lamb1 + len(gs))
                diff += abs(betaU[d] - new_betaU)
                betaU[d] = new_betaU
            diff = diff / len(betaU)
            if diff < 0.001:
                Uconv = True
        if Iconv is False:
            diff = 0
            # update beta i for all games by looping through all games
            for d in betaI:
                us = usersPerGame[d] # all users played g
                new_betaI = sum([playedHour[(user, d)] - (alpha + betaU[user]) for user in us]) / (lamb2 + len(us))
                diff += abs(betaI[d] - new_betaI)
                betaI[d] = new_betaI
            diff = diff / len(betaI)
            if diff <0.001:
                Iconv = True
        # calculate mse on the validation set after this iteration of updating offsets
        mse = sum([(alpha + betaU[t[0]] + betaI[t[1]] - t[2]['hours_transformed'])**2 for t in hoursValid]) / len(hoursValid)
        # first cycle of iteration, update mse record
        if oldmse is None:
            oldmse = mse
        else:
            diff = abs(oldmse - mse)
            # check for difference in change of mse on validation set, if not changing much (<0.0005), we terminate the iteration early before convergence
            if diff < 0.0005:
                # while terminating the iteration, write out predictions on test set in csv file
                predictions = open("predictions_Hours.csv", 'w')
                for l in open("pairs_Hours.csv"):
                    if l.startswith("userID"):
                        predictions.write(l)
                        continue
                    u,g = l.strip().split(',')
                    bu=0
                    bi=0
                    # handle cold start, currently just predicting alpha
                    if u in betaU:
                        bu = betaU[u]
                    if g in betaI:
                        bi = betaI[g]
    
                    _ = predictions.write(u + ',' + g + ',' + str(alpha + bu + bi) + '\n')
                predictions.close()
                return mse
            else:
                oldmse = mse
        # if all offsets converges, exit iteration loop
        if alphaConv and Iconv and Uconv:
            break

# call the iterating function, the optimized hyperparameters are 8, 1.71, gives out 
iterate(8, 1.71)
# calculate mse on validation set to confirm
sum([(alpha + betaU[t[0]] + betaI[t[1]] - t[2]['hours_transformed'])**2 for t in hoursValid]) / len(hoursValid)
