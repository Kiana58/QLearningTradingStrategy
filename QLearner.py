import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):

        self.verbose = verbose
        self.num_actions = num_actions
        self.s = 0
        self.a = 0

        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.Q = np.random.uniform(-1.0,1.0,(num_states,num_actions))

        if self.dyna != 0:
            self.T2 = dict()
            self.R = -1.0 * np.ones((num_states, num_actions))


    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        action = np.argmax(self.Q[s, :],axis=0)
        self.a = action
        if self.verbose: print "s =", s,"a =",action
        return action

    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        Q=self.Q
        alpha = self.alpha
        gamma = self.gamma
        
        
        num_states = self.num_states
        num_actions = self.num_actions
        Q[self.s, self.a] = (1 - self.alpha) * Q[self.s, self.a] + alpha * (r + gamma * Q[s_prime,np.argmax(Q[s_prime,:], axis=0)])
        

        if self.dyna != 0:
            T2 = self.T2
            R = self.R
            if T2.get((self.s, self.a)) is not None:
                T2[(self.s, self.a)].append(s_prime)
            else :
                T2[(self.s, self.a)] = [s_prime]
            
            R[self.s, self.a] = (1 - self.alpha) * R[self.s, self.a] + alpha * r
            
            for i in range(self.dyna):

                s_rand = int(num_states*np.random.random())
                a_rand = int(num_actions*np.random.random())

                if T2.get((s_rand, a_rand)) is None:
                    y = int(num_states*np.random.random())
                else:
                    y = rand.choice(T2[(s_rand, a_rand)])
                
                Q[s_rand, a_rand] = (1 - alpha) * Q[s_rand, a_rand] + alpha * (R[s_rand, a_rand] + gamma * np.max(Q[y,:]))
            self.T2 = T2
            self.R = R 
        
        if rand.random() > self.rar:
            action = np.argmax(Q[s_prime, :])
        else:
            action = int(num_actions*np.random.random())

        self.Q = Q   
        
        self.rar = self.rar * self.radr
        self.s = s_prime
        self.a = action
        if self.verbose: print "s =", s_prime,"a =",action,"r =",r
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
