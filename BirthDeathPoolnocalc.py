# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:03:57 2015

@author: Jasmin
"""

'''
Parameters
b: intrinsic birthrate
mu: intrinsic deathrate
k: carrying capacity
s: 
u: beneficial mutation rate
t: average time between envirnmental changes
n: initial population size
genome: length of initial genomes
beginf: beginning fitness class

0 --> adapted
1 --> not adapted
'''

class RandomChoiceDict(object):
    def __init__(self):
        self.mapping = {}  # wraps a dictionary
                           # e.g. {'a':'Alice', 'b':'Bob', 'c':'Carrie'}

        # the arbitrary mapping mentioned above
        self.idToKey = {}  # e.g. {0:'a', 1:'c' 2:'b'}, 
                           #      or {0:'b', 1:'a' 2:'c'}, etc.

        self.keyToId = {}  # needed to help delete elements
    
    def __getitem__(self, key):  # O(1)
        return self.mapping[key]

    def __setitem__(self, key, value):  # O(1)
        if key in self.mapping:
            self.mapping[key] = value
        else: # new item
            newId = len(self.mapping)

            self.mapping[key] = value

            # add it to the arbitrary bijection
            self.idToKey[newId] = key
            self.keyToId[key] = newId

    def __delitem__(self, key):  # O(1)
        del self.mapping[key]  # O(1) average case
                               # see http://wiki.python.org/moin/TimeComplexity

        emptyId = self.keyToId[key]
        largestId = len(self.mapping)  # about to be deleted
        largestIdKey = self.idToKey[largestId]  # going to store this in empty Id

        # swap deleted element with highest-id element in arbitrary map:
        self.idToKey[emptyId] = largestIdKey
        self.keyToId[largestIdKey] = emptyId

        del self.keyToId[key]
        del self.idToKey[largestId]

    def randomItem(self):  # O(1)
        r = np.random.randint(len(self.mapping))
        k = self.idToKey[r]
        return (k, self.mapping[k])
        
import numpy as np
import pickle
from multiprocessing import Pool 

distinfo = []

def ExtTimes(rstr):  
    
    # Parameters
    b = 2.              # intrinsic birth rate
    mu = 1.             # intrinsic death rate
    k = 1e5            # carrying capacity
    s = 0.01            # 
    u = 1e-5            # beneficial mutation rate
    n = k/2.            # initial population size
    genome = 45        # length of initial genomes
    beginf = 40         # beginning fitness class 
    #rstr =             # strength of recombination
    t = 120.
    
    parameters = [b,mu,k,s,u,n,genome,beginf,t]
    '''
    # compare v and i/T   
    v = [s*(2.*np.log(k*s*(b-mu-i*s)/b)-np.log(s/(u*i)))/np.log(s/(u*i))**2 for i in range(1,int((b-mu)/s))]
    v=np.concatenate([[0],v,[0]])    
    plt.plot([1/t]*102)
    plt.plot(v)
    plt.show()
    '''
    # create initial dictionary to hold possible fitness classes
    population = {}
                
    # array to track genomes
    tracking = [0] * genome
        
    # beginning fitness class
    population[beginf] = RandomChoiceDict() 
                
    # generate genome
    indiv = ''   
    indiv = '0' * 40
    for a in range(0,genome-beginf):
        indiv = indiv + '1'
    #for a in range (0,genome):
    #    indiv = indiv + str(np.random.randint(0, high=2))
    fitness = indiv.count('1')
            
    # make sure individual is in the specified fitness class
    while fitness != beginf:
        indiv = ''
        for a in range (0,genome):
            indiv = indiv + str(np.random.randint(0, high=2))
        fitness = indiv.count('1')  
    
    # copy genome into entire population        
    for count in range (0,int(n)):   
        # add individual to fitness class
        population[beginf][count] = indiv
            
        # update genome tracking
        for d in range (0,genome):
            if indiv[d] == '0':
                tracking[d] = tracking[d] + 1  
       
    # calculate initial birthrate
    birthrate = b * (1 - n/k) * n
            
    # calculate initial deathrate
    deathrate = 0
    locations = [None] * len(population)
    i = 0
    cdrSum = 0
    for key in population:
        cdr = mu + key * s
        cdrSum = cdrSum + cdr  
        deathrate = deathrate + cdr * len(population[key].mapping)
        # create array holding keys in corresponding locations
        locations[i] = key
        i = i + 1
            
    # increment counter to prevent overlap
    count = count + 1
    x = -1
    
    pt = int(np.random.exponential(scale=t))
    pt = 70
    
    while True:
        x = x + 1
        # determine if birth or death occurs 
        prbirth = birthrate / (birthrate + deathrate)
        prb = np.random.random()
        
        if prb <= prbirth: # birth occurs             
            
            # create random selection index
            selection = np.random.randint(0, high=2, size=genome)
        
            # select two random individuals to carry out recombination           
            parent1pool = population[list(population)[np.random.randint(len(population))]]
            #pooldict1 = RandomChoiceDict()
            #for key in parent1pool:
            #    pooldict1[key] = parent1pool[key]                               
            parent1 = parent1pool.randomItem()[1]
            
            parent2pool = population[list(population)[np.random.randint(len(population))]]
            #pooldict2 = RandomChoiceDict()
            #for key in parent1pool:
            #    pooldict1[key] = 
            parent2 = parent2pool.randomItem()[1]
              
            if np.random.randint(100) <= rstr:
                ofsp = ''   
                for y in range (0,genome):
                    if selection[y] == 0:
                        ofsp = ofsp + parent1[y]
                    else:
                        ofsp = ofsp + parent2[y]
            else:
                ofsp = parent1
            
            # apply random beneficial mutation  
            fitness = ofsp.count('1')       
            mutation = np.random.random()
            limit = fitness * u
            if mutation <= limit:
                location = np.random.randint(0, high=genome)
                if '1' in ofsp and ofsp[location] == '0':
                    while ofsp[location] == '0':
                        location = np.random.randint(0, high=genome)
                    ofsp = ofsp[:location] + '0' + ofsp[location + 1:]
                else:
                    ofsp = ofsp[:location] + '0' + ofsp[location + 1:]
                    
            # save offspring into fitness class                       
            if fitness in population:
                population[fitness][count] = ofsp
            else: # create new fitness class 
                population[fitness] = RandomChoiceDict()
                population[fitness][count] = ofsp
                    
                cdrSum = cdrSum + (mu + fitness * s)
                    
                # recreate locations array for new fitness class
                locations = [None] * len(population)
                i = 0
                for key in population:
                    locations[i] = key
                    i = i + 1
                        
            # update genome tracking
            for e in range (0,genome):
                if ofsp[e] == '0':
                    tracking[e] = tracking[e] + 1
          
            count = count + 1
            n = n + 1 
                
            # update birthrate/deathrate
            birthrate = b * (1 - n/k) * n
            deathrate = deathrate + (mu + fitness * s)
            
        else: # death occurs  
            # make cum. prb. array
            z = 0
            prbdeath = [None] * len(population)
            for key in population:
                prbdeath[z] = (mu + key * s) / cdrSum
                z = z + 1
                
            # select class that experiences death
            fitclass = np.random.choice(locations, p=prbdeath)
            # prbdeath holds probabilities associated with entries in locations
                
            # select random individual killed from class
            individual = population[fitclass].randomItem()[0]
                
            # update genome tracking
            for f in range (0,genome):
                genes = population[fitclass][individual]
                current = genes[f]
                if current == '0':
                    tracking[f] = tracking[f] - 1
             
            # delete individual
            del population[fitclass][individual]
            n = n-1
             
            # update birthrate/deathrate
            birthrate = b * (1 - n/k) * n
            deathrate = deathrate - (mu + fitclass * s)
     
            # erase class if left empty
            if len(population[fitclass].mapping) == 0: 
                del population[fitclass]
                # update locations array
                locations.remove(fitclass) 
                cdrSum = cdrSum - (mu + fitclass * s)                    
            
        # break if there are no living individuals
        if len(population) == 0 or x == 100:
            return [distinfo, parameters]
            
        # environmental change
        if x == pt :
            
            # add a '1' to end of each genome
            for g in population:
                for h in population[g].mapping:
                    population[g][h] = population[g][h] + '1'
            genome = genome + 1
            # update genome tracking, no one is adapted
            tracking.append(0)
            
            # check for fully adapted genes
            delete = []
            for g in range (0,genome-1):                
                i = 0
                if tracking[g] == n:
                    genome = genome - 1
                    # erase gene in every individual
                    for h in population:
                        for j in population[h].mapping:
                            patient = population[h][j]
                            population[h][j] = patient[:g] + patient[g + 1:]
                    # store locations of genes to be deleted
                    delete.append(g)
                    i = i + 1                   
            
            # erase fully adapted genes in tracking array
            if len(delete) != 0:
                for h in range(0,len(delete)):
                    del tracking[delete[h]] 
                
            # sample new 't'
            pt = x + int(np.random.exponential(scale=t))
            
        # check distribution of population
        if x % 1 == 0: #1000 == 0: 
                
            #Store Distribution Information
            pop = {}
            for key in population:
                pop[key] = len(population[key].mapping)
            pop['extra'] = [birthrate, deathrate]
            distinfo.append(pop)
          
# end function          
var = np.arange(0,1)

if __name__ == '__main__':
    pool = Pool(processes=1)
    result = pool.map(ExtTimes,var)

store = [result,var]
       
# store results
name = 'test'
pickle.dump(store,open(name, 'w'))
