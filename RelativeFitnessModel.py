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

locations: array; keys in population (fitness classes)
tracking: array; how many individuals in the popoulation are adapted to a 
          trait; same length as genome 

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
    b = 2.              # intrinsic birth rate\
    n = float(1e5)      # population size
    s = 0.01            # 
    u = 1e-5            # beneficial mutation rate   
    genome = 45         # length of initial genomes
    beginf = 40         # beginning fitness class 
    cleanUp = 1000      # frequency of genome cleanup (iterations)
    #rstr = 0                                                                   # change
    parameters = [b,n,s,u,genome,beginf,cleanUp]

    # create initial dictionary to hold possible fitness classes
    population = {}
                
    # array to track genomes
    tracking = [0] * genome
        
    # beginning fitness class
    population[genome-beginf] = RandomChoiceDict() 
                
    # generate genome in beginning fitness
    indiv = '1' * beginf
    for a in range(0,genome-beginf):
        indiv = indiv + '0'
    fitness = indiv.count('1')
           
    # copy genome into entire population        
    for count in range (0,int(n)):   
        # add individual to fitness class
        population[genome-beginf][count] = indiv
            
        # update genome tracking
        for d in range (0,genome):
            if indiv[d] == '1':
                tracking[d] = tracking[d] + 1         
    
    # calculate initial birthrate weights
    birthRate = 0
    locations = [None] * len(population)
    i = 0
    cdrSum = 0
    for key in population:
        cdr = b + key * s
        cdrSum = cdrSum + cdr  
        birthRate = birthRate + cdr * len(population[key].mapping)
        # create array holding keys in corresponding locations
        locations[i] = key
        i = i + 1
    # Number of iterations
    x = -1
                                   
    while True:                                                                
        x = x + 1 
        #DEATH ----------------------------------------------------------------
        # select class that experiences death
        fitclass = population[list(population)[np.random.randint(len(population))]]                            
        # select random individual killed from class
        place = np.random.randint(0, high=len(fitclass.mapping))
        individual = fitclass[place]
        
        # update genome tracking
        genes = population[fitclass][individual]
        for f in range (0,genome):
            if genes[f] == '1':
                tracking[f] = tracking[f] - 1           
             
        # delete individual
        del population[fitclass][individual]

        # erase class if left empty
        if len(population[fitclass].mapping) == 0: 
            del population[fitclass]
            # update locations array
            locations.remove(fitclass) 
            cdrSum = cdrSum - (b + fitclass * s)
            
        #BIRTH ----------------------------------------------------------------            
        ofsp = ''       
        # Weighted selection of parents 
        z = 0
        prbBirth = [None] * len(population)
        for key in population:
            prbBirth[z] = (b + key * s) / cdrSum
            z = z + 1
            
        # select class that experiences death
        fitclass = np.random.choice(locations, p=prbBirth)
        # prbBirth holds probabilities associated with entries in locations (fitness classes)
                
        # select random individual killed from class
        individual = population[fitclass].randomItem()[0]
            
        # If recombination is used
        if np.random.randint(100) < rstr:
            # create random selection index
            selection = np.random.randint(0, high=2, size=genome)
    
            # carry out recombination
            for y in range (0,genome):
                if selection[y] == 0:
                    ofsp = ofsp + parent1[y]
                else:
                    ofsp = ofsp + parent2[y]            
        # otherwise, clonal 
        else:
            #print "CLONAL"                                                 # printline
            parentpool = population[list(population)[np.random.randint(len(population))]]
            parent = parentpool.randomItem()[1]    
            ofsp = parent
            
        # Apply random beneficial mutation  
        chance = np.random.random()
        if chance <= u:        
            # Add one to offspring
            ofsp = ofsp + '1' 
            # Add 0 to every other individual
            for g in population:
                 for h in population[g].mapping:
                     population[g][h] = population[g][h] + '0'                     
            # Update genome length
            genome = genome + 1
            # update genome tracking, only one individual is adapted
            tracking.append(1)
                
        # save offspring into fitness class            
        if fitness in population:
            population[fitness][place] = ofsp
        else: # create new fitness class 
            population[fitness] = RandomChoiceDict()
            population[fitness][place] = ofsp
            
            cdrSum = cdrSum + (b + fitness * s)
                    
            # recreate locations array for new fitness class                   # Double Check
            locations = [None] * len(population)
            i = 0
            for key in population:
                locations[i] = key
                i = i + 1
                        
        # update genome tracking
        for e in range (0,genome):
            if ofsp[e] == '1':
                tracking[e] = tracking[e] + 1

        # break if there are no living individuals
        if len(population) == 0 or x == 100000:
            return [distinfo, parameters]
        
        # GENOME CLEAN UP------------------------------------------------------
        if x = cleanUp:                                    
            # store locations of genes to be deleted
            delete = []   
            for g in range (0,genome):  
                if tracking[g] == int(n):                                       
                    delete.append(g)   
                    
            # erase gene in every individual
            adjust = 0
            for j in range(0,len(delete)):
                select = delete[j]
                for fit in population:
                    for single in population[fit].mapping:
                        patient = population[fit][single]
                        population[fit][single] = patient[:select-adjust] + patient[select-adjust + 1:]
                adjust = adjust + 1
                
            # erase fully adapted genes in tracking array
            if len(delete) > 0:
                genome = genome - len(delete)
                delete.sort()
                tracking = [keep for j, keep in enumerate(tracking) if j not in delete]
            if len(locations) > 1:
                for key in population:
                    print key, ": ", population[key].mapping()
            
        # Store data
        if x % 1000 == 0:                
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
