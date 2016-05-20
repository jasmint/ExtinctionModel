# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:03:57 2015

@author: Jasmin
"""

'''
PARAMETERS
b: intrinsic birthrate
mu: intrinsic deathrate
k: carrying capacity
s: 
u: beneficial mutation rate
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
    b = 2.              # intrinsic birth rate
    k = 10000 #1e5             # carrying capacity                                 #shortened
    N = k/2.0           # population size
    s = 0.01            # 
    u = 1e-5            # beneficial mutation rate   
    genome = 50         # length of initial genomes
    beginf = 20         # beginning fitness class 
    cleanUp = 10000         # frequency of genome cleanup (every generation)
    distGap = 1
    #rstr = 10                                                                  # change
    parameters = [b,k,N,s,u,genome,beginf,cleanUp,rstr,distGap]

    # create initial dictionary to hold possible fitness classes
    population = {}
                
    # array to track genomes
    tracking = [0] * genome
        
    # beginning fitness class
    population[beginf] = RandomChoiceDict() 
                
    # generate genome in beginning fitness
    indiv = '1' * beginf
    for a in range(0,genome-beginf):
        indiv = indiv + '0'
           
    # copy genome into entire population        
    for count in range (0,int(N)):   
        # add individual to fitness class
        population[beginf][count] = indiv
            
        # update genome tracking
        for d in range (0,genome):
            if indiv[d] == '1':
                tracking[d] = tracking[d] + 1         
    
    # calculate initial average fitness
    i = 0
    sumNi = 0  #numerator, sum over ni
    for key in population:
        sumNi = sumNi + key * len(population[key].mapping)
        i = i + 1
    avgFit = sumNi/float(N)
    
    # Number of iterations
    x = -1
    while True:                                                                              
        x = x + 1 
        
        #DEATH ----------------------------------------------------------------       
        #place everyone in a dictionary NOT separated by fitness
        deathSelect = {}
        for key in population:
            for code in population[key].mapping:
                deathSelect[code] = [population[key][code], key]
                
        # select random individual to die
        iden = np.random.choice(deathSelect.keys()) # individual's code in original dicionary
        fitclass = deathSelect[iden][1] # individual's fitness
        individual = population[fitclass][iden]

        # update genome tracking for death
        for f in range (0,genome):
            if individual[f] == '1':
                tracking[f] = tracking[f] - 1
                
        #BIRTH ----------------------------------------------------------------            
        ofsp = ''    

        # Weighted selection of parents 
        z = 0
        # prbBirth holds probabilities associated with fitness classes (keys in population)
        prbBirth = [None] * len(population)
        for key in population:
            prbBirth[z] = (len(population[key].mapping)/float(N)) * ((key-avgFit)*s+1) # (ni/N) * ((i - muI)s + 1)
            z = z + 1
            
        # If recombination is used
        if np.random.randint(100) < rstr: 
            # select class and random individual from class
            fitclass = np.random.choice(population.keys(), p=prbBirth)                                    
            parent1 = population[fitclass][population[fitclass].randomItem()[0]]     
            
            # select second class and rndom individual
            fitclass = np.random.choice(population.keys(), p=prbBirth)
            parent2 = population[fitclass][population[fitclass].randomItem()[0]]
            
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
            fitclass = np.random.choice(population.keys(), p=prbBirth)
            ofsp = population[fitclass].randomItem()[1]   
            
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
            # extend genome tracking for mutation location
            tracking.append(0)
            
        #COMPLETE DEATH--------------------------------------------------------
        # delete individual
        del population[fitclass][iden]
        
        # erase class if left empty
        if len(population[fitclass].mapping) == 0: 
            del population[fitclass] 
                
        #RETURN TO BIRTH-------------------------------------------------------    
        # save offspring into fitness class 
        fitness = ofsp.count('1')
        if fitness in population:
            population[fitness][iden] = ofsp
        else: # create new fitness class 
            population[fitness] = RandomChoiceDict()
            population[fitness][iden] = ofsp
        
        # update sumNi and average fitness
        sumNi = sumNi - fitclass + fitness
        avgFit = sumNi / float(N)
        
        # update genome tracking for birth
        for e in range (0,genome):
            if ofsp[e] == '1':
                tracking[e] = tracking[e] + 1
                
        # break if there are no living individuals
        if len(population) == 0 or x == 500000:
            return [distinfo, parameters]
        
        # GENOME CLEAN UP------------------------------------------------------
        if x == cleanUp:                                   
            # store locations of genes to be deleted
            delete = []   
            for g in range (0,genome):  
                if tracking[g] == int(N):                                       
                    delete.append(g)   
                    
            # erase gene in every individual
            adjust = 0
            delete.sort()
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
     
            # Place everyone in new fitness class
            new = {}
            for key in population:
                for code in population[key].mapping:
                    fitness = population[key][code].count('1')
                    if fitness != key:
                        if fitness in population:
                            population[fitness][code] = population[key][code]
                            del population[key][code]
                        else: # create new class
                            if fitness in new:
                                new[fitness][code] = str(key) + population[key][code]
                            else: 
                                new[fitness] = {}
                                new[fitness][code] = str(key) + population[key][code]
            
            if len(new) != 0:
                for key in new:
                    population[key] = RandomChoiceDict()
                    for code in new[key]:
                        originalFit = int(new[key][code][:2])
                        genes = new[key][code][2:]
                        del population[originalFit][code]
                        population[key][code] = genes
                
            # check for empty classes
            delete = []
            for key in population:
                if len(population[key].mapping) == 0:
                    delete.append(key)
            for k in delete:
                population.pop(k, None)
            # recalculate sumNi
            i = 0
            sumNi = 0  #numerator, sum over ni
            for key in population:
                sumNi = sumNi + key * len(population[key].mapping)
                i = i + 1
            avgFit = sumNi/float(N)
        # Store data
        if x % distGap == 0:                
            pop = {}
            for a in population.keys():
                pop[a] = len(population[a].mapping)
            pop['extra'] = [x]
            distinfo.append(pop)
# end function          
#result = ExtTimes(0)

#var = np.arange(0,2)
var = [0,10]

if __name__ == '__main__':
    pool = Pool(processes=2)
    result = pool.map(ExtTimes,var)

store = [result,var]
       
# store results
name = 'Data-0,10TEST2'
pickle.dump(store,open(name, 'w'))
#pickle.dump(result,open(name, 'w'))