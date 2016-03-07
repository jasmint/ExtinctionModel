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
    mu = 1.             # intrinsic death rate
    k = 1e5            # carrying capacity
    s = 0.01            # 
    u = 1e-5            # beneficial mutation rate
    n = k/2.            # initial population size
    genome = 45        # length of initial genomes
    beginf = 40         # beginning fitness class 
    t = 120.
    #rstr = 0
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
    population[genome-beginf] = RandomChoiceDict() 
                
    # generate genome in beginning fitness
    indiv = '0' * beginf
    for a in range(0,genome-beginf):
        indiv = indiv + '1'
    fitness = indiv.count('1')
               
    # copy genome into entire population        
    for count in range (0,int(n)):   
        # add individual to fitness class
        population[genome-beginf][count] = indiv
            
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
    pt = 10
    
    while True:
        x = x + 1
        # determine if birth or death occurs 
        prbirth = birthrate / (birthrate + deathrate)
        prb = np.random.random()
        
        if prb <= prbirth: # birth occurs             
            ofsp = ''
            # If recombination is used
            if np.random.randint(100) < rstr:
                # create random selection index
                selection = np.random.randint(0, high=2, size=genome)
            
                # select two random inidividuals (parents)          
                parent1pool = population[list(population)[np.random.randint(len(population))]]
                parent1 = parent1pool.randomItem()[1]                
                parent2pool = population[list(population)[np.random.randint(len(population))]]
                parent2 = parent2pool.randomItem()[1]              
                
                # carry out recombination
                for y in range (0,genome):
                    if selection[y] == 0:
                        ofsp = ofsp + parent1[y]
                    else:
                        ofsp = ofsp + parent2[y]            
            # otherwise, clonal 
            else:
                parentpool = population[list(population)[np.random.randint(len(population))]]
                parent = parentpool.randomItem()[1]    
                ofsp = parent
            
            # apply random beneficial mutation  
            fitness = ofsp.count('1')       
            mutation = np.random.random()
            limit = fitness * u
            if mutation <= limit and '1' in ofsp:
                location = np.random.randint(0, high=genome)    
                while ofsp[location] == '0':
                    location = np.random.randint(0, high=genome)
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
            # prbdeath holds probabilities associated with entries in locations (fitness classes)
                
            # select random individual killed from class
            individual = population[fitclass].randomItem()[0]
                
            # update genome tracking
            genes = population[fitclass][individual]
            for f in range (0,genome):
                if genes[f] == '0':
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
        if len(population) == 0 or x == 20:
            return [distinfo, parameters]
            
        # environmental change
        if x == pt:
            
            # add a '1' to end of each genome
            for g in population:
                for h in population[g].mapping:
                    population[g][h] = population[g][h] + '1'
            genome = genome + 1
            # update genome tracking, no one is adapted
            tracking.append(0)
                        
            #ADDITION-----------------------------------------------------------------
            # Move every individual back one fitness class 
            old_keys = []
            new_keys = []
            
            # Record old fitness classes and new fitness classes
            for oldKey in population:
                old_keys.append(oldKey)
                newKey = oldKey + 1
                new_keys.append(newKey)
            
            #flip order of each array to prevent overlap
            def reverse_numeric(x, y):
                return y - x
            old_keys = sorted(old_keys, cmp=reverse_numeric)
            new_keys = sorted(new_keys, cmp=reverse_numeric)              
                       
            # Move every class back one.
            for w in range(0, len(new_keys)): 
                population[new_keys[w]] = population[old_keys[w]]
                
            # Delete fitness classes left unpopulated 
            for key in old_keys:
                if key not in new_keys:
                    del population[key]
                
            # recreate locations array
            locations = [None] * len(population)
            i = 0
            for key in population:
                locations[i] = key
                i = i + 1
                
            #Recalculate deathrate
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
            #ADDITION-----------------------------------------------------------------
                      
            # check for fully adapted genes
            delete = []
            for g in range (0,genome):  
                if tracking[g] == int(n):
                    # erase gene in every individual
                    for fit in population:
                        for single in population[fit].mapping:
                            patient = population[fit][single]
                            population[fit][single] = patient[:g] + patient[g + 1:]
                    # store locations of genes to be deleted
                    delete.append(g)         
            
            # erase fully adapted genes in tracking array
            if len(delete) > 0:
                genome = genome - len(delete)
                delete.sort()
                shift = 0
                for h in range(0,len(delete)):
                    trait = delete[h] - shift
                    del tracking[trait] 
                    shift = shift + 1
                
            # sample new 't'
            pt = x + int(np.random.exponential(scale=t))
            
        # check distribution of population
        if x % 1 == 0:                
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
