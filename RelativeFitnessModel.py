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

dictionary[new_key] = dictionary.pop(old_key)
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
    N = 10000          # population size
    s = 0.05            # 
    u = 1e-4            # beneficial mutation rate   
    genome = 50         # length of initial genomes
    beginf = 20         # beginning fitness class 
    cleanUp = N         # frequency of genome cleanup (every generation)
    distGap = float(N)/10  #frequency of snapshots
    #rstr = 50                                                                  # change
    parameters = [b,N,s,u,genome,beginf,cleanUp,rstr,distGap]
    end = 10000000       #iterations to run

    # create initial dictionary to hold possible fitness classes
    population = {}
    #birthPs = {}
    mutationCount = RandomChoiceDict()    # {iteration#:mutations before nose mutation}
                
    # array to track genomes
    tracking = [0] * genome
        
    # beginning fitness class
    population[str(beginf)+'-'+str(N)] = RandomChoiceDict()                       
                
    # generate genome in beginning fitness
    indiv = '1' * beginf
    for a in range(0,genome-beginf):
        indiv = indiv + '0'
           
    # copy genome into entire population        
    for count in range(0,int(N)):   
        # add individual to fitness class
        population[str(beginf)+'-'+str(N)][count] = indiv
            
        # update genome tracking
        for d in range (0,genome):
            if indiv[d] == '1':
                tracking[d] = tracking[d] + 1  
    noseFit = beginf
    # calculate initial average fitness
    i = 0
    sumNi = 0  #numerator, sum over ni
    for key in population:
        sumNi = sumNi + int(key[:key.index('-')]) * int(key[key.index('-')+1:])              
        i = i + 1
    avgFit = sumNi/float(N)
    
    # Number of iterations
    x = -1
    while True:                                                                              
        x = x + 1         
        #DEATH ----------------------------------------------------------------       
        #place everyone in a dictionary NOT separated by fitness
        deathSelect = []
        for key in population:
            deathSelect.append(float(key[key.index('-')+1:]) / float(N))
        
        total=0                                                                 #CHECK
        for key in population:
            total = total+int(key[key.index('-')+1:]) 
        if total!=10000:
            print total
            print population.keys()                
        # select random individual to die
        idenClass = np.random.choice(population.keys(), p=deathSelect)  # key in population
        idenCode = population[idenClass].randomItem()                   # tuple, code, genome 
        iden = idenCode[0]                                              # individual's code in original dicionary
        fitclassD = idenClass[:idenClass.index('-')]                    # individual's fitness
        individual = population[idenClass][iden]                        # genome

        #BIRTH ----------------------------------------------------------------            
        ofsp = ''            
        # Weighted selection of parents 
        z = 0        
        # prbBirth holds probabilities associated with fitness classes (keys in population)
        prbBirth = np.zeros(len(population))
        for key in population:                                                    
            prbBirth[z] = (int(key[key.index('-')+1:])/float(N)) * ((int(key[:key.index('-')])-avgFit)*s+1)    # MAPPING
                        # (ni/N) * ((i - muI)s + 1)
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
      
        #COMPLETE DEATH--------------------------------------------------------
        # update genome tracking for death
        for f in range (0,genome):
            if individual[f] == '1':
                tracking[f] = tracking[f] - 1
                 
        # delete individual
        del population[idenClass][iden]
        newKey = idenClass[:idenClass.index('-')+1] + str(int(idenClass[idenClass.index('-')+1:])-1)
        population[newKey] = population.pop(idenClass)
        
        # erase class if left empty
        if int(newKey[newKey.index('-')+1:]) == 0:                          
            del population[newKey] 
                
        #RETURN TO BIRTH-------------------------------------------------------                   
        # Apply random beneficial mutation  
        chance = np.random.random()
        if chance <= u:
            # Add one to offspring
            ofsp = ofsp + '1' 
            # Add 0 to every other individual
            for g in population:
                 for h in population[g].mapping:                                # MAPPING
                     population[g][h] = population[g][h] + '0'                     
            # Update genome length
            genome = genome + 1
            # extend genome tracking for mutation location
            tracking.append(0)  
           
        # save offspring into fitness class 
        fitness = ofsp.count('1')
        exist = False
        for key in population:
            keyFit = int(key[:key.index('-')])
            if fitness==keyFit:
                exist = True
                existingKey = key
        if exist == True:
            population[existingKey][iden] = ofsp
            newKey = existingKey[:existingKey.index('-')+1] + str(int(existingKey[existingKey.index('-')+1:])+1)
            population[newKey] = population.pop(existingKey)
        else: # create new fitness class             
            population[str(fitness)+'-1'] = RandomChoiceDict()
            population[str(fitness)+'-1'][iden] = ofsp
            if fitness > noseFit:              
                hold = {}
                for a in population.keys():
                    store = int(a[:a.index('-')])
                    hold[store] = int(a[a.index('-')+1:])   
                mutationCount[x] = hold         
                noseFit = fitness
                
        # update sumNi and average fitness
        sumNi = sumNi - int(fitclassD) + fitness
        avgFit = sumNi / float(N)
        
        # update genome tracking for birth
        for e in range (0,genome):
            if ofsp[e] == '1':
                tracking[e] = tracking[e] + 1
 
        # break if there are no living individuals
        if len(population) == 0 or x == end:
            return [distinfo, parameters, mutationCount] #birthPs]
            
        if x% (end/4)==0:
            percentDone = x/float(end) *100          
            print str(percentDone) + '% finished'
        '''   
        total=0
        for key in population:                                                  #CHECK
            total = total+int(key[key.index('-')+1:]) 
        if total!=10000:
            print 'FINISHED BIRTH'
            print total
            print population.keys()
        '''           
        # GENOME CLEAN UP------------------------------------------------------
        if x % cleanUp == 0: 
            if genome < 20:
                delete = []
            else:  
                # store locations of genes to be deleted
                delete = [] 
                fitLoss = 0
                for g in range(0,genome):  
                    if tracking[g] == int(N):                                       
                        delete.append(g)
                        fitLoss = fitLoss+1
                    if tracking[g] == 0:                                       
                        delete.append(g)
                
                if len(delete) == genome:                 
                    if delete[0] == '1':
                        fitLoss = fitLoss-1
                    delete.pop(0)
                # erase gene in every individual
                adjust = 0
                delete.sort()
                
                for j in range(0,len(delete)):
                    select = delete[j]
                    for fit in population:
                        for single in population[fit].mapping:                       # MAPPING
                            patient = population[fit][single]
                            population[fit][single] = patient[:select-adjust] + patient[select-adjust + 1:]
                    adjust = adjust + 1
                
                # erase fully adapted genes in tracking array
                if len(delete) > 0:
                    genome = genome - len(delete)
                    delete.sort()
                    tracking = [keep for j, keep in enumerate(tracking) if j not in delete]
            
                # Change fitness in keys in population
                pop2 = list(population.keys())
                for key in pop2:
                    newKey = str(int(key[:key.index('-')])-fitLoss) + key[key.index('-'):]
                    population[newKey] = population.pop(key)                
    
                # recalculate sumNi
                sumNi = 0  #numerator, sum over ni
                for key in population:
                    sumNi = sumNi + int(key[:key.index('-')]) * int(key[key.index('-')+1:])     
                avgFit = sumNi/float(N)
                
        # Store data
        if x % distGap == 0:              
            pop = {}
            for a in population.keys():
                store = int(a[:a.index('-')])
                pop[store] = int(a[a.index('-')+1:])                              
            pop['extra'] = [x,genome]
            distinfo.append(pop)
        '''
        total=0                                                                 #CHECK
        for key in population:
            total = total+int(key[key.index('-')+1:]) 
        if total!=10000:
            print 'AFTER CLEANUP'
            print total
            print population.keys()
        '''        
# end function          
      
result = ExtTimes(0)

'''
var = [0]

if __name__ == '__main__':
    pool = Pool(processes=1)
    result = pool.map(ExtTimes,var)

store = [result,var]
       
# store results
name = 'NoseMutationTest'
pickle.dump(store,open(name, 'w'))
'''
