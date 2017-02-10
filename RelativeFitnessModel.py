# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:03:57 2015

@author: Jasmin
"""

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
import random
import matplotlib.pyplot as plt
import matplotlib.colors as clr
norm=clr.Normalize(0,255)
#plt.ioff()

#Color key


distinfo = []                     

def ExtTimes(rstr):  

    file=open("backup.txt","w")

    # Parameters
    N = 100          # population size
    s = 0.01            
    u = .0001           # beneficial mutation rate  
    r = .05             #breaks/locus (locus = 1 or 0)
    
    genome = 50         # length of initial genomes
    beginf = 20         # beginning fitness class 
    cleanUp = N         # frequency of genome cleanup (every generation)
    distGap = float(N)/100  #frequency of snapshots
    
    parameters = [N,s,u,genome,beginf,cleanUp,rstr,distGap]
    
    end = 10000       #iterations to run

    # create initial dictionary to hold possible fitness classes
    population = {}
                
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
    
    # calculate initial average fitness
    i = 0
    sumNi = 0  #numerator, sum over ni
    for key in population:
        sumNi = sumNi + int(key[:key.index('-')]) * int(key[key.index('-')+1:])              
        i = i + 1
    avgFit = sumNi/float(N)
    
    #Graph first column, all identical
    genomesTrack = [indiv]*N
    fitnessTrack = [beginf]*N   
    colorBank = {}
    
    time=0
    x=[time]*N  

    colorCodes = np.random.randint(0,256,3)    
    c1=colorCodes[0]
    c2=colorCodes[1]
    c3=colorCodes[2] 
    color=str((c1,c2,c3))
    colorBank[beginf]=color
    
    plt.plot(x, range(1,N+1),color=color)
    plt.xlabel('Time')
    plt.ylabel('Individuals')
    time=time+1 #increase time     
    
    # Number of iterations
    x = -1
    while True:                                                                              
        x = x + 1         
        #DEATH ----------------------------------------------------------------       
        #place everyone in a dictionary NOT separated by fitness
        deathSelect = []
        for key in population:
            deathSelect.append(float(key[key.index('-')+1:]) / float(N))
        
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
        sortedKeys = population.keys()
        prbBirth = np.zeros(len(population))
        check = np.zeros(len(prbBirth))
        for key in sortedKeys:                                                    
            prbBirth[z] = (int(key[key.index('-')+1:])/float(N)) * ((int(key[:key.index('-')])-avgFit)*s+1) 
            check[z] = ((int(key[:key.index('-')])-avgFit)*s+1) # MAPPING
                        # (ni/N) * ((i - muI)s + 1)
            z = z + 1    
            
        # If recombination is used
        if np.random.randint(100) < rstr: 
            # select class and random individual from class
            fitclass = np.random.choice(sortedKeys, p=prbBirth)                                    
            parent1 = population[fitclass][population[fitclass].randomItem()[0]]     
        
            # select second class and rndom individual
            fitclass = np.random.choice(sortedKeys, p=prbBirth)
            parent2 = population[fitclass][population[fitclass].randomItem()[0]]
            '''
            # EXTREME RECOMBINATION selection index
            selection = np.random.randint(0, high=2, size=genome)
            '''
            # REALISTIC RECOMBINATION selection index
            breaks = r*genome #(L=#of loci OR genome length)
            
            if breaks==0:
                 breaks=int(np.random.normal(1,1))
                 if breaks<0:
                     breaks=np.abs(breaks)
            breaks = int(breaks)
            loc = random.sample(xrange(genome),breaks)
            beg=np.random.randint(0,high=2)
            
            if beg==0:
                zero=True
                #selection = '0'
            else:
                zero=False
                #selection='1'
            selection = ''
            for i in range(0,genome):
                if i in loc:
                    zero= not zero
                if zero:
                    selection=selection+'0' 
                else:
                    selection=selection+'1'                
            
            # Carry out recombination
            ofsp=''
            for y in range (0,genome):
                if selection[y] == '0':
                    ofsp = ofsp + parent1[y]
                else:
                    ofsp = ofsp + parent2[y]  
            
        # otherwise, clonal 
        else:
            fitclass = np.random.choice(sortedKeys, p=prbBirth)
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
                  
        # update sumNi and average fitness
        sumNi = sumNi - int(fitclassD) + fitness
        avgFit = sumNi / float(N)
        
        # update genome tracking for birth
        for e in range (0,genome):
            if ofsp[e] == '1':
                tracking[e] = tracking[e] + 1
                
        #Delete dead indiv from array
        fitLoc = genomesTrack.index(individual)
        genomesTrack.remove(individual)
        del fitnessTrack[fitLoc]
                
        #Place new individual into proper location
        found=False
        if ofsp in genomesTrack: #if genome exists
            insertLoc=genomesTrack.index(ofsp)
            fitnessTrack.insert(insertLoc,fitness)
            genomesTrack.insert(insertLoc,ofsp)
            found=True
        if found==False and fitness < fitnessTrack[len(fitnessTrack)-1]:
            fitnessTrack.append(fitness)
            genomesTrack.append(ofsp)
            found=True
        if found==False and fitness > fitnessTrack[0]:
            fitnessTrack.insert(0,fitness)
            genomesTrack.insert(0,ofsp)
            found=True
        if found==False: #gap in distribution check
            for f in range(0,len(fitnessTrack)-1):
                if found==False and fitness > fitnessTrack[f]:
                    insertLoc=f
                    found=True        
            fitnessTrack.insert(f,fitness)
            genomesTrack.append(f,ofsp)
            
        # break if there are no living individuals
        if len(population) == 0 or x == end:
            return [distinfo, parameters]
          
        # GENOME CLEAN UP------------------------------------------------------
        if x % cleanUp == 0 and x>5: 
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
         
            pop = {}
            file.write('[')
            for a in population.keys():
                store = int(a[:a.index('-')])
                pop[store] = int(a[a.index('-')+1:]) 
                file.write(str(store)+':'+str(pop[store])+']')                           
            pop['extra'] = [x,genome]
            distinfo.append(pop)
        
        # Store data
        if x % distGap == 0:              
            pop = {}
            for a in population.keys():
                store = int(a[:a.index('-')])
                pop[store] = int(a[a.index('-')+1:]) 
            pop['extra'] = [x,genome]
            distinfo.append(pop)   
            
            
            #graph new column        
            colorCodes = np.random.randint(0,256,3)            
            c1=str(colorCodes[0])
            c2=str(colorCodes[1])
            c3=str(colorCodes[2])
            color = 
            colorBank.append(color)
            
            start=fitnessTrack[0]
            y=[]
            ymarker=1
            
            for q in range(1,len(fitnessTrack)-2):
                if fitnessTrack[q]<start:
                    if fitnessTrack[q-1] in colorBank.keys():
                        color=colorBank[q-1]
                    else:
                        colorCodes = np.random.randint(0,256,3)    
                        c1=colorCodes[0]
                        c2=colorCodes[1]
                        c3=colorCodes[2] 
                        color=str((c1,c2,c3))
                        colorBank[q-1]=color
                    plt.plot([time]*len(y),y,color=color)
                    y=[]
                    start=fitnessTrack[q]
                else:
                    y.append(ymarker)
                    ymarker=ymarker+1
            
            time=time+1 #increase time      
            '''
            #move over on x-axis
            time=time+1
        
# end function          
'''
result = ExtTimes(10)
var = [0]
name = 'TEST'
store = [result,var]
#pickle.dump(store,open(name, 'w'))

'''
var = [10]

if __name__ == '__main__':
    pool = Pool(processes=1)
    result = pool.map(ExtTimes,var)

store = [result,var]
       
# store results
name = 'r=.03'
pickle.dump(store,open(name, 'w'))
