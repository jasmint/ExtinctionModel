# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:36:49 2015

@author: Jasmin
 
"""
import heapq
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

name = 'test'
runData = pickle.load(open('C:\Users\Jasmin\Documents\GitHub\ExtinctionModel\\' + name, 'r'))
   
functionData = runData[0]
variable = runData[1]

meanI = ['NONE']
maxI = ['NONE']
birthR = ['NONE']
deathR = ['NONE']
intervals = 1 # how often snapshots were taken

# Enter directory to save graphs for one run
parameters = functionData[0][len(functionData[0])-1]
os.mkdir(str(parameters[8]) + "'")
os.chdir(str(parameters[8]) + "'")
    
# Functiondata[x] where x is the rstr passed into the function(beginning at 0)
for i in range(min(variable), max(variable)+1):
    rstrVal = functionData[i] # data for one run at one rstr
    
    snapshots = rstrVal[0] # snapshots for one run at one rstr - list
    parameters = rstrVal[1] # parameters from one run at one rstr
        
    # Enter directory for one variable value
    os.mkdir(str(i))
    os.chdir(str(i))
    
    for l in range(0, len(snapshots)): # loop through every snapshot at one rstr value 
        onesnap = snapshots[l]# info from one snapshot - dict
        classes = []
        abund = []             
        
        birthRate = onesnap['extra'][0]
        deathRate = onesnap['extra'][1]
        birthR.append(onesnap['extra'][0])
        deathR.append(onesnap['extra'][1])
        
        for key in onesnap: # create array with classes and array with abundances
            if key != 'extra':            
                classes.append(key)
                abund.append(onesnap[key])

        # Population Distributions               
        fig = plt.figure()
        ax = fig.add_subplot(111)
        rects = ax.bar(classes, abund, width=1,align='center')
        #ax.autoscale(False)
        
        # Label bar abundances 
        def autolabel(rects):
            for rect in rects:
                height = rect.get_height()
                plt.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                        ha='center', va='bottom')
                        
        autolabel(rects)
        ax.set_ylabel('Abundance')
        ax.set_xlabel('Class')
        ax.set_xticks(range(min(classes)-2, max(classes)+2), minor=False)
        ax.set_xticklabels(range(min(classes)-2, max(classes)+2))
        
        # Save figure and clear screen
        plt.savefig('Recomb(' + str(i) + ') Iter(' + str(l * intervals + intervals) + ').png')
        plt.close();
        
        # calculations for meanI, maxI
        sumI = 0
        for m in range(0, len(classes)):
            sumI = sumI + classes[m] * abund[m]
        muI = sumI / float(sum(abund))
        maximum = max(classes)
        n = 2
        while onesnap[maximum] < 100:          
            largest = heapq.nlargest(n, classes)
            maximum = min(largest)
            n = n + 1
            if n == len(onesnap)-1:
                maximum = max(classes)
            break
        meanI.append(muI)
        maxI.append(maximum)
        '''
        # Equilibrium calculations
        N = float(sum(abund))
        sigSquared = 0
        for y in range(0,len(classes)):
            nI = abund[y]
            pI = nI / N
            sigSquared = sigSquared + (classes[y] - meanI[y])**2 * pI
            print sigSquared 
         '''   
    # add marker between variable values
    birthR.append('NONE')
    deathR.append('NONE')
    meanI.append('NONE')
    maxI.append('NONE')

# Enter directory to save graphs for one run
os.chdir("..")

print "Birth Rates: ", birthR
print "Mean fitnesses: ", meanI
print "Max fitnesses: ", maxI

# Split arrays
indicesBirth = [i for i, x in enumerate(birthR) if x == "NONE"]
indicesDeath = [i for i, x in enumerate(deathR) if x == "NONE"]
indicesMean = [i for i, x in enumerate(meanI) if x == "NONE"]
indicesMax = [i for i, x in enumerate(maxI) if x == "NONE"]

# Graphing 
# mean vs time for each var (recomb)
for p in range(min(variable), max(variable)+1):
    # Recomb value [x-value]
    var = variable[p]
    
    beg = indicesMean[p] + 1
    end = indicesMean[p+1]
    # mean data for var
    chunk1 = meanI[beg:end]     
    # max data for var
    chunk2 = maxI[beg:end]
    
    time = [j * intervals for j in range(0, len(chunk1))]
    #time[:] = [x/1000000. for x in time]
    
    #Plot average
    plt.plot(time, chunk1, '.r-')      
    plt.ylim(0,max(chunk2)+10)  
    plt.xlabel('Iterations')
    plt.ylabel('Lower Fitness --->')    
    plt.legend(['Avg fitness', 'Max fitness'], loc='upper left')
    plt.savefig('AVG-T' + str(parameters[8]) + '.png')
    plt.close();
    
    #Plot Max
    plt.plot(time, chunk2, '.b-')  
    plt.ylim(0,max(chunk2)+10)  
    plt.xlabel('Iterations')
    plt.ylabel('Lower Fitness --->')    
    plt.legend(['Avg fitness', 'Max fitness'], loc='upper left')
    plt.savefig('MAX-T' + str(parameters[8]) + '.png')
    plt.close();
