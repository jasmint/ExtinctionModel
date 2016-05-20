# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 14:36:49 2015

@author: Jasmin
 
"""
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.ioff()

name = 'Data-0,10TEST2'
runData = pickle.load(open('C:\Users\Jasmin\Documents\GitHub\ExtinctionModel\\' + name, 'r'))

#Organize Data
functionData = runData[0] #data returned from function: [[[{fitness:abund, extra:x},{}][parameters]]]
parameters = functionData[0][len(functionData[0])-1] #[b,k,n,s,u,genome,beginf,cleanUp,rstr,distGap]
variable = runData[1] #recombination values passed into function

# Enter directory to save graphs for one run
os.mkdir(str(min(variable))+'-'+str(max(variable)))
os.chdir(str(min(variable))+'-'+str(max(variable)))
intervals = parameters[9] #how often snapshots were taken

meanI = ['NONE']
maxI = ['NONE'] 
minI = ['NONE']
width = ['NONE']
fitVar = ['NONE']  
 
# Functiondata[x] where x is the rstr passed into the function(beginning at 0)
for i in range(0,len(variable)):
    rstr = variable[i]
    rstrData = functionData[i] # data for one run at one rstr
    
    snapshots = rstrData[0] # snapshots for one run at one rstr - list
    parameters = rstrData[1] # parameters from one run at one rstr:[b,k,n,s,u,genome,beginf,cleanUp,rstr,distGap]
        
    # Enter directory for one variable rstr value   
    os.mkdir(str(rstr)+'recomb')
    os.chdir(str(rstr)+'recomb')
    
    for l in range(0, len(snapshots)): # loop through every snapshot at one rstr value 
        onesnap = snapshots[l]# info from one snapshot - dict
        classes = []
        abund = []             
        
        iterationNum = l*intervals #onesnap['extra'][0]
        
        #Pull distribution fitness/abundances data into two arrays
        for key in onesnap: 
            if key != 'extra':            
                classes.append(key)
                abund.append(onesnap[key])

        # Plot population distribution graphs
        if l%1 == 0 and l>600000:              
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
            plt.savefig('Iter(' + str(l * intervals + intervals) + ').png')
            plt.close();
            
        # Calculate mean fitness
        sumI = 0
        for m in range(0, len(classes)):
            sumI = sumI + classes[m] * abund[m]
        muI = sumI / float(sum(abund))
        meanI.append(muI)
        
        # Find maximum fitness
        maximum = max(classes)
        #n = 2
        #while onesnap[maximum] < 100:          
        #    largest = heapq.nsmallest(n, classes)
        #    maximum = max(largest)
        #    n = n + 1
        #    if n == len(onesnap)-1:
        #        maximum = max(classes)
        #    break        
        maxI.append(maximum)        
        
        # Find lowest fitness
        minimum = min(classes)
        #n = 2
        #while onesnap[maximum] < 100:          
        #    largest = heapq.nsmallest(n, classes)
        #    minimum = min(largest)
        #    n = n + 1
        #    if n == len(onesnap)-1:
        #        minimum = min(classes)
        #    break        
        minI.append(minimum)        
        
        # Calculate width of distribution
        width.append(maximum-minimum)        
        
        # Calculate fitness variance 
        N = float(sum(abund))   # population size
        sigSquared = 0
        for y in range(0,len(classes)):
            fit = classes[y] # class
            nI = abund[y] # individuals in the class
            pI = nI / N     
            sigSquared = sigSquared + (float(fit) - muI)**2 * pI
        fitVar.append(sigSquared)        
        
    meanI.append('NONE')
    maxI.append('NONE') 
    minI.append('NONE')  
    width.append('NONE')
    fitVar.append('NONE')
    
    os.chdir("..")

#print means/maxes; NONE between runs of function at one rstr value 
#print "Mean fitnesses: ", meanI
#print "Max fitnesses: ", maxI

# Split arrays
indicesMean = [i for i, x in enumerate(meanI) if x == "NONE"]
indicesMax = [i for i, x in enumerate(maxI) if x == "NONE"]
indicesMin = [i for i, x in enumerate(minI) if x == "NONE"]
indicesWidth = [i for i, x in enumerate(width) if x == "NONE"]
indicesFitVar = [i for i, x in enumerate(fitVar) if x == "NONE"]

# Graphing 
# mean and max fitness vs time for each recom vlaue
for p in range(0,len(variable)):
    # Recomb value [x-value]
    var = variable[p]
    
    beg = indicesMean[p] + 1
    end = indicesMean[p+1]
    
    # Take data chunks
    meanChunk = meanI[beg:end]     
    maxChunk = maxI[beg:end]
    minChunk = minI[beg:end]
    widthChunk = width[beg:end]
    fitVarChunk = fitVar[beg:end]
    
    time = [j * intervals for j in range(0, len(meanChunk))]
    
    #Plot average
    plt.plot(time, meanChunk, '.r-')      
    plt.ylim(0,max(meanChunk)+10)  
    plt.xlim(0,max(time))
    plt.xlabel('Iterations ('+str(intervals)+')')
    plt.ylabel('Higher Fitness --->')    
    #plt.legend(['Avg fitness', 'Max fitness'], loc='upper left')
    plt.savefig('AVG-' + str(var) + 'recomb.png')
    plt.close();
    
    # Plot Width
    plt.plot(time, widthChunk, '.g-')      
    plt.ylim(0,max(widthChunk)+10)  
    plt.xlabel('Iterations ('+str(intervals)+')')
    plt.ylabel('Width of distribution')    
    #plt.legend(['Avg fitness', 'Max fitness'], loc='upper left')
    plt.savefig('WIDTH-' + str(var) + 'recomb.png')
    plt.close();
    
    # Plot fitness variance
    plt.plot(time, fitVarChunk, '.m-')      
    plt.ylim(0,max(fitVarChunk)+10)  
    plt.xlabel('Iterations ('+str(intervals)+')')
    plt.ylabel('Fitness variance')    
    #plt.legend(['Avg fitness', 'Max fitness'], loc='upper left')
    plt.savefig('FITVAR-' + str(var) + 'recomb.png')
    plt.close();
    
    
    #Plot Max
    plt.plot(time, maxChunk, '.b-')  
    plt.ylim(0,max(maxChunk)+10)  
    plt.xlabel('Iterations')
    plt.ylabel('Higher Fitness --->')    
    #plt.legend(['Avg fitness', 'Max fitness'], loc='upper left')
    plt.savefig('MAX-' + str(var) + 'recomb.png')
    plt.close();
 