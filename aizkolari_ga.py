#!/usr/bin/python

#------------------------------------------------------------------------------
#License GPL v3.0
#Author: Alexandre Manhaes Savio <alexsavio@gmail.com>
#Grupo de Inteligencia Computational <www.ehu.es/ccwintco>
#Universidad del Pais Vasco UPV/EHU
#Use this at your own risk!
#2012-06-18
#------------------------------------------------------------------------------

#from IPython.core.debugger import Tracer; debug_here = Tracer()

import numpy as np
import np.random as rand

class aizko_ga:

   #population size
   popsize  = 100

   #subject size
   subjsize = 22 #get from data

   #number of crossover operations
   cjumps = popsize;

   #number of mutations
   mjumps = popsize;

   #probability of crossover
   cprob = 0.5;

   #probability of mutation
   mprob = 0.25;
   genmprob = 0.1;

   #proportion of population to be selected (be careful changing this)
   sperc = 0.5;

   #random initial population
   pop = rand.random_integers(0, 1, [popsize, subjsize])

   #number of generations
   gn = 100;


   def cross (pop, cjumps=None, cprob=None):
   ##returns a new population from crossing cjumps pairs in pop with
   ##probability cprob
   ## if not given cjumps, then cjumps = popsize
   ## if not given cprob, then cprob = 1

      #random new pop size
      [pn,sn] = pop.shape

      if not cjumps:
         cjumps = pn

      if cprob > 0 and cprob <= 1:
         #if only a percentage of selected parents will produce siblings
         npn = np.sum(rand.random([cjumps,1]) > cprob)
      else:
         #if all selected parents will produce siblings
         npn = cjumps

      nupop = np.zeros([npn,sn], dtype=int);

      #Uniform Crossover
      for j in np.arange(npn):
         pi1 = rand.randint(0,pn)
         pi2 = rand.randint(0,pn)
         while pi1 == pi2:
            pi2 = rand.randint(0,pn)

         p1 = pop[pi1,:]
         p2 = pop[pi2,:]
         c = np.array([p1,p2])
         sel = rand.random_integers(0,1, [1,sn]);

         nupop[j,:] = np.choose(sel,c)

   #   #two-point crossover
   #   for j=1:npn
   #      p1 = pop(randi(pn),:);
   #      p2 = pop(randi(pn),:);
   #      x  = randi(sn-1,2,1);
   #      a  = min(x);
   #      b  = max(x);

   #      np(j,1:a)   = p1(1:a);
   #      np(j,a:b)   = p2(a+1:b);
   #      np(j,b:end) = p1(b+1:end);
   #   end

   #   #one-point crossover
   #   for j=1:npn
   #      p1 = pop(randi(pn),:);
   #      p2 = pop(randi(pn),:);
   #      a  = randi(sn-1);
   #      np(j,1:a)   = p1(1:a);
   #     np(j,a:end) = p2(a+1:end);
   #   end

   def fitness (pop):
   #returns a numpy array with the fitness value of each subject from pop
   #this function should be changed for any other fitness function you need
      return np.sum(pop, axis=1)

   def mutate (pop, mjumps, mprob, genmprob):
   ##returns a mutated population, where 
   #pop is the population [number of subjects, size of subject]
   #mjumps is the number of mutation trials
      [pn,sn] = pop.shape
      for j in np.arange(pn):
         if (rand.random() <= mprob):
            r = rand.random (sn)
            ng = rand.random_integers(0,1, np.sum(r <= genmprob))
            pop[j][r <= genmprob] = ng

      return pop


   def select_best (pop, fits, perc):
   #returns the index of the best  
      [pn,sn] = pop.shape;

      #calculating best fits
      fits = fits(:);
      en = np.floor(pn * perc);
      idx = np.argsort(fits);

      #index of best fits
      idxe  = idx(end-en+1:end);
      return idxe

