### agents.py

import matplotlib.pyplot as plt
import numpy as np
import random
import math
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from scipy.spatial import distance


# Create charging pole agents
class Charge_pole(Agent):
    def __init__(self, unique_id, pos, model):
        super().__init__(pos, model)
        max_charge = 3*model.vision
        max_sockets = 2
        self.charge = max_charge
        self.max_sockets = max_sockets
        self.pos = pos
        self.free_poles = 2
        self.attached = []
        

    def step(self):
        self.amount = min([self.max_sockets, self.amount + 1])

# Create the Electric Vehicles agents
class EV_Agent(Agent):
    """ An agent with fixed initial battery."""
    def __init__(self, unique_id, model, vision, home_pos, work_pos):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.vision = 1                                                    # taken from a slider input
        self.max_battery = np.random.randint(150,200)                      # maximum battery size, differs for different cars
        self.battery = np.random.randint(120,self.max_battery)             # starting battery 
        self.usual_charge_time = 10                                        # the time period for how long it usually charges
        self.time_charging = 0
        self.current_strategy = 0
        self.pole_count = 0                                                # counts the amount of charging_pole encounters (to calculate the 'age' of memories)
        self.strategies = [[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]] # different strategies used, which memories count in each strategyg
        self.initMemory() 
        self.possible_steps = []
        self.offLimits = []

        self.home_pos = home_pos            # Agent lives here
        self.work_pos = work_pos            # Agent works here
        self.shopping_pos = (0, 0)          # Will be set later
        self.target = "work"                # Sets off from home at first
        self.target_pos = self.work_pos[:]
        self.braveness = 1

    # can randomly move in the neighbourhood with radius = vision
    def move(self):
        print(self.unique_id, self.pos)
        self.checkTargets()
        if not (self.target == "charge_pole" and self.pos == self.target_pos):
            self.getNeighbourhood()
            self.chooseNextStep()
            self.moveEV()
            
        
    def getNeighbourhood(self):
        self.possible_steps = self.model.grid.get_neighborhood(self.pos,radius = 1,moore=True,include_center=True)
        self.polesInSight = self.model.grid.get_neighborhood(self.pos,radius = self.vision,moore=True,include_center=True)

        done = False
        for point in self.polesInSight:
            for agent in self.model.grid.get_cell_list_contents(point):
                if type(agent) == Charge_pole:
                    if agent.free_poles>0:
                        self.updateMemory(1,agent.pos)
                        if (self.battery < 50 and done == False) or (self.battery < 100 and self.target != "charge_pole"):
                            print("looking for charger",self.unique_id, self.battery, agent.pos)
                            print(self.unique_id,agent.free_poles)
                            agent.free_poles  = agent.free_poles - 1
                            print(self.unique_id,agent.free_poles)
                            if self.target != "charge_pole" and self.target != "searching": 
                                self.prev_target = self.target
                                self.prev_target_pos = self.target_pos
                            self.target_pos = agent.pos
                            self.target = "charge_pole"
                            done = True
                    else:
                        self.updateMemory(-1,agent.pos)
                        if self.battery < 100:
                            if len(self.offLimits) < 4:
                                self.offLimits.append(self.pos)
                            else:
                                self.offLimits = [self.pos]+self.offLimits[:-1]
    def charge(self):
        print(self.unique_id, self.battery, self.pos)
        self.time_charging = self.time_charging + 1
        if self.time_charging < self.usual_charge_time or self.battery < self.max_battery:
            if self.battery < self.max_battery:
                self.battery += 10
                if self.battery > self.max_battery: 
                    self.battery = self.max_battery
        else: 
            print("done", self.unique_id, self.battery, self.pos)
            self.target = self.prev_target
            self.target_pos = self.prev_target_pos
            self.time_charging = 0
            self.offLimits = []
            for agent in self.model.grid.get_cell_list_contents([self.pos]):
                if type(agent) is Charge_pole:
                    print(self.unique_id,agent.free_poles)
                    agent.free_poles  = agent.free_poles + 1
                    print(self.unique_id,agent.free_poles)

    def checkTargets(self):
        if self.battery < 100 and self.target != "charge_pole" and self.target != "searching":
            self.chooseTargetPole()

        if self.target_pos[0] == self.pos[0] and self.target_pos[1] == self.pos[1]:
            # Target: work -> shopping, shopping -> home, home -> work
            if self.target == "work":
                self.target = "shopping"
                # New coordinates
                hw_dist = np.sqrt((self.home_pos[0]-self.work_pos[0])**2 + (self.home_pos[1]-self.work_pos[1])**2)
                center_pos = ((self.home_pos[0]+self.work_pos[0])/2, (self.home_pos[1]+self.work_pos[1])/2)
                self.target_pos[0] = self.braveness*np.random.randint(np.max([center_pos[0] - hw_dist, 0]),np.min([center_pos[0] + hw_dist, self.model.grid.width]))
                self.target_pos[1] = self.braveness*np.random.randint(np.max([center_pos[1] - hw_dist, 0]),np.min([center_pos[1] + hw_dist, self.model.grid.height]))
            elif self.target == "shopping":
                # Goes home
                self.target = "home"
                self.target_pos = self.home_pos[:]
            elif self.target == "searching":
                self.chooseTargetPole()
            elif self.target == "charge_pole":
                if self.time_charging == 0:
                    print("target reached: charging", self.unique_id)
                self.charge()
            else:
                # Goes to work
                self.target = "work"
                self.target_pos = self.work_pos[:]

    def chooseNextStep(self):
        # Steps towards the target and chooses a position with the shortest remaining distance
        self.new_position = self.possible_steps[0]
        new_distance = (self.new_position[0]-self.target_pos[0])**2 + (self.new_position[1]-self.target_pos[1])**2
        for candidate_position in self.possible_steps:
            candidate_distance = (candidate_position[0]-self.target_pos[0])**2 + (candidate_position[1]-self.target_pos[1])**2
            if candidate_distance < new_distance:
                new_distance = candidate_distance
                self.new_position = candidate_position[:]
    
    def moveEV(self):
        self.use_battery()
        self.model.grid.move_agent(self,self.new_position)
    
    def initMemory(self):
        # for each strategy create a (neutral) memory
        self.memory = {}
        self.scores = {}
        for i in range(len(self.strategies)):
            self.memory[i+1]=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
        self.cpf = []
        self.updateStrategies()

    def updateMemory(self,succes,pos):
        self.pole_count += 1
        if pos in self.memory:
            self.memory[pos] = [[succes]+self.memory[pos][0][:-1],[self.pole_count]+self.memory[pos][1][:-1]]
        else:
            self.memory[pos] = [[succes]+[0,0,0,0,0,0,0,0,0],[self.pole_count]+[0,0,0,0,0,0,0,0,0]]
        if self.current_strategy > 0 and pos == self.target_pos:
            self.memory[self.current_strategy] = [[succes]+self.memory[self.current_strategy][0][:-1],[self.pole_count]+self.memory[self.current_strategy][1][:-1]]
        self.updateStrategies()
        self.updateScores(pos)
        
    def updateStrategies(self):
        prev=0
        self.cpf = []
        for i in range(len(self.strategies)):
            next = (sum(self.ageCompensation(i+1)) + 10) # does this make sense? minimal -10, maximum 10. Make positive, nonzero number.
            if next == 0:
                next = 0.000001
            self.cpf.append(prev + next)
            prev=self.cpf[i]
        
        for i in range(len(self.cpf)):
            self.cpf[i]  = self.cpf[i] / self.cpf[len(self.strategies)-1]
    
    def updateScores(self,pos):
        if pos not in self.scores:
            self.scores[pos] = [0,0,0,0]
        age = self.ageCompensation(pos)
        for i in range(len(self.strategies)):
            temp = 0
            for j in range(len(age)):
                temp += self.strategies[i][j]*age[j]
            self.scores[pos][i] = temp

    # call self.current_strategy = self.chooseStrategy() - chosen based on cumulative probability function
    def chooseStrategy(self):
        r = np.random.rand()
        for i in range(len(self.cpf)):
            if i<r:
                return i+1

    def ageCompensation(self,key):
        result=[]
        for i in range(len(self.memory[key][1])):
            result.append( self.memory[key][0][i] * math.pow(0.98,self.pole_count-self.memory[key][1][i]))
        return result

    def chooseTargetPole(self):
        print("choosing target pole",self.unique_id)
        if self.target != "searching" and self.target != "charge_pole":
            self.prev_target = self.target 
            self.prev_target_pos = self.target_pos
        self.current_strategy = self.chooseStrategy()
        
        # get scores for current strategy
        num=0
        options = []
        for key in self.scores:
            options.append([key,self.scores[key][self.current_strategy-1]])
            num += 1

        # check whether any of the scores are 'off limits'
        found = 0
        if num>0:
            for opt in options:
                if opt[0] not in self.offLimits:
                    found += 1
                else: 
                    options.remove(opt)

        # if no options, explore
        if found == 0:
            self.target = "searching"
            # I used this part from 'shopping'. thought it might have the highest chance of finding a new pole, as opposed to random movement
            hw_dist = np.sqrt((self.home_pos[0]-self.work_pos[0])**2 + (self.home_pos[1]-self.work_pos[1])**2)
            center_pos = ((self.home_pos[0]+self.work_pos[0])/2, (self.home_pos[1]+self.work_pos[1])/2)
            min_lim = np.max([center_pos[0] - hw_dist, 0])
            max_lim = np.min([center_pos[0] + hw_dist, self.model.grid.width])
            self.target_pos = (self.braveness*np.random.randint(np.max([center_pos[0] - hw_dist, 0]),np.min([center_pos[0] + hw_dist, self.model.grid.width])),self.braveness*np.random.randint(np.max([center_pos[1] - hw_dist, 0]),np.min([center_pos[1] + hw_dist, self.model.grid.height])))
        # until final score functions: choose random from choices
        else: 
            self.target_pos = random.choice(options)[0]
            self.target = "charge_pole"
            for p in self.polesInSight:
                if p == self.target_pos:
                    for Agent in self.model.grid.get_cell_list_contents(p):
                        if type(Agent) == Charge_pole:
                            Agent.free_poles = Agent.free_poles - 1
            print("target set",self.unique_id,self.target_pos)
                
            
    # function to decrease battery with the distance
    def use_battery(self):
        dist = (distance.euclidean(self.pos, self.new_position))
        cost = dist
        self.battery -= cost
    
    def step(self):
        #self.total_EV_in_cell = self.total_EV_in_cell
        if self.battery <= 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
        if self.battery > 0:
            self.move()
            #print(self.unique_id, self.battery)


