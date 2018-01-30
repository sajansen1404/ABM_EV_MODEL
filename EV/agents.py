#   agents.py

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
        self.free_poles = 2
        self.charge_speed = 20  

    def step(self):
        pass

# Create the Electric Vehicles agents
class EV_Agent(Agent):
    """ An agent with fixed initial battery."""
    def __init__(self, unique_id, model, vision, home_pos, work_pos, initial_bravery):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.vision = vision                                        # taken from a slider input
        self.max_battery = np.random.randint(70,80)                 # maximum battery size, differs for different cars is between 70 and 80 kwh (all tesla)
        self.battery = np.random.randint(50,self.max_battery)       # starting battery 
        self.usual_charge_time = np.random.normal(25,10) 			# the time period for how long it usually charges
        self.charge_speed = 3                                       # the battery increase for every timestep at a charge station
        self.time_charging = 0
        self.state = np.random.choice(["working", "shopping", "at_home", "traveling"])	#inital state
        self.time_in_state = np.random.randint(0,30)	#initial value to make sure not everyone moves at the same time
        self.how_long_at_work = np.random.normal(25, 3)  #initial value for time to stay at work
        self.how_long_shopping = np.random.normal(5, 3)
        self.how_long_at_home = np.random.normal(30, 5) #if at home, ususally stays for 30 timesteps
        self.minimum_battery_to_look_for_cp = abs(np.random.normal(30, 10))

        self.current_strategy = 0
        self.pole_count = 0                                                # counts the amount of charging_pole encounters (to calculate the 'age' of memories)
        self.strategies = [[1,1,1,1,1,1,1,1,1,1],[1,1,1,1,1,0,0,0,0,0],[1,1,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]] # different strategies used, which memories count in each strategyg
        self.initMemory() 
        self.possible_steps = []
        self.offLimits = []
        self.prev_target = ""
        self.prev_target_pos = []
        self.cpf = [0.25,0.5,0.75,1.0]

        self.home_pos = home_pos            # Agent lives here
        self.work_pos = work_pos            # Agent works here
        self.shopping_pos = (0, 0)          # Will be set later
        self.target = np.random.choice(["work", "home", "shopping"])             # has one of the targets first
        self.target_pos = self.work_pos[:]
        self.braveness = 1

        # the amount of tiles it will explore away from the middle between home and work is normally distributed
        # this will be lower the more poles found, and is exponentially distributed.
        self.initial_bravery = abs(round(np.random.normal(initial_bravery, 5))) 



    # can randomly move in the neighbourhood with radius = vision
    def move(self):

    	self.time_in_state = self.checkState(self.state, self.time_in_state)
    	# if self.unique_id == 10:
    	# 	print(len(self.memory), self.memory)
    	if self.time_in_state == 0:
	        self.checkTargets()                                                     # checks if needs to look for chargepole, and checks if target is reached
	        if not (self.target == "charge_pole" and self.pos == self.target_pos):  # if not charging (anymore):
	            self.getNeighbourhood()                                             # - find possible moves and register charging poles within vision
	            self.chooseNextStep()                                               # - choose next steps based on target and possible moves
	            self.moveEV()                                                       # move and use battery
            


	# checks what the agent is doing, and let it be in that state for a while
    def checkState(self, state, time_in_state):
        if state ==  "working":
            if time_in_state < self.how_long_at_work:
                return (time_in_state + 1)
            else:
                return 0
        elif state == "shopping":
            if time_in_state < self.how_long_shopping:
                return (time_in_state + 1)
            else:
                return 0
        elif state == "at_home":
            if time_in_state < self.how_long_at_home:
                return time_in_state + 1
            else:
                return 0
        else:
        	return 0


    # registers possible moves and charging pole positions    
    def getNeighbourhood(self):
        self.possible_steps = self.model.grid.get_neighborhood(self.pos,radius = 1,moore=True,include_center=True) # possible moves
        self.polesInSight = self.checkForPoles()  # returns positions of all poles within vision

        done = False
        neighbors = []    # array of neighbors, saved to prevent updating the memory of the same pole many times in a row, because it is close
        # for each pole in vision, if not just updated, check for free spaces and update memory
        for point in self.polesInSight:
            if self.inLastPoints(point) == False:
                neighbors.append(point)
                if self.checkIfFree(point)>0:
                    self.updateMemory(1,point)
                    # if the pole has space and the battery is verylow or no other pole was chosen yet, it sets the pole as target (to give the strategies a chance I don't always grab the closest pole)
                    if (self.battery < 5 and done == False) or (self.battery < self.minimum_battery_to_look_for_cp and self.target != "charge_pole"):
                        if self.target != "charge_pole" and self.target != "searching": 
                            self.prev_target = self.target           # store target and position to continue to after charging
                            self.prev_target_pos = self.target_pos
                        self.target_pos = point
                        self.target = "charge_pole"
                        done = True
                else:
                    # if the pole is full, store this information to memory so when looking for a pole it won't visit the 3 (at this point) last full poles it passed by
                    self.updateMemory(-1,point)
                    if self.battery < self.minimum_battery_to_look_for_cp:
                        if len(self.offLimits) < 4:
                            self.offLimits.append(point)
                        else:
                            self.offLimits = [point]+self.offLimits[:-1]
        self.neighborMemory(neighbors)

    # adds all new neighboring poles to memory, replacing older memories
    def neighborMemory(self,neighbors):  
        self.memory["neighborPoles"] = [neighbors], self.memory["neighborPoles"][:-1]

    # checks for poles within self.vision and returns the positions
    def checkForPoles(self):  
        poles = []
        # old method
        # cells = self.model.grid.get_neighborhood(self.pos,radius = self.vision,moore=True,include_center=True)

        # for cell in cells:
        #     for agent in self.model.grid.get_cell_list_contents(cell):
        #         if type(agent) == Charge_pole:
        #             poles.append(cell)

        # new method, is faster, does not loop trough the whole neighborhood
        neig = self.model.grid.get_neighbors(self.pos,radius = self.vision,moore=True,include_center=True)
        for agent in neig:
        	if type(agent) == Charge_pole:
        		poles.append(agent.pos)
        

        return poles

    # returns the free sockets of a pole at a given position
    def checkIfFree(self,pos):
        for agent in self.model.grid.get_cell_list_contents(pos):
            if type(agent) == Charge_pole:
                if agent.free_poles>2:
                    print("too many poles!!!!", agent.free_poles,pos)
                if agent.free_poles<0:
                    print("too little poles!!!!", agent.free_poles,pos)
                return agent.free_poles

    # registers that the car starts charging and a space at the pole is taken
    def takePlace(self):
        for agent in self.model.grid.get_cell_list_contents(self.pos):
            if type(agent) == Charge_pole:
                agent.free_poles = agent.free_poles - 1

    # registers that charging is complete and a space at the pole is freed
    def freePlace(self):
        for agent in self.model.grid.get_cell_list_contents(self.pos):
            if type(agent) == Charge_pole:
                agent.free_poles = agent.free_poles + 1

    # checks whether given position is in neighborMemory
    # to prevent from updating the same pole memory every step
    def inLastPoints(self,pos):
        for timepoint in self.memory["neighborPoles"]:
            for coordinate in timepoint:
                if coordinate == pos:
                    return True
        return False
    
    # charges and checks if conditions for charging complete are met
    def charge(self):
        self.time_charging = self.time_charging + 1
        if self.time_charging < self.usual_charge_time or self.battery < self.max_battery:  #self.time_charging < self.usual_charge_time or 
            if self.battery < self.max_battery:
                self.battery += self.charge_speed
                if self.battery > self.max_battery: 
                    self.battery = self.max_battery
        # if battery is done charging and minimum charging time is over, stop charging
        else: 
            # resets values, goes back to previous targets and frees socket space
            self.target = self.prev_target
            self.target_pos = self.prev_target_pos
            self.time_charging = 0
            self.current_strategy = 0
            self.offLimits = []
            self.freePlace()

    # checks whether EV needs to look for charger and whether targets are reached
    def checkTargets(self):
        if self.battery < self.minimum_battery_to_look_for_cp and self.target != "charge_pole" and self.target != "searching":
            self.chooseTargetPole()

        if self.target_pos[0] == self.pos[0] and self.target_pos[1] == self.pos[1]:
            # Target: work -> shopping, shopping -> home, home -> work, searching -> searching (new target position), charge_pole -> ____ -> prev_target
            if self.target == "work":
                self.target = "shopping"
                self.state = "working"
                self.how_long_shopping = np.random.normal(5, 3) #if at shopping center, stays around 5 timesteps
                self.newRandomPos() # self.target_pos is selected         

            elif self.target == "shopping":
                # Goes home
                self.target = "home"
                self.state = "shopping"
                self.how_long_at_home = np.random.normal(30, 5) #if at home, ususally stays for 30 timesteps
                self.target_pos = self.home_pos[:]

            elif self.target == "searching":
            	self.state = "searching"
            	self.chooseTargetPole()
            elif self.target == "charge_pole":
            	self.state = "searching"
            	if self.time_charging == 0:
                    if self.checkIfFree(self.pos) > 0:
                        self.takePlace()
                        #print("target reached: charging", self.unique_id)
                        self.charge()
                    else:
                        if len(self.offLimits) < 4:
                            self.offLimits.append(self.pos)
                        else:
                            self.offLimits = [self.pos]+self.offLimits[:-1]
                        self.chooseTargetPole()
            	else:
                    self.charge()
            else:
                # Goes to work
                self.target = "work"
                self.state = "at_home"
                self.how_long_at_work = np.random.normal(25, 3)  #if at work, ususally stays for 25 timesteps
                self.target_pos = self.work_pos[:]
        else:
            self.state = "traveling"

    # chooses new random position around center
    def newRandomPos(self):
        # Coordinates between home and work
        center_pos = (int((self.home_pos[0]+self.work_pos[0])/2),
                      int((self.home_pos[1]+self.work_pos[1])/2))
        hw_dist = np.sqrt((self.home_pos[0]-self.work_pos[0])**2 + (self.home_pos[1]-self.work_pos[1])**2)

        polesInMemory = len(self.memory)- 5 # initial memory with strategies is 5, every pole added in memory is one extra

        if polesInMemory == 0:
        	bravery = round(np.random.exponential(self.initial_bravery))
        else:
        	bravery = round(np.random.exponential(self.initial_bravery/polesInMemory)) # exponential function to get random shopping position distance

        # newPos = np.random.choice(self.model.grid.get_neighborhood(center_pos, radius = int(bravery), moore = True, include_center = True), 1) 
        if bravery != 0:
            newPos = (np.random.choice([np.max([center_pos[0] - bravery, 0]),
        		np.min([center_pos[0] + bravery, self.model.grid.width - 1])]),
        	np.random.choice([np.max([center_pos[1] - bravery, 0]),
        		np.min([center_pos[1] + bravery, self.model.grid.height - 1])]))
        else:
        	newPos = center_pos

	   
        self.target_pos = newPos
        

    def chooseNextStep(self):
        # Steps towards the target and chooses a position with the shortest remaining distance
        self.new_position = self.possible_steps[0]
        new_distance = distance.euclidean(self.new_position, self.target_pos)
        for candidate_position in self.possible_steps:
            candidate_distance = distance.euclidean(self.target_pos, candidate_position)
            if candidate_distance < new_distance:
                new_distance = candidate_distance
                self.new_position = candidate_position[:]
    
    # changes position an drains battery
    def moveEV(self):
        self.use_battery()
        self.model.grid.move_agent(self,self.new_position)
    
    # initiates memory dictionary and score dictionary
    def initMemory(self):
        self.memory = {}
        self.scores = {}
        for i in range(len(self.strategies)): # for each strategy create a (neutral) memory
            self.memory[i+1]=[[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]]
        self.memory["neighborPoles"] = [[0],[0],[0]]
        self.updateStrategies()

    # saves new memories
    def updateMemory(self,succes,pos):
        self.pole_count += 1
        if pos in self.memory:
            self.memory[pos] = [[succes]+self.memory[pos][0][:-1],[self.pole_count]+self.memory[pos][1][:-1]]
        else:
            self.memory[pos] = [[succes]+[0,0,0,0,0,0,0,0,0],[self.pole_count]+[0,0,0,0,0,0,0,0,0]]
        if self.current_strategy > 0 and pos[0] == self.target_pos[0] and pos[1] == self.target_pos[1]:
            self.memory[self.current_strategy] = [[succes]+self.memory[self.current_strategy][0][:-1],[self.pole_count]+self.memory[self.current_strategy][1][:-1]]
            self.updateStrategies()
        self.updateScores(pos)
    
    # updates cumulative probability function based on new memories for strategy    
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
    
    # updates scores when memory changes
    def updateScores(self,pos):
        if pos not in self.scores:
            self.scores[pos] = [0,0,0,0]
        age = self.ageCompensation(pos)
        for i in range(len(self.strategies)):
            temp = 0
            for j in range(len(age)):
                temp += self.strategies[i][j]*age[j]
            self.scores[pos][i] = temp

    # strategy chosen based on cumulative probability function
    def chooseStrategy(self):
        r = np.random.rand()
        for i in range(len(self.cpf)):
            if r<self.cpf[i]:
                return i+1

    # compensates the age of memories by formula y = score * 0.98 ^ (current pole_count - pole_count attached to memory)
    def ageCompensation(self,key):
        result=[]
        for i in range(len(self.memory[key][1])):
            result.append( self.memory[key][0][i] * math.pow(0.98,self.pole_count-self.memory[key][1][i]))
        return result

    # if possible, chooses target pole. Otherwise starts exploring
    def chooseTargetPole(self):
        if self.target != "searching" and self.target != "charge_pole":
            self.prev_target = self.target 
            self.prev_target_pos = self.target_pos
        self.current_strategy = self.chooseStrategy()
        
        options = self.checkOptions()
        
        # if no options, explore
        if len(options) == 0:
            self.target = "searching"
            #completely random position, and not somewhere close to its center
            self.target_pos = (np.random.randint(0,self.model.grid.width), np.random.randint(0, self.model.grid.height))
            #self.newRandomPos()
        else:
            OptionScores = []

            # The option array is an array with [[position, best sore], [position, best score], ...] depending on how
            # many options are available. In this for loop the distance between every CP and the agent is
            # calculated. This distance is put into a linear declining formula which is multiplied by the already
            # existing score. After every all the multiplications are done, the CP with the best score is chosen as
            # the new target position. We still have to discuss if we want it to be exp. of lin. declining function.
            # (depending on how strong we want the distance to affect the EV)
            for i in np.arange(np.shape(options)[0]):
                dist = abs(self.pos[0]-self.pos[1]) + abs(options[i][0][0]-options[i][0][1])
                a_lin = 1 / 50
                w_dist_lin = (-a_lin * dist) + 1
                new_CP_score = w_dist_lin * options[i][1]
                OptionScores.append(new_CP_score)

            ind_highest_score = np.argmax(OptionScores)
            self.target_pos = options[ind_highest_score][0]
            #self.target_pos = random.choice(options)[0]
            self.target = "charge_pole"
            #print("target set",self.unique_id,self.target_pos)

    # goes through known poles and checks if they're options as targets
    def checkOptions(self):
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
        return options
            
    # function to decrease battery with the distance
    def use_battery(self):
        dist = (distance.euclidean(self.pos, self.new_position))

        # average battery cost per km is between 0.08 and 0.3 kwh
        cost = dist * ((0.30 - 0.08) * np.random.random_sample() + 0.08)
        self.battery -= cost
    
    def step(self):
        if self.battery <= 0:
            self.model.grid._remove_agent(self.pos, self)
            self.model.schedule.remove(self)
        if self.battery > 0:
            self.move()
            #print(self.unique_id, self.battery)
