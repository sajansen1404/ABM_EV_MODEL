### agents.py

import matplotlib.pyplot as plt
import numpy as np
import random
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
        

    def step(self):
        self.amount = min([self.max_sockets, self.amount + 1])

# Create the Electric Vehicles agents
class EV_Agent(Agent):
    """ An agent with fixed initial battery."""
    def __init__(self, unique_id, model, vision, home_pos, work_pos):
        super().__init__(unique_id, model)
        self.unique_id = unique_id
        self.vision = vision            # taken from a slider input
        self.battery = 120              # starting battery
        self.max_battery = np.random.randint(150,200)   # maximum battery size, differs for different cars
        self.total_EV_in_cell = 0       # initial value
        self.usual_charge_time = 10      # the time period for how long it usually charges
        self.time_charging = 0

        self.home_pos = home_pos            # Agent lives here
        self.work_pos = work_pos            # Agent works here
        self.shopping_pos = (0, 0)          # Will be set later
        self.target = "work"                # Sets off from home at first
        self.target_pos = self.work_pos[:]
        self.braveness = 1

    # can randomly move in the neighbourhood with radius = vision
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            radius = self.vision,
            moore=True,
            include_center=True)

        #if self.unique_id == 0 and self.battery >= 100:
            #print("Target: {}, target_pos: {}, current_pos: {}".format(self.target, self.target_pos, self.pos))
            #print(self.target_pos in possible_steps)

        if len(possible_steps) > 0:
            # Steps towards the target and chooses a position with the shortest remaining distance
            new_position = possible_steps[0][:]
            new_distance = (new_position[0]-self.target_pos[0])**2 + (new_position[1]-self.target_pos[1])**2
            for candidate_position in possible_steps:
                candidate_distance = (candidate_position[0]-self.target_pos[0])**2 + (candidate_position[1]-self.target_pos[1])**2
                if candidate_distance < new_distance:
                    new_distance = candidate_distance
                    new_position = candidate_position[:]

            # Arrives, if target_pos == new_position. Need to reset the target.
            if self.target_pos[0] == new_position[0] and self.target_pos[1] == new_position[1]:
                # Target: work -> shopping, shopping -> home, home -> work
                if self.target == "work":
                    self.target = "shopping"
                    # New coordinates
                    hw_dist = np.sqrt((self.home_pos[0]-self.work_pos[0])**2 + (self.home_pos[1]-self.work_pos[1])**2)
                    center_pos = ((self.home_pos[0]+self.work_pos[0])/2, (self.home_pos[1]+self.work_pos[1])/2)
                    self.target_pos[0] = self.braveness*np.random.randint(np.max([center_pos[0] - hw_dist, 0]),
                                                           np.min([center_pos[0] + hw_dist, self.model.grid.width]))
                    self.target_pos[1] = self.braveness*np.random.randint(np.max([center_pos[1] - hw_dist, 0]),
                                                           np.min([center_pos[1] + hw_dist, self.model.grid.height]))

                elif self.target == "shopping":
                    # Goes home
                    self.target = "home"
                    self.target_pos = self.home_pos[:]

                else:
                    # Goes to work
                    self.target = "work"
                    self.target_pos = self.work_pos[:]



        # define a random new position
        # new_position = random.choice(possible_steps)
        # if agent is under a certain battery level it will find a CP
        if self.battery < 100:
            
            # if agent can find CP it will go to that position
            if self.find_CP(self.pos):
                new_position = self.find_CP(self.pos)
        
                #checks if occupied  
                while (self.is_occupied(new_position)):
                    new_position = random.choice(possible_steps)
        
        
        # adds up how long a car is at a charge_pole
        self.time_charging = self.time_at_charge_pole(self.pos, self.time_charging)

        # it will stay at a charge_pole for a certain time period, even if the car is already full
        current_cell = self.model.grid.get_cell_list_contents([self.pos])
        if any(isinstance(occupant, Charge_pole) for occupant in current_cell) and (self.time_charging < self.usual_charge_time):
            new_position = self.pos

        # if on charge_pole it will charge until full
        if self.charge_battery(new_position):
            if self.battery < self.max_battery:
                self.battery += self.charge_battery(new_position)

        # uses battery according to the distance traveled
        self.use_battery(new_position)


        # count total EVs in cell
        this_cell = self.model.grid.get_cell_list_contents([new_position])
        total_EV = 0
        for occupant in this_cell:
            if type(occupant) is EV_Agent:
                total_EV += 1
        self.total_EV_in_cell = total_EV

        self.model.grid.move_agent(self, new_position)


        
    # function taken from sugarscape: get_sugar
    def charge_battery(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        for agent in this_cell:
            if type(agent) is Charge_pole:
                return agent.charge
            
            
    def is_occupied(self, pos):
        this_cell = self.model.grid.get_cell_list_contents([pos])
        #print(this_cell)
        return len(this_cell) > 2

    def time_at_charge_pole(self, pos, time_charging):
        current_cell = self.model.grid.get_cell_list_contents([self.pos])
        if any(isinstance(occupant, Charge_pole) for occupant in current_cell):
            return time_charging + 1
        else:
            return 0
         
    
    
    def find_CP(self, pos):
        neig = self.model.grid.get_neighbors(
            self.pos,
            radius = self.vision, 
            moore = True,
            include_center = True)
        possible_charge_poles =[]
        
        #check if there is an agent in neighborhood, and what type of agents
        if neig:
            for agent in neig:
                if isinstance(agent, Charge_pole):
                    possible_charge_poles.append(agent)
                    
        # randomly select from the CP in the neighborhood -> not perfect yet
        
            if possible_charge_poles:
                
                #take a first random choice of CP in neighborhood
                chosen_charge_pole = random.choice(possible_charge_poles)
                
                #check if it works, else check all other charge_poles
                for count, charge_pole in enumerate(possible_charge_poles):
                    if self.is_occupied(chosen_charge_pole.pos):
                        chosen_charge_pole = charge_pole
                        
                #if there is no free charge_pole return False
                if self.is_occupied(chosen_charge_pole.pos):
                    return False
                        
                return chosen_charge_pole.pos
                
            
    # function to decrease battery with the distance
    def use_battery(self, new_position):
        dist = (distance.euclidean(self.pos, new_position))
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


