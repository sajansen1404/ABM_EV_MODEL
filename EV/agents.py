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
    def __init__(self, unique_id, model, vision):
        super().__init__(unique_id, model)
        self.vision = vision            # taken from a slider input
        self.battery = 120              # starting battery
        self.max_battery = np.random.randint(100,200)   # maximum battery size, differs for different cars
        self.total_EV_in_cell = 0       # initial value
        self.usual_charge_time = 10      # the time period for how long it usually charges
        self.time_charging = 0


    
    # can randomly move in the neighbourhood with radius = vision
    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            radius = self.vision,
            moore=True,
            include_center=True)
        
        # define a random new position
        new_position = random.choice(possible_steps)
        
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


