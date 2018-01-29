### model.py

import matplotlib.pyplot as plt
import numpy as np
import random
from mesa import Agent, Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from scipy.spatial import distance

from EV.agents import EV_Agent, Charge_pole
from EV.schedule import RandomActivationByBreed


# Mean value of all battery
def mean_all_battery(model):
    agent_battery_levels = [agent.battery for agent in model.schedule.agents]
    return np.mean(agent_battery_levels)


# Find 25. percentile
def lowest_25_percent(model):
    agent_battery_levels = [agent.battery for agent in model.schedule.agents]
    return np.percentile(agent_battery_levels, 25)


# Count the number of agents
def count_agents(model):
    return model.num_agents


# Create the model
class EV_Model(Model):
    def __init__(self, N=50, width=20, height=20, n_poles=10, vision=10, seed=None):
        super().__init__(seed)

        self.num_agents = N

        self.grid = MultiGrid(width, height, False)
        self.schedule = RandomActivationByBreed(self)
        self.vision = vision

        # All grid points: (Terribly inefficient, but works.)
        all_grid_points = []
        for i in np.arange(self.grid.width):
            for j in np.arange(self.grid.height):
                all_grid_points.append([i, j])
        np.random.shuffle(all_grid_points)

        # Create Charge Pole agents
        index = 0
        for i in range(n_poles):
            x, y = all_grid_points[index]
            index = index + 1

            charge_pole = Charge_pole(i, (x, y), self)
            self.grid.place_agent(charge_pole, (x, y))

        # Create EV agents
        for i in range(self.num_agents):
            home_pos = all_grid_points[index]
            index = index + 1
            work_pos = all_grid_points[index]
            index = index + 1

            # Parameters
            max_battery = np.random.randint(150, 200)
            battery = np.random.randint(120, max_battery)
            usual_charge_time = 10
            braveness = 10
            vision = 1

            EV = EV_Agent(i, self, home_pos, work_pos, vision, max_battery, battery, usual_charge_time, braveness)

            self.schedule.add(EV)
            # Add the agent to a random grid cell
            self.grid.place_agent(EV, home_pos)

        self.datacollector = DataCollector(agent_reporters={"Battery": lambda EV: EV.battery},
                                           model_reporters={"Avg_Battery": mean_all_battery,
                                                            "lower25":lowest_25_percent,
                                                            "Num_agents": count_agents,
                                                            "EVs": lambda m: m.schedule.get_breed_count(EV_Agent)})

        self.running = True
            
    def step(self):
        self.schedule.step()
        self.datacollector.collect(self)





