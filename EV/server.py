# server.py
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from EV.agents import EV_Agent, Charge_pole
from EV.model import EV_Model

import numpy as np



from mesa.visualization.ModularVisualization import VisualizationElement

class HistogramModule(VisualizationElement):
    package_includes = ["Chart.min.js"]
    local_includes = ["HistogramModule.js"]

    def __init__(self, bins, canvas_height, canvas_width):
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.bins = bins
        new_element = "new HistogramModule({}, {}, {})"
        new_element = new_element.format(bins,
                                         canvas_width,
                                         canvas_height)
        self.js_code = "elements.push(" + new_element + ");"


    def render(self, model):
        battery_vals = [agent.battery for agent in model.schedule.agents]
        hist = np.histogram(battery_vals, bins=self.bins)[0]
        print(hist)
        print(self.bins)
        return [int(x) for x in hist]



from mesa.visualization.UserParam import UserSettableParameter




# server.py
color_dic = {5:"#003A92",
             4: "#003A92",
             3: "#0066FF",
             2: "#287EFF",
             1: "#CAE0FF",
             0: "#CAE0FF"}


def agent_portrayal(agent):
    portrayal = {"Shape": "circle",
                 "Filled": "true",
                 "Layer": 1,
                 "r": 0.5}
    if type(agent) is Charge_pole:
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Layer"]= 0
        if agent.free_poles == 2:
            portrayal["Color"] = "green"
        elif agent.free_poles == 1:
            portrayal["Color"] = "orange"
        else:
            portrayal["Color"] = "red"

    elif type(agent) is EV_Agent:
        portrayal["Color"] = "black"
        portrayal["Layer"] = 1


        if agent.unique_id == 10:
            portrayal["Shape"] = "rect"
            portrayal["w"] = 0.8
            portrayal["h"] = 0.8
            portrayal["Color"] = "blue"
            portrayal["Layer"] = 2

    # portrayal["Shape"] = "rect"
    # portrayal["Filled"] = "true"
    # portrayal["Layer"] = 0
    # portrayal["w"] = 1
    # portrayal["h"] = 1
    
    return portrayal

grid_size = 80
grid_width = grid_size
grid_height = grid_size

grid = CanvasGrid(agent_portrayal, grid_width, grid_height)

#canvas_element = CanvasGrid(SsAgent_portrayal, 50, 50, 500, 500)
chart = ChartModule([{"Label": "Avg_Battery",
                      "Color": "Black"},
                      {"Label": "lower25",
                      "Color": "Red"},
                      {"Label": "unique_battery",
                      "Color": "Green"}],
                    data_collector_name='datacollector')

chart2 = ChartModule([{"Label": "Percentage_failed",
                      "Color": "Black"}])

chart_usage = ChartModule([{"Label": "Usage",
                      "Color": "Black"},
                      #{"Label": "High_Usage",
                      #"Color": "Green"},
                      #{"Label": "Low_Usage",
                      #"Color": "Red"},
                      {"Label": "Percentage_failed",
                      "Color": "Blue"}],
                    data_collector_name='datacollector')

histogram = HistogramModule(list(np.arange(0,121, 10)), 200, 500)
chart_element = ChartModule([{"Label": "EVs", "Color": "#AA0000"}])


#grid_slider = UserSettableParameter('slider', "grid_size", 100, 20, 200, 1)
n_slider = UserSettableParameter('slider', "N", 100, 2, 500, 10)
vision_slider = UserSettableParameter('slider', "vision", 4, 1, 20, 1)
n_poles_slider = UserSettableParameter('slider', "Number of Charge poles", 0.15, 0.05, 0.3, 0.05)
initial_bravery_slider = UserSettableParameter('slider', "Exploration rate (bravery)", 10, 5, 30, 1)
battery_size_slider = UserSettableParameter('slider', "Average battery size", 75, 15, 80, 5)


choice_option = UserSettableParameter('choice', 'Charge pole grid layout', value='random',
                                              choices=['random', 'circle','big circle', 'LHS'])


server = ModularServer(EV_Model,
                       [grid, chart_usage, chart_element],
                       "EV Model",

                       {"N": n_slider, "width": grid_width, "height": grid_height, "n_poles": n_poles_slider, 
                       "vision": vision_slider, "grid_positions": choice_option, "initial_bravery":initial_bravery_slider,"battery_size": battery_size_slider})



server.port = 8521 # The default
