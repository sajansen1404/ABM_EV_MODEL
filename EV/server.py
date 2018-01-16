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

n_slider = UserSettableParameter('slider', "N", 100, 2, 200, 1)
vision_slider = UserSettableParameter('slider', "vision", 10, 1, 20, 1)


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
        portrayal["Color"] = "red"
        portrayal["Shape"] = "rect"
        portrayal["w"] = 1
        portrayal["h"] = 1
        portrayal["Layer"]: 0

    elif type(agent) is EV_Agent:
        if agent.total_EV_in_cell >= 0:
            if agent.total_EV_in_cell > 5:
                agent.total_EV_in_cell = 5
            portrayal["Color"] = color_dic[agent.total_EV_in_cell]
    else:

        portrayal["Color"] = "gray"


    # portrayal["Shape"] = "rect"
    # portrayal["Filled"] = "true"
    # portrayal["Layer"] = 0
    # portrayal["w"] = 1
    # portrayal["h"] = 1
    
    return portrayal

grid = CanvasGrid(agent_portrayal, 50, 50)

#canvas_element = CanvasGrid(SsAgent_portrayal, 50, 50, 500, 500)
chart = ChartModule([{"Label": "Avg_Battery",
                      "Color": "Black"},
                      {"Label": "lower25",
                      "Color": "Red"}],
                    data_collector_name='datacollector')
chart2 = ChartModule([{"Label": "Num_agents",
                      "Color": "Black"}],
                    data_collector_name='datacollector')

histogram = HistogramModule(list(np.arange(0,121, 10)), 200, 500)
chart_element = ChartModule([{"Label": "Battery", "Color": "#AA0000"}])

server = ModularServer(EV_Model,
                       [grid, chart, chart2],
                       "EV Model",
                       {"N": n_slider, "width": 50, "height": 50, "n_poles": 20, "vision": vision_slider})


server.port = 8521 # The default
