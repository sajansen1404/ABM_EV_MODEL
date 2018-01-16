# server.py
from mesa.visualization.modules import CanvasGrid, ChartModule
from mesa.visualization.ModularVisualization import ModularServer

from EV.agents import EV_Agent, Charge_pole
from EV.model import EV_Model

# server.py
color_dic = {4: "#0066FF",
             3: "#287EFF",
             2: "#78AEFF",
             1: "#CAE0FF"}


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
      if agent.total_EV_in_cell > 0:

            portrayal["Color"] = color_dic[agent.total_EV_in_cell]
    else:

        portrayal["Color"] = "gray"


    # portrayal["Shape"] = "rect"
    # portrayal["Filled"] = "true"
    # portrayal["Layer"] = 0
    # portrayal["w"] = 1
    # portrayal["h"] = 1
    
    return portrayal

grid = CanvasGrid(agent_portrayal, 20, 20)

#canvas_element = CanvasGrid(SsAgent_portrayal, 50, 50, 500, 500)
chart_element = ChartModule([{"Label": "EV_Agent", "Color": "#AA0000"}])

server = ModularServer(EV_Model,
                       [grid, chart_element],
                       "EV Model",
                       {"N": 100, "width": 20, "height": 20, "n_poles": 10})


server.port = 8521 # The default
