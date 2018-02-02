
#run.py
from Batch import batchRunner
from EV.model import *
import multiprocessing as mp
import pandas as pd
from time import time
import os


cores = mp.cpu_count()
iterations = 100

def run_model():
    output = []
    fixed_params = {"N": [],
                    "n_poles": [],
                    "vision": [],
                    "grid_positions": [],
                    "open_grid": [],
                    "width": 80,
                    "height": 80,
                    "initial_bravery": 10,
                    "battery_size": 75}


    RandomParams = {"N": np.random.uniform(100,400,iterations),
                    "n_poles": np.random.uniform(0.1,0.25,iterations),
                    "vision": np.random.choice([1,2],iterations),
                    "grid_positions": np.random.choice(["LHS","circle"],iterations),
                    "open_grid": np.random.choice(["True","False"],iterations)}

    for i in range(iterations):
        keys = []
        for key in RandomParams:
            if key == "N":
                fixed_params[key] = int(RandomParams[key][i])
            else:
                fixed_params[key] = RandomParams[key][i]
            keys.append(fixed_params[key])

        batch_run = batchRunner(EV_Model,
                                fixed_parameters=fixed_params,
                                variable_parameters={},
                                iterations=1,
                                max_steps=2500,
                                model_reporters={"Usage": avg_usage,
                                                 "Total_attempts": totalAttempts,
                                                 "Percentage_failed": percentageFailed,
                                                 "Average_lifespan": averageLifespan})

        batch_run.run_all()
        run_data = batch_run.get_model_vars_dataframe().values.tolist()
        run_data[0].extend(keys)
        output.extend(run_data)
    return output


if __name__ == "__main__":
    results = []
    pool = mp.Pool(cores)
    for i in range(7):
        def callback(result):
            results.extend(result)
        pool.apply_async(run_model, callback=callback)
    pool.close()
    pool.join()
    df = pd.DataFrame(results, columns=["run","Average_lifespan","Percentage_failed","Total_attempts","Usage","N","N_poles","Vision","Grid_positions","Grid_open"])
    print(df)

    df.to_csv("SOBOL1.csv",sep=",",header=True)