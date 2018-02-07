
#run.py
from mesa.batchrunner import BatchRunner
from EV.model import *
import multiprocessing as mp
import pandas as pd




cores = mp.cpu_count()

results = []


def run_model():
    fixed_params = {"width": 80,
                    "height": 80,
                    "initial_bravery": 10,
                    "battery_size": 75}
    variable_params = {"N": np.arange(100,500,150),                   # 3
                       "n_poles": [1/10,1/8,1/6,1/4],                 # 4
                       "vision": [1,2],                      # 3
                       "grid_positions": ["LHS", "circle"],           # 2
                       "open_grid": ["True", "False"]}                # 2
                                                                      # 3*4*3*2*2 = 144
    batch_run = BatchRunner(EV_Model,
                            fixed_parameters=fixed_params,
                            variable_parameters=variable_params,
                            iterations=1,
                            max_steps=2500,
                            model_reporters={"Usage": avg_usage,
                                             "Total_attempts": totalAttempts,
                                             "Percentage_failed": percentageFailed,
                                             "Average_lifespan": averageLifespan})
    batch_run.run_all()

    run_data = batch_run.get_model_vars_dataframe()

    return run_data.values.tolist()


if __name__ == "__main__":
    results = []
    pool = mp.Pool(cores)
    for i in range(8):
        def callback(result):
            results.extend(result)
        pool.apply_async(run_model, callback=callback)
    pool.close()
    pool.join()
    df = pd.DataFrame(results, columns=["N","N_poles","Vision","Grid_positions","Grid_open","run","Average_lifespan","Percentage_failed","Total_attempts","Usage","Width","Height","Initial_bravery","Battery_size"])
    print(df)

    df.to_csv("180202_1.csv",sep=",",header=True)
