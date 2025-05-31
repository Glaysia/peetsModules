# import library
import pandas as pd                 # 1.5.3
import numpy as np                  # 1.23.0
import math
import matplotlib.pyplot as plt     # 3.8.0

import pymoo                        # 0.5.0
import lightgbm                     # 3.3.2
from pymoo.core.problem import Problem

import joblib                       # 1.3.2


from NSGA.problem_variables import ProblemVariables
from NSGA.variable_preprocess import variable_scaling, extract_variables

# # import openpyxl                     #
# import csv
# import os

# 작동 테스트, 사용 안할 예정
if __name__ == '__main__':
    # %%
    from os import path
    modelPath = path.join(
        "C:/Users/5600X2/IdeaProjects/peets_automation/models", "peetsRegression-0.0.1", "model")
    date = "250512"
    # ML_model_Ltx = joblib.load(f'..\\regression model\\model\\Ltx_{date}.pkl')
    # ML_model_Lrx1 = joblib.load(f'..\\regression model\\model\\Lrx1_{date}.pkl')
    # ML_model_Lrx2 = joblib.load(f'..\\regression model\\model\\Lrx2_{date}.pkl')
    # ML_model_M1 = joblib.load(f'..\\regression model\\model\\M1_{date}.pkl')
    # ML_model_M2 = joblib.load(f'..\\regression model\\model\\M2_{date}.pkl')
    # ML_model_k1 = joblib.load(f'..\\regression model\\model\\k1_{date}.pkl')
    # ML_model_k2 = joblib.load(f'..\\regression model\\model\\k2_{date}.pkl')
    # ML_model_Lmr1 = joblib.load(f'..\\regression model\\model\\Lmr1_{date}.pkl')
    # ML_model_Lmr2 = joblib.load(f'..\\regression model\\model\\Lmr2_{date}.pkl')
    # ML_model_Llt = joblib.load(f'..\\regression model\\model\\Llt_{date}.pkl')
    # ML_model_Llr1 = joblib.load(f'..\\regression model\\model\\Llr1_{date}.pkl')
    # ML_model_Llr2 = joblib.load(f'..\\regression model\\model\\Llr2_{date}.pkl')
    ML_model_Lmt = joblib.load(
        path.join(modelPath, f'Lmt_{date}.pkl'))
    ML_model_copperloss_Tx = joblib.load(
        path.join(modelPath, f'copperloss_Tx_{date}.pkl'))
    ML_model_copperloss_Rx1 = joblib.load(
        path.join(modelPath, f'copperloss_Rx1_{date}.pkl'))
    ML_model_copperloss_Rx2 = joblib.load(
        path.join(modelPath, f'copperloss_Rx2_{date}.pkl'))
    ML_model_coreloss = joblib.load(
        path.join(modelPath, f'coreloss_{date}.pkl'))
    ML_model_magnetizing_copperloss_Tx = joblib.load(
        path.join(modelPath, f'magnetizing_copperloss_Tx_{date}.pkl'))
    # ML_model_B_core = joblib.load(f'..\\regression model\\model\\B_core_{date}.pkl')
    ML_model_B_left = joblib.load(
        path.join(modelPath, f'B_left_{date}.pkl'))
    # ML_model_B_right = joblib.load(f'..\\regression model\\model\\B_right_{date}.pkl')
    ML_model_B_center = joblib.load(
        path.join(modelPath, f'B_center_{date}.pkl'))
    ML_model_B_top_left = joblib.load(
        path.join(modelPath, f'B_top_left_{date}.pkl'))
    # ML_model_B_bottom = joblib.load(f'..\\regression model\\model\\B_bottom_{date}.pkl')
    # set search parameter range

    # %% 

    # column_names = ["center_length", "center_width","Tx_height","Tx_width","Rx_height","Rx_width","layer_gap_Tx","layer_gap_Rx","N_Tx","N_Rx","move_Rx","space_height"]
    def pre_processing_data(X) :
        # "w1","g1","g2","l1_leg","l1_top","l2","h1","w1_ratio","l1_center","ratio","pri_turns","width_ratio","pri_width","pri_height","pri_space_x","pri_x","pri_preg","sec_width","sec_x","sec_space_y",
        #                                "sec_y","sec_preg","sec_height","sec_space_x","pri_space_y","pri_y"

        w1 = X[:, 0]
        l1_leg = X[:, 1]
        l1_top = X[:, 2]
        l2 = X[:, 3]

        h1 = X[:, 4]
        l1_center = X[:, 5]
        
        Tx_turns = X[:, 6]

        Tx_width = X[:, 7]
        Tx_height = (X[:, 8]-0.105)*35 + 0.105  #연산 : 105 + (X-105)*35
        Tx_space_x = X[:, 9]
        Tx_space_y = X[:, 10]
        Tx_preg = X[:, 11]

        Rx_width = X[:, 12]
        Rx_height = (X[:, 13]-0.105)*35 + 0.105 #연산 : 105 + (X-105)*35
        Rx_space_x = X[:, 14]
        Rx_space_y = X[:, 15]
        Rx_preg = X[:,16]

        g2 = X[:,17]

        Tx_layer_space_x = X[:,18]
        Tx_layer_space_y = X[:,19]


        



        #new_data = np.hstack([np.array(N1), np.array(N2), np.array(N1_left), np.array(N1_right), np.array(N2_left), np.array(N2_right), np.array(struct), np.array(freq), np.array(per), np.array(coil_width_inner), np.array(coil_width_outer), np.array(coil_width_x), np.array(coil_width_ter2), np.array(coil_height), np.array(space_height), np.array(move_z,space_inner), np.array(space_outer), np.array(space_x), np.array(A), np.array(B), np.array(C), np.array(D), np.array(E), np.array(F), np.array(g1), np.array(g2), np.array(ratio), np.array(ter_length)])
        new_data = np.column_stack((w1, l1_leg, l1_top, l2, h1, l1_center, Tx_turns, Tx_width, Tx_height, Tx_space_x, Tx_space_y, Tx_preg, Rx_width, Rx_height, 
                                    Rx_space_x, Rx_space_y, Rx_preg, g2, Tx_layer_space_x, Tx_layer_space_y))
        
        return new_data








    # print("Names: ", my_vars1.get_names())
    # print("First Values: ", my_vars1.get_first_values())
    # print("Second Values: ", my_vars1.get_second_values())

    # print(my_vars1.get_num_of_variables())

    # print(my_vars1.get_scale_values())

    # %%
    # "w1","l1_leg","l1_top","l2","h1","l1_center","Tx_turns","Tx_width","Tx_height","Tx_space_x","Tx_space_y",
    # "Tx_preg","Rx_width","Rx_height","Rx_space_x","Rx_space_y","Rx_preg",
    # "g2","Tx_layer_space_x","Tx_layer_space_y","wire_diameter","strand_number"
    from pymoo.config import Config
    Config.show_compile_hint = False

    vars1 = { # under, upper, resolution

        "w1_range": [20, 200, 1, 0],
        "l1_leg_range": [20, 150, 1, 0], # *0.1
        "l1_top_range": [5, 20, 1, 0], # *0.1
        "l2_range": [50, 300, 1, 0], # *0.1

        "h1_range": [10, 300, 1, 0], # *0.01 
        "l1_center_range": [1, 25, 1, 0],

        "Tx_turns_range": [14, 14, 1, 0],

        "Tx_width_range": [5,30,1,0],#*0.1
        "Tx_height_range": [105,107,35,0],#0.001 연산 : 105 + (X-105)*35
        "Tx_space_x_range": [1, 50, 1, 0],# *0.1 
        "Tx_space_y_range": [1, 50, 1, 0],# *0.1
        "Tx_preg_range": [1, 30, 1, 0],# *0.01
        
        "Rx_width_range": [40, 200, 1, 0], # *0.1
        "Rx_height_range": [105, 107, 1, 0],# *0.001 연산 : 105 + (X-105)*35
        "Rx_space_x_range": [1, 50, 1, 0],# *0.1
        "Rx_space_y_range": [1, 50, 1, 0],# *0.1
        "Rx_preg_range": [1, 30, 1, 0],# *0.01

        "g2_range": [10, 300, 1, 0],# *0.01

        "Tx_layer_space_x_range": [1, 50, 1, 0],# *0.1
        "Tx_layer_space_y_range": [1, 50, 1, 0],# *0.1
        

    }


    my_vars1 = ProblemVariables(vars1)

    class MyProblem(Problem) :

        def __init__(self, vars, ML_model_Lmt,ML_model_copperloss_Tx,ML_model_copperloss_Rx1,ML_model_copperloss_Rx2,ML_model_coreloss,ML_model_B_center,ML_model_B_left,ML_model_B_top_left,ML_model_magnetizing_copperloss_Tx):
            super().__init__(n_var=vars.get_num_of_variables(),     #number of inputs
                            n_obj=2,     #number of outputs
                            n_constr=40,  #nubmer of constraints
                            xl=np.array(vars.get_first_values()), #input lower bounds
                            xu=np.array(vars.get_second_values())) #input upper bounds
            self.vars = vars

            # self.ML_model_Ltx = ML_model_Ltx
            # self.ML_model_Lrx1 = ML_model_Lrx1
            # self.ML_model_Lrx2 = ML_model_Lrx2
            # self.ML_model_M1 = ML_model_M1
            # self.ML_model_M2 = ML_model_M2
            # self.ML_model_k1 = ML_model_k1
            # self.ML_model_k2 = ML_model_k2
            self.ML_model_Lmt = ML_model_Lmt
            # self.ML_model_Lmr1 = ML_model_Lmr1
            # self.ML_model_Lmr2 = ML_model_Lmr2
            # self.ML_model_Llt = ML_model_Llt
            # self.ML_model_Llr1 = ML_model_Llr1
            # self.ML_model_Llr2 = ML_model_Llr2
            self.ML_model_copperloss_Tx = ML_model_copperloss_Tx
            self.ML_model_copperloss_Rx1 = ML_model_copperloss_Rx1
            self.ML_model_copperloss_Rx2 = ML_model_copperloss_Rx2
            self.ML_model_coreloss = ML_model_coreloss
            # self.ML_model_B_core = ML_model_B_core
            self.ML_model_B_left = ML_model_B_left
            # self.ML_model_B_right = ML_model_B_right
            self.ML_model_B_center = ML_model_B_center
            self.ML_model_B_top_left = ML_model_B_top_left
            self.ML_model_magnetizing_copperloss_Tx = ML_model_magnetizing_copperloss_Tx
            # self.ML_model_B_bottom = ML_model_B_bottom
    
        #[N1, N2, freq, per, w1, l1, l2, h1, l1_leg, l1_top, g1, g2, coil_width_inner_ratio, coil_width_outer_ratio, coil_width_x_ratio, coil_height, space_height,
        # space_inner_ratio, space_outer_ratio, space_x_ratio, move_z, ratio]
        
        def _evaluate(self, X, out, *args, **kwargs) :

            X[:,0] = X[:,0]  #w1
            X[:,1] = X[:,1] * 0.1 #l1_leg
            X[:,2] = X[:,2] * 0.1 #l1_top
            X[:,3] = X[:,3] * 0.1 #l2

            X[:,4] = X[:,4] * 0.01 #h1
            X[:,5] = X[:,5]  #l1_center

            X[:,6] = X[:,6]  #Tx_turns

            X[:,7] = X[:,7] * 0.1 #Tx_width
            X[:,8] = X[:,8] * 0.001 #Tx_height
            X[:,9] = X[:,9] * 0.01 #Tx_space_x
            X[:,10] = X[:,10] * 0.1 #Tx_space_y
            X[:,11] = X[:,11] * 0.01 #Tx_preg

            X[:,12] = X[:,12] * 0.1 #Rx_width
            X[:,13] = X[:,13] * 0.001 #Rx_height
            X[:,14] = X[:,14] * 0.1 #Rx_space_x
            X[:,15] = X[:,15] * 0.1 #Rx_space_y
            X[:,16] = X[:,16] * 0.01 #Rx_preg

            X[:,17] = X[:,17] * 0.01 #g2
            X[:,18] = X[:,18] * 0.1 #Tx_layer_space_x
            X[:,19] = X[:,19] * 0.1 #Tx_layer_space_y
            
        
            w1 = X[:,0] * 1e-3
            l1_leg = X[:,1] * 1e-3
            l1_top = X[:,2] * 1e-3
            l2 = X[:,3] * 1e-3
    
            h1 = X[:,4] * 1e-3
            l1_center = X[:,5] * 1e-3

            Tx_turns = X[:,6]

            Tx_width = X[:,7] * 1e-3
            Tx_height = ((X[:, 8]-0.105)*35 + 0.105)*1e-3
            Tx_space_x = X[:,9] * 1e-3
            Tx_space_y = X[:,10] * 1e-3
            Tx_preg = X[:,11] * 1e-3

            Rx_width = X[:,12] * 1e-3
            Rx_height = ((X[:, 13]-0.105)*35 + 0.105)*1e-3
        
            Rx_space_x = X[:,14] * 1e-3
            Rx_space_y = X[:,15] * 1e-3
            Rx_preg = X[:,16] * 1e-3

            g2 = X[:,17] * 1e-3

            Tx_layer_space_x = X[:,18] * 1e-3
            Tx_layer_space_y = X[:,19] * 1e-3


    
        


            V_Tx = 390
            current = 25
            new_data = pre_processing_data(X)
            
            freq=130e+3
        
            copperloss_Tx = self.ML_model_copperloss_Tx.predict(new_data) 
            copperloss_Rx1 = self.ML_model_copperloss_Rx1.predict(new_data)
            copperloss_Rx2 = self.ML_model_copperloss_Rx2.predict(new_data)
            
        
            Lmt = self.ML_model_Lmt.predict(new_data)
            magnetizing_current = V_Tx*math.sqrt(2)/2/freq/2/math.pi/Lmt/(1e-6)

            coreloss_new_data = np.column_stack((new_data,magnetizing_current))
            coreloss = self.ML_model_coreloss.predict(coreloss_new_data) 
            magnetizing_copperloss_Tx = self.ML_model_magnetizing_copperloss_Tx.predict(coreloss_new_data)

            
            # Core volume
            A = np.maximum((l1_leg + l2 + l1_center/2)*2 * (w1+Tx_layer_space_x*6+Tx_width*7+Tx_space_x),(l1_leg + l2 + l1_center/2)*2 * (w1+Rx_width+Rx_space_x))
            height = l1_top*2 + h1
            coil_height = Tx_height*2+Rx_height*4+Tx_preg*2+Rx_preg*4
            hh = h1  - coil_height       
            Tx_Rx = Tx_preg/2 + Rx_preg

            #Total loss
            total_loss = copperloss_Tx + copperloss_Rx1 + copperloss_Rx2 + coreloss + magnetizing_copperloss_Tx

            #calculate B field

            B_center = self.ML_model_B_center.predict(coreloss_new_data)
            B_left = self.ML_model_B_left.predict(coreloss_new_data)
            B_top_left = self.ML_model_B_top_left.predict(coreloss_new_data)

            Tx = Tx_width * 7 + Tx_layer_space_y * 6 + Tx_space_y
            Rx = Rx_width + Rx_space_y
            max_length = l2 - np.maximum(Tx,Rx)
            gap = h1 - g2


            gLmt = -(Lmt - 64)*(Lmt - 70)/0.01
            gh = -(height - 0.0045)*(height-0.005)/0.0005
            gcopperloss_Tx = (copperloss_Tx-0)/0.001
            gcopperloss_Rx1 = (copperloss_Rx1-0)/0.001
            gcopperloss_Rx2 = (copperloss_Rx2-0)/0.001
            
            gB_center =  -(B_center-0)*(B_center-0.3)/0.3
            gB_left =  -(B_left-0)*(B_left-0.3)/0.3
            gB_top_left =  -(B_top_left-0)*(B_top_left-0.3)/0.3

            gcoreloss = -(coreloss-15)*(coreloss-30)/15
            
            gmax_length = (max_length-0.00001)/0.00001
            ghh = (hh - 0.00001)/0.00001
            ggap = (gap - 0.00001)/0.00001
            gTx_Rx = (Tx_Rx - 0.0004)/0.0004



            out["F"] = np.column_stack([A,total_loss]) # "Minimize" values (volume, coreloss)
            #out["G"] = np.column_stack([gB, gxgap, gygap1, gygap2, gAT, gAR, gBT, gBR])
            out["G"] = np.column_stack([gLmt,gh,gB_center,ghh,gmax_length,ggap,gB_left,gB_top_left,gTx_Rx])

            out["G"] = - out["G"] # Actually < 0

    sim = MyProblem(my_vars1,ML_model_Lmt,ML_model_copperloss_Tx,ML_model_copperloss_Rx1,ML_model_copperloss_Rx2,ML_model_coreloss,ML_model_B_center,ML_model_B_left,ML_model_B_top_left,ML_model_magnetizing_copperloss_Tx)

    from pymoo.algorithms.moo.nsga2 import NSGA2
    from pymoo.factory import get_sampling, get_crossover, get_mutation
    # from pymoo.operators.MixedVariableOperator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover
    from pymoo.operators.mixed_variable_operator import MixedVariableSampling, MixedVariableMutation, MixedVariableCrossover


    mask = ["int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int", "int","int", "int"]

    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=1000,

        sampling = MixedVariableSampling(mask, {
            "int": get_sampling("int_random")
        }),

        crossover = MixedVariableCrossover(mask, {
            "int": get_crossover("int_sbx", prob=0.8, eta=50)
        }),

        #mutation=get_mutation("real_pm", eta=40),
        mutation = MixedVariableMutation(mask, {
            "int": get_mutation("int_pm", eta=10)
        }),

        eliminate_duplicates=True
    )


    from pymoo.factory import get_termination
    termination = get_termination("n_gen", 100)

    from pymoo.optimize import minimize
    res = minimize(sim,
                algorithm,
                termination,
                seed=15, #RANDOM SEED
                save_history=False,
                verbose=True)
    res
        
    # %%
    if res.X is not None:

        import copy
        XX = copy.deepcopy(res.X)

        #XX = res.X
        XX[:,0] = XX[:,0]  #w1
        XX[:,1] = XX[:,1] * 0.1 #l1_leg
        XX[:,2] = XX[:,2] * 0.1 #l1_top
        XX[:,3] = XX[:,3] * 0.1 #l2

        XX[:,4] = XX[:,4] * 0.01 #h1
        XX[:,5] = XX[:,5]  #l1_center

        XX[:,6] = XX[:,6]  #Tx_turns

        XX[:,7] = XX[:,7] * 0.1 #Tx_width
        XX[:,8] = XX[:,8] * 0.001 #Tx_height
        XX[:,9] = XX[:,9] * 0.01 #Tx_space_x
        XX[:,10] = XX[:,10] * 0.1 #Tx_space_y
        XX[:,11] = XX[:,11] * 0.01 #Tx_preg

        XX[:,12] = XX[:,12] * 0.1 #Rx_width
        XX[:,13] = XX[:,13] * 0.001 #Rx_height
        XX[:,14] = XX[:,14] * 0.1 #Rx_space_x
        XX[:,15] = XX[:,15] * 0.1 #Rx_space_y
        XX[:,16] = XX[:,16] * 0.01 #Rx_preg

        XX[:,17] = XX[:,17] * 0.01 #g2
        XX[:,18] = XX[:,18] * 0.1 #Tx_layer_space_x
        XX[:,19] = XX[:,19] * 0.1 #Tx_layer_space_y
        

        w1 = XX[:,0] * 1e-3
        l1_leg = XX[:,1] * 1e-3
        l1_top = XX[:,2] * 1e-3
        l2 = XX[:,3] * 1e-3

        h1 = XX[:,4] * 1e-3
        l1_center = XX[:,5] * 1e-3

        Tx_turns = XX[:,6]

        Tx_width = XX[:,7] * 1e-3
        Tx_height = ((XX[:,8] - 0.105) * 35 + 0.105) * 1e-3
        Tx_space_x = XX[:,9] * 1e-3
        Tx_space_y = XX[:,10] * 1e-3
        Tx_preg = XX[:,11] * 1e-3

        Rx_width = XX[:,12] * 1e-3
        Rx_height = ((XX[:,13] - 0.105) * 35 + 0.105) * 1e-3

        Rx_space_x = XX[:,14] * 1e-3
        Rx_space_y = XX[:,15] * 1e-3
        Rx_preg = XX[:,16] * 1e-3

        g2 = XX[:,17] * 1e-3

        Tx_layer_space_x = XX[:,18] * 1e-3
        Tx_layer_space_y = XX[:,19] * 1e-3


        V_Tx = 390
        current = 25
        new_data2 = pre_processing_data(XX)

        freq = 140e+3

        copperloss_Tx = ML_model_copperloss_Tx.predict(new_data2)
        copperloss_Rx1 = ML_model_copperloss_Rx1.predict(new_data2)
        copperloss_Rx2 = ML_model_copperloss_Rx2.predict(new_data2)     

        Lmt = ML_model_Lmt.predict(new_data2)
        magnetizing_current = V_Tx*math.sqrt(2)/2/freq/2/math.pi/Lmt/(1e-6)
        coreloss_new_data2 = np.column_stack((new_data2, magnetizing_current))
        coreloss = ML_model_coreloss.predict(coreloss_new_data2) 
        magnetizing_copperloss_Tx = ML_model_magnetizing_copperloss_Tx.predict(coreloss_new_data2)


        # Core volume
        core_center = (l1_leg*2)*w1*h1 + l1_center*w1*h1
        core_tb = (l1_leg + l2 + l1_center/2)*w1*l1_top * 2
        V_core = core_center + core_tb

        V = (l1_leg + l2 + l1_center/2)*2 * w1 * (h1+l1_top*2)
        A = np.maximum((l1_leg + l2 + l1_center/2)*2 * (w1+Tx_layer_space_x*6+Tx_width*7+Tx_space_x),(l1_leg + l2 + l1_center/2)*2 * (w1+Rx_width+Rx_space_x))
        height = l1_top*2 + h1
        coil_height = Tx_height*2 + Rx_height*4 + Tx_preg*2 + Rx_preg*4
        hh = h1 - coil_height       
        

        #Total loss
        total_loss = coreloss + copperloss_Tx + copperloss_Rx1 + copperloss_Rx2 + magnetizing_copperloss_Tx

        # calculate B field
        B_center = ML_model_B_center.predict(coreloss_new_data2)
        B_left = ML_model_B_left.predict(coreloss_new_data2)
        B_top_left = ML_model_B_top_left.predict(coreloss_new_data2)




        X_column = ["w1","l1_leg","l1_top","l2","h1","l1_center","Tx_turns","Tx_width","Tx_height","Tx_space_x","Tx_space_y","Tx_preg","Rx_width","Rx_height","Rx_space_x","Rx_space_y","Rx_preg",
                    "g2","Tx_layer_space_x","Tx_layer_space_y","magnetizing_current"]

        # print(f"new_data : {new_data}")
        # loss
        # Tx_loss = self.ML_model_Tx_loss.predict( new_data ) / (10*math.sqrt(2))**2 * current**2 # [unit : W] # 기본 전류 설정 : 10루트2
        # Rx_loss = self.ML_model_Rx_loss.predict( new_data ) / (10*math.sqrt(2))**2 * current**2 # [unit : W] # 기본 전류 설정 : 10루트2


        X_data = pd.DataFrame(coreloss_new_data2, columns=X_column)
        X_data = X_data.astype(float)
        result_array = np.vstack([
                    total_loss,
                    coreloss,
                    copperloss_Tx,
                    copperloss_Rx1,
                    copperloss_Rx2,
                    magnetizing_copperloss_Tx,
                    B_center,
                    A*1e6,
                    Lmt,
                    hh,
                    h1,
                    coil_height,
                    Tx_height,
                    Rx_height,
                    Tx_preg,
                    Rx_preg,
                    B_left,
                    B_top_left
                ])

        result_data = pd.concat([
            X_data,
            pd.DataFrame(result_array.T, columns=["total_loss","coreloss","copperloss_Tx","copperloss_Rx1","copperloss_Rx2","magnetizing_copperloss_Tx","B_center","A","Lmt","hh","h1","coil_height","Tx1","Rx1","Txp1","Rxp1","B_left","B_top_left"])
        ], axis=1)

        
        result_data = result_data.sort_values('coreloss', ascending=True) # 정렬
        # result_data.to_csv(f'./result/core{name}.csv')

        
    # %% 
    import copy


    res.X[0]



    X = copy.deepcopy(res.X[0])


    X[0] = X[0]  #w1
    X[1] = X[1] * 0.1 #l1_leg
    X[2] = X[2] * 0.1 #l1_top
    X[3] = X[3] * 0.1 #l2

    X[4] = X[4] * 0.01 #h1
    X[5] = X[5]  #l1_center

    X[6] = X[6]  #Tx_turns

    X[7] = X[7] * 0.1 #Tx_space_x
    X[8] = X[8] * 0.1 #Tx_space_y
    X[9] = X[9] * 0.01 #Tx_preg

    X[10] = X[10] * 0.1 #Rx_width
    X[11] = X[11] * 0.001 #Rx_height
    X[12] = X[12] * 0.1 #Rx_space_x
    X[13] = X[13] * 0.1 #Rx_space_y
    X[14] = X[14] * 0.01 #Rx_preg

    X[15] = X[15] * 0.01 #g2

    X[16] = X[16] * 0.1 #Tx_layer_space_x
    X[17] = X[17] * 0.1 #Tx_layer_space_y
    X[18] = X[18] * 0.01 #wire_diameter
    X[19] = X[19] #strand_number

    X

    # %%
    
    plt.rcParams["figure.figsize"] = (8,7)

    parameters = {'xtick.labelsize' : 20,
            'ytick.labelsize' : 20}
    plt.rcParams.update(parameters)

    plt.scatter((result_data["A"]), result_data["total_loss"] ,s=60)



    #plt.scatter(data3["Ac"], data3["eff"] ,s=60)

    plt.xlabel("A[mm$^2$]", fontsize=20)
    #plt.xlim([0, 5000])
    #plt.ylim([98.0, 99.8])
    plt.ylabel("loss[W]", fontsize=20)
    plt.grid(True)

    plt.show()

    result_data.to_csv("tmp.csv")
# %%
from dataclasses import dataclass
from typing import Tuple
from pathlib import Path

@dataclass
class DimensionRange:
    width: Tuple[float, float]
    depth: Tuple[float, float]
    height: Tuple[float, float]

@dataclass
class CoilParams:
    Lm: float
    Lk: float
    size: DimensionRange
    freq: float
    iv_in: Tuple[float, float]
    iv_out: Tuple[float, float]

def executeModel(params: CoilParams)->Path:
    print(params)
    return Path("result1_v2_68uH.csv").absolute()