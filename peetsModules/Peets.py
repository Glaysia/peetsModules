import msvcrt
import sys
import numpy as np
import pandas as pd
import csv
import os
import math
import time
import shutil
from datetime import datetime

import traceback
import logging
import platform

from ansys.aedt.core import Desktop, Maxwell3d, Icepak
from ansys.aedt.core.application.design import Design
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.cad.polylines import Polyline

import portalocker
import copy


import sys
import numpy as np
import pandas as pd
import csv
import os
import math
import time
import shutil
from datetime import datetime

import traceback
import logging
import platform

from ansys.aedt.core import Desktop, Maxwell3d, Icepak
from ansys.aedt.core.application.design import Design
from ansys.aedt.core.modeler.cad.object_3d import Object3d
from ansys.aedt.core.modeler.cad.polylines import Polyline

import portalocker
# import fcntl
import copy

# geometry parameter class
class Parameter() :

    def __init__(self) :

        self.a = None

    def _random_choice(self, X) :
        return round(np.random.choice( np.arange( X[0] , X[1]+X[2] , X[2]) ),X[3])

    def get_random_variable(self) :

        # ===============
        # Range setup
        # ===============

        w1_range = [20, 200, 1, 0]
        l1_leg_range = [2, 15, 0.1, 1]
        l1_top_range = [0.5, 2, 0.1, 1]
        l2_range = [5, 30, 0.1, 1] # under, upper, resolution

        h1_range = [0.1,3, 0.01, 2]

        Tx_turns_range = [14, 14, 1, 0]

        Tx_height_range = [0.105, 0.175, 0.035, 3] 
        Tx_preg_range = [0.01, 0.2, 0.01, 2] 
        
        Rx_preg_range = [0.01, 0.2, 0.01, 2]
        Rx_height_range = [0.105, 0.175, 0.035, 3]

        g1_range = [0, 0, 0.01, 2]
        g2_range = [0.1, 3, 0.01, 2]

        l1_center_range = [1,25,1,0]
        l2_tap_range = [0,0,1,0]

        Tx_space_x_range = [0.1, 5, 0.1, 1]
        Tx_space_y_range = [0.1, 5, 0.1, 1]
        Rx_space_x_range = [0.1, 5, 0.1, 1]
        Rx_space_y_range = [0.1, 5, 0.1, 1]

        core_N_w1_range = [0,0,1,0]
        core_P_w1_range = [0,0,1,0]

        Tx_layer_space_x_range = [0.1, 5, 0.1, 1]
        Tx_layer_space_y_range = [0.1, 5, 0.1, 1]
        Rx_layer_space_x_range = [0.1, 5, 0.1, 1]
        Rx_layer_space_y_range = [0.1, 5, 0.1, 1]

        Tx_width_range = [0.5,3,0.1,1]
        Rx_width_range = [4,20,0.1,1]

        # ===============
        # Get values
        # ===============

        self.w1 = self._random_choice(w1_range)
        self.l1_leg= self._random_choice(l1_leg_range)

        self.l1_top = self._random_choice(l1_top_range)
        self.l2 = self._random_choice(l2_range)

        self.h1 = self._random_choice(h1_range)


        self.Tx_turns = self._random_choice(Tx_turns_range)

        self.Tx_height = self._random_choice(Tx_height_range)
        self.Tx_preg = self._random_choice(Tx_preg_range)

        self.Rx_space_y = self._random_choice(Rx_space_y_range)
        self.Rx_preg = self._random_choice(Rx_preg_range)

        self.Rx_height = self._random_choice(Rx_height_range)
        self.Rx_space_x = self._random_choice(Rx_space_x_range)

        self.g1 = 0
        self.g2 = self._random_choice(g2_range)
        self.l1_center = self._random_choice(l1_center_range)
        self.l2_tap = 0

        self.Tx_space_x = self._random_choice(Tx_space_x_range)
        self.Tx_space_y = self._random_choice(Tx_space_y_range)
        self.Rx_space_x = self._random_choice(Rx_space_x_range)
        self.Rx_space_y = self._random_choice(Rx_space_y_range)

        self.core_N_w1 = 0
        self.core_P_w1 = 0

        self.Tx_layer_space_x = self._random_choice(Tx_layer_space_x_range)
        self.Tx_layer_space_y = self._random_choice(Tx_layer_space_y_range)
        self.Rx_layer_space_x = self._random_choice(Rx_layer_space_x_range)
        self.Rx_layer_space_y = self._random_choice(Rx_layer_space_y_range)

        self.Tx_width = self._random_choice(Tx_width_range)
        self.Rx_width = self._random_choice(Rx_width_range)

        Tx_max = max(((self.Tx_layer_space_x + self.Tx_width)*math.ceil(self.Tx_turns/2) + self.Tx_space_x), ((self.Tx_layer_space_y + self.Tx_width)*math.ceil(self.Tx_turns/2) + self.Tx_space_y))
        Rx_max = max((self.Rx_width + self.Rx_space_x),(self.Rx_width+self.Rx_space_y) )

        while(True) :
            if self.Tx_height*2 + self.Tx_preg*2 + self.Rx_height*4 + self.Rx_preg * 4 >= self.h1:
                self.Tx_height = self._random_choice(Tx_height_range)
                self.Tx_preg = self._random_choice(Tx_preg_range)
                self.Rx_height = self._random_choice(Rx_height_range)
                self.Rx_preg = self._random_choice(Rx_preg_range)
                self.h1 = self._random_choice(h1_range)

            elif  Tx_max >= self.l2+self.l2_tap :
                self.Tx_layer_space_x = self._random_choice(Tx_layer_space_x_range)
                self.Tx_layer_space_y = self._random_choice(Tx_layer_space_y_range)
                self.Tx_width = self._random_choice(Tx_width_range)
                self.l2 = self._random_choice(l2_range)
                Tx_max = max(((self.Tx_layer_space_x + self.Tx_width)*math.ceil(self.Tx_turns/2) + self.Tx_space_x), ((self.Tx_layer_space_y + self.Tx_width)*math.ceil(self.Tx_turns/2) + self.Tx_space_y))    

            elif  Rx_max >= self.l2+self.l2_tap :
                self.Rx_layer_space_x = self._random_choice(Rx_layer_space_x_range)
                self.Rx_width = self._random_choice(Rx_width_range)
                Rx_max = max((self.Rx_width + self.Rx_space_x),(self.Rx_width+self.Rx_space_y) )

            elif  self.g2 >= self.h1 :
                self.g2 = self._random_choice(g2_range)
                self.h1 = self._random_choice(h1_range)

            else :
                break


def extract_data_from_last_line(filename):
    
    with open(filename, 'r') as file:
        lines = file.readlines()

    # 공백이 아닌 마지막 줄을 찾기
    for line in reversed(lines):
        if line.strip():  # 줄이 공백이 아닐 경우
            last_data_line = line
            break

    parts = last_data_line.split('|')
    pass_number = parts[0].strip()
    tetrahedra = parts[1].strip()
    total_energy = parts[2].strip()
    energy_error = parts[3].strip()
    delta_energy = parts[4].strip()

    return pass_number, tetrahedra, total_energy, energy_error, delta_energy

def write_to_csv(filename, pd_data):

    num_retries = 10
    delay = 3

    # check file existence
    file_exists = os.path.isfile(filename)

    if not file_exists:
        pd_data.to_csv(filename, header=True, mode='a')
        return True

    for i in range(num_retries) :

        try :
            pd_data.to_csv(filename, header=False, mode='a')
            return True
        except Exception as e :
            print(f"An error occurred while writing: {e}. Retrying... ({i+1}/{num_retries})")
            time.sleep(delay)
    
    print("Failed to read the file after multiple attempts.")
    return None


class Sim(Parameter) :

    def __init__(self) :
        # super().get_random_variable() # parameter overide
        self.project_name = "script1"
        self.flag = 1
        self.Proj = 0
        self.itr = 0
        self.freq = 130

        self.computer_name = "5950X1"
        self.create_desktop()

    def create_desktop(self) :
        
        # open desktop
        self.desktop = Desktop(
            version = "2024.2",
            non_graphical = True)
        self.desktop.disable_autosave()


    def create_project(self) :
        # project property
        self.project_name = f"script{self.num}"
        self.solution_type = "EddyCurrent"

        self.dir_temp = os.getcwd()

        self.dir = os.path.join(self.dir_temp,'script',f'script{self.num}')
        self.dir_project = os.path.join(self.dir, f'{self.project_name}.aedt')

        # delete and make dir
        if os.path.exists(self.dir) :
            shutil.rmtree(self.dir)
        if os.path.exists(self.dir) == False :
            os.mkdir(self.dir)

        self.design_name = f'script{self.num}_{self.itr}'
    
        self.M3D = Maxwell3d(
            project=self.dir_project,
            design=self.design_name,
            solution_type=self.solution_type
        )
        self.project = self.M3D.oproject
        oDesign = self.M3D.odesign
        oDesign.SetDesignSettings(
            [
                "NAME:Design Settings Data",
                "Allow Material Override:=", False,
                "Perform Minimal validation:=", False,
                "EnabledObjects:="	, [],
                "PerfectConductorThreshold:=", 1E+30,
                "InsulatorThreshold:="	, 1,
                "SolveFraction:="	, False,
                "Multiplier:="		, "1",
                "SkipMeshChecks:="	, True
            ], 
            [
                "NAME:Model Validation Settings",
                "EntityCheckLevel:="	, "Strict",
                "IgnoreUnclassifiedObjects:=", False,
                "SkipIntersectionChecks:=", False
            ])

        # self.M3D.analyze()

        # self.pid = self.desktop.aedt_process_id
        # self._save_pid_info()

    def set_material(self) : 
        self.mat = self.M3D.materials.duplicate_material(material="ferrite", name="ferrite_simulation")
        self.mat.permeability = 3000
        self.mat.set_power_ferrite_coreloss(cm=0.012866,x=1.7893,y=2.52296) #GP98 material

    def set_variable(self) :

        self.M3D["w1"] = f'{self.w1}mm'
        self.M3D["l1_leg"] = f'{self.l1_leg}mm'
        self.M3D["l1_top"] = f'{self.l1_top}mm'
        self.M3D["l2"] = f'{self.l2}mm'
        self.M3D["h1"] = f'{self.h1}mm'
        self.M3D["w1_ratio"] = f'1'
        self.M3D["l1_center"] = f'{self.l1_center}mm'
        self.M3D["ratio"] = f'1'
        self.M3D["Tx_turns"] = f'{self.Tx_turns}'
        self.M3D["Rx_turns"] = f'{2}'
        self.M3D["Tx_width"] = f'{self.Tx_width}mm'
        self.M3D["Tx_height"] = f'{self.Tx_height}mm'
        self.M3D["Tx_space_x"] = f'{self.Tx_space_x}mm'
        self.M3D["Tx_space_y"] = f'{self.Tx_space_y}mm'
        self.M3D["Tx_preg"] = f'{self.Tx_preg}mm'
        self.M3D["Rx_width"] = f'{self.Rx_width}mm'
        self.M3D["Rx_preg"] = f'{self.Rx_preg}mm'
        self.M3D["Rx_height"] = f'{self.Rx_height}mm'
        self.M3D["Rx_space_x"] = f'{self.Rx_space_x}mm'
        self.M3D["Rx_space_y"] = f'{self.Rx_space_y}mm'
        self.M3D["g1"] = f'{self.g1}mm'
        self.M3D["g2"] = f'{self.g2}mm'
        self.M3D["l2_tap"] = f'{self.l2_tap}mm'
        self.M3D["core_N_w1"] = f'{self.core_N_w1}mm'
        self.M3D["core_P_w1"] = f'{self.core_P_w1}mm'
        self.M3D["Tx_layer_space_x"] = f'{self.Tx_layer_space_x}mm'
        self.M3D["Tx_layer_space_y"] = f'{self.Tx_layer_space_y}mm'

    def set_analysis(self) :
        setup = self.M3D.create_setup(name = "Setup1")
        setup.props["MaximumPasses"] = 5
        setup.props["MinimumPasses"] = 2
        setup.props["PercentError"] = 2.5
        setup.props["Frequency"] = f'{self.freq}kHz'
            
    def create_core(self) :

        # make core (main part)
        origin = ["-(w1)/2*w1_ratio","-(2*l1_leg+2*l2+2*l2_tap+l1_center)/2",  "-(2*l1_top+h1)/2"]
        
        dimension = [ "(w1)*w1_ratio","(2*l1_leg+2*l2+2*l2_tap+l1_center)", "(2*l1_top+h1)"]
        
        self.core_base = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core",
            matname = self.mat
        )


        origin = ["-(w1)/2*w1_ratio" ,"l1_center/2" ,"-(h1)/2"]
        
        dimension = [ "w1","l2+l2_tap", "h1"]

        self.core_sub1 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_sub1",
            matname = "ferrite"
        )


        origin = ["-(w1)/2*w1_ratio" ,"-l1_center/2" ,"-(h1)/2"]
        
        dimension = [ "w1","-(l2+l2_tap)", "h1"]

        self.core_sub2 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_sub2",
            matname = "ferrite"
        )


        origin = ["-(w1)/2*w1_ratio" ,"-l1_center/2" ,"-(h1)/2"]
        
        dimension = [ "w1","l1_center", "h1"]

        self.core_sub3 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_sub3",
            matname = "ferrite"
        )


        origin = [ "-(w1)/2*w1_ratio", "-(2*l1_leg+2*l2+2*l2_tap+l1_center)/2","-(h1)/2"]
        
        dimension = [ "(w1)","(l1_leg)", "(g1)"]
        
        self.core_sub_g1 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_sub_g1",
            matname = self.mat
        )

        origin = [ "-(w1)/2*w1_ratio","(2*l1_leg+2*l2+2*l2_tap+l1_center)/2", "-(h1)/2"]
        
        dimension = [ "(w1)","-(l1_leg)", "(g1)"]
        
        self.core_sub_g2 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_sub_g2",
            matname = self.mat
        )

        

        origin = ["-(w1)/2*w1_ratio" ,"-l1_center/2" ,"-(h1)/2"]
        
        dimension = [ "w1*w1_ratio","l1_center", "h1"]

        
        self.core_unite1 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_unite1",
            matname = "ferrite"
        )


        origin = ["-(w1)/2*w1_ratio" ,"-l1_center/2" ,"-(h1)/2"]
        
        dimension = [ "w1*w1_ratio","l1_center", "g2"]

        self.core_unite_sub_g1 = self.M3D.modeler.create_box(
            position = origin,
            dimensions_list = dimension,
            name = "core_sub_g3",
            matname = "ferrite"
        )

        # subtract core
        blank_list = [self.core_base.name]
        tool_list = [self.core_sub1.name, 
                    self.core_sub2.name, 
                    self.core_sub3.name,
                    self.core_sub_g1.name,
                    self.core_sub_g2.name]


        self.M3D.modeler.subtract(
            blank_list = blank_list,
            tool_list = tool_list,
            keep_originals = False
        )

        # subtract core
        blank_list = [self.core_unite1.name]
        tool_list = [self.core_unite_sub_g1.name]


        self.M3D.modeler.subtract(
            blank_list = blank_list,
            tool_list = tool_list,
            keep_originals = False
        )

        self.core_list =[self.core_base, self.core_unite1]

        self.core = self.M3D.modeler.unite(unite_list=self.core_list)

        self.core_base.transparency = 0.6

    def create_winding(self) :
        self.temp = [["(w1*w1_ratio/2 + core_P_w1 + 40mm)","-(l1_center/2 + Rx_space_y + Rx_width/2)","0mm"],
                     ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","-(l1_center/2 + Rx_space_y + Rx_width/2)","0mm"],
                     ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","-(l1_center/2 + Rx_space_y + Rx_width/2)","0mm"],
                     ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","(l1_center/2 + Rx_space_y + Rx_width/2)","0mm"],
                     ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","(l1_center/2 + Rx_space_y + Rx_width/2)","0mm"],
                     ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","0","0mm"],
                     ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","0","-(Rx_height+Rx_preg)"],
                     ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","-(l1_center/2 + Rx_space_y + Rx_width/2)","-(Rx_height+Rx_preg)"],
                     ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","-(l1_center/2 + Rx_space_y + Rx_width/2)","-(Rx_height+Rx_preg)"],
                     ["-(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","(l1_center/2 + Rx_space_y + Rx_width/2)","-(Rx_height+Rx_preg)"],
                     ["(w1*w1_ratio/2 + Rx_space_x + Rx_width/2)","(l1_center/2 + Rx_space_y + Rx_width/2)","-(Rx_height+Rx_preg)"],
                     ["(w1*w1_ratio/2 + core_P_w1 + 40mm)","(l1_center/2 + Rx_space_y + Rx_width/2)","-(Rx_height+Rx_preg)"]]

        self.Rx_1 = self._create_polyline(points = self.temp, name = f"Rx_1", coil_width = "Rx_width", coil_height = "Rx_height")
        self.M3D.modeler.mirror(objid=self.Rx_1,position=[0,0,0],vector=[1,0,0])
        Rx_1_move = ["0mm" ,"0mm" ,"(Tx_preg+Tx_height+Rx_preg+Rx_height/2+Rx_preg+Rx_height)"]
        self.M3D.modeler.move(objid=self.Rx_1,vector=Rx_1_move)

        self.Rx_2 = self._create_polyline(points = self.temp, name = f"Rx_2", coil_width = "Rx_width", coil_height = "Rx_height")
        self.M3D.modeler.mirror(objid=self.Rx_2,position=[0,0,0],vector=[1,0,0])
        Rx_2_move = ["0mm" ,"0mm" ,"-(Tx_preg+Tx_height+Rx_preg+Rx_height/2)"]
        self.M3D.modeler.move(objid=self.Rx_2,vector=Rx_2_move)

        self.temp_Tx = [["(w1*w1_ratio/2 + core_P_w1 + 40mm)" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1)*(Tx_layer_space_y + Tx_width))" ,"0mm"],
                        [f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2)" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1)*(Tx_layer_space_y + Tx_width))" ,f"0mm"]]

        if self.Tx_turns % 2 != 0 :
            for i in range(0,math.ceil(self.Tx_turns/2)) :
                if i == math.ceil(self.Tx_turns/2) - 1 :
                    self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                    self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"(Tx_width/2)" ,f"0mm"])
                else :
                    self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-{i})*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                    self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-{i})*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                    self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-{i})*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                    self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}+1))*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                    
            self.temp_connect = [[f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"0"  ,"(Tx_preg/2+Tx_height/2)"],
                                 [f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"0"  ,"-(Tx_preg/2+Tx_height/2)"]]
            self.Tx_connect = self.M3D.modeler.create_polyline(self.temp_connect, name = f"Tx_connect",xsection_type="Circle", xsection_width= "Tx_width*0.8",xsection_num_seg=12)
        else :
            for i in range(0,math.ceil(self.Tx_turns/2)) :
                self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-{i})*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                self.temp_Tx.append([f"-(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-{i})*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-{i})*(Tx_layer_space_y + Tx_width))" ,f"0mm"])
                if i == math.ceil(self.Tx_turns/2) - 1 :   
                    self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"-(Tx_width/2)" ,f"0mm"])
                else :
                    self.temp_Tx.append([f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}+1))*(Tx_layer_space_y + Tx_width))" ,f"0mm"])

            self.temp_connect = [[f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"0" ,f"(Tx_preg/2+Tx_height/2)"],
                                 [f"(w1*w1_ratio/2 + Tx_space_x + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1-({i}))*(Tx_layer_space_x + Tx_width))" ,f"0" ,f"-(Tx_preg/2+Tx_height/2)"]]
            self.Tx_connect = self.M3D.modeler.create_polyline(self.temp_connect, name = f"Tx_connect",xsection_type="Circle", xsection_width= "Tx_width*0.8",xsection_num_seg=12)




        self.Tx_1 = self._create_polyline(points = self.temp_Tx, name = f"Tx_1", coil_width = "Tx_width", coil_height = "Tx_height")
        self.M3D.modeler.copy(object_list=self.Tx_1)
        self.M3D.modeler.paste()
        self.Tx_2 = self.M3D.modeler.get_object_from_name(objname="Tx_2")
        self.M3D.modeler.mirror(objid=self.Tx_2,position=[0,0,0],vector=[0,1,0])
        self.M3D.modeler.move(objid=self.Tx_2,vector=["0mm" ,"0mm" ,"-(Tx_preg/2+Tx_height/2)"])

        self.M3D.modeler.move(objid=self.Tx_1,vector=["0mm" ,"0mm" ,"Tx_preg/2+Tx_height/2"])
        
        

        self.M3D.modeler.unite(unite_list=[self.Tx_1,self.Tx_2,self.Tx_connect])


        self.Rx_1.color = [0,0,255]
        self.Rx_1.transparency = 0
        self.Rx_2.color = [0,0,255]
        self.Rx_2.transparency = 0
        self.Tx_1.color = [255,0,0]
        self.Tx_1.transparency = 0

    def create_exctation(self) :
        self.Tx_in = self.M3D.modeler.get_faceid_from_position(position=["(w1*w1_ratio/2 + core_P_w1 + 40mm)" ,f"-(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1)*(Tx_layer_space_y + Tx_width))" ,"(Tx_preg/2 + Tx_height/2)"])
        self.Tx_out = self.M3D.modeler.get_faceid_from_position(position=["(w1*w1_ratio/2 + core_P_w1 + 40mm)" ,f"(l1_center/2 + Tx_space_y + Tx_width/2 + ({math.ceil(self.Tx_turns/2)}-1)*(Tx_layer_space_y + Tx_width))" ,"-(Tx_preg/2 + Tx_height/2)"])

        self.Rx2_in = self.M3D.modeler.get_faceid_from_position(position=["-(w1*w1_ratio/2 + core_P_w1 + 40mm)","-(l1_center/2 + Rx_space_y + Rx_width/2)","-(Tx_preg+Tx_height+Rx_preg+Rx_height/2)"])
        self.Rx2_out = self.M3D.modeler.get_faceid_from_position(position=["-(w1*w1_ratio/2 + core_P_w1 + 40mm)","(l1_center/2 + Rx_space_y + Rx_width/2)","-(Tx_preg+Tx_height+Rx_preg+Rx_height/2 + Rx_preg+Rx_height)"])

        self.Rx1_in = self.M3D.modeler.get_faceid_from_position(position=["-(w1*w1_ratio/2 + core_P_w1 + 40mm)","-(l1_center/2 + Rx_space_y + Rx_width/2)","(Tx_preg+Tx_height+Rx_preg+Rx_height/2+Rx_preg+Rx_height)"])
        self.Rx1_out = self.M3D.modeler.get_faceid_from_position(position=["-(w1*w1_ratio/2 + core_P_w1 + 40mm)","(l1_center/2 + Rx_space_y + Rx_width/2)","(Tx_preg+Tx_height+Rx_preg+Rx_height/2)"])

        # assign coil terminal
        self.M3D.assign_coil(self.Tx_in, conductor_number=1,polarity="Positive",name="Tx_in")
        self.M3D.assign_coil(self.Tx_out, conductor_number=1,polarity="Negative",name="Tx_out")

        self.M3D.assign_coil(self.Rx1_in, conductor_number=1,polarity="Positive",name="Rx1_in")
        self.M3D.assign_coil(self.Rx1_out, conductor_number=1,polarity="Negative",name="Rx1_out")

        self.M3D.assign_coil(self.Rx2_in, conductor_number=1,polarity="Positive",name="Rx2_in")
        self.M3D.assign_coil(self.Rx2_out, conductor_number=1,polarity="Negative",name="Rx2_out")





        Tx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current",is_solid=True,current_value= 3*math.sqrt(2),name="Tx")
        Rx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current",is_solid=True,current_value= 8.9*math.sqrt(2),name="Rx1")
        Rx_winding2 = self.M3D.assign_winding(coil_terminals=[], winding_type="Current",is_solid=True,current_value=0 ,name="Rx2")
        
        self.M3D.add_winding_coils(Tx_winding.name, coil_names=["Tx_in","Tx_out"])
        self.M3D.add_winding_coils(Rx_winding.name, coil_names=["Rx1_in","Rx1_out"])
        self.M3D.add_winding_coils(Rx_winding2.name, coil_names=["Rx2_in","Rx2_out"])
        self.M3D.assign_matrix(sources=[Tx_winding.name,Rx_winding.name,Rx_winding2.name],matrix_name="Matrix1")
       
    def _create_polyline(self, points, name, coil_width, coil_height) :

        polyline_obj = self.M3D.modeler.create_polyline(
            points, 
            name = name,
            material = "copper",
            xsection_type = "Rectangle",
            xsection_width = coil_width,
            xsection_height = coil_height)    
        
        return polyline_obj

    def _get_mean_Bfield(self, obj) :

        assignment = obj.name

        oModule = self.M3D.ofieldsreporter
        oModule.CalcStack("clear")
        oModule.CopyNamedExprToStack("Mag_B")
        oModule.EnterVol(assignment) if obj.is3d else oModule.EnterSurf(assignment)
        oModule.CalcOp("Mean")
        name = "B_mean_{}".format(assignment)  # Need to check for uniqueness !
        oModule.AddNamedExpression(name, "Fields")

        return name
    
    def _get_max_Bfield(self, obj) :

        assignment = obj.name

        oModule = self.M3D.ofieldsreporter
        oModule.CalcStack("clear")
        oModule.CopyNamedExprToStack("Mag_B")
        oModule.EnterVol(assignment) if obj.is3d else oModule.EnterSurf(assignment)
        oModule.CalcOp("Maximum")
        name = "B_max_{}".format(assignment)  # Need to check for uniqueness !
        oModule.AddNamedExpression(name, "Fields")

        return name

    def _create_B_field(self) :
        self.leg_left = self.M3D.modeler.create_rectangle(
            orientation = "XY",
            origin = ['-(w1)/2' ,'-(2*l1_leg+2*l2+l1_center)/2' ,'g1/2'],
            sizes = ['w1','l1_leg']
        )
        self.leg_left.model = False
        self.leg_left.name = "leg_left"

        self.leg_center = self.M3D.modeler.create_rectangle(
            orientation = "XY",
            origin = ['-(w1)/2','-l1_center/2', 'g2/2'],
            sizes = ['w1','l1_center']
        )
        self.leg_center.model = False
        self.leg_center.name = "leg_center"

        self.leg_right = self.M3D.modeler.create_rectangle(
            orientation = "XY",
            origin = ['-(w1)/2' ,'(2*l1_leg+2*l2+l1_center)/2' ,'g1/2'],
            sizes = ['w1','-l1_leg']
        )
        self.leg_right.model = False
        self.leg_right.name = "leg_right"

        self.leg_top_left = self.M3D.modeler.create_rectangle(
            orientation = "XZ",
            origin = ['-(w1)/2' ,'-(l1_center+l2)/2' ,'h1/2+l1_top'],
            sizes = ['-l1_top','w1']
        )
        self.leg_top_left.model = False
        self.leg_top_left.name = "leg_top_left"

        self.leg_top_right = self.M3D.modeler.create_rectangle(
            orientation = "XZ",
            origin = ['-(w1)/2' ,'(l1_center+l2)/2' ,'h1/2+l1_top'],
            sizes = ['-l1_top','w1']
        )
        self.leg_top_right.model = False
        self.leg_top_right.name = "leg_top_right"

        self.leg_bottom_left = self.M3D.modeler.create_rectangle(
            orientation = "XZ",
            origin = ['-(w1)/2','-(l1_center+l2)/2'  ,'-(h1/2+l1_top)'],
            sizes = [ 'l1_top','w1']
        )
        self.leg_bottom_left.model = False
        self.leg_bottom_left.name = "leg_bottom_left"

        self.leg_bottom_right = self.M3D.modeler.create_rectangle(
            orientation = "XZ",
            origin = ['-(w1)/2','(l1_center+l2)/2'  ,'-(h1/2+l1_top)'],
            sizes = ['l1_top','w1']
        )
        self.leg_bottom_right.model = False
        self.leg_bottom_right.name = "leg_bottom_right"


    def _get_B_field(self) :
        parameters2 = []
        parameters2.append([self.core_base, "B_mean", "B_mean_core"])
        parameters2.append([self.leg_left, "B_mean", "B_mean_leg_left"])
        parameters2.append([self.leg_right, "B_mean", "B_mean_leg_right"])
        parameters2.append([self.leg_center, "B_mean", "B_mean_leg_center"])
        parameters2.append([self.leg_top_left, "B_mean", "B_mean_leg_top_left"])
        parameters2.append([self.leg_bottom_left, "B_mean", "B_mean_leg_bottom_left"])
        parameters2.append([self.leg_top_right, "B_mean", "B_mean_leg_top_right"])
        parameters2.append([self.leg_bottom_right, "B_mean", "B_mean_leg_bottom_right"])

        self.result_expressions = []
        self.name_list = []
        self.report_list = {}
        for obj, expression, name in parameters2:
            if expression == "B_mean" :
                self.result_expressions.append(self._get_mean_Bfield(obj))
            self.name_list.append(name)

    def create_region(self) :

        region = self.M3D.modeler.create_air_region(z_pos = "800", z_neg="800", y_pos="300",y_neg="300",x_pos="0",x_neg="0")
        self.M3D.assign_material(obj= region,mat="vacuum")
        region_face = self.M3D.modeler.get_object_faces("Region")
        region_face
        self.M3D.assign_radiation(assignment=region_face,radiation="Radiation")
             
    def assign_mesh(self) :
        temp_list = list()
        temp_list.append(f"Tx_1")
        skin_depth = f"{math.sqrt(1.7*10**(-8)/math.pi/80/10**3/0.999991/4/math.pi/10**(-7))*10**3}mm"


        self.M3D.mesh.assign_skin_depth(assignment=temp_list,skin_depth=skin_depth,triangulation_max_length="12.2mm")

        air_list = list()
        air_list.append(f"air_box")
        self.M3D.mesh.assign_length_mesh(assignment=air_list,maximum_length="20mm")

    def _close_project(self) : 

        solution_dir = os.path.join(self.dir, f'script{self.num}.aedtresults')
        aedt_dir = os.path.join(self.dir, f'script{self.num}.aedt')

        if os.path.isdir(aedt_dir):
            shutil.rmtree(aedt_dir)

        self.M3D.close_project()
        self.desktop.release_desktop()

    def _get_magnetic_report(self) :
        get_result_list = []
        get_result_list.append(["Matrix1.L(Tx,Tx)","Ltx"])
        get_result_list.append(["Matrix1.L(Rx1,Rx1)","Lrx1"])
        get_result_list.append(["Matrix1.L(Rx2,Rx2)","Lrx2"])
        get_result_list.append(["Matrix1.L(Tx,Rx1)","M1"])
        get_result_list.append(["Matrix1.L(Tx,Rx2)","M2"])
        get_result_list.append(["Matrix1.CplCoef(Tx,Rx1)","k1"])
        get_result_list.append(["Matrix1.CplCoef(Tx,Rx2)","k2"])
        get_result_list.append(["Matrix1.L(Tx,Tx)*(Matrix1.CplCoef(Tx,Rx1)^2)","Lmt"])
        get_result_list.append(["Matrix1.L(Rx1,Rx1)*(Matrix1.CplCoef(Tx,Rx1)^2)","Lmr1"])
        get_result_list.append(["Matrix1.L(Rx2,Rx2)*(Matrix1.CplCoef(Tx,Rx2)^2)","Lmr2"])
        get_result_list.append(["Matrix1.L(Tx,Tx)*(1-Matrix1.CplCoef(Tx,Rx1)^2)","Llt"])
        get_result_list.append(["Matrix1.L(Rx1,Rx1)*(1-Matrix1.CplCoef(Tx,Rx1)^2)","Llr1"])
        get_result_list.append(["Matrix1.L(Rx2,Rx2)*(1-Matrix1.CplCoef(Tx,Rx2)^2)","Llr2"])
        get_result_list.append(["Matrix1.R(Tx,Tx)","Rtx"])
        get_result_list.append(["Matrix1.R(Rx1,Rx1)","Rrx1"])
        get_result_list.append(["Matrix1.R(Rx2,Rx2)","Rrx2"])



        result_expressions = [item[0] for item in get_result_list]

        report = self.M3D.post.create_report(expressions=result_expressions, setup_sweep_name=None, domain='Sweep', 
                            variations={"Freq": ["All"]}, primary_sweep_variable=None, secondary_sweep_variable=None, 
                            report_category=None, plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plotname="simulation parameter")
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        
        print(dir_data)
        
        import re

        # --- 1) 결과 CSV 읽기 -------------------------------------------------
        self.data1 = pd.read_csv(dir_data)

        # --- 2) 첫 번째 "Freq [kHz]" 열 제거 -----------------------------------
        if "Freq" in self.data1.columns[0]:
            self.data1.drop(columns=self.data1.columns[0], inplace=True)

        # --- 3) 단위 → 변환 계수 사전 ----------------------------------------
        factor = {
            "ph":1e-6, "nh":1e-3, "uh":1, "mh":1e3, "h":1e6,
            "mohm":1e-3, "ohm":1, "":1          # '' = 무단위
        }

        new_names = {}
        for col in self.data1.columns:
            col_main = col.split(" -", 1)[0].strip()          # 뒤 꼬리 제거

            # 대괄호 안의 단위 추출 (없으면 빈 문자열)
            m = re.search(r"\[([a-zA-Z]*)\]", col_main)
            unit_raw = m.group(1) if m else ""
            unit = unit_raw.lower()

            # factor에 있으면 변환, 없으면 그대로 둔다
            if unit in factor:
                self.data1[col] = self.data1[col] * factor[unit]

            # 단위 표기를 떼고 깔끔한 열 이름으로
            base = re.sub(r"\s*\[[^\]]*\]", "", col_main).strip()
            new_names[col] = base

        # 새 열 이름 적용
        self.data1.rename(columns=new_names, inplace=True)

        # --- 4) 열 순서·별칭 고정 --------------------------------------------
        self.data1.columns = ["Ltx","Lrx1","Lrx2","M1","M2","k1","k2",
                            "Lmt","Lmr1","Lmr2","Llt","Llr1","Llr2",
                            "Rtx","Rrx1","Rrx2"]

        # --- 5) 값 사용 -------------------------------------------------------
        self.Lmt = self.data1.loc[0, "Lmt"]

    def _get_copper_loss_parameter(self) :

        # ==============================
        # get copper loss data
        # ==============================
        Tx = self.M3D.modeler.get_object_from_name(objname = "Tx_1")
        Rx1 = self.M3D.modeler.get_object_from_name(objname = "Rx_1")
        Rx2 = self.M3D.modeler.get_object_from_name(objname = "Rx_2")

        n_Tx_loss = self.M3D.post.volumetric_loss(object_name=Tx.name)
        n_Rx1_loss = self.M3D.post.volumetric_loss(object_name=Rx1.name)
        n_Rx2_loss = self.M3D.post.volumetric_loss(object_name=Rx2.name)

        get_result_list = []
        get_result_list.append([f'P_{Tx.name}',"copperloss_Tx"])
        get_result_list.append([f'P_{Rx1.name}',"copperloss_Rx1"])
        get_result_list.append([f'P_{Rx2.name}',"copperloss_Rx2"])

        result_expressions = [item[0] for item in get_result_list]

        report = self.M3D.post.create_report(expressions=result_expressions, setup_sweep_name=None, domain='Sweep', 
                                        variations={"Phase": ["0deg"]}, primary_sweep_variable=None, secondary_sweep_variable=None, 
                                        report_category="Fields", plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plotname="copper loss data")
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        
        print(dir_data)
        
        self.data2 = pd.read_csv(dir_data)

        for itr, (column_name) in enumerate(self.data2.columns) : 

            self.data2[column_name] = abs(self.data2[column_name])

            if f'P_{Tx.name}' not in column_name and f'P_{Rx1.name}' not in column_name and f'P_{Rx2.name}' not in column_name :
                self.data2 = self.data2.drop(columns=column_name)
                                          
        self.data2.columns = ["copperloss_Tx", "copperloss_Rx1","copperloss_Rx2"]

    def get_input_parameter(self) :
        # ==============================
        # get input parameter
        # ==============================

    


        self.magnetizing_current = 390*math.sqrt(2)/2/math.pi/(self.freq*10**(3))/self.Lmt/10**(-6)/2


        input_parameter_array = np.array([self.w1, self.l1_leg, self.l1_top, self.l2, self.h1,self.l1_center,self.Tx_turns,
                                          self.Tx_width,self.Tx_height,self.Tx_space_x,self.Tx_space_y,self.Tx_preg,self.Rx_width,self.Rx_height,self.Rx_space_x,self.Rx_space_y,self.Rx_preg, self.g2,
                                          self.Tx_layer_space_x,self.Tx_layer_space_y,self.magnetizing_current])  
        input_parameter_array = input_parameter_array.reshape(1,len(input_parameter_array))
        
        input_parameter_columns = ["w1","l1_leg","l1_top","l2","h1","l1_center","Tx_turns",
                                   "Tx_width","Tx_height","Tx_space_x","Tx_space_y","Tx_preg","Rx_width","Rx_height","Rx_space_x","Rx_space_y","Rx_preg","g2",
                                   "Tx_layer_space_x","Tx_layer_space_y","magnetizing_current"]

        # transform pandas data form
        self.input_parameter = pd.DataFrame(data=input_parameter_array, columns=input_parameter_columns)
    
    def set_coreloss(self) :
        # 코어에 코어로스 세팅
        self.M3D.set_core_losses(assignment="core",core_loss_on_field=True)
        self._create_B_field()

    def write_data(self) :
        
        self.new_data = pd.concat([self.input_parameter, self.data1.round(4),self.data2.round(4),self.data3.round(4),self.data4.round(4),self.data5.round(5)], axis=1)
        
        current_dir = os.getcwd()
        csv_file = os.path.join(current_dir,f"output_data.csv")
        
        if os.path.isfile(csv_file):
            self.new_data.to_csv(csv_file, mode='a', index=False, header=False)
        else:
            self.new_data.to_csv(csv_file, mode='w', index=False, header=True)


    def coreloss_project(self) :
        self.M3D.duplicate_design(label=f'script1_{self.itr}_coreloss')
        self.M3D.set_active_design(name=f'script1_{self.itr}_coreloss')
        
        to_delete = [exc for exc in self.M3D.design_excitations.values() if exc.name in ["Tx", "Rx1", "Rx2"]]

        # 미리 수집한 대상에 대해 delete 호출
        for exc in to_delete:
            exc.delete()
       
        # 코어손실 계산할 자화 전류 인가
        self.magnetizing_current = 390*math.sqrt(2)/2/math.pi/(self.freq*10**(3))/self.Lmt/10**(-6)/2 #V*sqrt(2)/wL/2
        
        Tx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current",is_solid=True,current_value= self.magnetizing_current,name="Tx")
        Rx_winding = self.M3D.assign_winding(coil_terminals=[], winding_type="Current",is_solid=True,current_value= 0,name="Rx1")
        Rx_winding2 = self.M3D.assign_winding(coil_terminals=[], winding_type="Current",is_solid=True,current_value= 0,name="Rx2")

        self.M3D.add_winding_coils(Tx_winding.name, coil_names=["Tx_in","Tx_out"])
        self.M3D.add_winding_coils(Rx_winding.name, coil_names=["Rx1_in","Rx1_out"])
        self.M3D.add_winding_coils(Rx_winding2.name, coil_names=["Rx2_in","Rx2_out"])
        self.M3D.assign_matrix(sources=[Tx_winding.name,Rx_winding.name,Rx_winding2.name],matrix_name="Matrix1")

        # 코어에 코어로스 세팅
        self.M3D.set_core_losses(objects="core",value=True)
        self._create_B_field()

        # 시뮬레이션
        self.M3D.analyze()

        # 코어손실 리포트 작성 후 저장 및 데이터로 가져오기
        get_result_list_coreloss = [f'Coreloss']
        report = self.M3D.post.create_report(expressions=get_result_list_coreloss, setup_sweep_name=None, domain='Sweep', 
                            variations={"Freq": ["All"]}, primary_sweep_variable=None, secondary_sweep_variable=None, 
                            report_category=None, plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plotname="coreloss parameter")
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        
        print(dir_data)
        
        self.data3 = pd.read_csv(dir_data)

        for itr, (column_name) in enumerate(self.data3.columns) : 

            self.data3[column_name] = abs(self.data3[column_name])

            if itr == 0 : # delete "Freq [kHz]" columns
                self.data3 = self.data3.drop(columns=column_name)
                continue
            elif "[mW]" in column_name :
                self.data3[column_name] = self.data3[column_name] * 1e-3
            elif "[W]" in column_name :
                self.data3[column_name] = self.data3[column_name] * 1e+0
            elif "[kW]" in column_name :
                self.data3[column_name] = self.data3[column_name] * 1e+3
        
        self.data3.columns = ["coreloss"]

        # get B field
        parameters2 = []
        parameters2.append([self.core_base, "B_mean", "B_mean_core"])
        parameters2.append([self.leg_left, "B_mean", "B_mean_leg_left"])
        parameters2.append([self.leg_right, "B_mean", "B_mean_leg_right"])
        parameters2.append([self.leg_center, "B_mean", "B_mean_leg_center"])
        parameters2.append([self.leg_top_left, "B_mean", "B_mean_leg_top_left"])
        parameters2.append([self.leg_bottom_left, "B_mean", "B_mean_leg_bottom_left"])
        parameters2.append([self.leg_top_right, "B_mean", "B_mean_leg_top_right"])
        parameters2.append([self.leg_bottom_right, "B_mean", "B_mean_leg_bottom_right"])

        self.result_expressions = []
        self.name_list = []
        self.report_list = {}
        for obj, expression, name in parameters2:
            if expression == "B_mean" :
                self.result_expressions.append(self._get_mean_Bfield(obj))
            self.name_list.append(name)

        report = self.M3D.post.create_report(expressions=self.result_expressions, setup_sweep_name=None, domain='Sweep', 
                    variations={"Phase": ["0deg"]}, primary_sweep_variable=None, secondary_sweep_variable=None, 
                    report_category= "Fields", plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plotname="calculator_report")
        
        export_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        self.data4 = pd.read_csv(export_data,skiprows=1,header=None)

        self.data4 = self.data4.iloc[:, [3, 5, 7, 9, 11, 13, 15, 17]]  # 필요한 열 선택
        self.data4.columns = ["B_core","B_left","B_right","B_center","B_top_left","B_bottom_left","B_top_right","B_bottom_right"] # new column name

        data5_result = [f'P_Tx_1']
        report = self.M3D.post.create_report(expressions=data5_result, setup_sweep_name=None, domain='Sweep', 
                                        variations={"Phase": ["0deg"]}, primary_sweep_variable=None, secondary_sweep_variable=None, 
                                        report_category="Fields", plot_type='Data Table', context=None, subdesign_id=None, polyline_points=1001, plotname="magnetizing copper loss data")
        dir_data = self.M3D.post.export_report_to_csv(project_dir=self.dir, plot_name=report.plot_name)
        
        print(dir_data)
        
        self.data5 = pd.read_csv(dir_data)

        for itr, (column_name) in enumerate(self.data5.columns) : 

            self.data5[column_name] = abs(self.data5[column_name])

            if f'P_Tx_1' not in column_name:
                self.data5 = self.data5.drop(columns=column_name)

        self.data5.columns = ["magnetizing_copperloss_Tx"]

    def data_remove(self) :
        

       # 삭제할 파일 경로
        file_path = os.path.join(f'script',f'script{self.num}','simulation parameter.csv')

        # 파일 존재 여부 확인 후 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        file_path = os.path.join(f'script',f'script{self.num}','copper loss data.csv')

        # 파일 존재 여부 확인 후 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        file_path = os.path.join(f'script',f'script{self.num}','coreloss parameter.csv')

        # 파일 존재 여부 확인 후 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        file_path = os.path.join(f'script',f'script{self.num}','calculator_report.csv')

        # 파일 존재 여부 확인 후 삭제
        if os.path.exists(file_path):
            os.remove(file_path)

        file_path = os.path.join(f'script',f'script{self.num}','magnetizing copper loss data.csv')

        # 파일 존재 여부 확인 후 삭제
        if os.path.exists(file_path):
            os.remove(file_path)


        self.data1 = []
        self.data2 = []
        self.data3 = []
        self.data4 = []
        self.data5 = []
        self.new_data = []
        self.input_parameter = []

    def simulation(self) :
        self.start_time = time.time()
        file_path = "simulation_num.txt"

        # 파일이 존재하지 않으면 생성
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as file:
                file.write("1")

        # 읽기/쓰기 모드로 파일 열기
        with open(file_path, "r+", encoding="utf-8") as file:
            # 파일 잠금: LOCK_EX는 배타적 잠금,  블로킹 모드로 실행
            fcntl.flock(file, fcntl.LOCK_EX)

            # 파일에서 값 읽기
            content = int(file.read().strip())
            self.num = content
            self.PROJECT_NAME = f"simulation{content}"
            content += 1

            # 파일 포인터를 처음으로 되돌리고, 파일 내용 초기화 후 새 값 쓰기
            file.seek(0)
            file.truncate()
            file.write(str(content))

            # 파일은 with 블록 종료 시 자동으로 닫히며, 잠금도 해제됨

        log_simulation(number=self.num, pid=self.desktop.aedt_process_id)


        self.create_project()
        self.get_random_variable()
        self.set_variable()
        self.set_analysis()
        self.set_material()


        self.create_core()
        self.create_winding()
        self.assign_mesh()
        self.create_region()

        self.create_exctation()

        self.M3D.analyze()

        self._get_magnetic_report()
        self.get_input_parameter()
        self._get_copper_loss_parameter()

        self.coreloss_project()
        self.write_data()


def loging(msg):

    file_path = "log.txt"
    max_attempts = 5
    attempt = 0

    # 파일이 없으면 새로 생성하고, 있으면 append 모드로 엽니다.
    while attempt < max_attempts:
        try:
            with open(file_path, "a", encoding="utf-8") as file:
                file.write(msg + "\n")
            break  # 성공하면 루프 탈출
        except Exception as e:
            attempt += 1
            print(f"파일 쓰기 오류 발생: {e}. 재시도 {attempt}/{max_attempts}...")
            time.sleep(1)
    else:
        print("파일 쓰기에 계속 실패했습니다.")


def safe_open(filename, mode, retries=5, delay=1):
    """
    filename: 열 파일명
    mode: 열기 모드 (예: 'r', 'w', 'a')
    retries: 재시도 횟수
    delay: 재시도 전 대기 시간(초)
    """
    for i in range(retries):
        try:
            return open(filename, mode, newline='')
        except (IOError, OSError) as e:
            if i == retries - 1:
                raise e
            time.sleep(delay)


def log_simulation(number, state=None, pid=None, filename='log.csv'):
    """
    number: 기록할 숫자 값
    state: None이면 초기 기록, "fail"이면 Error, 그 외는 Finished로 업데이트
    pid: 기록할 프로세스 아이디 값 (인자로 받음)
    filename: 로그 파일명 (기본 'log.csv')

    파일이 없으면 헤더( Number, Status, StartTime, PID )와 함께 생성한 후,
    초기 호출 시 새로운 레코드를 추가하고, state가 전달되면 기존 레코드의 Status를 업데이트합니다.
    """
    lock_timeout = 10  # 락 타임아웃 시간(초)

    # 파일이 없으면 헤더를 포함하여 생성
    if not os.path.exists(filename):
        with portalocker.Lock(filename, 'w', timeout=lock_timeout, newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Number', 'Status', 'StartTime', 'PID'])
    
    # 초기 기록인 경우: state가 None이면 해당 번호의 레코드가 있는지 확인 후 없으면 추가
    if state is None:
        exists = False
        with portalocker.Lock(filename, 'r', timeout=lock_timeout, newline='') as f:
            reader = csv.reader(f)
            for row in reader:
                if row and row[0] == str(number):
                    exists = True
                    break
        if not exists:
            start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            with portalocker.Lock(filename, 'a', timeout=lock_timeout, newline='') as f:
                writer = csv.writer(f)
                writer.writerow([number, 'Simulation', start_time, pid])
    else:
        # state가 전달된 경우: 기존 레코드의 상태 업데이트
        new_status = "Error" if state.lower() == "fail" else "Finished"
        with portalocker.Lock(filename, 'r+', timeout=lock_timeout, newline='') as f:
            # 파일의 모든 행을 읽고 리스트로 저장
            rows = list(csv.reader(f))
            updated_rows = []
            for row in rows:
                # 헤더나, 해당 번호의 상태가 "Simulation"인 경우만 업데이트
                if row and row[0] == str(number) and row[1] == "Simulation":
                    row[1] = new_status
                updated_rows.append(row)
            # 파일 포인터를 맨 앞으로 돌리고 내용을 덮어씌운 후 파일 내용을 잘라냅니다.
            f.seek(0)
            writer = csv.writer(f)
            writer.writerows(updated_rows)
            f.truncate()

def save_error_log(project_name, error_info):
    error_folder = "error"
    if not os.path.exists(error_folder):
        os.makedirs(error_folder)
    error_file = os.path.join(error_folder, f"{project_name}_error.txt")
    with open(error_file, "w", encoding="utf-8") as f:
        f.write(error_info)


# sim = Sim()

class NoSim(Sim):
    def __init__(self):
        super().__init__()
    def simulation(self, no_analyze:bool = True) :
        self.start_time = time.time()
        file_path = "simulation_num.txt"

        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                f.write("1")

        # 읽기/쓰기 모드로 열고 배타 잠금
        with open(file_path, "r+", encoding="utf-8") as file:
            fd = file.fileno()
            # 첫 1바이트를 블로킹 모드로 잠금
            msvcrt.locking(fd, msvcrt.LK_LOCK, 1)

            # 현재 값 읽기
            raw = file.read().strip()
            content = int(raw) if raw else 0

            # 인스턴스 변수에 반영
            self.num = content
            self.PROJECT_NAME = f"simulation{content}"

            # 값 증가
            content += 1

            # 파일 갱신
            file.seek(0)
            file.truncate()
            file.write(str(content))
            file.flush()

            # 잠금 해제
            file.seek(0)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

        log_simulation(number=self.num, pid=self.desktop.aedt_process_id)


        self.create_project()
        self.get_random_variable()
        self.set_variable()
        self.set_analysis()
        self.set_material()


        self.create_core()
        self.create_winding()
        self.assign_mesh()
        self.create_region()

        self.create_exctation()

        if no_analyze == False:
            self.M3D.analyze()

            self._get_magnetic_report()
            self.get_input_parameter()
            self._get_copper_loss_parameter()

            self.coreloss_project()
            self.write_data()



    def simulation(self, no_analyze:bool = True):
        self.start_time = time.time()
        file_path = "simulation_num.txt"

        # # 파일이 존재하지 않으면 생성
        # if not os.path.exists(file_path):
        #     with open(file_path, "w", encoding="utf-8") as file:
        #         file.write("1")

        # # 읽기/쓰기 모드로 파일 열기
        # with open(file_path, "r+", encoding="utf-8") as file:
        #     # 파일 잠금: LOCK_EX는 배타적 잠금,  블로킹 모드로 실행
        #     fcntl.flock(file, fcntl.LOCK_EX)

        #     # 파일에서 값 읽기
        #     content = int(file.read().strip())
        #     self.num = content
        #     self.PROJECT_NAME = f"simulation{content}"
        #     content += 1

        #     # 파일 포인터를 처음으로 되돌리고, 파일 내용 초기화 후 새 값 쓰기
        #     file.seek(0)
        #     file.truncate()
        #     file.write(str(content))

        #     # 파일은 with 블록 종료 시 자동으로 닫히며, 잠금도 해제됨
        if not os.path.exists(file_path):
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write('1')

        # 파일 열기
        with open(file_path, 'r+', encoding='utf-8') as f:
            # 전체 파일 크기를 구합니다.
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(0)

            # 배타적 잠금 (BLOCKING)
            # LK_LOCK: 블로킹 모드로 잠금 시도
            msvcrt.locking(f.fileno(), msvcrt.LK_LOCK, size)

            # 파일에서 값 읽기
            content = int(f.read().strip())
            self.num = content
            self.PROJECT_NAME = f"simulation{content}"
            content += 1

            # 파일 갱신
            f.seek(0)
            f.truncate()
            f.write(str(content))
            f.flush()  # 반드시 flush()

            # 잠금 해제
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, size)

        log_simulation(number=self.num, pid=self.desktop.pid)

        self.desktop()
        log_simulation(number=self.num, pid=self.desktop.pid)


        self.create_project()
        self.get_random_variable()
        self.set_variable()
        self.set_analysis()
        self.set_material()


        self.create_core()
        self.create_winding()
        self.assign_mesh()
        self.create_region()

        self.create_exctation()
        if no_analyze == False:
            self.M3D.analyze()

            self._get_magnetic_report()
            self.get_input_parameter()
            self._get_copper_loss_parameter()

            self.coreloss_project()
            self.write_data()

class peetsModules:
    @staticmethod
    def openAedt():
        sim = NoSim()
        sim.simulation(True)
# %%
if __name__=='__main__':
    from nsgaTest import CoilParams, executeModel

    print(model_result:=executeModel(CoilParams(10.1,3,(10,20,5),14,6.78e9,(220,0.45),(24,4))))

    import pandas as pd
    import matplotlib.pyplot as plt
    model_result = pd.read_csv(model_result)
    
    plt.rcParams["figure.figsize"] = (8,7)

    parameters = {'xtick.labelsize' : 20,
            'ytick.labelsize' : 20}
    plt.rcParams.update(parameters)

    plt.scatter((model_result["A"]), model_result["total_loss"] ,s=60)



    #plt.scatter(data3["Ac"], data3["eff"] ,s=60)

    plt.xlabel("A[mm$^2$]", fontsize=20)
    #plt.xlim([0, 5000])
    #plt.ylim([98.0, 99.8])
    plt.ylabel("loss[W]", fontsize=20)
    plt.grid(True)

    plt.show()
# %%
