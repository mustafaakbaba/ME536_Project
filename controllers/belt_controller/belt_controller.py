"""belt_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor
from controller import Supervisor
from controller import DistanceSensor
import random
thresh = 0
# create the Robot instance.
robot = Supervisor()

# get the time step of the current world.
timestep = int(robot.getBasicTimeStep())
root_node = robot.getRoot()
lr_l = robot.getDevice("lr_l")
children_field = root_node.getField('children')
#lr_l.setPosition(5.5)


obj_list = ["RubberDuck", "E-puck", "PaperBoat"]
obj_list2 = ["RubberDuck", "E-puck", "PaperBoat", "Rock10cm", "FlowerPot", "ComputerMouse"]
#obj_list2 = ["ComputerMouse", "Rock10cm"]
sensor_name = 'sens'
dist = DistanceSensor(sensor_name)
DistanceSensor.enable(dist, timestep)
#prox_sensor = robot.getDevice(sensor_name)
#prox_sensor.enable(timestep)
    
pos_value = [-1, 0, 1.75]
 
i = 1
while robot.step(timestep) != -1:
    
    #print(dist.getValue())
    if i%280 == 0 and i <= thresh:
        obj = random.choice(obj_list)
        str_1 = obj + '{ translation ' + str(pos_value[0]) + ' ' + str(pos_value[1])+ ' ' + str(pos_value[2])
        if obj == "E-puck":
            str_2 = ' controller "<generic>"}'
            children_field.importMFNodeFromString(-1, str_1 + str_2)
        else:
            children_field.importMFNodeFromString(-1, str_1 + '}')
        
        
    if i%280 == 0 and i > thresh:
        obj = random.choice(obj_list2)
        str_1 = obj + '{ translation ' + str(pos_value[0]) + ' ' + str(pos_value[1])+ ' ' + str(pos_value[2])
        if obj == "E-puck":
            #print("epuck")
            str_2 = ' controller "<generic>"}'
            children_field.importMFNodeFromString(-1, str_1 + str_2)
        elif obj == "Rock10cm":
            #print("rock")
            str_1 = obj + '{ translation ' + str(pos_value[0]) + ' ' + str(pos_value[1])+ ' ' + str(pos_value[2]+0.04)
            str_2 = ' rotation 0 1 0 -2.88 color 0 1 1 physics Physics{ density 1e+03 mass -1}}'
            children_field.importMFNodeFromString(-1, str_1 + str_2)
        elif obj == "FlowerPot":
            #print("pot")
            str_1 = obj + '{ translation ' + str(pos_value[0]) + ' ' + str(pos_value[1])+ ' ' + str(pos_value[2]+0.05)
            str_2 = ' rotation 0 1 0 -1.571 physics Physics{ density 1e+03 mass -1}}'
            children_field.importMFNodeFromString(-1, str_1 + str_2)
        elif obj == "ComputerMouse":
            str_2 = ' topColor 0.3 0.7 0.7 bottomColor 0.7 0.7 0.06}'
            children_field.importMFNodeFromString(-1, str_1 + str_2)
        else:
            children_field.importMFNodeFromString(-1, str_1 + '}')
        #print(obj)
    i = i+1
    pass

# Enter here exit cleanup code.
