"""camera_ros controller."""

from controller import Robot, Camera


robot = Robot()
timestep = int(robot.getBasicTimeStep())
camera = Camera('camera')
camera.enable(100)

while robot.step(timestep) != -1:
    img = camera.getImage()
    pass


    

