from roborollout_vla.robots import droid, yam
from roborollout_vla.robots.base import Arm, Robot

# robots that policy servers can be run for, by name
ROBOTS = {robot.name: robot for robot in (droid.ROBOT, yam.ROBOT)}
