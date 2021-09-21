#!/usr/bin/env python

# 加载需要的包和环境
from __future__ import print_function
import argparse
import collections
import datetime
import glob
import logging
import math
import os
import signal
import random
import re
import sys
import weakref
import numpy as np
import time
import torch
from collections import deque

os.chdir('/home/cckklt/CARLA_0.9.10.1/PythonAPI/examples')


# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================
try:
	sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
		sys.version_info.major,
		sys.version_info.minor,
		'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
	pass

try:
	sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/carla')
except IndexError:
	pass

import carla
from carla import ColorConverter as cc
from carla import ColorConverter

from agents.navigation.roaming_agent import RoamingAgent
from agents.navigation.basic_agent import BasicAgent  # 创造自己的运动模型，可以设置不同的速度参数
from agents.navigation.behavior_agent import BehaviorAgent 

from models.model_supervised import Model_Segmentation_Traffic_Light_Supervised
import deep_util
from deep_util import CarlaSyncMode,SupervisedEncoder,CameraManager,CollisionManager,SafetyEstimator,VehicleManager,PygDisplay,set_all_traffic_lights
from global_risk import fuc_risk,getX,compute_total_distance,write_data

import math
import random
import statistics



# ==============================================================================
# -- add PythonAPI for release mode --------------------------------------------
# ==============================================================================
try:
	sys.path.append(glob.glob('PythonAPI')[0])
except IndexError:
	pass

try:
	import queue
except ImportError:
	import Queue as queue

try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')



global X, vehicle_ego, number, alpha, D, X_original

thre_risk = 0.5  # risk model中某个系数
alpha = 1.0  # risk model中某个系数
D = 20  # risk model中最小的距离阈值
D = D * D
SD = 15  # 距离目的地的距离

address = '/home/cckklt/Downloads/TMP/Task-Motion-Interaction/'
log = open(address + 'middle(0.5)_37_47.txt', 'a')

X = getX()
X_original = X

def game_loop():
	global X, X_original
	vehicles_list = []
	sensors_list=[]
	npc_list=[]
	pygame.init()
	#display = pygame.display.set_mode((1952, 600))
	#font = deep_util.get_font()
	
	#clock = pygame.time.Clock()
	print("X is",X)
	client = carla.Client('localhost', 2000)
	client.set_timeout(15.0)
	traj = []

	try:
		
		#change map to town5
		world = client.load_world('Town05')
		time.sleep(2)
		world = client.get_world()
		
		vehicle_manager=VehicleManager(world,client)
		time.sleep(1.5)
		vehicle_ego = vehicle_manager.generate_town5_agent()
		log.write('Id of hero car is %d!\n' % vehicle_ego.id)
		recordfile_name = address + 'saved/' + "middle(0.5)_37_47_recording_" + str(time.time()) + ".log"
		log.write('new filename is %s\n' % recordfile_name)

		timestamp=int(datetime.datetime.utcnow().timestamp())

		print("time stamp is",timestamp)
		client.start_recorder("/home/cckklt/Downloads/TMP/MotionPlanner_TMPUD/logs/"+str(timestamp)+".log")
		print("Start Recording")
		
		vehicles_list.append(vehicle_ego)
		agent_ego = BasicAgent(vehicle_ego)
		
		vehicles = world.get_actors().filter('vehicle.*')  # 获得所有车辆信息
		log.write('此时有多少辆车？%d' % len(vehicles))

		#world.wait_for_tick()
		pyg_display=PygDisplay()
		
		camera_manager=CameraManager(world)
		camera,*cameras_288=camera_manager.set_up_sensors(vehicle_ego)
		supervised_encoder_front=SupervisedEncoder()
		supervised_encoder_back=SupervisedEncoder()
		supervised_encoder_right=SupervisedEncoder()
		supervised_encoder_left=SupervisedEncoder()
		safety_estimator=SafetyEstimator(state_dict_path="safe_estimation_town5.pt")

		collision_manager=CollisionManager(world)
		collision_sensor=collision_manager.set_up_sensors(vehicle_ego)
		sensors_list.append(collision_sensor)
		sensors_list.append(camera)
		for cam in cameras_288:
			sensors_list.append(cam)

		tm = client.get_trafficmanager(port=8000)
		tm_port = tm.get_port()

		"""
		bots=vehicle_manager.generate_bot_vehicles()
		for bot in bots:
			vehicles_list.append(bot)
			bot.set_autopilot()
		"""

		
		vehicle_manager.generate_bot_control()

		with CarlaSyncMode(world, camera,*cameras_288, fps=30) as sync_mode:
			#world.tick()
			for second in range(4):
				snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)
			
				front=supervised_encoder_front.run_step(images[0],"Front",str(snapshot))
				back=supervised_encoder_back.run_step(images[1],"Back",str(snapshot))
				right=supervised_encoder_right.run_step(images[2],"Right",str(snapshot))
				left=supervised_encoder_left.run_step(images[3],"Left",str(snapshot))

				pyg_display.py_show(image_rgb,*images)
			#print('ego starts at x: %3.3f, y: %3.3f, yaw:%3.3f\n' % (X[0][3], X[0][4], X[0][5]))
			#spawn other vehicles
			npc_list=vehicle_manager.spawn_npc(180)


			log.write('ego starts at x: %3.3f, y: %3.3f, yaw:%3.3f\n' % (X[0][3], X[0][4], X[0][5]))
			i = 0
			behavior = X[i][1]  # merge lane (1) or not(0)
			
			while i < len(X) - 1:
				if behavior == 0:  # do not merge lane, not use risk model
					i = i + 1
					#agent_ego.set_destination(vehicle_ego.get_location(),(X[i][3], X[i][4], 1.2))
					#agent_ego.set_destination(vehicle_ego.get_location(),carla.Location(X[i][3], X[i][4], 1.2))
					agent_ego.set_destination((X[i][3], X[i][4], 1.2))
					log.write('current lane is %d, ego does not merge lane, performs action %d to get next lane %d\n' % (
						X[i - 1][2], X[i - 1][0], X[i][2]))

					# ego的当前坐标
					e_l = vehicle_ego.get_location()
					# 直到当前坐标与目的地重合
					while math.sqrt((e_l.x - X[i][3]) ** 2 + (e_l.y - X[i][4]) ** 2) > SD:
						#if not world.wait_for_tick():  # as soon as the server is ready continue!
							#continue
						
						snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)

						front=supervised_encoder_front.run_step(images[0],"Front",str(snapshot))
						back=supervised_encoder_back.run_step(images[1],"Back",str(snapshot))
						right=supervised_encoder_right.run_step(images[2],"Right",str(snapshot))
						left=supervised_encoder_left.run_step(images[3],"Left",str(snapshot))

						#deep_util.py_show(display,clock,image_rgb,*images)collision_flag
						pyg_display.py_show(image_rgb,*images)

						control = agent_ego.run_step()
						control.manual_gear_shift = False
						vehicle_ego.apply_control(control)
						traj.append(vehicle_ego.get_location())
						e_l = vehicle_ego.get_location()
						#vehicle_manager.bot_run_step()

					behavior = X[i][1]  # 已经到达lane，下一个action是behavior
					log.write('ego vehicle reaches lane %d and prepare to take action %d (1 or 0) !\n' % (X[i][2], behavior))

				else:  # merge lane, apply risk model and compute the p_risk
					# ==============================================================================
					for second in range(8):
						snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)

						front=supervised_encoder_front.run_step(images[0],"Front",str(snapshot))
						back=supervised_encoder_back.run_step(images[1],"Back",str(snapshot))
						right=supervised_encoder_right.run_step(images[2],"Right",str(snapshot))
						left=supervised_encoder_left.run_step(images[3],"Left",str(snapshot))
						pyg_display.py_show(image_rgb,*images)
					
					snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)
					
					front=supervised_encoder_front.run_step(images[0],"Front",str(snapshot),save_flag=True)
					back=supervised_encoder_back.run_step(images[1],"Back",str(snapshot),save_flag=True)
					right=supervised_encoder_right.run_step(images[2],"Right",str(snapshot),save_flag=True)
					left=supervised_encoder_left.run_step(images[3],"Left",str(snapshot),save_flag=True)

					pyg_display.py_show(image_rgb,*images)
					control = agent_ego.run_step()
					control.manual_gear_shift = False
					vehicle_ego.apply_control(control)
					traj.append(vehicle_ego.get_location())
					e_l = vehicle_ego.get_location()
			
					print(sync_mode.times_of_ticks)
					com=np.concatenate((front,back,right,left))
					risk=round(safety_estimator.predict(com),3)
					print("#########Risk is################",risk)
					
					currentlane = X[i][2]
					p_risk = fuc_risk(i, vehicle_ego, world, currentlane,X=X,log=log)  # this step is to calculate p_risk, i 是当前step
					print(p_risk)
					log.write('current lane is %d and p_risk value is %3.3f when merging lane\n' % (X[i][2], p_risk))
					print('current lane is %d and p_risk value is %3.3f when merging lane\n' % (X[i][2], p_risk))

					pyg_display.py_show(image_rgb,*images,risk_traj=p_risk,risk_deep=risk)
					
					# ==============================================================================
					#if p_risk < thre_risk:  # smaller thre_risk is, safer system is
					if risk < thre_risk:
						# still merge lane
						i = i + 1
						#agent_ego.set_destination(vehicle_ego.get_location(),carla.Location(X[i][3], X[i][4], 1.2))
						agent_ego.set_destination((X[i][3], X[i][4], 1.2))
						e_l = vehicle_ego.get_location()
						while math.sqrt((e_l.x - X[i][3]) ** 2 + (e_l.y - X[i][4]) ** 2) > SD:
							#if not world.wait_for_tick():  # as soon as the server is ready continue!
								#continue
							snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)
							#deep_util.py_show(display,clock,image_rgb,*images)
							pyg_display.py_show(image_rgb,*images)
							
							control = agent_ego.run_step()
							control.manual_gear_shift = False
							vehicle_ego.apply_control(control)
							traj.append(vehicle_ego.get_location())
							# print('simutaneous speed is : \n')
							# print(vehicle_ego.get_velocity())
							
							e_l = vehicle_ego.get_location()
							traj.append(e_l)

							#vehicle_manager.bot_run_step()

						behavior = X[i][1]
						log.write(
							'ego vehicle reaches lane %d and prepare to take action %d (1 or 0) !\n' % (X[i][2], behavior))
							
					#elif p_risk > thre_risk:
					elif risk > thre_risk:
						#p_risk = 1.0
						log.write("Risk! merge lane cannot perform, please do task planning again!\n ")
						log.write('Here to update X (coordinate for motion planner), and original X is %s\n' % X)
						with open(address + 'test_risk.txt', 'r+') as f:
							test_risk = [line.rstrip() for line in f]
						log.write('risk value of merge lane in lane %d needs to update!\n' % int(X[i][2]))
						log.write('its original value is %d, now its value is %d!\n' % (
							int(test_risk[int(X[i][2] - 1) * 12 + 6 - 1]), int(p_risk * 100)))
						test_risk[int(X[i][2] - 1) * 12 + 6 - 1] = int(p_risk * 100)
						test_risk[int(X[i][2] - 1) * 12 + 8 - 1] = int(p_risk * 100)
						with open(address + 'test_risk.txt', 'w+') as f:
							for item in test_risk:
								f.write(str(item))
								f.write('\n')
						f.close()

						start_lane = int(X[i][2])
						desination_lane = int(X[-1][2])

						log.write('re do task planner, start lane and the desination lane are %d and %d, respectively\n' % (
							start_lane, desination_lane))

						commandtaskplanner = 'python' + ' ' + address + 'grounding.py' + ' ' + str(start_lane) + ' ' + str(desination_lane)
						os.system(commandtaskplanner)

						X = getX()
						print("Replanned X is",X)
						if not (X == X_original):
							i = 0
							behavior = X[i][1]  # temporaily do not change
							X_original = X
						else:
							i = i + 1
							behavior = 0  # temporaily do not change
			log.write('###### end! #####\n\n\n\n')
			if collision_manager.flag == 0:
				print("No collision")
			else:
				print("There is collision")
			distance=compute_total_distance(traj)
			print("Traveled distance is",distance)
			write_data(timestamp,distance,collision_manager.flag)

			print("end")

	finally:
		client.stop_recorder()
		log.close()
		print("Start to destroy")
		print("Actor list length",len(vehicles_list),len(sensors_list))
		for s in sensors_list:
			s.stop()

		client.apply_batch([carla.command.DestroyActor(x) for x in sensors_list])
		client.apply_batch([carla.command.DestroyActor(x) for x in vehicles_list])
		client.apply_batch([carla.command.DestroyActor(x) for x in npc_list])
		
		print("ALL cleaned up!")





# ==============================================================================
# -- main() --------------------------------------------------------------
# ==============================================================================
def main():

	game_loop()


if __name__ == '__main__':
	try:
		main()
	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
