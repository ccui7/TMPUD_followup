#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys
import random
import weakref
import torch
from collections import deque
import cv2
import time
import datetime

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
from carla import ColorConverter

import random

try:
	import pygame
except ImportError:
	raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
	import numpy as np
except ImportError:
	raise RuntimeError('cannot import numpy, make sure numpy package is installed')

try:
	import queue
except ImportError:
	import Queue as queue

from agents.navigation.behavior_agent import BehaviorAgent  # pylint: disable=import-error
from agents.navigation.roaming_agent import RoamingAgent  # pylint: disable=import-error
from agents.navigation.basic_agent import BasicAgent  # pylint: disable=import-error
from models.model_supervised import Model_Segmentation_Traffic_Light_Supervised

from deep_util import CarlaSyncMode,CameraManager,CollisionManager,VehicleManager,SupervisedEncoder,SafetyEstimator
from deep_util import get_coord,get_font,draw_image,draw_images,py_show

def should_quit():
	for event in pygame.event.get():
		if event.type == pygame.QUIT:
			return True
		elif event.type == pygame.KEYUP:
			if event.key == pygame.K_ESCAPE:
				return True
	return False


def main():
	#os.environ["CUDA_VISIBLE_DEVICES"] = '0'
	actor_list = []
	pygame.init()

	display = pygame.display.set_mode(
		(1952, 600))


	#display_288 = pygame.display.set_mode(
		#(288, 288))
	clock = pygame.time.Clock()
	client = carla.Client('localhost', 2000)
	client.set_timeout(2.0)

	#world = client.get_world()
	world = client.load_world('Town05')

	try:
		m = world.get_map()
		blueprint_library = world.get_blueprint_library()

		vehicle_manager=VehicleManager(world)
		#vehicle=vehicle_manager.generate_agent_vehicle()
		vehicle=vehicle_manager.generate_town5_agent()
		actor_list.append(vehicle)

		#(x,y,z) = (30.070419311523438,-108.70726013183594,2.461855888366699)
		#x=27.532440, y=-134.615311, z=1.200000
		#merge_location=carla.Location(-10.178999900817871,65.29945373535156,2.618715524673462)
		merge_location=carla.Location(x=27.532440, y=-128.615311, z=1.200000)
		merge_waypoint=m.get_waypoint(merge_location)

		#target_location=carla.Location(-6.330255508422852,108.34795379638672,2.5465259552001953)
		target_location=carla.Location(30.070419311523438,-108.70726013183594,2.461855888366699)
		target_waypoint=m.get_waypoint(target_location)


		camera_manager=CameraManager(world)
		camera,*cameras_288=camera_manager.set_up_sensors(vehicle)
		actor_list.append(camera)
		for cam in cameras_288:
			actor_list.append(cam)

		collision_manager=CollisionManager(world)
		collision_sensor=collision_manager.set_up_sensors(vehicle)
		print("collision flag is",collision_manager.flag)
		actor_list.append(collision_sensor)

		supervised_encoder_front=SupervisedEncoder()
		supervised_encoder_back=SupervisedEncoder()
		supervised_encoder_right=SupervisedEncoder()
		supervised_encoder_left=SupervisedEncoder()

		#safety_estimator=SafetyEstimator()
		
		agent = BasicAgent(vehicle)
		agent.set_destination((merge_location.x,merge_location.y,merge_location.z))


		
		bots=vehicle_manager.generate_bot_vehicles()

		for bot in bots:
			actor_list.append(bot)
		
		print("There are",len(actor_list),"actors")
		

		control_highspeed=carla.VehicleControl(throttle=0.9)
		control_middlespeed=carla.VehicleControl(throttle=0.5)
		control_lowspeed=carla.VehicleControl(throttle=0.3)
		bot_control=random.choice([control_highspeed,control_middlespeed,control_lowspeed])

		tm = client.get_trafficmanager(8000)
		tm_port=tm.get_port()
		tm.set_synchronous_mode(True)
		# Create a synchronous mode context.

		start_time=datetime.datetime.utcnow()
		with CarlaSyncMode(world, camera,*cameras_288, fps=30) as sync_mode:
			#while True:
			"""
			flag=random.choice([0,1])
			
			if flag==1:
				for bot in bots:
					bot.set_autopilot(True)
			"""

			for sec in range(16):
				snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)
				py_show(display,clock,image_rgb,*images)
				supervised_encoder_front.run_step(images[0],"Front",str(snapshot))
				supervised_encoder_back.run_step(images[1],"Back",str(snapshot))
				supervised_encoder_right.run_step(images[2],"Right",str(snapshot))
				supervised_encoder_left.run_step(images[3],"Left",str(snapshot))

			while agent.done()==False and collision_manager.flag==0:
				if (datetime.datetime.utcnow()-start_time).total_seconds() > 240:
					break
				if should_quit():
					return
				snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)
				py_show(display,clock,image_rgb,*images)
				fps = round(1.0 / snapshot.timestamp.delta_seconds)
				
				supervised_encoder_front.run_step(images[0],"Front",str(snapshot))
				supervised_encoder_back.run_step(images[1],"Back",str(snapshot))
				supervised_encoder_right.run_step(images[2],"Right",str(snapshot))
				supervised_encoder_left.run_step(images[3],"Left",str(snapshot))

				control = agent.run_step()
				control.manual_gear_shift = False
				control.brake=0
				control.throttle=0.4
				vehicle.apply_control(control)

				
				for bot in bots:
					bot.apply_control(bot_control)
				


			print("At merge point or collision happened",collision_manager.flag)
			agent.set_destination((target_location.x,target_location.y,target_location.z))
			
			snapshot, image_rgb, *images = sync_mode.tick(timeout=2.0)
			print("snapshot is",snapshot)
			py_show(display,clock,image_rgb,*images)
			front=supervised_encoder_front.run_step(images[0],"Front",str(snapshot),save_flag=True)
			back=supervised_encoder_back.run_step(images[1],"Back",str(snapshot),save_flag=True)
			right=supervised_encoder_right.run_step(images[2],"Right",str(snapshot),save_flag=True)
			left=supervised_encoder_left.run_step(images[3],"Left",str(snapshot),save_flag=True)
			print("ticks",sync_mode.times_of_ticks)
			com=np.concatenate((front,back,right,left))
			#risk=safety_estimator.predict(com)
			#print("XXXXXXXXXXXXXRISKXXXXXXXXXXX",risk)

			while agent.done()==False and collision_manager.flag==0:
				
				if (datetime.datetime.utcnow()-start_time).total_seconds() > 240:
					break


				# Advance the simulation and wait for the data.
				#snapshot, image_rgb, image_rgb_288, image_semseg = sync_mode.tick(timeout=2.0)
				snapshot_2, image_rgb, *images = sync_mode.tick(timeout=1.0)
				py_show(display,clock,image_rgb,*images)
				fps = round(1.0 / snapshot.timestamp.delta_seconds)
				control = agent.run_step()
				control.manual_gear_shift = False
				vehicle.apply_control(control)
				#print(agent.done())
				
				for bot in bots:
					control.steer=0.0
					control.brake=random.random()
					control.throttle=random.random()
					bot.apply_control(bot_control)
			
			if agent.done()==True:
					print("Destination has arrived")

			if collision_manager.flag == 0:
				#image_rgb.save_to_disk("images/class_A"+str(snapshot))
				print("No collision")
				np.save("town5_class_A/"+str(snapshot),com)
				#print("No collision")
			
			elif collision_manager.flag == 1:
				#image_rgb.save_to_disk("images/class_B"+str(snapshot))
				np.save("town5_class_B/"+str(snapshot),com)
				
			#print("Output length is",len(com))

		
				
	finally:

		print('destroying actors.')
		print("Finally there are",len(actor_list),"actors")
		for actor in actor_list:
			actor.destroy()

		pygame.quit()
		print('done.')


if __name__ == '__main__':
	try:
		main()

	except KeyboardInterrupt:
		print('\nCancelled by user. Bye!')
