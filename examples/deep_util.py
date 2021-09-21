import pygame
import numpy as np
import sys
import torch
import glob
import os
import cv2
import weakref
import random
import logging
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from torchvision import datasets, transforms

try:
	import queue
except ImportError:
	import Queue as queue

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
from carla import VehicleLightState as vls
from models.model_supervised import Model_Segmentation_Traffic_Light_Supervised
from models.model_se import Net

def set_all_traffic_lights(curr_map,world):
	for ld in curr_map.get_all_landmarks():
		tlight=world.get_traffic_light(ld)
		if tlight:
			tlight.set_state(carla.TrafficLightState.Green)
			#print(tlight.get_red_time(),tlight.get_green_time())
			#tlight.set_red_time(2)
			#tlight.set_green_time(10)
			#print(tlight,"is set")


def get_coord(start_transform):
	x=start_transform.location.x
	y=start_transform.location.y
	z=start_transform.location.z
	return x,y,z

def get_font():
	fonts = [x for x in pygame.font.get_fonts()]
	default_font = 'ubuntumono'
	font = default_font if default_font in fonts else fonts[0]
	font = pygame.font.match_font(font)
	return pygame.font.Font(font, 14)
#Synchronous mode
def draw_image(surface, image, blend=False):
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))
	array = array[:, :, :3]
	array = array[:, :, ::-1]
	image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
	if blend:
		image_surface.set_alpha(100)
	surface.blit(image_surface, (0, 0))

def draw_image_288(surface, image, blend=False):
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))
	array = array[:, :, :3]
	array = array[:, :, ::-1]
	image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
	if blend:
		image_surface.set_alpha(100)
	surface.blit(image_surface, (800, 0))

def draw_image_ps(surface, image, x, blend=False):
	array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
	array = np.reshape(array, (image.height, image.width, 4))
	array = array[:, :, :3]
	array = array[:, :, ::-1]
	image_surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
	if blend:
		image_surface.set_alpha(100)
	surface.blit(image_surface, (x, 0))

def draw_images(surface,*images,blend=False):
	x=800
	for image in images:
		draw_image_ps(surface,image,x,blend)
		x+=288

class PygDisplay:
	def __init__(self):
		pygame.init()
		self.display=pygame.display.set_mode((1952, 600))
		self.clock = pygame.time.Clock()
		self.font_big= pygame.font.Font(None, 40)
		self.font_small=self.get_font()
		self.red=(255,0,0)
		self.green=(0,128,0)
		self.risk_traj=0
		self.risk_deep=0

	def get_font(self):
		fonts = [x for x in pygame.font.get_fonts()]
		default_font = 'ubuntumono'
		font = default_font if default_font in fonts else fonts[0]
		font = pygame.font.match_font(font)
		return pygame.font.Font(font, 14)

	def py_show(self,image_rgb,*images,risk_traj=None,risk_deep=None):
		self.clock.tick()
		draw_image(self.display, image_rgb)
		draw_images(self.display,*images)
		self.display.blit(self.font_small.render('% 5d FPS (real)' % self.clock.get_fps(), True, (255, 255, 255)),(8, 10))

		if risk_traj != None:
			self.risk_traj=risk_traj
		if risk_deep != None:
			self.risk_deep=risk_deep

		if self.risk_traj<0.5:
			traj_color=self.green
		if self.risk_traj>=0.5:
			traj_color=self.red

		if self.risk_deep<0.5:
			deep_color=self.green
		if self.risk_deep>=0.5:
			deep_color=self.red

		self.display.blit(self.font_big.render("Risk (global view):"+str(self.risk_traj), False, traj_color),(220,20))
		self.display.blit(self.font_big.render("Risk (local view):"+str(self.risk_deep), False, deep_color),(220,60))
		pygame.display.flip()

class CarlaSyncMode(object):
	"""
	Context manager to synchronize output from different sensors. Synchronous
	mode is enabled as long as we are inside this context

		with CarlaSyncMode(world, sensors) as sync_mode:
			while True:
				data = sync_mode.tick(timeout=1.0)

	"""
	def __init__(self, world, *sensors, **kwargs):
		self.world = world
		self.sensors = sensors
		self.actors=[]
		self.frame = None
		self.delta_seconds = 1.0 / kwargs.get('fps', 20)
		self._queues = []
		self._settings = None
		self.times_of_ticks=0

	def __enter__(self):
		self._settings = self.world.get_settings()
		self.frame = self.world.apply_settings(carla.WorldSettings(
			no_rendering_mode=False,
			synchronous_mode=True,
			fixed_delta_seconds=self.delta_seconds))

		def make_queue(register_event):
			q = queue.Queue()
			register_event(q.put)
			self._queues.append(q)

		make_queue(self.world.on_tick)
		for sensor in self.sensors:
			make_queue(sensor.listen)
		return self


	def tick(self, timeout):
		self.frame = self.world.tick()
		self.times_of_ticks+=1
		data = [self._retrieve_data(q, timeout) for q in self._queues]
		assert all(x.frame == self.frame for x in data)
		return data

	def __exit__(self, *args, **kwargs):
		self.world.apply_settings(self._settings)

	def _retrieve_data(self, sensor_queue, timeout):
		while True:
			data = sensor_queue.get(timeout=timeout)
			if data.frame == self.frame:
				return data

class CameraManager:
	def __init__(self,world):
		self.world = world
		self.actors=[]

	def set_up_sensors(self,vehicle):
		blueprint_library=self.world.get_blueprint_library()
		bp_camera=blueprint_library.find('sensor.camera.rgb')
		camera_rgb = self.world.spawn_actor(
			bp_camera,
			carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
			attach_to=vehicle)
		self.actors.append(camera_rgb)

		bp_camera_288=blueprint_library.find('sensor.camera.rgb')
		bp_camera_288.set_attribute('image_size_x', "288")
		bp_camera_288.set_attribute('image_size_y', '288')

		camera_front = self.world.spawn_actor(
			bp_camera_288,
			carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0)),
			attach_to=vehicle)
		self.actors.append(camera_front)

		camera_back = self.world.spawn_actor(
			bp_camera_288,
			carla.Transform(carla.Location(x=0, z=2.4), carla.Rotation(pitch=0,yaw=180)),
			attach_to=vehicle)
		self.actors.append(camera_back)

		camera_left = self.world.spawn_actor(
			bp_camera_288,
			carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0,yaw=90)),
			attach_to=vehicle)
		self.actors.append(camera_left)

		camera_right = self.world.spawn_actor(
			bp_camera_288,
			carla.Transform(carla.Location(x=1.5, z=2.4), carla.Rotation(pitch=0,yaw=270)),
			attach_to=vehicle)
		self.actors.append(camera_right)
		return camera_rgb,camera_front,camera_back,camera_left,camera_right

class LidarManager:
	def __init__(self,world):
		self.world = world
		self.actors=[]
		self.bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')

	def set_up_sensors(self,vehicle):
		self.bp.set_attribute('range', '30')
		self.bp.set_attribute('rotation_frequency', '10')
		self.bp.set_attribute('channels', '32')
		self.bp.set_attribute('lower_fov', '-30')
		self.bp.set_attribute('upper_fov', '30')
		self.bp.set_attribute('points_per_second', '56000')
		transform_lidar = carla.Transform(carla.Location(x=0.0, z=5.0))
		lidar = self.world.spawn_actor(self.bp, transform_lidar, attach_to=vehicle)
		lidar.listen(lambda data: do_something(data))


class CollisionManager:
	flag=0
	def __init__(self,world):
		self.world = world

		self.blueprints=self.world.get_blueprint_library()

	def set_up_sensors(self,vehicle):
		self.collision_sensor = self.world.spawn_actor(
			self.blueprints.find("sensor.other.collision"),
			carla.Transform(),
			attach_to=vehicle,
		)
		weak_self = weakref.ref(self)
		self.collision_sensor.listen(lambda event: CollisionManager._on_collision(weak_self, event))
		return self.collision_sensor

	@staticmethod
	def _on_collision(weak_self,event):
		print("collision happened")
		#weak_self.flag=1
		CollisionManager.flag=1

class VehicleManager:
	def __init__(self,world,client):
		self.client=client
		self.world = world
		self.bp=self.world.get_blueprint_library()
		self.bp_agent=self.world.get_blueprint_library().find('vehicle.audi.tt')
		self.bp_agent.set_attribute('role_name', 'ego')
		self.bp_agent.set_attribute('color', "0,0,255")
		#self.bp_bots=self.world.get_blueprint_library().find('vehicle.audi.tt')
		self.bots=[]
		self.agent_location=carla.Location(x=27.532440, y=-134.615311, z=1.200000)
		self.agent_rotation=carla.Rotation(pitch=0.000000, yaw=91.532082, roll=0.000000)
		self.agent_transform=carla.Transform(self.agent_location,self.agent_rotation)
		self.norm_rotation=carla.Rotation(pitch=0.000000, yaw=91.532082, roll=0.000000)
		self.bot_control=None

	def generate_town5_agent(self,coord=None):
		#self.agent_transform = carla.Transform(carla.Location(x=X[0][3], y=X[0][4], z=1.2),carla.Rotation(pitch=0, yaw=X[0][5], roll=0))
		if coord != None:
			self.agent_transform = carla.Transform(carla.Location(x=coord[0][3], y=coord[0][4], z=1.2),carla.Rotation(pitch=0, yaw=coord[0][5], roll=0))
		self.agent_vehicle=self.world.spawn_actor(self.bp_agent,self.agent_transform)
		return self.agent_vehicle

	def generate_agent_vehicle(self):

		"""
		blueprint_library = self.world.get_blueprint_library()
		bp_agent=blueprint_library.find('vehicle.audi.tt')
		"""
		spawn_points = self.world.get_map().get_spawn_points()
		start_transform=spawn_points[0]
		print("Start",start_transform)
		x_start,y_start,z_start=get_coord(start_transform)
		y_start=y_start+145
		self.norm_rotation=carla.Rotation(start_transform.rotation.pitch,start_transform.rotation.yaw-3,start_transform.rotation.roll)
		x_agent=x_start+1.5-5.5
		y_agent=y_start-20
		z_agent=z_start
		self.agent_location=carla.Location(x_agent,y_agent,z_agent)
		self.agent_transform=carla.Transform(self.agent_location,self.norm_rotation)
		self.agent_vehicle=self.world.spawn_actor(self.bp_agent,self.agent_transform)
		return self.agent_vehicle

	def get_bots_left_transforms(self):
		x_react=self.agent_location.x+3.5
		y_react=self.agent_location.y-6
		z_react=self.agent_location.z
		self.react_location=carla.Location(x_react,y_react,z_react)
		self.react_transform=carla.Transform(self.react_location,self.norm_rotation)

		x,y,z=get_coord(self.react_transform)
		#variation=random.randint(0,3)-random.randint(0,3)
		#y=y+variation
		transform_list=[]
		transform_list.append(self.react_transform)
		for i in range(1,3):
			vehicle_location=carla.Location(x,y+i*8,z)
			vehicle_transform=carla.Transform(vehicle_location,self.norm_rotation)
			transform_list.append(vehicle_transform)
		#sampled_transforms=random.sample(transform_list,num_vehicles)
		num_left_bots=random.choice([0,1,2,3])
		new_transform_list=random.sample(transform_list,k=num_left_bots)
		return new_transform_list

	def get_bots_right_transforms(self):
		x,y,z=get_coord(self.agent_transform)
		#variation=random.randint(0,3)-random.randint(0,3)
		#y=y+variation
		transform_list=[]
		for i in range(2,5):
			vehicle_location=carla.Location(x,y+i*7.5,z)
			vehicle_transform=carla.Transform(vehicle_location,self.norm_rotation)
			transform_list.append(vehicle_transform)
		return transform_list

	def generate_bot_vehicles(self):
		#self.bp_bots.set_attribute('color','255,255,255')

		bots_left_transforms=self.get_bots_left_transforms()
		for transform in bots_left_transforms:
			#self.bp_pots=random.choice(self.bp.filter('vehicle.*'))
			bp_bots=random.choice(self.bp.filter('vehicle.*'))

			bot=self.world.spawn_actor(bp_bots,transform)

			self.bots.append(bot)

		print(len(bots_left_transforms),"Bots on left is spawned")


		bots_right_transforms=self.get_bots_right_transforms()
		print(len(bots_right_transforms),"Bots on right transforms are gotten")
		for transform in bots_right_transforms:
			bp_bots=random.choice(self.bp.filter('vehicle.*'))

			bot=self.world.spawn_actor(bp_bots,transform)

			#print("Bot on right is spawn")
			self.bots.append(bot)
		print("Bots on right are spawned")

		#self.react_vehicle=self.world.spawn_actor(bp_bots,self.react_transform)
		#self.bots.append(self.react_vehicle)
		return self.bots

	def generate_bot_control(self):
		control_highspeed=carla.VehicleControl(throttle=0.9)
		control_middlespeed=carla.VehicleControl(throttle=0.5)
		control_lowspeed=carla.VehicleControl(throttle=0.3)
		self.bot_control=random.choice([control_highspeed,control_middlespeed,control_lowspeed])
		return self.bot_control

	def bot_run_step(self):
		for bot in self.bots:
			bot.apply_control(self.bot_control)
	
	def spawn_npc(self,number_of_vehicles):
		batch=[]
		npc_list=[]

		traffic_manager = self.client.get_trafficmanager(port=8000)
		traffic_manager.set_global_distance_to_leading_vehicle(1.0)
		spawn_points = self.world.get_map().get_spawn_points()

		SpawnActor = carla.command.SpawnActor
		SetAutopilot = carla.command.SetAutopilot
		SetVehicleLightState = carla.command.SetVehicleLightState
		FutureActor = carla.command.FutureActor
		light_state = vls.NONE
		
		for n, transform in enumerate(spawn_points):
			if n >= number_of_vehicles:
				break
			blueprints=self.bp.filter('vehicle.*')
			blueprints = [x for x in blueprints if int(x.get_attribute('number_of_wheels')) == 4]
			blueprint = random.choice(blueprints)
			#print("got blurprint")
			batch.append(SpawnActor(blueprint, transform)
					.then(SetAutopilot(FutureActor, True, traffic_manager.get_port()))
					.then(SetVehicleLightState(FutureActor, light_state)))

		print("start to spawn")
		for response in self.client.apply_batch_sync(batch,True):
			#print("spwnning")
			if response.error:
				logging.error(response.error)
			else:
				npc_list.append(response.actor_id)
		
		return npc_list
	
	def spawn_walkers(sefl,number_of_walker):
		pass

class SupervisedEncoder:
	def __init__(self):
		self.steps_image = [-10,-2,-1,0,]
		self.model_supervised=Model_Segmentation_Traffic_Light_Supervised(len(self.steps_image),len(self.steps_image),1024, 6, 4,crop_sky=False)
		self.device = torch.device("cuda")
		path_to_encoder_dict="models/model_epoch_34.pth"
		self.model_supervised.load_state_dict(torch.load(path_to_encoder_dict,map_location=self.device))
		self.model_supervised.to(device=self.device)
		self.encoder=self.model_supervised.encoder
		self.last_conv_downsample = self.model_supervised.last_conv_downsample
		self._rgb_queue=None
		self.window=(max([abs(number) for number in self.steps_image]) + 1)
		self.RGB_image_buffer = deque([], maxlen=self.window)
		#print("initial RGB length",len(self.RGB_image_buffer))
		for _ in range(self.window):
			self.RGB_image_buffer.append(np.zeros((3, 288, 288)))
		self.render=True


	def carla_img_to_np(self,carla_img):
		carla_img.convert(ColorConverter.Raw)

		#numpy.frombuffer(buffer, dtype=float, count=-1, offset=0, *, like=None)¶
		#Interpret a buffer as a 1-dimensional array.
		img = np.frombuffer(carla_img.raw_data, dtype=np.dtype("uint8"))

		img = np.reshape(img, (carla_img.height, carla_img.width, 4))

		img = img[:, :, :3]

		#carla uses bgr so need to convert it to rgb
		img = img[:, :, ::-1]

		return img

	def run_step(self,image,name,snapshot,save_flag=False):
		#self.rgb_image = self._rgb_queue.get()
		#rgb = observations["rgb"].copy()
		rgb=self.carla_img_to_np(image).copy()
		rgb = np.array(rgb)
		if self.render:
			#opencv uses bgr, need to convert rgb back to bgr
			bgr = rgb[:, :, ::-1]
			#cv2.imshow("network input", bgr)
			#cv2.imshow(name, bgr)
			#cv2.waitKey(1)
		#print("Shape before roll",rgb.shape)
		rgb = np.rollaxis(rgb, 2, 0)
		#print("Shape after roll",rgb.shape)
		self.RGB_image_buffer.append(rgb)
		#print(len(self.RGB_image_buffer))

		#The axis along which the arrays will be joined. If axis is None, arrays are flattened before use. Default is 0.
		np_array_RGB_input = np.concatenate(
			[
				self.RGB_image_buffer[indice_image + self.window - 1]
				for indice_image in self.steps_image
			]
		)

		if save_flag:
			for indice_image in self.steps_image:
				new_image=self.RGB_image_buffer[indice_image + self.window - 1].copy()
				#print("new_image_shape",new_image.shape)
				#new_image=np.rollaxis(new_image,2,0)
				#new_image=np.rollaxis(new_image,2,0)
				new_image=np.rollaxis(new_image,0,3)
				#print("new_image_shape after roll",new_image.shape)
				new_bgr=new_image[:, :, ::-1]
				#cv2.imwrite("images_2/"+snapshot+"_"+str(abs(indice_image))+"_"+name+".png",new_bgr)
				cv2.imwrite("integration/"+snapshot+"_"+str(abs(indice_image))+"_"+name+".png",new_bgr)

		torch_tensor_input = (
			torch.from_numpy(np_array_RGB_input)
			.to(dtype=torch.float32, device=self.device)
			.div_(255)
			.unsqueeze(0)
		)
		with torch.no_grad():
			current_encoding = self.encoder(torch_tensor_input)
			#print(current_encoding)
			current_encoding = self.last_conv_downsample(current_encoding)
		#cv2.imshow("Output", current_encoding)

		#cv2.waitKey(1)
		#print("size is",current_encoding.size())
		current_encoding_np = current_encoding.cpu().numpy().flatten()
		#print(len(current_encoding_np))
		return current_encoding_np

class SafetyEstimator:
	def __init__(self,state_dict_path="safe_estimation.pt") -> None:
		self.device=torch.device("cuda")
		self.sftmx=nn.Softmax(dim=1)
		self.estimate_model=Net()
		self.estimate_model.load_state_dict(torch.load(state_dict_path))
		self.transform=transforms.Compose([transforms.ToTensor()])

	def predict(self,sample):
		new_arr=np.expand_dims(sample,axis=0)
		new_sample=torch.from_numpy(new_arr)
		print(new_sample.size())
		new_sample.to(self.device)
		#label=self.estimate_model(new_sample).argmax(dim=1, keepdim=True)
		value=self.sftmx(self.estimate_model(new_sample))
		return float(value[0][1])


	
