#------------------------------------------------------------------------------+
#
#	Parsa Dahesh (dinoco.parsa23@gmail.com or parsa.dahesh@studio.unibo.it)
#	Racing Line Optimization with PSO
#	MIT License, Copyright (c) 2021 Parsa Dahesh
#
#------------------------------------------------------------------------------+

#------------------------------------------------------------------------------+
#
#   Modified by Hardik Lalla (2025)
#   - Adapted for integration with Trackmania-based racing line data
#   - Adjusted JSON input format and spline logic
#   - Added interactive file path input
#   - Implemented Trackmania-realistic physics model
#   - Enhanced visualization with better color mapping
#
#------------------------------------------------------------------------------+

import json
import math
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import os
import sys

import pso

from scipy import interpolate
from shapely.geometry import LineString
from shapely.geometry import MultiLineString

from utils import plot_lines, get_closet_points

def main():
	# PARAMETERS
	N_SECTORS = 30
	N_PARTICLES = 60
	N_ITERATIONS = 150
	W = -0.2256
	CP = -0.1564
	CG = 3.8876
	PLOT = True
	
	# Interactive file path input
	print("=" * 60)
	print("  TRACKMANIA RACING LINE OPTIMIZER")
	print("=" * 60)
	print()
	
	# Get file path from user
	while True:
		file_path = input("Enter the path to your track JSON file: ").strip()
		
		# Remove quotes if user copied path with quotes
		file_path = file_path.strip('"').strip("'")
		
		if not file_path:
			print("Error: File path cannot be empty. Please try again.")
			continue
			
		if not os.path.exists(file_path):
			print(f"Error: File '{file_path}' not found. Please check the path and try again.")
			retry = input("Would you like to try again? (y/n): ").strip().lower()
			if retry != 'y':
				print("Exiting...")
				sys.exit(0)
			continue
			
		if not file_path.endswith('.json'):
			print("Warning: File doesn't have .json extension. Proceeding anyway...")
			
		break
	
	print(f"\nLoading track from: {file_path}")
	
	# Read tracks from json file
	try:
		with open(file_path, 'r') as file:
			json_data = json.load(file)
	except json.JSONDecodeError as e:
		print(f"Error: Invalid JSON file - {e}")
		sys.exit(1)
	except Exception as e:
		print(f"Error reading file: {e}")
		sys.exit(1)

	# Validate JSON structure
	if 'layout' not in json_data or 'width' not in json_data:
		print("Error: JSON file must contain 'layout' and 'width' fields")
		sys.exit(1)

	track_layout = json_data['layout']
	track_width = json_data['width']
	
	print(f"Track loaded successfully!")
	print(f"  - Track points: {len(track_layout)}")
	print(f"  - Track width: {track_width}")
	print()

	# Compute inner and outer tracks borders
	center_line = LineString(track_layout)
	inside_offset = center_line.parallel_offset(track_width / 2, 'left')
	outside_offset = center_line.parallel_offset(track_width / 2, 'right')

	inside_line = max(inside_offset.geoms, key=lambda g: g.length) if isinstance(inside_offset, MultiLineString) else inside_offset
	outside_line = max(outside_offset.geoms, key=lambda g: g.length) if isinstance(outside_offset, MultiLineString) else outside_offset

	if PLOT:
		fig = plt.figure(figsize=(12, 10))
		plt.title("Track Layout Points", fontsize=14, fontweight='bold')
		for p in track_layout:
			plt.plot(p[0], p[1], 'r.', markersize=4)
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()
		
		fig = plt.figure(figsize=(12, 10))
		plt.title("Track Boundaries", fontsize=14, fontweight='bold')
		plot_lines([outside_line, inside_line])
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()
	
	# Define sectors' extreme points (in coordinates)
	inside_points, outside_points = define_sectors(center_line, inside_line, outside_line, N_SECTORS)

	if PLOT:
		fig = plt.figure(figsize=(12, 10))
		plt.title("Optimization Sectors", fontsize=14, fontweight='bold')
		for i in range(N_SECTORS):
			plt.plot([inside_points[i][0], outside_points[i][0]], 
					[inside_points[i][1], outside_points[i][1]], 
					'g-', alpha=0.3, linewidth=1)
		plot_lines([outside_line, inside_line])
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()

	# Define the boundaries for PSO
	boundaries = []
	for i in range(N_SECTORS):
		boundaries.append(np.linalg.norm(inside_points[i]-outside_points[i]))

	def myCostFunc(sectors):
		return get_lap_time(sectors_to_racing_line(sectors, inside_points, outside_points))

	print("Starting PSO optimization...")
	global_solution, gs_eval, gs_history, gs_eval_history = pso.optimize(
		cost_func=myCostFunc,
		n_dimensions=N_SECTORS,
		boundaries=boundaries,
		n_particles=N_PARTICLES,
		n_iterations=N_ITERATIONS,
		w=W, cp=CP, cg=CG,
		verbose=True
	)

	_, v, x, y = get_lap_time(sectors_to_racing_line(global_solution, inside_points, outside_points), return_all=True)

	if PLOT:
		# Animation of optimization progress
		fig = plt.figure(figsize=(14, 10))
		plt.title("Racing Line Optimization Progress", fontsize=14, fontweight='bold')
		plt.ion()
		
		for i in range(0, len(np.array(gs_history)), max(1, int(N_ITERATIONS/100))):
			plt.clf()
			lth, vh, xh, yh = get_lap_time(sectors_to_racing_line(gs_history[i], inside_points, outside_points), return_all=True)
			
			# Enhanced color mapping
			scatter = plt.scatter(xh, yh, marker='.', c=vh, 
								cmap='turbo', s=20, vmin=0, vmax=max(vh) if max(vh) > 0 else 1)
			
			plot_lines([outside_line, inside_line])
			plt.colorbar(scatter, label='Speed (km/h)')
			plt.title(f"Optimization Progress - Iteration {i}/{len(gs_history)-1}\nLap Time: {lth:.3f}s", 
					 fontsize=14, fontweight='bold')
			plt.grid(True, alpha=0.3)
			plt.axis('equal')
			plt.tight_layout()
			plt.draw()
			plt.pause(0.05)
		
		plt.ioff()
		
		# Final racing line visualization
		fig = plt.figure(figsize=(14, 10))
		rl = np.array(sectors_to_racing_line(global_solution, inside_points, outside_points))
		
		# Create enhanced color map
		norm = mcolors.Normalize(vmin=0, vmax=max(v))
		scatter = plt.scatter(x, y, marker='o', c=v, cmap='turbo', 
							 s=30, norm=norm, edgecolors='black', linewidths=0.5)
		
		# Plot sector lines
		for i in range(N_SECTORS):
			plt.plot([inside_points[i][0], outside_points[i][0]], 
					[inside_points[i][1], outside_points[i][1]], 
					'gray', alpha=0.2, linewidth=0.8)
		
		# Plot racing line points
		plt.plot(rl[:,0], rl[:,1], 'ko', markersize=6, 
				markerfacecolor='yellow', markeredgewidth=1.5, 
				label='Sector Points', zorder=5)
		
		plot_lines([outside_line, inside_line])
		
		# Enhanced colorbar
		cbar = plt.colorbar(scatter, label='Speed (km/h)', pad=0.02)
		cbar.ax.tick_params(labelsize=10)
		
		plt.title(f"Optimized Racing Line\nLap Time: {gs_eval:.3f}s | Avg Speed: {np.mean(v):.1f} km/h", 
				 fontsize=14, fontweight='bold')
		plt.legend(loc='best', fontsize=10)
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()

		# Optimization convergence plot
		fig = plt.figure(figsize=(12, 6))
		plt.plot(gs_eval_history, linewidth=2, color='#2E86AB')
		plt.fill_between(range(len(gs_eval_history)), gs_eval_history, 
						alpha=0.3, color='#2E86AB')
		plt.title("Optimization Convergence", fontsize=14, fontweight='bold')
		plt.ylabel("Lap Time (s)", fontsize=12)
		plt.xlabel("Iteration", fontsize=12)
		plt.grid(True, alpha=0.3)
		plt.tight_layout()
		plt.show()
		
		# Speed distribution along track
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
		
		# Speed profile
		distance = np.cumsum([0] + [math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2) 
									for i in range(len(x)-1)])
		ax1.plot(distance, v, linewidth=2, color='#A23B72')
		ax1.fill_between(distance, v, alpha=0.3, color='#A23B72')
		ax1.set_ylabel('Speed (km/h)', fontsize=12)
		ax1.set_xlabel('Distance (m)', fontsize=12)
		ax1.set_title('Speed Profile Along Track', fontsize=12, fontweight='bold')
		ax1.grid(True, alpha=0.3)
		
		# Speed histogram
		ax2.hist(v, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
		ax2.set_xlabel('Speed (km/h)', fontsize=12)
		ax2.set_ylabel('Frequency', fontsize=12)
		ax2.set_title('Speed Distribution', fontsize=12, fontweight='bold')
		ax2.axvline(np.mean(v), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(v):.1f} km/h')
		ax2.legend(fontsize=10)
		ax2.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		plt.show()

	# Print final statistics
	print()
	print("=" * 60)
	print("  OPTIMIZATION RESULTS")
	print("=" * 60)
	print(f"Final Lap Time:     {gs_eval:.3f} seconds")
	print(f"Average Speed:      {np.mean(v):.1f} km/h")
	print(f"Maximum Speed:      {max(v):.1f} km/h")
	print(f"Minimum Speed:      {min(v):.1f} km/h")
	print(f"Speed Variance:     {np.var(v):.1f}")
	print("=" * 60)

def sectors_to_racing_line(sectors:list, inside_points:list, outside_points:list):
	'''Sectors to racing line coordinate
	
	Converts sector values to cartesian coordinates using parametric functions.

	Parameters
	----------
	sectors : list
		Position value of the sector inside the sector segment
	inside_points : list
		List coordinates corresponding to the internal point of each sector segment
	outside_points : list
		List coordinates corresponding to the external point of each sector segment

	Returns
	-------
	racing_line : list
		List of coordinates corresponding to the sectors' position
	'''
	
	racing_line = []
	for i in range(len(sectors)):
		x1, y1 = inside_points[i][0], inside_points[i][1]
		x2, y2 = outside_points[i][0], outside_points[i][1]
		m = (y2-y1)/(x2-x1)

		a = math.cos(math.atan(m))
		b = math.sin(math.atan(m))

		xp = x1 - sectors[i]*a
		yp = y1 - sectors[i]*b

		if abs(math.dist(inside_points[i], [xp,yp])) + abs(math.dist(outside_points[i], [xp,yp])) - \
				abs(math.dist(outside_points[i], inside_points[i])) > 0.1:
			xp = x1 + sectors[i]*a
			yp = y1 + sectors[i]*b

		racing_line.append([xp, yp])
	return racing_line

def get_lap_time(racing_line:list, return_all=False):
	'''Fitness function with Trackmania-realistic physics
	
	Computes lap time using physics model inspired by Trackmania:
	- Realistic grip model with speed-dependent friction
	- Better acceleration/deceleration model
	- More accurate corner speed calculations

	Parameters
	----------
	racing_line : array
		Racing line in sector points
	return_all : boolean
		Flag to return optional values (default is False)

	Returns
	-------
	lap_time : float
		Lap time in seconds
	v : list[float], optional
		Speed value (km/h) for each point
	x : int, optional
		x coordinate for each point
	y : int, optional
		y coordinate for each point
	'''
	
	rl = np.array(racing_line)

	# Close the loop if needed
	if np.linalg.norm(rl[0] - rl[-1]) > 1e-3:
		rl = np.vstack([rl, rl[0]])

	# Remove duplicates
	_, unique_idx = np.unique(rl, axis=0, return_index=True)
	rl = rl[np.sort(unique_idx)]

	if len(rl) < 4:
		raise ValueError("Too few points to compute a racing line spline.")

	# Create smooth spline
	tck, _ = interpolate.splprep([rl[:, 0], rl[:, 1]], s=0.0, per=0)
	x, y = interpolate.splev(np.linspace(0, 1, 1000), tck)

	# Computing derivatives for curvature
	dx, dy = np.gradient(x), np.gradient(y)
	d2x, d2y = np.gradient(dx), np.gradient(dy)

	curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
	
	# Avoid division by zero
	curvature = np.maximum(curvature, 1e-6)
	radius = 1.0 / curvature

	# TRACKMANIA-INSPIRED PHYSICS MODEL
	# ================================
	
	# Car parameters (inspired by Trackmania Stadium car)
	MAX_SPEED = 500.0          # km/h - Trackmania cars can go very fast
	MAX_ACCELERATION = 15.0    # m/s² - Strong acceleration
	MAX_BRAKING = 25.0         # m/s² - Strong braking
	GRIP_COEFFICIENT = 1.8     # Higher than normal cars (arcade physics)
	DRAG_COEFFICIENT = 0.003   # Air resistance
	
	# Convert to m/s for calculations
	max_speed_ms = MAX_SPEED / 3.6
	
	# Calculate corner speeds based on grip and radius
	g = 9.81
	v_corner = []
	
	for r in radius:
		# Speed-dependent grip model (Trackmania has better grip at higher speeds)
		effective_grip = GRIP_COEFFICIENT * (1 + 0.0002 * r)
		
		# Maximum cornering speed based on lateral grip
		v_max_corner = math.sqrt(effective_grip * g * r)
		
		# Apply drag at high speeds
		v_max_corner = min(v_max_corner, max_speed_ms)
		
		# Convert to km/h
		v_corner.append(v_max_corner * 3.6)
	
	# Forward pass - acceleration limited
	v = [v_corner[0]]
	for i in range(1, len(x)):
		ds = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
		
		# Maximum speed we could reach by accelerating
		v_accel = math.sqrt(v[i-1]**2 + 2 * (MAX_ACCELERATION / 3.6**2) * ds * 3.6**2)
		
		# Take minimum of acceleration-limited and corner-limited speed
		v.append(min(v_accel, v_corner[i], MAX_SPEED))
	
	# Backward pass - braking limited
	for i in range(len(x) - 2, -1, -1):
		ds = math.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
		
		# Maximum speed we could have while still being able to brake for next point
		v_brake = math.sqrt(v[i+1]**2 + 2 * (MAX_BRAKING / 3.6**2) * ds * 3.6**2)
		
		# Take minimum of current speed and braking-limited speed
		v[i] = min(v[i], v_brake)
	
	# Apply drag losses
	for i in range(len(v)):
		drag_factor = 1.0 - DRAG_COEFFICIENT * v[i]
		v[i] = max(v[i] * drag_factor, 10.0)  # Minimum speed 10 km/h
	
	# Calculate lap time
	lap_time = 0
	for i in range(len(x) - 1):
		ds = math.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
		avg_speed = (v[i] + v[i+1]) / 2.0
		
		if avg_speed > 0:
			lap_time += ds / (avg_speed / 3.6)  # Convert km/h to m/s
	
	if return_all:
		return lap_time, v, x, y
	return lap_time

def define_sectors(center_line : LineString, inside_line : LineString, outside_line : LineString, n_sectors : int):
	'''Defines sectors' search space
	
	Parameters
	----------
	center_line : LineString
		Center line of the track
	inside_line : LineString
		Inside line of the track
	outside_line : LineString
		Outside line of the track
	n_sectors : int
		Number of sectors

	Returns
	-------
	inside_points : list
		List coordinates corresponding to the internal point of each sector segment
	outside_points : list
		List coordinates corresponding to the external point of each sector segment
	'''
	
	distances = np.linspace(0, center_line.length, n_sectors)
	center_points_temp = [center_line.interpolate(distance) for distance in distances]
	center_points = np.array([[center_points_temp[i].x, center_points_temp[i].y] for i in range(len(center_points_temp)-1)])
	center_points = np.append(center_points, [center_points[0]], axis=0)

	distances = np.linspace(0, inside_line.length, 1000)
	inside_border = [inside_line.interpolate(distance) for distance in distances]
	inside_border = np.array([[e.x, e.y] for e in inside_border])
	inside_points = np.array([get_closet_points([center_points[i][0], center_points[i][1]], inside_border) for i in range(len(center_points))]) 

	distances = np.linspace(0, outside_line.length, 1000)
	outside_border = [outside_line.interpolate(distance) for distance in distances]
	outside_border = np.array([[e.x, e.y] for e in outside_border])
	outside_points = np.array([get_closet_points([center_points[i][0], center_points[i][1]], outside_border) for i in range(len(center_points))])

	return inside_points, outside_points

if __name__ == "__main__":	
	main()
