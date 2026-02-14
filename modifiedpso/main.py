#------------------------------------------------------------------------------+
#
#	Parsa Dahesh (dinoco.parsa23@gmail.com or parsa.dahesh@studio.unibo.it)
#	Racing Line Optimization with PSO
#	MIT License, Copyright (c) 2021 Parsa Dahesh
#
#------------------------------------------------------------------------------+

#------------------------------------------------------------------------------+
#
#   Modified by Hardik Lalla (2026)
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

def remove_duplicate_points(points, tolerance=1e-6):
	'''Remove consecutive duplicate points from track layout
	
	Parameters
	----------
	points : list
		List of [x, y] coordinates
	tolerance : float
		Distance threshold for considering points as duplicates
		
	Returns
	-------
	cleaned_points : list
		Points with consecutive duplicates removed
	'''
	if len(points) < 2:
		return points
	
	cleaned = [points[0]]
	for i in range(1, len(points)):
		dist = math.sqrt((points[i][0] - cleaned[-1][0])**2 + 
						(points[i][1] - cleaned[-1][1])**2)
		if dist > tolerance:
			cleaned.append(points[i])
	
	print(f"  - Cleaned {len(points) - len(cleaned)} duplicate points")
	return cleaned

def main():
	N_SECTORS = 30
	N_PARTICLES = 60
	N_ITERATIONS = 150
	W = -0.2256
	CP = -0.1564
	CG = 3.8876
	PLOT = True
	
	print("=" * 60)
	print("  TRACKMANIA RACING LINE OPTIMIZER")
	print("=" * 60)
	print()
	
	while True:
		file_path = input("Enter the path to your track JSON file: ").strip()
		
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
	
	try:
		with open(file_path, 'r') as file:
			json_data = json.load(file)
	except json.JSONDecodeError as e:
		print(f"Error: Invalid JSON file - {e}")
		sys.exit(1)
	except Exception as e:
		print(f"Error reading file: {e}")
		sys.exit(1)

	if 'layout' not in json_data or 'width' not in json_data:
		print("Error: JSON file must contain 'layout' and 'width' fields")
		sys.exit(1)

	track_layout = json_data['layout']
	track_width = json_data['width']
	
	print(f"Track loaded - initial points: {len(track_layout)}")
	
	track_layout = remove_duplicate_points(track_layout, tolerance=0.01)
	
	if len(track_layout) < 4:
		print(f"Error: Track has too few unique points ({len(track_layout)}). Need at least 4 points.")
		sys.exit(1)
	
	start_point = np.array(track_layout[0])
	end_point = np.array(track_layout[-1])
	gap_distance = np.linalg.norm(start_point - end_point)
	
	is_open_track = gap_distance > 5.0  
	
	print(f"Track processed successfully!")
	print(f"  - Unique track points: {len(track_layout)}")
	print(f"  - Track width: {track_width}")
	print(f"  - Track type: {'OPEN (Sprint/Point-to-Point)' if is_open_track else 'CLOSED (Circuit/Loop)'}")
	print(f"  - Start-to-finish gap: {gap_distance:.2f} meters")
	print()

	center_line = LineString(track_layout)
	inside_offset = center_line.parallel_offset(track_width / 2, 'left')
	outside_offset = center_line.parallel_offset(track_width / 2, 'right')

	inside_line = max(inside_offset.geoms, key=lambda g: g.length) if isinstance(inside_offset, MultiLineString) else inside_offset
	outside_line = max(outside_offset.geoms, key=lambda g: g.length) if isinstance(outside_offset, MultiLineString) else outside_offset

	if PLOT:
		fig = plt.figure(figsize=(12, 10))
		plt.title("Track Layout Points", fontsize=14, fontweight='bold')
		for i, p in enumerate(track_layout):
			if i == 0:
				plt.plot(p[0], p[1], 'go', markersize=12, label='START', zorder=5)
			elif i == len(track_layout) - 1:
				plt.plot(p[0], p[1], 'rs', markersize=12, label='FINISH', zorder=5)
			else:
				plt.plot(p[0], p[1], 'r.', markersize=4)
		
		if is_open_track:
			plt.plot([track_layout[0][0], track_layout[-1][0]], 
					[track_layout[0][1], track_layout[-1][1]], 
					'b--', linewidth=2, alpha=0.5, label=f'Gap: {gap_distance:.1f}m')
		
		plt.legend(loc='best', fontsize=10)
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()
		
		fig = plt.figure(figsize=(12, 10))
		plt.title("Track Boundaries", fontsize=14, fontweight='bold')
		plot_lines([outside_line, inside_line])
		
		plt.plot(track_layout[0][0], track_layout[0][1], 'go', markersize=15, 
				label='START', zorder=5, markeredgecolor='black', markeredgewidth=2)
		plt.plot(track_layout[-1][0], track_layout[-1][1], 'rs', markersize=15, 
				label='FINISH', zorder=5, markeredgecolor='black', markeredgewidth=2)
		
		plt.legend(loc='best', fontsize=10)
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()
	
	inside_points, outside_points = define_sectors(center_line, inside_line, outside_line, N_SECTORS)

	boundaries = []
	valid_sectors = []
	
	for i in range(N_SECTORS):
		boundary_width = np.linalg.norm(inside_points[i] - outside_points[i])
		
		if boundary_width > 1.0:
			boundaries.append(boundary_width)
			valid_sectors.append(i)
		else:
			print(f"Warning: Skipping sector {i} - boundary too narrow ({boundary_width:.2f}m)")
	
	if len(valid_sectors) < 4:
		print(f"Error: Too few valid sectors ({len(valid_sectors)}). Need at least 4.")
		print("This usually means the track has too many duplicate points or sharp angles.")
		sys.exit(1)
	
	print(f"Using {len(valid_sectors)}/{N_SECTORS} sectors for optimization")
	
	inside_points_filtered = inside_points[valid_sectors]
	outside_points_filtered = outside_points[valid_sectors]

	if PLOT:
		fig = plt.figure(figsize=(12, 10))
		plt.title("Optimization Sectors", fontsize=14, fontweight='bold')
		for idx in valid_sectors:
			plt.plot([inside_points[idx][0], outside_points[idx][0]], 
					[inside_points[idx][1], outside_points[idx][1]], 
					'g-', alpha=0.5, linewidth=1.5, label='Valid Sector' if idx == valid_sectors[0] else '')
		
		invalid_sectors = [i for i in range(N_SECTORS) if i not in valid_sectors]
		if invalid_sectors:
			for idx in invalid_sectors:
				plt.plot([inside_points[idx][0], outside_points[idx][0]], 
						[inside_points[idx][1], outside_points[idx][1]], 
						'r-', alpha=0.3, linewidth=0.8, label='Invalid Sector' if idx == invalid_sectors[0] else '')
		
		plot_lines([outside_line, inside_line])
		plt.legend(loc='best', fontsize=10)
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()

	def myCostFunc(sectors):
		return get_lap_time(sectors_to_racing_line(sectors, inside_points_filtered, outside_points_filtered))

	print("Starting PSO optimization...")
	global_solution, gs_eval, gs_history, gs_eval_history = pso.optimize(
		cost_func=myCostFunc,
		n_dimensions=len(valid_sectors),
		boundaries=boundaries,
		n_particles=N_PARTICLES,
		n_iterations=N_ITERATIONS,
		w=W, cp=CP, cg=CG,
		verbose=True
	)

	_, v, x, y = get_lap_time(sectors_to_racing_line(global_solution, inside_points_filtered, outside_points_filtered), return_all=True)

	if PLOT:
		fig = plt.figure(figsize=(14, 10))
		plt.title("Racing Line Optimization Progress", fontsize=14, fontweight='bold')
		plt.ion()
		
		for i in range(0, len(np.array(gs_history)), max(1, int(N_ITERATIONS/100))):
			plt.clf()
			lth, vh, xh, yh = get_lap_time(sectors_to_racing_line(gs_history[i], inside_points_filtered, outside_points_filtered), return_all=True)
			
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
		
		fig = plt.figure(figsize=(14, 10))
		rl = np.array(sectors_to_racing_line(global_solution, inside_points_filtered, outside_points_filtered))
		
		norm = mcolors.Normalize(vmin=0, vmax=max(v))
		scatter = plt.scatter(x, y, marker='o', c=v, cmap='turbo', 
							 s=30, norm=norm, edgecolors='black', linewidths=0.5)
		
		for idx in valid_sectors:
			plt.plot([inside_points[idx][0], outside_points[idx][0]], 
					[inside_points[idx][1], outside_points[idx][1]], 
					'gray', alpha=0.2, linewidth=0.8)
		
		plt.plot(rl[:,0], rl[:,1], 'ko', markersize=6, 
				markerfacecolor='yellow', markeredgewidth=1.5, 
				label='Sector Points', zorder=5)
		
		if is_open_track:
			plt.plot(x[0], y[0], 'go', markersize=20, 
					label='START', zorder=10, markeredgecolor='black', markeredgewidth=2)
			plt.plot(x[-1], y[-1], 'rs', markersize=20, 
					label='FINISH', zorder=10, markeredgecolor='black', markeredgewidth=2)
		
		plot_lines([outside_line, inside_line])
		
		cbar = plt.colorbar(scatter, label='Speed (km/h)', pad=0.02)
		cbar.ax.tick_params(labelsize=10)
		
		track_type_str = "Sprint Track" if is_open_track else "Circuit"
		plt.title(f"Optimized Racing Line ({track_type_str})\nLap Time: {gs_eval:.3f}s | Avg Speed: {np.mean(v):.1f} km/h | Valid Sectors: {len(valid_sectors)}/{N_SECTORS}", 
				 fontsize=14, fontweight='bold')
		plt.legend(loc='best', fontsize=10)
		plt.grid(True, alpha=0.3)
		plt.axis('equal')
		plt.tight_layout()
		plt.show()

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
		
		fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
		
		distance = np.cumsum([0] + [math.sqrt((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2) 
									for i in range(len(x)-1)])
		ax1.plot(distance, v, linewidth=2, color='#A23B72')
		ax1.fill_between(distance, v, alpha=0.3, color='#A23B72')
		ax1.set_ylabel('Speed (km/h)', fontsize=12)
		ax1.set_xlabel('Distance (m)', fontsize=12)
		ax1.set_title('Speed Profile Along Track', fontsize=12, fontweight='bold')
		ax1.grid(True, alpha=0.3)
		
		ax2.hist(v, bins=30, color='#F18F01', alpha=0.7, edgecolor='black')
		ax2.set_xlabel('Speed (km/h)', fontsize=12)
		ax2.set_ylabel('Frequency', fontsize=12)
		ax2.set_title('Speed Distribution', fontsize=12, fontweight='bold')
		ax2.axvline(np.mean(v), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(v):.1f} km/h')
		ax2.legend(fontsize=10)
		ax2.grid(True, alpha=0.3, axis='y')
		
		plt.tight_layout()
		plt.show()

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

	distance_start_end = np.linalg.norm(rl[0] - rl[-1])
	is_closed_loop = distance_start_end < 5.0
	
	if is_closed_loop and distance_start_end > 1e-3:
		rl = np.vstack([rl, rl[0]])
		periodic_spline = 1
	else:
		periodic_spline = 0 

	_, unique_idx = np.unique(rl, axis=0, return_index=True)
	rl = rl[np.sort(unique_idx)]

	if len(rl) < 4:
		raise ValueError("Too few points to compute a racing line spline.")

	tck, _ = interpolate.splprep([rl[:, 0], rl[:, 1]], s=0.0, per=periodic_spline)
	x, y = interpolate.splev(np.linspace(0, 1, 1000), tck)

	dx, dy = np.gradient(x), np.gradient(y)
	d2x, d2y = np.gradient(dx), np.gradient(dy)

	curvature = np.abs(dx * d2y - d2x * dy) / (dx * dx + dy * dy)**1.5
	
	curvature = np.maximum(curvature, 1e-6)
	radius = 1.0 / curvature

	# TRACKMANIA-INSPIRED PHYSICS MODEL

	MAX_SPEED = 1000.0
	MAX_ACCELERATION = 22.0
	MAX_BRAKING = 40.0
	GRIP_COEFFICIENT = 2.8     
	DRAG_COEFFICIENT = 0.0006 
	
	max_speed_ms = MAX_SPEED / 3.6
	
	g = 9.81
	v_corner = []
	
	for r in radius:
		effective_grip = GRIP_COEFFICIENT * (1 + 0.0005 * r)
		
		v_max_corner = math.sqrt(max(effective_grip * g * r, 0.0))
		
		v_max_corner = min(v_max_corner, max_speed_ms)
		
		v_corner.append(v_max_corner)
	
	v = [v_corner[0]]
	for i in range(1, len(x)):
		ds = math.sqrt((x[i] - x[i-1])**2 + (y[i] - y[i-1])**2)
		
		a_drag = DRAG_COEFFICIENT * v[i-1]**2
		
		a_net = max(MAX_ACCELERATION - a_drag, 0.0)
		
		v_accel = math.sqrt(max(v[i-1]**2 + 2 * a_net * ds, 0.0))
		
		v.append(min(v_accel, v_corner[i], max_speed_ms))
	
	for i in range(len(x) - 2, -1, -1):
		ds = math.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
		
		v_brake = math.sqrt(max(v[i+1]**2 + 2 * MAX_BRAKING * ds, 0.0))
		
		v[i] = min(v[i], v_brake)
	
	for i in range(len(v)):
		v[i] = max(v[i] * 3.6, 10.0)
	
	lap_time = 0
	for i in range(len(x) - 1):
		ds = math.sqrt((x[i+1] - x[i])**2 + (y[i+1] - y[i])**2)
		
		avg_speed = ((v[i] / 3.6) + (v[i+1] / 3.6)) / 2.0
		
		if avg_speed > 0:
			lap_time += ds / avg_speed
	
	if return_all:
		return lap_time, v, x, y
	return lap_time


def define_sectors(center_line : LineString, inside_line : LineString, outside_line : LineString, n_sectors : int):
	'''Defines sectors' search space for open or closed tracks
	
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
	
	distances = np.linspace(0, center_line.length, n_sectors + 1)[:-1]  
	center_points_temp = [center_line.interpolate(distance) for distance in distances]
	center_points = np.array([[center_points_temp[i].x, center_points_temp[i].y] for i in range(len(center_points_temp))])
	
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