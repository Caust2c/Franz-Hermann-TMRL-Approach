# Franz-Hermann-TMRL-Approach
A more guided approach at Trackmania AI and a real life application using Particle Swarm Optimization

This project was initally intended to be done with a real-life RC Car churning out the best lap times, however due to a lack of LIDAR sensor i had to shift to a game based implementation.

Instead of relying on brute-force reinforcement learning, this project takes a smarter approach by encouraging the agent to follow a precomputed racing line. By rewarding it for staying close to the ideal path, the agent can learn faster and more efficiently, potentially cutting down training time a lot while still driving competitively.

The project includes:
- 📍 Path extraction and geometry analysis (adapted from Trackmania map data)
- 🧠 Particle Swarm Optimization (PSO) for finding the optimal racing trajectory
- 🎮 Simulation in Trackmania for visual and analytical feedback

I have essentially combined two repositories for this task with some changes and additions.

- [TrackMania_AI](https://github.com/AndrejGobeX/TrackMania_AI) (GPLv3)
- [Racing-Line-Optimization-with-PSO](https://github.com/ParsaD23/Racing-Line-Optimization-with-PSO) (MIT)

As such, this project is licensed under the terms of the [GNU General Public License v3.0](LICENSE). Also, please go through their repos the work they've done is amazing

As for the name, “Franz Hermann” was an alias used by Max Verstappen during a GT3 test at the Nürburgring Nordschleife, part of the official Nürburgring Endurance Series (he did it to avoid media attention). Hence, the name was given to the repo as admiration as the admired as one of the fastest, arguably greatest racing driver in F1 history.

## General Working
So its as simple as, put in a gbx file in the mapextractor.

We get a text file for output

Put it in the txtjsonconverter.py for the .json

and run that in the modified main.py for the pso repo.

## Requirements

* ```matplotlib```
* ```numpy```
* ```scipy```
* ```shapely```
* ```You need dotnet installed for the mapextractor, with GBX.NET packages```
* ```You might have to add GBX.NET.LZO package seperately in case you have an error specifiyng LZO```

## ⚠️ Limitations

- **Designed for closed loops**: The original racing line optimization was built for tracks where the start and end points are the same. Since many Trackmania maps aren't like that, some bugs or issues may occur when generating racing lines.

- **Physics are simplified**: It's not possible to fully replicate Trackmania’s driving physics in this system. However, I've done my best to approximate the behavior using adjustable parameters like friction and speed.

- **Works best with clean 2D maps**: This approach assumes simple 2D maps with clearly defined track boundaries and no extra environment clutter. Complex or messy layouts may cause bugs, plotting errors, or broken centerline paths in the JSON.

- **Limited to 2D representation**: Track elements like elevation, jumps, or banked turns are not supported, as the racing line logic is based on flat 2D coordinates.

