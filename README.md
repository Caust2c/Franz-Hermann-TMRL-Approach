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

As such, this project is licensed under the terms of the [GNU General Public License v3.0](LICENSE). Also, please go through their repos, the work they've done is amazing.

As for the name, “Franz Hermann” was an alias used by Max Verstappen during a GT3 test at the Nürburgring Nordschleife, part of the official Nürburgring Endurance Series (he did it to avoid media attention). Hence, the name was given to the repo as admiration as the admired as one of the fastest, arguably greatest racing driver in F1 history.

## General Working
So its as simple as, put in a gbx file in the mapextractor.

We get a text file for output

Put it in the converter.py for the .json

and run that in the modified main.py for the pso repo.

## Requirements

* ```matplotlib```
* ```numpy```
* ```scipy```
* ```shapely```
* ```You need dotnet installed for the mapextractor, with GBX.NET packages```
* ```You might have to add GBX.NET.LZO package seperately in case you have an error specifiyng LZO```

## Sample Showcase

This is a showcase of the nascar.gbx map in the samplemaps folder

The map layout (.json)

![nascarlayout](https://github.com/user-attachments/assets/be7183b1-7c60-4804-9b06-69501bacee01)

The map layout with sectors (n_sectors is set to 30)

![nascarsectors](https://github.com/user-attachments/assets/015e7a7d-2a55-4dca-bc2b-e9b8f98a662f)

The final racing line plot

![nascarracingline](https://github.com/user-attachments/assets/8b5ea474-a82f-45d3-bb50-4f3cfe974862)


## ⚠️ Limitations

- **Designed for closed loops**: The original racing line optimization was built for tracks where the start and end points are the same. Since many Trackmania maps aren't like that, some bugs or issues may occur when generating racing lines.

- **Physics are simplified**: It's not possible to fully replicate Trackmania’s driving physics in this system. However, I've done my best to approximate the behavior using adjustable parameters like friction and speed.

- **Works best with clean 2D maps**: This approach assumes simple 2D maps with clearly defined track boundaries and no extra environment clutter. Complex or messy layouts may cause bugs, plotting errors, or broken centerline paths in the JSON.

- **Limited to 2D representation**: Track elements like elevation, jumps, or banked turns are not supported, as the racing line logic is based on flat 2D coordinates.

- Even in the sample showcase above, some issues are noticeable like the gap between the start and end points, and a few spots where the racing line could clearly be improved. That said, this approach still feels more efficient than letting the agent figure everything out from scratch and constantly tweaking rewards or weights for every new track.


