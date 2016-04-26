# Voxel Terrain Generator
Built as my final project for ECEN 4003 - Concurrent Programming.

Generates a psuedo infinite terrain populated with stone, grass, water, and trees. Camera can either hover slightly above the terrain or can show a general overview. Terrain is limited to 128 blocks in Z, but limited to 2^35 blocks in both X and Y.

Contains the ability to generate, populate, and mesh in serial or parallel.

## Arguments

`-s=<integer>` Sets the seed for the world generator.
`-tc=<integer>` Sets the number of background worker threads used.
`-tm` Enables testing mode.

Testing mode disables user input, sets the camera to a fixed high speed, enables overhead view, and runs the same terrain twice for 4096 blocks, once completely serial and once completely parallel.

## Controls

`1,2` Sets Generation to serial/parallel respectively
`3,4` Sets Population to serial/parallel respectively
`5,6` Sets Meshing to serial/parallel respectively
`P` Toggles the camera between terrain view and overhead view
`Space` Toggles camera movement
`;` Sets camera velocity to 0
`=,-` Adds to camera Y velocity, Forward/Backward respectively
`[,]` Adds to camera X velocity, Left/Right respectively (Will make the camera slightly jittery in Z movement)
`Escape` Exits the program

## License
Licensed under the MIT license.
Don't copy this and market it as your own class project, that's a really bad idea. Feel free to read the code and learn from it though.
See the [LICENSE](LICENSE.md) file for details.
