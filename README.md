# tetrisdqn
A self-learning Tetris agent.

08/12/18 - 5PM
Start the project. We know we need a couple of things. The agent that learns, a gym to play in, and a way to test the gym as a human.
We create the basic classes.

08/12/18 - 6PM
We have some standard bits done, we have a few different pieces and we've defined the downward motion and the collision detection.
Upon collision, the piece becomes part of the scene, and then generates a new one, which starts falling.

We chose 3 block types, no block, solid block, and a block of the piece that's moving. We're ignoring the colour so we have 3 types for the neural network.
We can expand this out again later.

08/12/18 - 7PM
Added a way to test it by hand from the console, we've got left and right working, need to do rotation and scoring. Haven't made this particularly easy to do the rotations.

08/12/18 - 8PM
Rewritten the piece management, rotations work. Need to set the start position for the corner of the piece's space, and add the block removal and failure cases.

08/12/18 - 9PM
Tetris gym seems to operate as expected.