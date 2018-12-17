# TetrisDQN
A Tetris environment with a reinforcement learning agent to play it.

*This is not currently finished, but the goal is to get something that can at least play for a little while.*

The goal is to develop an agent that can learn to play Tetris with a reinforcement learning model. Designing reward functions appropriate to Tetris are difficult because good Tetris strategies rely on carefully building up large sections of blocks with the intention of clearing as many as possible in one go for an increased score.

Initially I took the standard approach of using just the rewards from the game, but random exploration is not completing any lines.
The current approach at the time of writing uses a combination of the height of the populated area, the density and some other calculations as a reward. It also waits until the block has landed before evaluating the success of the placement, and populating the experience with a weaker or stronger reward depending on how close the block is to being placed in that particular space.

I intent to make further adjustments so that the bot can be pre-trained by observing player behaviour for some amount of time before starting it's random exploration.

<p align="center">
  <img src="https://raw.githubusercontent.com/bretcolloff/tetrisdqn/master/screenshot.png" alt="screenshot" />
</p>
