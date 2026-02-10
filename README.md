### Simple 1v1 Bonk.io AI 
This is a project to train AI to learn how to play simple 1v1 with itself.
Could use same code to train AI on other maps.
Uses the PPO algorithm.

### RUN
To install dependencies, run `npm install`
To build `script.js` run `npm run build`
Open bonk.io and create a lobby with teams LOCKED.
Then copy paste `script.js` into bonk.io via inspect element / console
To save models at any point, in console you can type `top.saveModels()`

### TEST
To build `play.js` run `npm run build`
Copy the code from play.js into bonk.io inspect element / console
The AI will automatically play as you.
It fill fight against a random player. Please use for 1v1s.
Will only work on the map it was trained for (currently simple 1v1).

### Requirements
`Node.js`
My `Bonk Commands` Mod is required to get player positions and velocities.

[Node.js](https://nodejs.org/)
[Bonk Commands](https://greasyfork.org/en/scripts/451341-bonk-commands)


https://github.com/LEGENDBOSS123/Simple1v1AiBonkPPO