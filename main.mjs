import { CONFIG, updateConfig } from "./config.mjs";
import { log } from "./log.mjs";
import { move } from "./move.mjs";
import { Random } from "./Random.mjs";
import { ReplayBuffer } from "./ReplayBuffer.mjs";
import { setupLobby } from "./setupLobby.mjs";
import { cloneModel, deserializeModels, loadBrowserFile, setupModel } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";
import { train } from "./train.mjs";
import { actionToArray, arrayToAction, logProbabilities, predictActionArray, randomAction } from "./utils.mjs";

let models = [];
let currentModel = null;
let memory = new ReplayBuffer();

let losses = [];

top.models = function () { return models; };
top.currentModel = function () { return currentModel; };
top.memory = function () { return memory; };
top.losses = function () { return losses; };
top.paused = false;

async function setup() {
    const filePrompt = prompt("Do you want to load a model from file? (y/n)", "n") == "y";
    const filedata = filePrompt ? await loadBrowserFile() : null;
    if (filedata) {
        updateConfig(filedata.CONFIG);
        const deserialized = await deserializeModels(filedata.models);
        models = deserialized.models;
        currentModel = deserialized.currentModel;
        log(`Loaded ${models.length} models from file.`);
    }
    else {
        currentModel = setupModel();
    }
    log("TensorFlow.js version:", tf.version.tfjs);
    log(`Actor Model initialized with ${currentModel.actor.countParams()} parameters.`);
    log(`Critic Model initialized with ${currentModel.critic.countParams()} parameters.`);
}

async function main() {

    await setupLobby();

    log("Lobby setup complete.");


    async function gameLoop() {

        let p2Model = currentModel;

        while (true) {
            CONFIG.episodes++;
            if (memory.size() >= CONFIG.ROLLOUT_STEPS) {
                log("Training...");

                const batch = memory.getAll();

                let result = null;
                result = await train(currentModel, batch, CONFIG.EPOCHS, true);
                if (result) {
                    losses.push(result.lossActor);
                    log(`Loss: ${result.lossActor.toFixed(4)}`);
                    log(`Critic Loss: ${result.lossCritic.toFixed(4)}`);
                }
                
                memory.clear();
            }


            if (CONFIG.episodes % CONFIG.SAVE_AFTER_EPISODES === 0 && CONFIG.episodes > 0) {
                models.push(cloneModel(currentModel));
                if (models.length > 20) {
                    models.shift();
                }
                log(`Model checkpoint saved. Total checkpoints: ${models.length}`);
            }
            lastEpisodeLength = 0;

            // match start
            top.startGame();
            await Time.sleep(1500);

            // 20 TPS
            let TPS = 1000 / 20;

            let lastState = new State();
            lastState.fetch();
            await Time.sleep(TPS);

            let newState;

            let safeFrames = 0;


            if (models.length > 0) {
                if (CONFIG.episodes % 200 === 0) {
                    p2Model = Random.choose(models);
                }
            }

            let lastActionP1 = null;
            let lastLogProbP1 = null;
            let lastActionP2 = null;
            while (true) {
                CONFIG.steps++;
                newState = new State();
                newState.fetch();

                let rewardCurrentFrame = newState.reward();
                let rewardP1 = rewardCurrentFrame.p1;


                if (safeFrames > 500) {
                    newState.done = true;
                }

                if (lastActionP1 !== null) {
                    memory.add(lastState.toArray(),
                        lastActionP1,
                        lastLogProbP1,
                        rewardP1,
                        newState.toArray(),
                        newState.done
                    );
                }

                // DO NOT STORE P2 EXPERIENCES
                // if (lastActionP2 !== null) {
                //     memory.add(lastState.flip().toArray(),
                //         lastActionP2,
                //         rewardP2,
                //         newState.flip().toArray(),
                //         newState.done
                //     );
                // }

                if (newState.done) {
                    break;
                }

                let logits = predictActionArray(currentModel, newState.toArray());
                top.logits = logits;
                let actionP1 = arrayToAction(logits);
                move(CONFIG.PLAYER_ONE_ID, actionP1);
                lastActionP1 = actionToArray(actionP1);
                lastLogProbP1 = logProbabilities(logits, lastActionP1);


                let logits2 = predictActionArray(p2Model, newState.flip().toArray());
                let actionP2 = arrayToAction(logits2);
                if (CONFIG.episodes < CONFIG.WARM_UP_EPISODES) {
                    actionP2 = randomAction(0);
                    logits2 = actionToArray(actionP2);
                }
                move(CONFIG.PLAYER_TWO_ID, actionP2);
                lastActionP2 = actionToArray(actionP2);

                safeFrames++;
                lastState = newState;
                lastEpisodeLength++;

                // 20 FPS
                await Time.sleep(50);
            }

            if (top.paused) {
                log("Paused. Press OK to continue.");
                top.paused = false;
                return;
            }
        }
    }


    gameLoop();
}

function pause() {
    top.paused = true;
}

top.main = main;
setup().then(() => { main() });