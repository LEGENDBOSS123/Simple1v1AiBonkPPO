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
import { actionToArray, arrayToAction, logProbabilities, mean, nonTFgetEntropy, predictActionArray, randomAction } from "./utils.mjs";

let models = [];
let currentModel = null;
let memory = new ReplayBuffer();
top.lastSave = "";
let losses = [];

top.models = function () { return models; };
top.currentModel = function () { return currentModel; };
top.memory = function () { return memory; };
top.losses = function () { return losses; };

function expectedScore(eloA, eloB) {
    return 1 / (1 + Math.pow(10, (eloB - eloA) / 400));
}

function updateElo(modelA, modelB, scoreA) {
    const scoreB = 1 - scoreA;
    const expectedA = expectedScore(modelA.elo, modelB.elo);
    const expectedB = 1 - expectedA;
    const k = CONFIG.ELO_K;
    modelA.elo = modelA.elo + k * (scoreA - expectedA);
    // do not update opponent elo to keep it frozen
    // modelB.elo = modelB.elo + k * (scoreB - expectedB);
}

function chooseOpponentByElo(modelsPool, targetElo) {
    if (modelsPool.length === 0) return null;
    const weights = modelsPool.map(m => {
        const diff = Math.abs((m.elo ?? 1000) - targetElo);
        return Math.exp(-diff / CONFIG.ELO_DIFF_SCALE);
    });
    const total = weights.reduce((sum, w) => sum + w, 0);
    let r = Math.random() * total;
    for (let i = 0; i < modelsPool.length; i++) {
        r -= weights[i];
        if (r <= 0) return modelsPool[i];
    }
    return modelsPool[modelsPool.length - 1];
}

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
            CONFIG.ENTROPY_COEFFICIENT = Math.max(
                CONFIG.MIN_ENTROPY,
                CONFIG.ENTROPY_COEFFICIENT * CONFIG.ENTROPY_DECAY
            );

            if (memory.episodes() >= CONFIG.ROLLOUT_EPISODES) {
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
                if (models.length > CONFIG.MAX_SAVED_MODELS) {
                    models.shift();
                }
                log(`Model checkpoint saved. Total checkpoints: ${models.length}`);
                top.lastSave = await top.saveModelsString();
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
                if (CONFIG.episodes % CONFIG.SWITCH_OPPONENT_EVERY_EPISODES === 0) {
                    const rand = Math.random();
                    if (rand < 0.7) {
                        p2Model = chooseOpponentByElo(models, currentModel.elo);
                    }
                    else if (rand < 0.85) {
                        p2Model = Random.choose(modelsPool);
                    }
                    else {
                        p2Model = currentModel;
                    }
                }
            }

            let lastActionP1 = null;
            let lastLogProbP1 = null;
            let lastActionP2 = null;
            let entropies = [];
            while (true) {
                CONFIG.steps++;
                newState = new State();
                newState.fetch();

                let rewardCurrentFrame = newState.reward();
                let rewardP1 = rewardCurrentFrame.p1;


                if (safeFrames > CONFIG.MAX_SECONDS * 20) {
                    newState.done = true;
                    rewardP1 -= CONFIG.DRAW_PENALTY;
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

                // DO NOT STORE P2 EXPERIENCES SINCE IT IS FROM A OLDER POLICY
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
                entropies.push(nonTFgetEntropy(logits));
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

            if (newState?.done && p2Model && p2Model !== currentModel) {
                const scoreP1 = newState.winnerId === CONFIG.PLAYER_ONE_ID
                    ? 1
                    : newState.winnerId === CONFIG.PLAYER_TWO_ID
                        ? 0
                        : 0.5;
                updateElo(currentModel, p2Model, scoreP1);
                CONFIG.elos.push(currentModel.elo);
                CONFIG.entropies.push(mean(entropies));
            }
        }
    }


    gameLoop();
}

top.main = main;
setup().then(() => { main() });