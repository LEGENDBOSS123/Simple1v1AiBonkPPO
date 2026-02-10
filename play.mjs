import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
import { keyMap } from "./move.mjs";
import { deserializeModels, loadBrowserFile } from "./setupModel.mjs";
import { State } from "./State.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";


top.actorModel = null;

top.RECIEVEFUNCTION = function (args) {
    if (args.startsWith("42[7,")) {
        const data = JSON.parse(args.slice(2));
        keyMap.set(data[1], data[2].i);
    }
    return args;
}
let lastMove = { left: false, right: false, up: false, down: false, heavy: false, special: false };
function playMove(moves) {
    // moves = [left, right, up, down, heavy]
    const keys = {
        left: moves[0],
        right: moves[1],
        up: moves[2],
        down: moves[3],
        heavy: moves[4],
        special: false
    };

    top.presskeys(lastMove, keys);
    keyMap.set(top.myid, top.MAKE_KEYS(keys));
    lastMove = structuredClone(keys);
}

function isAlive(playerId) {
    if (!top.playerids[playerId]) {
        return false;
    }
    if (!top.playerids[playerId].playerData) {
        return false;
    }
    if (!top.playerids[playerId].playerData2.alive) {
        return false;
    }
    return true;
}

function getPlayerIds() {
    const myid = top.myid;
    if (!isAlive(myid)) {
        return null;
    }
    let otherId = null;
    for (const pid in top.playerids) {
        if (pid != myid && isAlive(pid)) {
            otherId = parseInt(pid);
            break;
        }
    }
    if (otherId === null) {
        return null;
    }
    return [myid, otherId];
}

async function main() {
    const filedata = await loadBrowserFile()
    if (filedata) {
        const deserialized = await deserializeModels(filedata.models);
        top.actorModel = deserialized.currentModel.actor;
        log(`Loaded AI from file.`);
    }
    else {
        log("No AI model file found.");
    }


    while (true) {
        if (top.actorModel) {
            const ids = getPlayerIds();
            if (ids) {
                [CONFIG.PLAYER_ONE_ID, CONFIG.PLAYER_TWO_ID] = ids;
                const state = new State();
                state.fetch();
                const stateArray = state.toArray();

                // 1. Create tensor manually
                const stateTensor = tf.tensor2d([stateArray]);

                try {
                    const logits = top.actorModel.predict(stateTensor);
                    const logitsArray = await logits.array();
                    const firstBatch = logitsArray[0];
                    const moves = firstBatch.map(x => x > 0);
                    playMove(moves);

                } catch (err) {
                    console.error("Prediction error:", err);
                } finally {
                    stateTensor.dispose();
                }
            }
        }
        await Time.sleep(80);
    }
}

main();