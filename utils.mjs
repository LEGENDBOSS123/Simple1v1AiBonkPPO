import { tf } from "./tf.mjs";



export function logProbsBernoulli(logits, sampled) {
    // log(sigmoid(x)) = -softplus(-x)
    const logSigmoid = tf.softplus(logits.neg()).neg();
    const logOneMinusSigmoid = tf.softplus(logits).neg();
    const logProbs = sampled.mul(logSigmoid).add(tf.scalar(1).sub(sampled).mul(logOneMinusSigmoid));
    return logProbs.sum(-1);
}

export async function logProbabilities(logitsArr, sampledArr) {
    const logProbs = tf.tidy(() => {
        const logits = tf.tensor2d([logitsArr]);
        const sampled = tf.tensor2d([sampledArr]);
        return logProbsBernoulli(logits, sampled);
    });
    const result = await logProbs.array();
    logProbs.dispose();
    return result[0];
}

function sigmoid(x) {
    return 1 / (1 + Math.exp(-x));
}


export function nonTFgetEntropy(logits) {
    let entropy = 0;
    for (let i = 0; i < logits.length; i++) {
        const p = sigmoid(logits[i]);
        // H = -p*log(p) - (1-p)*log(1-p)
        entropy += -p * Math.log(p + 1e-10) - (1 - p) * Math.log(1 - p + 1e-10);
    }
    return entropy;
}

export function mean(x) {
    const sum = x.reduce((a, b) => a + b, 0);
    return sum / x.length;
}

export async function predictActionArray(model, state) {
    const rawAction = tf.tidy(() => {
        const stateTensor = tf.tensor2d([state]);
        const prediction = model.actor.predict(stateTensor);
        return prediction;
    });
    const result = await rawAction.array();
    rawAction.dispose();
    return result[0];
}

export function sampleBernoulli(p) {
    return Math.random() < p ? 1 : 0;
}


export function actionToArray(action) {
    return [
        action.left ? 1 : 0,
        action.right ? 1 : 0,
        action.up ? 1 : 0,
        action.down ? 1 : 0,
        action.heavy ? 1 : 0,
        // action.special ? 1 : 0,
    ];
}

export function arrayToAction(logitsArr) {
    return {
        left: sampleBernoulli(sigmoid(logitsArr[0])) === 1,
        right: sampleBernoulli(sigmoid(logitsArr[1])) === 1,
        up: sampleBernoulli(sigmoid(logitsArr[2])) === 1,
        down: sampleBernoulli(sigmoid(logitsArr[3])) === 1,
        heavy: sampleBernoulli(sigmoid(logitsArr[4])) === 1,
        special: false//sampleBernoulli(sigmoid(logitsArr[5])) === 1
    }
    // return {
    //     left: arr[0] > 0,
    //     right: arr[1] > 0,
    //     up: arr[2] > 0,
    //     down: arr[3] > 0,
    //     heavy: arr[4] > 0,
    //     special: false//arr[5] > 0
    // };
}

export function randomAction(p = 0.5) {
    return {
        left: Math.random() < p,
        right: Math.random() < p,
        up: Math.random() < p,
        down: Math.random() < p,
        heavy: Math.random() < p,
        special: false//Math.random() < p
    };
}


export function entropyBernoulli(logits) {
    const p = logits.sigmoid();
    // log(sigmoid(x)) = -softplus(-x)
    const logSigmoid = tf.softplus(logits.neg()).neg();
    const logOneMinusSigmoid = tf.softplus(logits).neg();
    // H = -p*log(p) - (1-p)*log(1-p)
    const entropy = p.mul(logSigmoid).add(tf.scalar(1).sub(p).mul(logOneMinusSigmoid)).neg();
    return entropy.sum(-1);
}