import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";
import { Time } from "./Time.mjs";
import { entropyBernoulli, logProbsBernoulli } from "./utils.mjs";


function calculateGAE(rewards, values, nextValues, notDones, gamma, lambda) {
    const advantages = new Array(rewards.length).fill(0);
    let lastAdvantage = 0;

    for (let i = rewards.length - 1; i >= 0; i--) {
        const nextVal = (i == rewards.length - 1) ? nextValues : values[i + 1];
        const notDone = notDones[i];
        const delta = rewards[i] + gamma * nextVal * notDone - values[i];
        advantages[i] = delta + gamma * lambda * notDone * lastAdvantage;
        lastAdvantage = advantages[i];
    }

    return advantages;
}

export async function train(models, batch, epochs, sleep = false) {
    if (batch.length === 0) return null;

    const { actor, critic, optimizerActor, optimizerCritic } = models;

    const gamma = CONFIG.DISCOUNT_FACTOR;
    const clip = CONFIG.CLIP || 0.2;
    const lambda = CONFIG.LAMBDA;
    const eps = 1e-6;

    const states = tf.tensor2d(batch.map(b => b.state));
    const nextStates = tf.tensor2d(batch.map(b => b.nextState));
    const actions = tf.tensor2d(batch.map(b => b.action));
    const rewards = tf.tensor1d(batch.map(b => b.reward));
    const notDones = tf.tensor1d(batch.map(b => b.done ? 0 : 1));
    const oldLogProbs = tf.tensor1d(batch.map(b => b.logProb));

    const values = critic.predict(states).squeeze();
    const nextValues = critic.predict(nextStates).squeeze();

    const advantages = tf.tensor1d(calculateGAE(
        batch.map(b => b.reward),
        values.arraySync(),
        nextValues.arraySync().at(-1),
        batch.map(b => b.done ? 0 : 1),
        gamma,
        lambda
    ));

    const normalizedAdvantages = tf.tidy(() => {
        // normalized_a = (a-mean)/std
        const { mean, variance } = tf.moments(advantages);
        return advantages.sub(mean).div(variance.sqrt().add(eps));
    });

    let lastLossActor = 0;
    let lastLossCritic = 0;

    const returns = advantages.add(values);

    for (let epoch_i = 0; epoch_i < epochs; epoch_i++) {
        optimizerActor.minimize(() => {
            const logits = actor.apply(states);
            const newLogProbs = logProbsBernoulli(logits, actions);
            const ratio = newLogProbs.sub(oldLogProbs).exp();

            const surrogate1 = ratio.mul(normalizedAdvantages);
            const surrogate2 = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(normalizedAdvantages);
            const minSurrogate = tf.minimum(surrogate1, surrogate2);
            const policyLoss = minSurrogate.mean().neg();

            const entropyBonus = entropyBernoulli(logits).mean().mul(CONFIG.ENTROPY_COEFFICIENT);

            const totalLoss = policyLoss.sub(entropyBonus);

            if (epoch_i === epochs - 1) {
                lastLossActor = totalLoss.dataSync()[0];
            }
            return totalLoss;
        });
        if (sleep) {
            // let the tab breathe
            await Time.sleep(25);
        }
        optimizerCritic.minimize(() => {
            const vals = critic.predict(states).squeeze();

            const closs = tf.losses.meanSquaredError(returns, vals);
            if (epoch_i === epochs - 1) {
                lastLossCritic = closs.dataSync()[0];
            }
            return closs;
        });
        if (sleep) {
            await Time.sleep(25);
        }
    }

    tf.dispose([states, nextStates, actions, rewards, notDones, oldLogProbs,
        values, nextValues, advantages, normalizedAdvantages, returns]);

    return {
        lossActor: lastLossActor,
        lossCritic: lastLossCritic
    };
}