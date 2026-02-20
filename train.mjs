import { CONFIG } from "./config.mjs";
import { log } from "./log.mjs";
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
    const miniBatchSize = CONFIG.MINI_BATCH_SIZE || batch.length;
    const eps = 1e-6;

    const states = tf.tensor2d(batch.map(b => b.state));
    const nextStates = tf.tensor2d(batch.map(b => b.nextState));
    const actions = tf.tensor2d(batch.map(b => b.action));
    const oldLogProbs = tf.tensor1d(batch.map(b => b.logProb));

    // Track all tensors for guaranteed cleanup
    const outerTensors = [states, nextStates, actions, oldLogProbs];

    try {

        const values = tf.tidy(() => critic.predict(states).squeeze());
        const nextValues = tf.tidy(() => critic.predict(nextStates).squeeze());
        outerTensors.push(values, nextValues);

        if (sleep) {
            await Time.sleep(25);
        }

        log("Calculating advantages");

        const advantages = tf.tensor1d(calculateGAE(
            batch.map(b => b.reward),
            await values.array(),
            (await nextValues.array()).at(-1),
            batch.map(b => b.done ? 0 : 1),
            gamma,
            lambda
        ));
        outerTensors.push(advantages);

        if (sleep) {
            await Time.sleep(25);
        }

        log("Normalizing advantages");
        const normalizedAdvantages = tf.tidy(() => {
            // normalized_a = (a-mean)/std
            const { mean, variance } = tf.moments(advantages);
            return advantages.sub(mean).div(variance.sqrt().add(eps));
        });
        outerTensors.push(normalizedAdvantages);

        if (sleep) {
            await Time.sleep(25);
        }

        let lastLossActor = 0;
        let lastLossCritic = 0;

        const returns = advantages.add(values);
        outerTensors.push(returns);

        log("Starting training epochs", "| Tensors:", tf.memory().numTensors);

        for (let epoch_i = 0; epoch_i < epochs; epoch_i++) {
            // Shuffle indices for mini-batching
            const indices = Array.from({ length: batch.length }, (_, i) => i);
            for (let i = indices.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [indices[i], indices[j]] = [indices[j], indices[i]];
            }

            // Process mini-batches
            const numMiniBatches = Math.floor(batch.length / miniBatchSize);
            let epochActorLoss = 0;
            let epochCriticLoss = 0;

            for (let mb = 0; mb < numMiniBatches; mb++) {
                log("Epoch", epoch_i + 1, "Mini-batch", mb + 1, "of", numMiniBatches);
                const start = mb * miniBatchSize;
                const end = Math.min(start + miniBatchSize, batch.length);
                const mbIndices = indices.slice(start, end);
                const mbIndicesTensor = tf.tensor1d(mbIndices, 'int32');

                const mbStates = tf.gather(states, mbIndicesTensor);
                const mbActions = tf.gather(actions, mbIndicesTensor);
                const mbOldLogProbs = tf.gather(oldLogProbs, mbIndicesTensor);
                const mbNormalizedAdvantages = tf.gather(normalizedAdvantages, mbIndicesTensor);
                const mbReturns = tf.gather(returns, mbIndicesTensor);
                const actorLoss = optimizerActor.minimize(() => {
                    const logits = actor.apply(mbStates);
                    const newLogProbs = logProbsBernoulli(logits, mbActions);
                    const ratio = newLogProbs.sub(mbOldLogProbs).exp();

                    const surrogate1 = ratio.mul(mbNormalizedAdvantages);
                    const surrogate2 = tf.clipByValue(ratio, 1 - clip, 1 + clip).mul(mbNormalizedAdvantages);
                    const minSurrogate = tf.minimum(surrogate1, surrogate2);
                    const policyLoss = minSurrogate.mean().neg();

                    const entropyBonus = entropyBernoulli(logits).mean().mul(CONFIG.ENTROPY_COEFFICIENT);

                    const totalLoss = policyLoss.sub(entropyBonus);

                    return totalLoss;
                }, true);
                if (epoch_i === epochs - 1 && mb === numMiniBatches - 1) {
                    epochActorLoss = (await actorLoss.data())[0];
                }
                tf.dispose(actorLoss);
                if (sleep) {
                    await Time.sleep(25);
                }
                const criticLoss = optimizerCritic.minimize(() => {
                    const vals = critic.apply(mbStates).squeeze();
                    const closs = tf.losses.meanSquaredError(mbReturns, vals);
                    return closs;
                }, true);


                if (epoch_i === epochs - 1 && mb === numMiniBatches - 1) {
                    epochCriticLoss = (await criticLoss.data())[0];
                }
                tf.dispose(criticLoss);

                tf.dispose([mbStates, mbActions, mbOldLogProbs, mbNormalizedAdvantages, mbReturns, mbIndicesTensor]);

                if (sleep) {
                    await Time.sleep(25);
                }
            }
            lastLossActor = epochActorLoss;
            lastLossCritic = epochCriticLoss;
        }
        if (sleep) {
            await Time.sleep(25);
        }
        log("Training complete", "| Tensors:", tf.memory().numTensors);
        return {
            lossActor: lastLossActor,
            lossCritic: lastLossCritic
        };

    } finally {
        log("Disposing tensors");
        tf.dispose(outerTensors);
        if (sleep) {
            await Time.sleep(25);
        }
    }
}
