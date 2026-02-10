import { CONFIG } from "./config.mjs";
import { tf } from "./tf.mjs";


function setupPPOActor() {
    const model = tf.sequential();
    model.add(tf.layers.inputLayer({ inputShape: [CONFIG.GAME_STATE_SIZE], name: 'actor_input' }));
    let i = 0;
    for (const layerLength of CONFIG.ACTOR_LAYER_LENGTHS) {
        model.add(tf.layers.dense({
            units: layerLength,
            activation: 'tanh',
            kernelInitializer: 'glorotUniform',
            name: `actor_dense_${i++}`
        }));
        if (CONFIG.USE_LAYER_NORM) {
            model.add(tf.layers.layerNormalization({ name: `actor_norm_${i - 1}` }));
        }
    }
    model.add(tf.layers.dense({
        units: CONFIG.ACTION_SIZE,
        activation: 'linear',
        kernelInitializer: 'glorotUniform',
        name: 'actor_output'
    }));
    return model;
}

function setupPPOCritic() {
    let model = tf.sequential();
    model.add(tf.layers.inputLayer({ inputShape: [CONFIG.GAME_STATE_SIZE], name: 'critic_input' }));
    let i = 0;
    for (const layerLength of CONFIG.CRITIC_LAYER_LENGTHS) {
        model.add(tf.layers.dense({
            units: layerLength,
            activation: 'tanh',
            kernelInitializer: 'glorotUniform',
            name: `critic_dense_${i++}`
        }));
        if (CONFIG.USE_LAYER_NORM) {
            model.add(tf.layers.layerNormalization({ name: `critic_norm_${i - 1}` }));
        }
    }
    model.add(tf.layers.dense({
        units: 1,
        activation: 'linear',
        kernelInitializer: 'glorotUniform',
        name: 'critic_output'
    }));
    return model;
}

export function setupModel() {
    const actor = setupPPOActor();
    const critic = setupPPOCritic();

    const optimizerActor = tf.train.adam(CONFIG.LEARNING_RATE_ACTOR);
    const optimizerCritic = tf.train.adam(CONFIG.LEARNING_RATE_CRITIC);
    return {
        actor,
        critic,
        optimizerActor,
        optimizerCritic,
        elo: CONFIG.START_ELO
    };
}


export function cloneModel(model) {
    const newModel = setupModel();
    newModel.actor.setWeights(model.actor.getWeights());
    newModel.critic.setWeights(model.critic.getWeights());
    newModel.elo = model.elo ?? newModel.elo;
    return newModel;
}

export async function downloadModel(model) {
    const artifacts = await model.save(
        tf.io.withSaveHandler(async (artifacts) => {
            return artifacts;
        })
    );
    artifacts.weightData = Array.from(new Float32Array(artifacts.weightData));
    return artifacts;
}
export async function downloadOptimizer(optimizer) {
    // 1. Get the configuration (learning rate, decay, beta1, etc.)
    const config = optimizer.getConfig();

    // 2. Get the current state weights (tensors)
    // These contain the iteration count, momentum, accumulated gradients, etc.
    const weightTensors = optimizer.getWeights();

    // 3. Serialize the weights
    // We map over the tensors to extract their data asynchronously
    const weightData = await Promise.all(
        weightTensors.map(async (tensor) => {
            const data = await tensor.data();
            // Convert Float32Array to standard Array for JSON compatibility
            return Array.from(data);
        })
    );
    
    // We don't need to save shapes explicitly because optimizer weights 
    // are loaded by index, not by name or shape matching.
    
    return {
        className: optimizer.getClassName(), // e.g., 'Adam', 'SGD'
        config: config,
        weightData: weightData
    };
}
export function restoreOptimizer(optimizerArtifacts) {
    // 1. Recreate the optimizer using the saved config and class name
    // (This works for built-in optimizers like Adam, SGD, RMSProp)
    const optimizerConstructor = tf.train[optimizerArtifacts.className.toLowerCase()]; // e.g. tf.train.adam
    
    if (!optimizerConstructor) {
        throw new Error(`Unknown optimizer: ${optimizerArtifacts.className}`);
    }

    const optimizer = optimizerConstructor(optimizerArtifacts.config);

    // 2. Convert standard arrays back to Tensors
    // Note: Optimizer weights are often scalars or 1D, so we generally rely on 
    // setWeights handling the shape inference or just passing the data.
    const weightTensors = optimizerArtifacts.weightData.map(data => 
        tf.tensor(data)
    );

    // 3. Set the optimizer weights (restoring the state)
    optimizer.setWeights(weightTensors);
    
    // Clean up the temporary tensors created for setWeights
    weightTensors.forEach(t => t.dispose());
    
    return optimizer;
}
export async function serializeModels(models, currentModel) {
    let models2 = [...models];
    if (currentModel) {
        models2.push(currentModel);
    }
    const arrayOfModels = models2;

    const serializedModels = [];
    for (const model of arrayOfModels) {
        const artifactsActor = await downloadModel(model.actor);
        const artifactsCritic = await downloadModel(model.critic);
        const artifactsOptimizerActor = await downloadOptimizer(model.optimizerActor);
        const artifactsOptimizerCritic = await downloadOptimizer(model.optimizerCritic);
        const artifacts = {
            actor: artifactsActor,
            critic: artifactsCritic,
            actor_optimizer: artifactsOptimizerActor,
            critic_optimizer: artifactsOptimizerCritic,
            elo: model.elo
        };
        serializedModels.push(artifacts);
    }
    return serializedModels;
}

top.saveModels = async function () {
    const serializedModels = await serializeModels(top.models(), top.currentModel());
    const json = {
        models: serializedModels,
        CONFIG: CONFIG
    }
    await saveBrowserFile(json, "models.json");
}

top.saveModelsString = async function () {
    const serializedModels = await serializeModels(top.models(), top.currentModel());
    const json = {
        models: serializedModels,
        CONFIG: CONFIG
    }
    return json;
}

export async function loadModelFromArtifacts(artifacts) {
    const model = setupModel();

    const actorWeights = tf.io.decodeWeights(new Float32Array(artifacts.actor.weightData).buffer, artifacts.actor.weightSpecs);
    const criticWeights = tf.io.decodeWeights(new Float32Array(artifacts.critic.weightData).buffer, artifacts.critic.weightSpecs);
    model.actor.loadWeights(actorWeights);
    model.critic.loadWeights(criticWeights);

    if(artifacts.actor_optimizer && artifacts.critic_optimizer) {
        model.optimizerActor = restoreOptimizer(artifacts.actor_optimizer);
        model.optimizerCritic = restoreOptimizer(artifacts.critic_optimizer);
    }

    if (typeof artifacts.elo === "number") {
        model.elo = artifacts.elo;
    }

    return model;
}

export async function deserializeModels(arrayOfArtifacts) {
    const models = [];
    for (const artifacts of arrayOfArtifacts) {
        const model = await loadModelFromArtifacts(artifacts);
        models.push(model);
    }
    const currentModel = models.pop();
    return { models, currentModel };
}


export async function saveBrowserFile(filedata, filename) {
    // 1. Convert JSON to string
    const jsonString = JSON.stringify(filedata);

    // 2. Create a stream from the string data
    const stream = new Blob([jsonString]).stream();

    // 3. Pipe through GZIP compression
    const compressedStream = stream.pipeThrough(new CompressionStream("gzip"));

    // 4. Convert stream back to a Blob
    const compressedBlob = await new Response(compressedStream).blob();

    // 5. Download logic (Standard)
    const url = URL.createObjectURL(compressedBlob);
    const a = document.createElement("a");
    a.href = url;
    // Append .gz to indicate compression, or keep original if you handle extension logic elsewhere
    a.download = filename.endsWith('.gz') ? filename : filename + '.gz';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);

    URL.revokeObjectURL(url);
}

/**
 * Loads a GZIP compressed file from the user and parses it back to JSON.
 */
export function loadBrowserFile() {
    return new Promise((resolve, reject) => {
        const input = document.createElement("input");
        input.type = "file";
        // Accept .json or .gz files
        input.accept = "application/json, .gz";

        input.onchange = async (event) => {
            const file = event.target.files[0];
            if (!file) return;

            try {
                // 1. Get the stream from the file
                const stream = file.stream();

                // 2. Pipe through GZIP decompression
                const decompressedStream = stream.pipeThrough(new DecompressionStream("gzip"));

                // 3. Read the decompressed stream as text
                const text = await new Response(decompressedStream).text();

                // 4. Parse JSON
                const filedata = JSON.parse(text);
                resolve(filedata);
            } catch (error) {
                console.error("Decompression failed. Was the file actually compressed?", error);
                // Fallback: Try reading as plain JSON in case the user uploaded an uncompressed file
                try {
                    const text = await file.text();
                    resolve(JSON.parse(text));
                } catch (fallbackError) {
                    reject(error);
                }
            }
        };

        input.click();
    });
}