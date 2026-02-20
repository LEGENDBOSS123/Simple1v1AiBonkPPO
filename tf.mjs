import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

async function loadBackend(defaultBackend = "best") {

    await tf.ready();

    if (defaultBackend === "best") {
        if (tf.findBackend('webgpu')) {
            tf.setBackend('webgpu');
            console.log("🚀 Using WebGPU backend!");
        } else {
            tf.setBackend('webgl');
            console.log("⚠️ WebGPU not found, falling back to WebGL.");
        }
    }
    else {
        await tf.setBackend(defaultBackend);
        console.log(`Using ${defaultBackend} backend.`);
    }
}

loadBackend();
top.tf = tf;

export { tf };