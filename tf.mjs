import * as tf from '@tensorflow/tfjs';
import '@tensorflow/tfjs-backend-webgpu';

async function loadBackend() {
    await tf.ready();
    if (tf.findBackend('webgpu')) {
        tf.setBackend('webgpu');
        console.log("🚀 Using WebGPU backend!");
    } else {
        tf.setBackend('webgl');
        console.log("⚠️ WebGPU not found, falling back to WebGL.");
    }
}
loadBackend();
top.tf = tf;

export { tf };