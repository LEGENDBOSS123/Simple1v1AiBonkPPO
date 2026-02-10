import '@tensorflow/tfjs-backend-webgpu';
import * as tf from '@tensorflow/tfjs';
if (tf.findBackend('webgpu')) {
    tf.setBackend('webgpu');
    console.log("🚀 Using WebGPU backend!");
} else {
    tf.setBackend('webgl');
    console.log("⚠️ WebGPU not found, falling back to WebGL.");
}
top.tf = tf;

export { tf };