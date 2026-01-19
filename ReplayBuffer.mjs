import { Memory } from "./Memory.mjs";

export class ReplayBuffer {
    constructor() {
        this.buffer = [];
    }

    add(state, action, logProb, reward, nextState, done) {
        this.buffer.push(
            new Memory(state, action, logProb, reward, nextState, done)
        );
    }

    size() {
        return this.buffer.length;
    }

    getAll() {
        return this.buffer;
    }

    clear() {
        this.buffer = [];
    }
}