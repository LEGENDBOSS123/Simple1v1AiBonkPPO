import { Memory } from "./Memory.mjs";

export class ReplayBuffer {
    constructor() {
        this.buffer = [];
        this.episodeCount = 0;
    }

    add(state, action, logProb, reward, nextState, done) {
        if(done){
            this.episodeCount++;
        }
        this.buffer.push(
            new Memory(state, action, logProb, reward, nextState, done)
        );
    }

    episodes(){
        return this.episodeCount;
    }

    size() {
        return this.buffer.length;
    }

    getAll() {
        return this.buffer;
    }

    clear() {
        this.buffer = [];
        this.episodeCount = 0;
    }
}