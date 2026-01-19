export class Memory {
    constructor(state = [], action = [], logProb = 0, reward = 0, nextState = [], done = false) {
        this.state = state;
        this.action = action;
        this.logProb = logProb;
        this.reward = reward;
        this.nextState = nextState;
        this.done = done;
    }
}