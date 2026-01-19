export class Time {
    static sleep(ms = 4) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}