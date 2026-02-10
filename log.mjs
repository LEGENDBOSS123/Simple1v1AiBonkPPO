export function log(...args) {
    console.log(...args);
}

export function logChat(...args) {
    console.log(...args);
    top.displayInChat(args.join(" "));
}