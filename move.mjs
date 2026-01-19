
export const cValueMap = new Map();
export const keyMap = new Map();

export function getAction(id) {
    if (!keyMap.has(id)) {
        return actionToArray(top.GET_KEYS(0));
    }
    return actionToArray(top.GET_KEYS(keyMap.get(id)));
}



export function move(id, keys) {
    let cvalue = 100;
    if (cValueMap.has(id)) {
        cvalue = cValueMap.get(id);
        cValueMap.set(id, cvalue + 1);
    }
    else {
        cValueMap.set(id, cvalue);
    }
    keyMap.set(id, top.MAKE_KEYS(keys));
    let packet = `42[7,${id},{"i":${top.MAKE_KEYS(keys)},"f":${top.getCurrentFrame()},"c":${cvalue}}]`;
    top.SEND("42" + JSON.stringify([4, { "type": "fakerecieve", "from": top.playerids[myid].userName, "packet": [packet], to: [-1] }]));
    top.RECIEVE(packet);
}