export const CONFIG = {
    ACTOR_LAYER_LENGTHS: [512, 256, 128],
    CRITIC_LAYER_LENGTHS: [512, 256, 128],

    GAME_STATE_SIZE: 27,
    ACTION_SIZE: 5,

    LEARNING_RATE_ACTOR: 0.0003,
    LEARNING_RATE_CRITIC: 0.001,

    WARM_UP_EPISODES: 100,
    SAVE_AFTER_EPISODES: 2000,
    MAX_SAVED_MODELS: 30,
    SWITCH_OPPONENT_EVERY_EPISODES: 10,
    DISCOUNT_FACTOR: 0.993,
    LAMBDA: 0.95,
    EPOCHS: 4,
    ROLLOUT_STEPS: 32000,
    MINI_BATCH_SIZE: 512,

    ENTROPY_COEFFICIENT: 0.03,
    ENTROPY_DECAY: 0.9995,
    MIN_ENTROPY: 0.001,

    CLIP: 0.2,

    USE_LAYER_NORM: true,

    // ELO system
    ELO_K: 32,
    ELO_DIFF_SCALE: 200,
    START_ELO: 400,

    PLAYER_ONE_ID: 200,
    PLAYER_TWO_ID: 201,
    POSITION_NORMALIZATION: 1 / 500,
    VELOCITY_NORMALIZATION: 3,

    MAX_SECONDS: 60, // 1200 steps

    // reward
    WIN: 1,
    LOSS: -1,
    TIME_PENALTY: -0.001,
    DRAW_PENALTY: -1,
    
    // runtime params
    steps: 0,
    episodes: 0,
    elos: [],
    entropies: [],
};

export function updateConfig(newConfig) {
    Object.assign(CONFIG, newConfig);
}

top.CONFIG = CONFIG;