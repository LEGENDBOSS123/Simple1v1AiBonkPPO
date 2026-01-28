export const CONFIG = {
    ACTOR_LAYER_LENGTHS: [256, 256, 256],
    CRITIC_LAYER_LENGTHS: [512, 256, 256],

    GAME_STATE_SIZE: 27,
    ACTION_SIZE: 5,

    LEARNING_RATE_ACTOR: 0.0002,
    LEARNING_RATE_CRITIC: 0.0009,

    WARM_UP_EPISODES: 100,
    SAVE_AFTER_EPISODES: 1000,
    MAX_SAVED_MODELS: 30,
    SWITCH_OPPONENT_EVERY_EPISODES: 10,
    DISCOUNT_FACTOR: 0.995,
    LAMBDA: 0.95,
    EPOCHS: 4,
    ROLLOUT_EPISODES: 10,

    ENTROPY_COEFFICIENT: 0.03,
    ENTROPY_DECAY: 0.9993,
    MIN_ENTROPY: 0.003,

    CLIP: 0.2,

    USE_LAYER_NORM: true,

    ELO_K: 32,
    ELO_DIFF_SCALE: 200,
    START_ELO: 400,

    PLAYER_ONE_ID: 200,
    PLAYER_TWO_ID: 201,
    POSITION_NORMALIZATION: 1 / 500,
    VELOCITY_NORMALIZATION: 3,

    MAX_SECONDS: 45, // 900 steps

    // reward
    WIN: 1,
    LOSS: -0.67,
    TIME_PENALTY: 0,
    DRAW_PENALTY: -1.0,

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