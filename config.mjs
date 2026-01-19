export const CONFIG = {
    ACTOR_LAYER_LENGTHS: [256, 256, 256],
    CRITIC_LAYER_LENGTHS: [256, 256, 256],

    GAME_STATE_SIZE: 24,
    ACTION_SIZE: 5,

    LEARNING_RATE_ACTOR: 0.0002,
    LEARNING_RATE_CRITIC: 0.0002,

    WARM_UP_EPISODES: 10,
    SAVE_AFTER_EPISODES: 1000,
    DISCOUNT_FACTOR: 0.99,
    LAMBDA: 0.97,
    EPOCHS: 10,
    ROLLOUT_EPISODES: 10,

    ENTROPY_COEFFICIENT: 0.01,

    CLIP: 0.2,

    PLAYER_ONE_ID: 200,
    PLAYER_TWO_ID: 201,
    POSITION_NORMALIZATION: 1 / 500,
    VELOCITY_NORMALIZATION: 2,

    // runtime params
    steps: 0,
    episodes: 0,
};

export function updateConfig(newConfig) {
    Object.assign(CONFIG, newConfig);
}

top.CONFIG = CONFIG;