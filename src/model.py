import tensorflow as tf
from utils.logger import get_logger

logger = get_logger(__name__)

def build_model(config):

    model_config = config["model"]

    input_length = model_config["input_length"]
    forecast_horizon = model_config["forecast_horizon"]
    hidden_size = model_config["hidden_size"]
    num_layers = model_config["num_layers"]
    dropout = model_config["dropout"]

    inputs = tf.keras.layers.Input(shape=(input_length, 1)) 

    x = inputs
    for i in range(num_layers):
        attn_output = tf.keras.layers.MultiHeadAttention(
            num_heads=4, key_dim=hidden_size, dropout=dropout
        )(x, x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)

        ffn = tf.keras.Sequential([
            tf.keras.layers.Dense(hidden_size * 4, activation="relu"),
            tf.keras.layers.Dense(hidden_size),
        ])
        ffn_output = ffn(x)
        x = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + ffn_output)

        logger.info(f"Added Transformer encoder block {i+1}")

    x = tf.keras.layers.GlobalAveragePooling1D()(x)

    outputs = tf.keras.layers.Dense(forecast_horizon)(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config["training"]["learning_rate"]),
        loss="mse",
        metrics=["mae"]
    )

    logger.info("TSFM model compiled with Adam optimizer, MSE loss, and MAE metric")

    return model
