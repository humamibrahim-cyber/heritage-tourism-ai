"""Backbone registry and classifier assembly for Part 1.

Covers brief tasks 2-5:
  * task 2 - select a CNN architecture, configure for transfer learning,
             load pre-trained ImageNet weights
  * task 3 - freeze all convolutional layers
  * task 4 - new top: dense layer(s) + activation + dropout
  * task 5 - compile with optimizer / loss / metrics
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class BackboneSpec:
    """Everything that differs between the candidate architectures.

    ``needs_preprocessing`` is the subtle one. Keras EfficientNet / EfficientNetV2
    bake normalisation into the graph and expect raw [0, 255] pixels, whereas
    ResNet50V2 and MobileNetV2 expect their own ``preprocess_input`` applied
    first. Getting this backwards silently costs several accuracy points, which
    is why it is encoded here rather than left to the caller.
    """

    name: str
    builder: str            # dotted path under tf.keras.applications
    preprocess: str | None  # dotted path to preprocess_input, or None
    params_millions: float


BACKBONES: dict[str, BackboneSpec] = {
    "efficientnetv2b0": BackboneSpec(
        name="EfficientNetV2B0",
        builder="tensorflow.keras.applications.EfficientNetV2B0",
        preprocess=None,  # normalisation is inside the model
        params_millions=7.1,
    ),
    "efficientnetb0": BackboneSpec(
        name="EfficientNetB0",
        builder="tensorflow.keras.applications.EfficientNetB0",
        preprocess=None,
        params_millions=5.3,
    ),
    "resnet50v2": BackboneSpec(
        name="ResNet50V2",
        builder="tensorflow.keras.applications.ResNet50V2",
        preprocess="tensorflow.keras.applications.resnet_v2.preprocess_input",
        params_millions=25.6,
    ),
    "mobilenetv2": BackboneSpec(
        name="MobileNetV2",
        builder="tensorflow.keras.applications.mobilenet_v2.MobileNetV2",
        preprocess="tensorflow.keras.applications.mobilenet_v2.preprocess_input",
        params_millions=3.5,
    ),
    "densenet121": BackboneSpec(
        name="DenseNet121",
        builder="tensorflow.keras.applications.DenseNet121",
        preprocess="tensorflow.keras.applications.densenet.preprocess_input",
        params_millions=8.1,
    ),
}


def _resolve(dotted: str) -> Callable:
    import importlib

    module_path, attr = dotted.rsplit(".", 1)
    return getattr(importlib.import_module(module_path), attr)


def get_spec(key: str) -> BackboneSpec:
    key = key.lower().replace("-", "").replace("_", "")
    if key not in BACKBONES:
        raise KeyError(f"Unknown backbone '{key}'. Choose from {sorted(BACKBONES)}")
    return BACKBONES[key]


def build_backbone(
    key: str,
    input_shape: tuple[int, int, int],
    trainable: bool = False,
    weights: str | None = "imagenet",
):
    """Instantiate an ImageNet-pretrained convolutional base (brief tasks 2-3).

    ``weights=None`` builds the same graph with random initialisation - used by
    the unit tests so they run without downloading weight files.
    """
    spec = get_spec(key)
    base = _resolve(spec.builder)(
        include_top=False,          # we supply our own classifier head
        weights=weights,            # transfer learning: pre-trained weights
        input_shape=input_shape,
        pooling=None,
    )
    base.trainable = trainable      # False => every conv layer frozen
    return base, spec


def build_classifier(cfg, backbone: str | None = None, augmentation=None,
                     weights: str | None = "imagenet"):
    """Assemble the full transfer-learning model.

    Architecture:
        input -> [augmentation] -> [preprocess] -> frozen conv base
              -> GlobalAveragePooling -> BatchNorm
              -> Dense(units, relu) -> Dropout -> Dense(n_classes, softmax)

    The final Dense is forced to float32 so the model stays numerically stable
    when mixed precision is enabled on the GPU.
    """
    import tensorflow as tf

    from ..config import CLASS_NAMES

    key = backbone or cfg.backbone
    base, spec = build_backbone(key, cfg.input_shape, trainable=False, weights=weights)

    inputs = tf.keras.Input(shape=cfg.input_shape, name="image")
    x = inputs
    if augmentation is not None:
        x = augmentation(x)
    if spec.preprocess:
        x = tf.keras.layers.Lambda(
            _resolve(spec.preprocess), name="preprocess"
        )(x)

    x = base(x, training=False)  # keep BatchNorm in inference mode while frozen
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.BatchNormalization(name="head_bn")(x)
    x = tf.keras.layers.Dense(
        cfg.dense_units, activation="relu", name="head_dense"
    )(x)
    x = tf.keras.layers.Dropout(cfg.dropout_rate, name="head_dropout")(x)
    outputs = tf.keras.layers.Dense(
        len(CLASS_NAMES), activation="softmax", dtype="float32", name="predictions"
    )(x)

    return tf.keras.Model(inputs, outputs, name=f"{spec.name}_heritage")


def get_base(model):
    """Retrieve the nested convolutional base from an assembled classifier.

    We look the layer up by type rather than stashing it as an attribute on the
    Model - assigning a Layer to a Model attribute makes Keras track it twice.
    """
    import tensorflow as tf

    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name != model.name:
            return layer
    raise ValueError(
        f"No nested backbone found in '{model.name}'. Was it built by build_classifier()?"
    )


def compile_model(model, lr: float, label_smoothing: float = 0.0):
    """Brief task 5 - optimizer, loss and metrics.

    Default path is SparseCategoricalCrossentropy, matching the integer labels
    produced by ``image_dataset_from_directory(label_mode="int")``.

    Label smoothing requires one-hot targets. If you set ``label_smoothing > 0``
    you MUST also rebuild the datasets with ``label_mode="categorical"``,
    otherwise the shapes will not match at fit() time.
    """
    import tensorflow as tf

    if label_smoothing > 0:
        loss = tf.keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
        metrics = [tf.keras.metrics.CategoricalAccuracy(name="accuracy")]
    else:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
        metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=loss,
        metrics=metrics,
    )
    return model


def unfreeze_top(model, fraction: float = 0.30, verbose: bool = True):
    """Stage B - unfreeze the top ``fraction`` of the convolutional base.

    BatchNormalization layers are deliberately kept frozen. Updating their
    running statistics on a small batch during fine-tuning is a well-known way
    to destroy a pre-trained model's accuracy.
    """
    import tensorflow as tf

    base = get_base(model)
    base.trainable = True
    n_layers = len(base.layers)
    cutoff = int(n_layers * (1 - fraction))

    trainable_count = 0
    for i, layer in enumerate(base.layers):
        if i < cutoff or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
            trainable_count += 1

    if verbose:
        print(
            f"Unfroze {trainable_count}/{n_layers} layers "
            f"(from index {cutoff}); BatchNorm kept frozen."
        )
    return model


def enable_mixed_precision(enabled: bool = True):
    """float16 compute on GPU - roughly 1.5-2x faster on A100/L4/V100."""
    import tensorflow as tf

    if not enabled:
        return "float32"
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        print("No GPU detected - staying in float32.")
        return "float32"
    tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"Mixed precision enabled on {gpus[0].name}")
    return "mixed_float16"
