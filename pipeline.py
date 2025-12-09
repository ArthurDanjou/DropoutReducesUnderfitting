%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tensorflow import keras
import tensorflow as tf
from keras import callbacks, layers, models, mixed_precision
from sklearn.model_selection import train_test_split

policy = mixed_precision.Policy("mixed_float16")
mixed_precision.set_global_policy(policy)

sns.set(style="whitegrid")

class DynamicDropout(layers.Layer):
    """Dropout layer with a dynamically adjustable dropout rate.

    Args:
        rate_variable: A TensorFlow variable representing the dropout rate.

    """
    def __init__(self, rate_variable, **kwargs):
        super(DynamicDropout, self).__init__(**kwargs)
        self.rate_var = rate_variable

    def call(self, inputs, training=None):
        if training:
            return tf.nn.dropout(inputs, rate=self.rate_var)
        return inputs


class DropoutScheduler(callbacks.Callback):
    """
    Keras Callback to dynamically adjust the dropout rate.
    Implements strategies from the paper 'Dropout Reduces Underfitting'.
    Simplified version for dense networks (MLP).
    
    Args:
        mode (str): 'standard', 'early', or 'late'.
        switch_epoch (int): The epoch where the behavior switches.
        rate (float): The dropout rate to apply when active (e.g., 0.2).
        verbose (int): 1 to log changes, 0 for silence.
        
    Usage:
        dropout_scheduler = DropoutScheduler(mode='early', switch_epoch=15, rate=0.2, verbose=1)
        model.fit(X_train, y_train, epochs=50, callbacks=[dropout_scheduler])
    """ 
    def __init__(self, rate_variable, mode, switch_epoch, active_rate, verbose=0):
        super(DropoutScheduler, self).__init__()
        
        valid_modes = ["standard", "early", "late", "none"]
        if mode not in valid_modes:
            raise ValueError(f"Mode must be one of {valid_modes}")
        
        self.rate_var = rate_variable
        self.mode = mode
        self.switch_epoch = switch_epoch
        self.active_rate = active_rate
        self.verbose = verbose

    def on_train_begin(self, logs=None):
        self.dropout_layers = []
        for layer in self.model.layers:
            if isinstance(layer, layers.Dropout):
                self.dropout_layers.append(layer)
            if hasattr(layer, 'layers'):
                for sub_layer in layer.layers:
                    if isinstance(sub_layer, layers.Dropout):
                        self.dropout_layers.append(sub_layer)
        
        if self.verbose > 0:
            print(f"[DropoutScheduler] {len(self.dropout_layers)} Dropout layers tracked.")

    def on_epoch_begin(self, epoch, logs=None):
        new_rate = 0.0

        if self.mode == "standard":
            new_rate = self.active_rate
        elif self.mode == "none":
            new_rate = 0.0
        elif self.mode == "early":
            if epoch < self.switch_epoch:
                new_rate = self.active_rate
            else:
                new_rate = 0.0
        elif self.mode == "late":
            if epoch < self.switch_epoch:
                new_rate = 0.0
            else:
                new_rate = self.active_rate

        self.rate_var.assign(new_rate)

        if self.verbose > 0:
            status = "ACTIF" if new_rate > 0 else "INACTIF"
            print(
                f"\n[Epoch {epoch + 1}] Mode '{self.mode}': {status} (Taux={new_rate:.2f})"
            )
            
class ExperimentPipeline:
    """Pipeline to run experiments with different datasets and models.

    Args:
        dataset_name (str): 'mnist', 'cifar10', or 'cifar100'.
        model_type (str): 'dense' or 'cnn'.
        subset_fraction (float): Fraction of training data to use (0 < fraction <= 1.0).

    Usage:
        pipeline = ExperimentPipeline(dataset_name='mnist', model_type='dense', subset_fraction=0.5)
        history = pipeline.train(mode='early', switch_epoch=15, rate=0.2, epochs=50)
    """
    def __init__(
        self, dataset_name: str, model_type: str, subset_fraction: float = 1.0
    ):
        self.dataset_name = dataset_name
        self.model_type = model_type
        self.subset_fraction = subset_fraction
        self.dropout_var = tf.Variable(0.0, trainable=False, dtype=tf.float32)

        self._load_data(dataset_name, subset_fraction)
        self._check_gpu()

        if model_type == "dense":
            self.model_factory = self._create_dense_model
        elif model_type == "cnn":
            self.model_factory = self._create_convolutional_model
        else:
            raise ValueError(f"Model type '{model_type}' non reconnu.")

        print(f"✅ Pipeline Ready: {dataset_name.upper()} | {model_type.upper()}")

    def _check_gpu(self):
        """Check for GPU availability and print status."""
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            print(f"✅ GPU detected : {len(gpus)} available(s)")
        else:
            print("⚠️ NO GPU DETECTED. Training might be slow.")
    
    def _load_data(self, name: str, subset_fraction: float = 1.0) -> None:
        """Load dataset and prepare train/val splits.

        Args:
            name (str): Dataset name.
            subset_fraction (float): Fraction of training data to use.
        """
        if name == "mnist":
            (X_tr, y_tr), (X_te, y_te) = keras.datasets.mnist.load_data()
            X_tr = np.expand_dims(X_tr, -1).astype("float32") / 255.0
            X_te = np.expand_dims(X_te, -1).astype("float32") / 255.0
        elif name == "cifar10":
            (X_tr, y_tr), (X_te, y_te) = keras.datasets.cifar10.load_data()
            X_tr = X_tr.astype("float32") / 255.0
            X_te = X_te.astype("float32") / 255.0
        elif name == "cifar100":
            (X_tr, y_tr), (X_te, y_te) = keras.datasets.cifar100.load_data()
            X_tr = X_tr.astype("float32") / 255.0
            X_te = X_te.astype("float32") / 255.0
        elif name == "fashion_mnist":
            (X_tr, y_tr), (X_te, y_te) = keras.datasets.fashion_mnist.load_data()
            X_tr = np.expand_dims(X_tr, -1).astype("float32") / 255.0
            X_te = np.expand_dims(X_te, -1).astype("float32") / 255.0
        else:
            raise ValueError("Dataset unknown")

        if subset_fraction < 1.0:
            subset_size = int(len(X_tr) * subset_fraction)
            X_tr = X_tr[:subset_size]
            y_tr = y_tr[:subset_size]
            print(
                f"⚠️ Using subset: {subset_fraction * 100:.1f}% of training data ({subset_size} samples)"
            )

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_tr, y_tr, test_size=0.2, random_state=42
        )
        self.input_shape = self.X_train.shape[1:]
        self.num_classes = len(np.unique(self.y_train))

    def _create_dense_model(self) -> models.Sequential:
        """Create a simple dense MLP model with dropout.

        Args:
            dropout_rate (float): Dropout rate to use in Dropout layers.

        Returns:
            models.Sequential: Compiled Keras model.
        """
        return models.Sequential(
            [
                layers.Input(shape=self.input_shape),
                layers.Flatten(),
                layers.Dense(256, activation="relu"),
                DynamicDropout(self.dropout_var),
                layers.Dense(128, activation="relu"),
                DynamicDropout(self.dropout_var),
                layers.Dense(64, activation="relu"),
                DynamicDropout(self.dropout_var),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

    def _create_convolutional_model(self) -> models.Sequential:
        """Create a simple CNN model with dropout.

        Args:
            dropout_rate (float): Dropout rate to use in Dropout layers.

        Returns:
            models.Sequential: Compiled Keras model.
        """
        return models.Sequential(
            [
                layers.Input(shape=self.input_shape),
                layers.Conv2D(32, 3, activation="relu", padding="same"),
                DynamicDropout(self.dropout_var),
                layers.Conv2D(32, 3, activation="relu", padding="same"),
                layers.MaxPooling2D(2),
                DynamicDropout(self.dropout_var),
                layers.Conv2D(64, 3, activation="relu", padding="same"),
                DynamicDropout(self.dropout_var),
                layers.Flatten(),
                layers.Dense(64, activation="relu"),
                DynamicDropout(self.dropout_var),
                layers.Dense(self.num_classes, activation="softmax"),
            ]
        )

    def train(
        self,
        mode: str,
        switch_epoch: int,
        rate: float,
        epochs: int,
        batch_size: int = 64,
        verbose: int = 0,
    ) -> keras.callbacks.History:
        """Train the model with specified dropout scheduling.

        Args:
            mode (str): 'standard', 'early', or 'late'.
            switch_epoch (int): Epoch to switch dropout behavior.
            rate (float): Dropout rate to use when active.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            verbose (int): Verbosity level for training output.

        Returns:
            keras.callbacks.History: Training history object.
        """
        keras.backend.clear_session()

        self.dropout_var.assign(0.0)

        model = self.model_factory()
        model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        scheduler = DropoutScheduler(
            rate_variable=self.dropout_var,
            mode=mode,
            switch_epoch=switch_epoch,
            active_rate=rate,
            verbose=verbose,
        )

        return model.fit(
            self.X_train,
            self.y_train,
            validation_data=(self.X_val, self.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[scheduler],
            verbose=verbose,
        )

    def compare_learning_curves(
        self,
        modes: list = ["standard", "early", "late"],
        switch_epoch: int = 10,
        rate: float = 0.3,
        epochs: int = 20,
    ) -> None:
        """Train multiple modes and plot them on the same graph.

        Args:
            modes (list): List of modes to compare.
            switch_epoch (int): Epoch to switch dropout behavior.
            rate (float): Dropout rate to use when active.
            epochs (int): Number of training epochs.
        """
        colors = sns.color_palette("magma", n_colors=len(modes))
        histories = {}
        print(
            f"\n📊 Comparing Learning Curves: {modes} (Switch={switch_epoch}, Rate={rate})"
        )

        for mode in modes:
            print(f"   Running {mode}...")
            histories[mode] = self.train(
                mode=mode, switch_epoch=switch_epoch, rate=rate, epochs=epochs
            )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        for mode, h in histories.items():
            color = colors[modes.index(mode)]
            ax1.plot(
                h.history["val_accuracy"],
                label=f"{mode} (Val)",
                linewidth=2,
                color=color,
            )
            ax1.plot(
                h.history["accuracy"],
                label=f"{mode} (Train)",
                linestyle="--",
                alpha=0.4,
                color=color,
            )

        ax1.axvline(x=switch_epoch, color="gray", linestyle=":", label="Switch")
        ax1.set_title("Validation Accuracy vs Epochs")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for mode, h in histories.items():
            color = colors[modes.index(mode)]
            ax2.plot(
                h.history["val_loss"], label=f"{mode} (Val)", linewidth=2, color=color
            )
            ax2.plot(
                h.history["loss"],
                label=f"{mode} (Train)",
                linestyle="--",
                alpha=0.4,
                color=color,
            )

        ax2.axvline(x=switch_epoch, color="gray", linestyle=":", label="Switch")
        ax2.set_title("Validation Loss vs Epochs")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_drop_rates(
        self, rates: list, modes: list, switch_epoch: int, epochs: int
    ) -> None:
        """Ablation: Accuracy vs Drop Rate (Train & Val gaps).

        Args:
            rates (list): List of dropout rates to test.
            modes (list): List of modes to compare.
            switch_epoch (int): Epoch to switch dropout behavior.
            epochs (int): Number of training epochs.
        """
        print("\n📊 Comparing Dropout Rates Impact (Accuracy & Loss)")
        colors = sns.color_palette("magma", n_colors=len(modes))

        res_acc = {mode: [] for mode in modes}
        res_loss = {mode: [] for mode in modes}

        for rate in rates:
            for mode in modes:
                print(f"   Running {mode} with rate={rate}...")
                h = self.train(
                    mode=mode, switch_epoch=switch_epoch, rate=rate, epochs=epochs
                )

                final_acc = np.mean(h.history["val_accuracy"][-3:])
                final_loss = np.mean(h.history["val_loss"][-3:])

                res_acc[mode].append(final_acc)
                res_loss[mode].append(final_loss)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        for mode in modes:
            ax1.plot(
                rates,
                res_acc[mode],
                marker="o",
                label=f"{mode}",
                color=colors[modes.index(mode)],
                linewidth=2,
            )
        ax1.set_title(
            f"Final Validation Accuracy vs Drop Rate\n(Switch={switch_epoch})"
        )
        ax1.set_xlabel("Dropout Rate")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for mode in modes:
            ax2.plot(
                rates,
                res_loss[mode],
                marker="s",
                label=f"{mode}",
                color=colors[modes.index(mode)],
                linewidth=2,
                linestyle="--",
            )
        ax2.set_title(f"Final Validation Loss vs Drop Rate\n(Switch={switch_epoch})")
        ax2.set_xlabel("Dropout Rate")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    def compare_switch_epochs(
        self, switch_epochs: list, modes: list, rate: float, epochs: int
    ) -> None:
        print("\n📊 Comparing Switch Epochs Impact (Accuracy & Loss)")
        colors = sns.color_palette("magma", n_colors=len(modes))

        res_acc = {mode: [] for mode in modes}
        res_loss = {mode: [] for mode in modes}

        for s in switch_epochs:
            for mode in modes:
                print(f"   Running {mode} with switch_epoch={s}...")
                h = self.train(mode=mode, switch_epoch=s, rate=rate, epochs=epochs)

                final_acc = np.mean(h.history["val_accuracy"][-3:])
                final_loss = np.mean(h.history["val_loss"][-3:])

                res_acc[mode].append(final_acc)
                res_loss[mode].append(final_loss)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        for mode in modes:
            ax1.plot(
                switch_epochs,
                res_acc[mode],
                marker="o",
                label=mode,
                color=colors[modes.index(mode)],
                linewidth=2,
            )
        ax1.set_title(f"Final Val Accuracy vs Switch Epoch\n(Rate={rate})")
        ax1.set_xlabel("Switch Epoch")
        ax1.set_ylabel("Accuracy")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        for mode in modes:
            ax2.plot(
                switch_epochs,
                res_loss[mode],
                marker="s",
                label=mode,
                color=colors[modes.index(mode)],
                linewidth=2,
                linestyle="--",
            )
        ax2.set_title(f"Final Val Loss vs Switch Epoch\n(Rate={rate})")
        ax2.set_xlabel("Switch Epoch")
        ax2.set_ylabel("Loss")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

def run_dataset_size_comparison(
    dataset_name: str,
    model_type: str,
    fractions=[0.1, 0.5, 1.0],
    modes=["standard", "early"],
    rate: float = 0.3,
    switch_epoch: int =10,
    epochs: int =20,
) -> None:
    """
    Compare Early vs Standard dropout on variable dataset sizes.
    
    Args:
        dataset_name (str): Dataset to use ('mnist', 'cifar10', 'cifar100').
        model_type (str): Model type ('dense' or 'cnn').
        fractions (list): List of dataset size fractions to test.
        modes (list): List of dropout modes to compare.
        rate (float): Dropout rate to use when active.
        switch_epoch (int): Epoch to switch dropout behavior.
        epochs (int): Number of training epochs.
    """
    print("\n📊 Comparing Dataset Sizes Impact (Accuracy & Loss)")
    results_acc = {mode: [] for mode in modes}
    results_loss = {mode: [] for mode in modes}

    for frac in fractions:
        print(f"\n>> Testing with {frac * 100}% of data...")

        pipe = ExperimentPipeline(dataset_name, model_type, subset_fraction=frac)

        for mode in modes:
            print(f"   Running {mode}...")
            hist = pipe.train(
                mode=mode,
                switch_epoch=switch_epoch,
                rate=rate,
                epochs=epochs,
                verbose=0,
            )

            final_acc = np.mean(hist.history["val_accuracy"][-3:])
            final_loss = np.mean(hist.history["val_loss"][-3:])

            results_acc[mode].append(final_acc)
            results_loss[mode].append(final_loss)

            print(f"   -> {mode}: Acc={final_acc:.4f} | Loss={final_loss:.4f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    colors = sns.color_palette("magma", n_colors=len(modes))
    markers = {"standard": "s", "early": "o", "late": "^", "none": "x"}

    for mode, accs in results_acc.items():
        color = colors[modes.index(mode)]
        marker = markers.get(mode, "o")
        ax1.plot(
            fractions,
            accs,
            marker=marker,
            label=f"{mode}",
            linewidth=2,
            markersize=8,
            color=color,
        )

    ax1.set_title(
        f"Validation Accuracy vs Dataset Size\n({dataset_name} - {model_type})"
    )
    ax1.set_xlabel("Fraction of Training Data")
    ax1.set_ylabel("Final Accuracy")
    ax1.set_xticks(fractions)
    ax1.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    for mode, losses in results_loss.items():
        color = colors[modes.index(mode)]
        marker = markers.get(mode, "o")
        ax2.plot(
            fractions,
            losses,
            marker=marker,
            label=f"{mode}",
            linewidth=2,
            markersize=8,
            color=color,
            linestyle="--",
        )

    ax2.set_title(f"Validation Loss vs Dataset Size\n({dataset_name} - {model_type})")
    ax2.set_xlabel("Fraction of Training Data")
    ax2.set_ylabel("Final Loss")
    ax2.set_xticks(fractions)
    ax2.set_xticklabels([f"{int(f * 100)}%" for f in fractions])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()