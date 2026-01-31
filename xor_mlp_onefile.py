import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


# ---------- aktywacja ----------
def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))

def sigmoid_deriv_from_activation(a: np.ndarray) -> np.ndarray:
    return a * (1.0 - a)


# ---------- sieć ----------
class MLP:
    """
    2-warstwowy MLP: in -> hidden -> out (sigmoid w obu warstwach)
    Uczenie: backprop dla MSE.
    """
    def __init__(self, n_in: int, n_hidden: int, n_out: int, seed: int = 42):
        rng = np.random.default_rng(seed)

        # Xavier/Glorot init
        limit1 = np.sqrt(6 / (n_in + n_hidden))
        limit2 = np.sqrt(6 / (n_hidden + n_out))

        self.W1 = rng.uniform(-limit1, limit1, size=(n_hidden, n_in))
        self.b1 = np.zeros((n_hidden, 1))
        self.W2 = rng.uniform(-limit2, limit2, size=(n_out, n_hidden))
        self.b2 = np.zeros((n_out, 1))

        # momentum buffers
        self.vW1 = np.zeros_like(self.W1)
        self.vb1 = np.zeros_like(self.b1)
        self.vW2 = np.zeros_like(self.W2)
        self.vb2 = np.zeros_like(self.b2)

    def forward(self, X: np.ndarray) -> dict:
        Z1 = self.W1 @ X + self.b1
        A1 = sigmoid(Z1)
        Z2 = self.W2 @ A1 + self.b2
        A2 = sigmoid(Z2)
        return {"X": X, "Z1": Z1, "A1": A1, "Z2": Z2, "A2": A2}

    @staticmethod
    def mse(y_hat: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((y_hat - y) ** 2))

    def backward(self, cache: dict, y: np.ndarray) -> dict:
        X  = cache["X"]
        A1 = cache["A1"]
        A2 = cache["A2"]
        batch = X.shape[1]

        # L = mean((A2 - y)^2) -> dL/dA2 = 2*(A2-y)/N, N = liczba elementów
        dA2 = (2.0 / A2.size) * (A2 - y)
        dZ2 = dA2 * sigmoid_deriv_from_activation(A2)  # (n_out, batch)

        dW2 = (dZ2 @ A1.T) / batch
        db2 = np.mean(dZ2, axis=1, keepdims=True)

        dA1 = self.W2.T @ dZ2
        dZ1 = dA1 * sigmoid_deriv_from_activation(A1)  # (n_hidden, batch)

        dW1 = (dZ1 @ X.T) / batch
        db1 = np.mean(dZ1, axis=1, keepdims=True)

        return {
            "dW1": dW1, "db1": db1,
            "dW2": dW2, "db2": db2,
            "delta1": dZ1,
            "delta2": dZ2,
        }

    def step(self, grads: dict, lr: float, momentum: float = 0.0):
        mu = float(momentum)
        self.vW1 = mu * self.vW1 - lr * grads["dW1"]
        self.vb1 = mu * self.vb1 - lr * grads["db1"]
        self.vW2 = mu * self.vW2 - lr * grads["dW2"]
        self.vb2 = mu * self.vb2 - lr * grads["db2"]

        self.W1 += self.vW1
        self.b1 += self.vb1
        self.W2 += self.vW2
        self.b2 += self.vb2

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)["A2"]

    def predict_label(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


# ---------- dane XOR ----------
def make_xor():
    X = np.array([[0, 0, 1, 1],
                  [0, 1, 0, 1]], dtype=float)  # (2,4)
    y = np.array([[0, 1, 1, 0]], dtype=float)  # (1,4)
    return X, y

def classification_error(y_label: np.ndarray, y_true: np.ndarray) -> float:
    return float(1.0 - np.mean(y_label == y_true))

def iterate_minibatches(X, y, batch_size, rng):
    n = X.shape[1]
    idx = np.arange(n)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        b = idx[start:start + batch_size]
        yield X[:, b], y[:, b]

def snapshot(net: MLP):
    return (net.W1.copy(), net.b1.copy(), net.W2.copy(), net.b2.copy(),
            net.vW1.copy(), net.vb1.copy(), net.vW2.copy(), net.vb2.copy())

def restore(net: MLP, snap):
    net.W1, net.b1, net.W2, net.b2, net.vW1, net.vb1, net.vW2, net.vb2 = snap


# ---------- pomoc: epoka przełomu ----------
def find_first_epoch_below(epoch_arr: np.ndarray, values: list, threshold: float):
    """Zwraca numer epoki, gdy wartości pierwszy raz <= threshold, inaczej None."""
    v = np.array(values, dtype=float)
    idx = np.where(v <= threshold)[0]
    if idx.size == 0:
        return None
    return int(epoch_arr[idx[0]])


# ---------- wykresy ----------
def save_plots(outdir, logs, mark_epoch=None):
    os.makedirs(outdir, exist_ok=True)
    epoch = np.array(logs["epoch"])

    def maybe_mark():
        if mark_epoch is not None:
            plt.axvline(mark_epoch, linestyle="--", linewidth=1.0, alpha=0.7, label=f"milestone @ {mark_epoch}")

    # MSE + delta^2 w warstwach (diagnostyka) - log skala, żeby delta była widoczna
    plt.figure()
    plt.plot(epoch, logs["mse"], label="MSE (output)")
    plt.plot(epoch, logs["delta1_mse"], label="mean(delta1^2) (hidden)")
    plt.plot(epoch, logs["delta2_mse"], label="mean(delta2^2) (output)")
    maybe_mark()
    plt.xlabel("Epoch"); plt.ylabel("Value (log)")
    plt.title("MSE and layer error signals (log scale)")
    plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "mse_and_layer_errors.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # Błąd klasyfikacji
    plt.figure()
    plt.plot(epoch, logs["clf_err"], label="Classification error")
    maybe_mark()
    plt.xlabel("Epoch"); plt.ylabel("Error")
    plt.title("Classification error (threshold)")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "classification_error.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # LR
    plt.figure()
    plt.plot(epoch, logs["lr"], label="lr")
    maybe_mark()
    plt.xlabel("Epoch"); plt.ylabel("Learning rate")
    plt.title("Learning rate over epochs")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "learning_rate.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # Wagi warstwy 1
    W1_hist = np.stack(logs["W1_flat"], axis=0)  # (E, n_w1)
    plt.figure()
    for i in range(W1_hist.shape[1]):
        plt.plot(epoch, W1_hist[:, i], label=f"W1[{i}]")
    maybe_mark()
    plt.xlabel("Epoch"); plt.ylabel("Weight")
    plt.title("Weights: layer 1 (input->hidden)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "weights_layer1.png"), dpi=170, bbox_inches="tight")
    plt.close()

    # Wagi warstwy 2
    W2_hist = np.stack(logs["W2_flat"], axis=0)  # (E, n_w2)
    plt.figure()
    for i in range(W2_hist.shape[1]):
        plt.plot(epoch, W2_hist[:, i], label=f"W2[{i}]")
    maybe_mark()
    plt.xlabel("Epoch"); plt.ylabel("Weight")
    plt.title("Weights: layer 2 (hidden->output)")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "weights_layer2.png"), dpi=170, bbox_inches="tight")
    plt.close()

    print("Saved plots to:", outdir)
    for f in ["mse_and_layer_errors.png", "classification_error.png", "learning_rate.png",
              "weights_layer1.png", "weights_layer2.png"]:
        print(" -", f)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5000)
    ap.add_argument("--lr", type=float, default=1.0)
    ap.add_argument("--momentum", type=float, default=0.0)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--threshold", type=float, default=0.5)

    # early stopping
    ap.add_argument("--early_stop", action="store_true")
    ap.add_argument("--target_mse", type=float, default=1e-3)
    ap.add_argument("--target_clf_err", type=float, default=0.0)

    # adaptive LR (bold driver)
    ap.add_argument("--adaptive_lr", action="store_true")
    ap.add_argument("--lr_inc", type=float, default=1.02)
    ap.add_argument("--lr_dec", type=float, default=0.7)
    ap.add_argument("--lr_min", type=float, default=1e-5)
    ap.add_argument("--lr_max", type=float, default=10.0)
    ap.add_argument("--rollback_on_worse", action="store_true")

    ap.add_argument("--outdir", type=str, default="outputs")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    X, y = make_xor()
    net = MLP(n_in=2, n_hidden=args.hidden, n_out=1, seed=args.seed)
    rng = np.random.default_rng(args.seed)

    logs = {
        "epoch": [], "lr": [],
        "mse": [], "clf_err": [],
        "delta1_mse": [], "delta2_mse": [],
        "W1_flat": [], "W2_flat": [],
    }

    lr = float(args.lr)
    prev_mse = None

    for epoch in range(1, args.epochs + 1):
        snap = snapshot(net) if (args.adaptive_lr and args.rollback_on_worse) else None

        # trening mini-batch
        for Xb, yb in iterate_minibatches(X, y, args.batch_size, rng):
            cache = net.forward(Xb)
            grads = net.backward(cache, yb)
            net.step(grads, lr=lr, momentum=args.momentum)

        # ewaluacja na całym zbiorze
        cache_full = net.forward(X)
        y_hat = cache_full["A2"]
        mse = net.mse(y_hat, y)
        y_label = (y_hat >= args.threshold).astype(int)
        clf_err = classification_error(y_label, y.astype(int))

        grads_full = net.backward(cache_full, y)
        delta1_mse = float(np.mean(grads_full["delta1"] ** 2))
        delta2_mse = float(np.mean(grads_full["delta2"] ** 2))

        # adaptacyjny LR (bold driver)
        if args.adaptive_lr and prev_mse is not None:
            if mse > prev_mse:
                lr = max(args.lr_min, lr * args.lr_dec)
                if args.rollback_on_worse and snap is not None:
                    restore(net, snap)
                    # policz metryki po rollback
                    cache_full = net.forward(X)
                    y_hat = cache_full["A2"]
                    mse = net.mse(y_hat, y)
                    y_label = (y_hat >= args.threshold).astype(int)
                    clf_err = classification_error(y_label, y.astype(int))
                    grads_full = net.backward(cache_full, y)
                    delta1_mse = float(np.mean(grads_full["delta1"] ** 2))
                    delta2_mse = float(np.mean(grads_full["delta2"] ** 2))
            else:
                lr = min(args.lr_max, lr * args.lr_inc)

        prev_mse = mse

        # logi
        logs["epoch"].append(epoch)
        logs["lr"].append(lr)
        logs["mse"].append(mse)
        logs["clf_err"].append(clf_err)
        logs["delta1_mse"].append(delta1_mse)
        logs["delta2_mse"].append(delta2_mse)
        logs["W1_flat"].append(net.W1.reshape(-1).copy())
        logs["W2_flat"].append(net.W2.reshape(-1).copy())

        if epoch == 1 or epoch % 200 == 0:
            print(f"epoch={epoch:5d} mse={mse:.6f} clf_err={clf_err:.3f} lr={lr:.6f}")

        # early stopping
        if args.early_stop and (mse <= args.target_mse) and (clf_err <= args.target_clf_err):
            print(f"[STOP] epoch={epoch} mse={mse:.6f} clf_err={clf_err:.3f} lr={lr:.6f}")
            break

    # zapisz logi do npz (przyda się, jak chcesz potem dorobić inne wykresy)
    np.savez(
        os.path.join(args.outdir, "logs.npz"),
        epoch=np.array(logs["epoch"]),
        lr=np.array(logs["lr"]),
        mse=np.array(logs["mse"]),
        clf_err=np.array(logs["clf_err"]),
        delta1_mse=np.array(logs["delta1_mse"]),
        delta2_mse=np.array(logs["delta2_mse"]),
        W1_hist=np.stack(logs["W1_flat"], axis=0),
        W2_hist=np.stack(logs["W2_flat"], axis=0),
    )

    # predykcje końcowe
    y_hat = net.predict_proba(X)
    print("\nXOR predictions (proba):")
    for i in range(X.shape[1]):
        print(f"X={X[:, i]}  y={int(y[0, i])}  y_hat={float(y_hat[0, i]):.4f}")

    # epoka przełomu: kiedy clf_err osiąga target (domyślnie 0.0)
    epoch_arr = np.array(logs["epoch"])
    milestone = find_first_epoch_below(epoch_arr, logs["clf_err"], threshold=args.target_clf_err)

    if milestone is not None:
        print(f"\nMilestone: clf_err <= {args.target_clf_err} at epoch {milestone}")
    else:
        print(f"\nMilestone: clf_err never went <= {args.target_clf_err} within epochs")

    save_plots(args.outdir, logs, mark_epoch=milestone)
    print("\nDone.")

if __name__ == "__main__":
    main()
