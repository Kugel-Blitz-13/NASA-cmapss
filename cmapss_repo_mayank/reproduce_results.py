
import argparse
import json
from pathlib import Path

import torch
from train import (
    SequenceWindowDataset,
    evaluate_regression,
    evaluate_test_last_window,
    make_model,
    maybe_download_dataset,
    plot_learning_curves,
    plot_scatter,
    prepare_data,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/CMAPSS")
    parser.add_argument("--val_frac", type=float, default=0.2)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu")

    maybe_download_dataset(Path(args.data_dir))
    prepared = prepare_data(Path(args.data_dir), max_rul=checkpoint["max_rul"], val_frac=args.val_frac)

    val_ds = SequenceWindowDataset(prepared.val_df, prepared.feature_cols, window_size=checkpoint["window_size"], train_mode=True)
    test_ds = SequenceWindowDataset(prepared.test_df, prepared.feature_cols, window_size=checkpoint["window_size"], train_mode=False)

    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=256, shuffle=False)

    model = make_model(checkpoint["model_name"], input_dim=len(prepared.feature_cols))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    val_metrics = evaluate_regression(model, val_loader, device="cpu")
    test_metrics = evaluate_test_last_window(model, test_ds, prepared.test_rul, device="cpu")

    history_path = run_dir / "history.json"
    if history_path.exists():
        history = json.loads(history_path.read_text())
        plot_learning_curves(history, run_dir / "learning_curve_reproduced.png")

    plot_scatter(test_metrics["y_true"], test_metrics["y_pred"], run_dir / "test_scatter_reproduced.png")

    out = {"val": val_metrics, "test": test_metrics}
    with open(run_dir / "reproduced_metrics.json", "w") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
