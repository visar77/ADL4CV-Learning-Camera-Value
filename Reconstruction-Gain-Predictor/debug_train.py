"""Run the CPU toy-CNN pipeline smoke test on a preprocessed dataset."""

from train import main

if __name__ == "__main__":
    main(
        defaults={
            "model": "toy",
            "mode": "overfit",
            "scheduler": "none",
            "early_stopping_patience": 0,
            "device": "cpu",
            "model_name": "psnr_toy_debug",
            "batch_size": 1,
            "image_size": 32,
            "num_samples": 10,
        }
    )
