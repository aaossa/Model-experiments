import copy
import os
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm.notebook import tqdm

from models.utils import get_cpu_copy, save_checkpoint


def train_with_dataloaders(
    model, device, criterion, optimizer, scheduler, dataloaders,
    max_epochs=1, max_lrs=float("inf"), train_per_valid_times=1,
    model_version=None, checkpoint_dir=None, writer_dir=None,
    non_blocking=True, save_last=False,
):
    # Initialization
    model = model.to(device)
    start = time.time()

    # Model version
    if not model_version:
        model_version = (
            f"{model.__class__.__name__}_"
            f"{time.strftime('%Y-%m-%d-%H-%M-%S')}"
        )

    # Checkpoints
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join("checkpoints")
    assert os.path.isdir(checkpoint_dir)
    checkpoint_dst = os.path.join(checkpoint_dir, f"{model_version}.tar")
    print(f">> Checkpoints stored at... {checkpoint_dst}")

    # Save first checkpoint
    save_checkpoint(
        checkpoint_dst, model=model,
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        epoch=0, accuracy=None,
    )

    # Summary writer
    if writer_dir is None:
        writer_dir = os.path.join("runs")
    assert os.path.isdir(writer_dir)
    writer_dst = os.path.join(writer_dir, model_version)
    print(f">> Summary writer data stored at... {writer_dst}")

    # Create Summary writer
    summary_writer = SummaryWriter(writer_dst, flush_secs=20)

    # Training loop setup
    best_validation_acc = 0.0
    used_lrs = [optimizer.param_groups[0]["lr"]]

    # Progress bars
    pbar_epochs = tqdm(
        total=max_epochs,
        desc="Epoch", unit="epoch",
        postfix={
            "used_lrs": len(used_lrs),
            "current_lr": used_lrs[0],
        },
    )
    pbar_train = tqdm(
        total=train_per_valid_times*len(dataloaders["train"]),
        desc="Train",
        postfix={
            "last_acc": None,
            "last_loss": None,
        },
    )
    pbar_valid = tqdm(
        total=len(dataloaders["validation"]),
        desc="Valid",
        postfix={
            "best_acc": None,
            "best_loss": None,
            "best_epoch": None,
            "bad_epochs": f"{scheduler.num_bad_epochs}",
        },
    )

    # Training loop
    for epoch in range(1, max_epochs + 1):

        # Each epoch has a training and a validation phase
        for phase in ["train", "validation"]:
            # Update model mode and progress bar
            if phase == "train":
                model.train()
                pbar_train.reset()
                pbar_valid.reset()
            elif phase == "validation":
                model.eval()

            # Value accumulators
            running_acc = torch.tensor(0, dtype=int, device=device)
            running_loss = torch.tensor(0.0, dtype=torch.double, device=device)

            # Iterate over data
            dataset = dataloaders[phase].dataset
            loop_times = train_per_valid_times if phase == "train" else 1
            for _ in range(loop_times):
                for i_batch, data in enumerate(dataloaders[phase]):
                    # Load data into tensors
                    profile = data[0].to(device, non_blocking=non_blocking).squeeze(dim=0)
                    pi = data[1].to(device, non_blocking=non_blocking).squeeze(dim=0)
                    ni = data[2].to(device, non_blocking=non_blocking).squeeze(dim=0)
                    target = torch.ones(pi.size(0), 1, 1, device=device)

                    # Restart params gradients
                    optimizer.zero_grad()

                    # Forward pass
                    with torch.set_grad_enabled(phase == "train"):
                        output = model(profile, pi, ni)
                        loss = criterion(output, target)
                        # Backward pass
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # Statistics
                    running_acc.add_((output > 0).sum())
                    running_loss.add_(loss.detach() * output.size(0))

                    # Update progress bar
                    if phase == "train":
                        pbar_train.update()
                    else:
                        pbar_valid.update()

                    # Synchronize GPU (debugging)
                    # torch.cuda.synchronize()

            # Aggregate statistics
            dataset_size = loop_times * len(dataset)
            epoch_acc = running_acc.item() / dataset_size
            epoch_loss = running_loss.item() / dataset_size
            # tqdm.write(f">> Epoch {epoch} ({phase.title()}) | ACC {100 * epoch_acc:.3f} - Loss {epoch_loss:.6f}")

            if phase == "train":
                # Update progress bar
                pbar_train.set_postfix({
                    "last_acc": f"{100 * epoch_acc:.3f}",
                    "last_loss": f"{epoch_loss:.6f}",
                })
            elif phase == "validation":
                if epoch_acc > best_validation_acc:
                    # Save best model
                    best_validation_acc = epoch_acc
                    best_validation_loss = epoch_loss
                    save_checkpoint(
                        checkpoint_dst, model=get_cpu_copy(model),
                        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
                        epoch=scheduler.last_epoch, accuracy=best_validation_acc,
                    )
                    # tqdm.write(f">> New best model (Epoch: {epoch}) | ACC {100 * epoch_acc:.3f} ({epoch_acc})")
                # Scheduler step
                scheduler.step(epoch_acc)
                next_lr = optimizer.param_groups[0]["lr"]
                if next_lr not in used_lrs:
                    # tqdm.write(f">> Next lr: {next_lr} (Already used {used_lrs})")
                    used_lrs.append(next_lr)
                    pbar_epochs.set_postfix({
                        "used_lrs": len(used_lrs),
                        "current_lr": next_lr,
                    })
                # Update progress bar
                pbar_valid.set_postfix({
                    "best_acc": f"{100 * best_validation_acc:.3f}",
                    "best_loss": f"{best_validation_loss:.6f}",
                    "best_epoch": f"{epoch}",
                    "bad_epochs": f"{scheduler.num_bad_epochs}",
                })

            # Memory report
            # memory_report()

            # Write to SummaryWriter
            summary_writer.add_scalar(f"{phase} loss", epoch_loss, epoch)
            summary_writer.add_scalar(f"{phase} accuracy", epoch_acc, epoch)
            summary_writer.flush()

        # Update epochs pbar at the end
        pbar_epochs.update()
        # tqdm.write("\n")

        # Check if used all available learning rates
        if len(used_lrs) > max_lrs:
            print(f">> Reached max different lrs ({used_lrs[:-1]})")
            break

    # Close SummaryWriter
    summary_writer.close()

    # Complete progress bars
    pbar_epochs.close()
    pbar_train.close()
    pbar_valid.close()

    # Report status
    elapsed = time.time() - start
    print(f">> Training completed in {elapsed // 60:.0f}m {elapsed % 60:.0f}s")
    print(f">> Best validation accuracy: ~{100 * best_validation_acc:.3f}%")

    if save_last:
        # Copy last model weights
        print(">> Copy last model")
        last_model_weights = copy.deepcopy(get_cpu_copy(model))
    else:
        epoch_acc = None
        last_model_weights = None

    # Load best model weights
    print(">> Load best model")
    checkpoint = torch.load(checkpoint_dst, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model"])

    # Move model back to device
    model.to(device)

    # Save last state
    print(">> Save last state")
    save_checkpoint(
        checkpoint_dst, model=get_cpu_copy(model),
        criterion=criterion, optimizer=optimizer, scheduler=scheduler,
        epoch=scheduler.last_epoch, accuracy=best_validation_acc,
        last_model=last_model_weights, last_accuracy=epoch_acc,
    )

    return model, checkpoint["accuracy"], checkpoint["epoch"]
