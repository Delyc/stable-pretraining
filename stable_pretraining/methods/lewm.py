
"""LeWM: Latent Embedding World Model."""

from typing import Any

import torch
from torch import nn

from stable_pretraining.methods.lewm_components.architecture import SIGReg


def lewm_forward(
    self,
    batch: dict[str, Any],
    stage: str,
) -> dict[str, torch.Tensor]:
    """Compute the LeWM prediction and regularization losses.

    Args:
        self: Stable Pretraining module containing the world model.
        batch: Dictionary containing image sequences and actions.
        stage: Training, validation, or test stage.

    Returns:
        Dictionary containing embeddings and training losses.

    Note:
        Both context and target embeddings receive gradients,
        matching the official LeWM implementation.
    """
    batch["action"] = torch.nan_to_num(batch["action"], nan=0.0)

    output = self.model.encode(batch)

    emb = output["emb"]
    act_emb = output["act_emb"]

    ctx_emb = emb[:, : self.history_size]
    ctx_act = act_emb[:, : self.history_size]

    tgt_emb = emb[:, self.num_preds :]
    pred_emb = self.model.predict(ctx_emb, ctx_act)

    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))

    output["loss"] = (
        output["pred_loss"]
        + self.sigreg_weight * output["sigreg_loss"]
    )

    output["embedding"] = emb

    # Log prediction, regularization, and total losses
    self.log_dict(
        {
            f"{stage}/pred_loss": output["pred_loss"].detach(),
            f"{stage}/sigreg_loss": output["sigreg_loss"].detach(),
            f"{stage}/loss": output["loss"].detach(),
        },
        on_step=True,
        sync_dist=True,
    )

    return output


class LeWM:
    """Create a Stable Pretraining module for LeWM."""

    def __new__(
        cls,
        model: nn.Module,
        history_size: int,
        num_preds: int,
        sigreg_weight: float = 0.1,
        sigreg: nn.Module | None = None,
        optim: dict | None = None,
    ):
        from stable_pretraining.module import Module

        return Module(
            forward=lewm_forward,
            model=model,
            sigreg=sigreg if sigreg is not None else SIGReg(),
            history_size=history_size,
            num_preds=num_preds,
            sigreg_weight=sigreg_weight,
            optim=optim,
        )
