"""A simple MLP model for binary classification."""

import lightning as L
import torch
import torchmetrics


class InputLayer(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedder = torch.nn.EmbeddingBag(
            num_embeddings=vocab_size,
            embedding_dim=hidden_size,
            padding_idx=0,
            mode="sum",
        )

    def _embed(self, ids, values, values_mask, event_mask=None):
        per_sample_weights = torch.where(
            values_mask, torch.nan_to_num(values, nan=0), torch.ones_like(values)
        )
        embedded = self.embedder(ids, per_sample_weights=per_sample_weights)
        if event_mask is not None:
            embedded = torch.where(
                event_mask.unsqueeze(-1).expand_as(embedded), embedded, torch.zeros_like(embedded)
            )
        return embedded

    def _3D_embed(self, ids, event_mask, values, values_mask):
        batch_size, seq_len, n_measurements = ids.shape

        ids = ids.reshape((batch_size * seq_len, n_measurements))
        event_mask = event_mask.reshape(
            (batch_size * seq_len,),
        )
        values = values.reshape((batch_size * seq_len, n_measurements))
        values_mask = values_mask.reshape((batch_size * seq_len, n_measurements))

        embed = self._embed(ids, values, values_mask, event_mask)
        return embed.reshape((batch_size, seq_len, self.hidden_size))

    def forward(
        self,
        static_ids,
        static_values,
        static_values_mask,
        dynamic_ids,
        dynamic_values,
        dynamic_values_mask,
        event_mask,
    ):
        static_embedding = self._embed(static_ids, static_values, static_values_mask)
        dynamic_embedding = self._3D_embed(dynamic_ids, event_mask, dynamic_values, dynamic_values_mask)

        return static_embedding, dynamic_embedding


class Model(torch.nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int):
        super().__init__()

        self.input_layer = InputLayer(vocab_size, hidden_size)
        self.layers = []
        for _ in range(n_layers):
            self.layers.append(torch.nn.Linear(hidden_size, hidden_size))
            self.layers.append(torch.nn.ReLU())

        self.layers.append(torch.nn.Linear(hidden_size, 1))

        self.layers = torch.nn.Sequential(*self.layers)
        self.criterion = torch.nn.BCEWithLogitsLoss()

    def forward(
        self,
        static_ids,
        static_values,
        static_values_mask,
        dynamic_ids,
        dynamic_values,
        dynamic_values_mask,
        event_mask,
        **labels,
    ):
        static_embedding, dynamic_embedding = self.input_layer(
            static_ids,
            static_values,
            static_values_mask,
            dynamic_ids,
            dynamic_values,
            dynamic_values_mask,
            event_mask,
        )

        assert len(labels) == 1
        label_name, label = next(iter(labels.items()))


        seq_embed = dynamic_embedding.sum(1) + static_embedding
        logit = self.layers(seq_embed).squeeze(-1)
        loss = self.criterion(logit, label.float())

        return (loss, logit, seq_embed)


class LightningModel(L.LightningModule):
    def __init__(self, vocab_size: int, hidden_size: int, n_layers: int, lr: float):
        super().__init__()

        self.model = Model(vocab_size, hidden_size, n_layers)
        self.lr = lr
        self.AUROC = torchmetrics.AUROC(task="binary", thresholds=50)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def _step(self, batch, prefix: str):
        loss, logit, seq_embed = self.model(*batch)
        self.log(f"{prefix}_loss", loss)
        self.AUROC(logit, batch[-1])
        self.log(f"{prefix}_AUROC", self.AUROC, on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "tuning")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "held_out")
