import torch


class _ValidationLoop:
    def run(self, model, val_segment):
        model.eval()
        with torch.no_grad():
            for batch in val_segment:
                model.validation_step(batch)

    @property
    def done(self) -> bool:
        return NotImplementedError
