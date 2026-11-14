"""MolmoAct2 with a 128-token static text length (prompt + discretised state + setup/control tokens).

Studio's MolmoAct2 pads the text to `tokenizer_max_length` (default 256) and has no init argument for it. Our
prompts use ~100 tokens, so ~158 of the 256 positions are padding that the LLM prefill and all 10 action-expert
steps pay for (XPS profile). Changing it only at export moves outputs by ~1 deg, so it must be trained in.

The override is applied where Studio builds the config (`_initialize_from_config`), so:
* training, validation and the saved checkpoint all use 128 (Studio stores the resolved config in the checkpoint);
* no new __init__ argument: the hyperparameters stay those of the stock class, so the checkpoint loads with
  `physicalai.policies.MolmoAct2.load_from_checkpoint` for export, which restores tokenizer_max_length=128.
"""

from __future__ import annotations

from dataclasses import replace

from physicalai.policies import MolmoAct2

TEXT_LEN = 128


class MolmoAct2Text128(MolmoAct2):
    """`physicalai.policies.MolmoAct2` with tokenizer_max_length fixed to TEXT_LEN."""

    def _initialize_from_config(self, config, **kwargs):  # noqa: ANN001, ANN003
        if getattr(config, "tokenizer_max_length", TEXT_LEN) != TEXT_LEN:
            config = replace(config, tokenizer_max_length=TEXT_LEN)
        return super()._initialize_from_config(config, **kwargs)
