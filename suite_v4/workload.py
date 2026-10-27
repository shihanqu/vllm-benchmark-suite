"""
Synthetic workload generation, following LLM-Inference-Bench.

LLM-Inference-Bench feeds every request `input_len` random token ids and forces exactly
`output_len` generated tokens (ignore_eos=True), so every framework does identical work.
v4 does the same against an OpenAI-compatible endpoint by sending token-id prompts to
/v1/completions. Prompts are re-drawn for every iteration (seeded, so runs are reproducible):
with prefix caching on by default in current vLLM, re-sending the same batch would skip prefill.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np

# Plain, common English words for text prompts (chat endpoints or servers without token-id prompts)
_WORDS = (
    "the of and to in is was for on are with as his they at be this from have or by one had "
    "not but what all were when we there can an your which their said if do will each about how "
    "up out them then she many some so these would other into has more her two like him see time "
    "could no make than first been its who now people my made over did down only way find use may "
    "water long little very after words called just where most know get through back much before "
    "go good new write our used me man too any day same right look think also around another came "
    "come work three word must because does part even place well such here take why things help "
    "put years different away again off went old number great tell men say small every found still "
    "between name should home big give air line set own under read last never us left end along "
    "while might next sound below saw something thought both few those always looked show large "
    "often together asked house world going want school important until form food keep children "
    "feet land side without boy once animals life enough took sometimes four head above kind began "
    "almost live page got earth need far hand high year mother light parts country father let night "
    "following picture being study second eyes soon times story boys since white days ever paper "
    "hard near sentence better best across during today others however sure means knew river mountain"
).split()

# Typical BPE tokenizers spend ~1.3 tokens per common English word plus its leading space.
DEFAULT_TOKENS_PER_WORD = 1.3

# Phase codes keep warmup, measured and probe prompts independent of each other.
PHASE_WARMUP = 0
PHASE_MEASURE = 1
PHASE_PROBE = 2


class PromptFactory:
    """Reproducible synthetic prompts keyed by (shape, batch size, phase, iteration)."""

    def __init__(
        self,
        seed: int = 0,
        token_id_range: Sequence[int] = (1000, 10000),
        tokens_per_word: Optional[float] = None,
    ):
        self.seed = int(seed)
        self.low, self.high = int(token_id_range[0]), int(token_id_range[1])
        self.tokens_per_word = tokens_per_word or DEFAULT_TOKENS_PER_WORD

    def _rng(self, input_len: int, output_len: int, batch_size: int, phase: int, iteration: int):
        return np.random.default_rng([self.seed, input_len, output_len, batch_size, phase, iteration])

    def token_prompts(
        self, input_len: int, output_len: int, batch_size: int, phase: int, iteration: int
    ) -> List[np.ndarray]:
        """batch_size prompts of exactly input_len random token ids (LLM-Inference-Bench dummy inputs).

        Rows stay numpy arrays until a request is built: a 64 x 256K batch as Python lists
        would cost gigabytes.
        """
        rng = self._rng(input_len, output_len, batch_size, phase, iteration)
        return list(rng.integers(self.low, self.high, size=(batch_size, input_len), dtype=np.int32))

    def text_prompts(
        self, input_len: int, output_len: int, batch_size: int, phase: int, iteration: int
    ) -> List[str]:
        """batch_size prompts of random common words sized to roughly input_len tokens."""
        rng = self._rng(input_len, output_len, batch_size, phase, iteration)
        n_words = max(1, int(round(input_len / self.tokens_per_word)))
        idx = rng.integers(0, len(_WORDS), size=(batch_size, n_words))
        return [" ".join(_WORDS[i] for i in row) for row in idx]

    @staticmethod
    def calibration_text(n_words: int = 2000) -> str:
        rng = np.random.default_rng(12345)
        return " ".join(_WORDS[i] for i in rng.integers(0, len(_WORDS), size=n_words))
