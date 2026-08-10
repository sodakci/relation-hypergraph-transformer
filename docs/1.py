
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter
from pathlib import Path
from statistics import mean, median

import torch

from isolation_gnn import (
    DecodeStatus,
    LabeledVeriStrongHistoryDataset,
    decode_with_beam,
    discover_dbcop_histories,
    load_veristrong_checkpoint,
    scores_from_outputs,
    validate_witness,
    veristrong_decision_loss,
    VeriStrongDecisionNetwork
)

model = VeriStrongDecisionNetwork(
    hidden_dim=128,
    processor_steps=9,
    aggregation="mean_sum",
)

print(model.aggregation)
print(model.processor.aggregation)

print(
    hasattr(
        model.processor,
        "relation_embeddings",
    )
)

print(
    hasattr(
        model.processor,
        "relation_gate",
    )
)