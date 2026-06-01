"""TDC activity oracle for JNK3/GSK3B.

[[component.TDCActivity.endpoint]]
name = "TDC Activity Score"
weight = 0.5

param.target = "jnk3"
param.cwd = "."
"""

from __future__ import annotations

__all__ = ["TDCActivity"]

import os
import sys
import logging
from typing import List

import numpy as np

from .component_results import ComponentResults
from .add_tag import add_tag
from pydantic.dataclasses import dataclass

logger = logging.getLogger("reinvent")


@add_tag("__parameters")
@dataclass
class Parameters:
    target: List[str]
    cwd: List[str]


@add_tag("__component")
class TDCActivity:
    def __init__(self, params: Parameters):
        target = params.target[0]
        cwd = params.cwd[0] if params.cwd else "."

        # TDC expects oracle/ model files relative to CWD
        original_cwd = os.getcwd()
        try:
            os.chdir(cwd)
            from tdc import Oracle
            self.oracle = Oracle(name=target)
        finally:
            os.chdir(original_cwd)

        self.target = target
        logger.info(f"TDCActivity initialized for target={target}")

    def __call__(self, smilies: List[str]) -> np.array:
        scores = []
        for smiles in smilies:
            try:
                score = self.oracle(smiles)
                scores.append(score)
            except Exception:
                scores.append(0.0)
        return ComponentResults([np.array(scores)])
