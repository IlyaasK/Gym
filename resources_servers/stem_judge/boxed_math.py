# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run STEM's configured Math-Verify metric in a bounded worker process."""

import asyncio
import ctypes
import multiprocessing as mp
import os
import signal
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING

from math_verify import parse, verify
from math_verify.errors import TimeoutException
from math_verify.metric import math_metric
from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig
from sympy import And, Set

from nemo_gym.telemetry import MeteredSemaphore, resources_operation


if TYPE_CHECKING:
    from resources_servers.stem_judge.app import MathVerifyConfig


def _boxed_math(text: str) -> str:
    """Wrap a reference or reverse-comparison operand in the box expected by Math-Verify."""
    text = text.strip()
    if text.startswith(r"\(") and text.endswith(r"\)"):
        text = text[2:-2].strip()
    if len(text) > 1 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()
    return r"\boxed{" + text + "}"


def _reduce_chain(expr):
    if isinstance(expr, And):
        return And(*(link.doit() for link in expr.args))
    return expr


def _reverse_match(box: str, reference: str, latex: LatexExtractionConfig, precision: int) -> bool:
    """Compare in reverse, retaining set membership and all constraints of a chained relation."""

    candidate = parse(_boxed_math(box), (latex,), fallback_mode="no_fallback")
    expected = parse(_boxed_math(reference), (ExprExtractionConfig(), latex), fallback_mode="no_fallback")
    if len(candidate) != 1 or len(expected) != 1:
        return False
    left, right = candidate[0], expected[0]
    if not verify(left, right, float_rounding=precision):
        return False
    if isinstance(left, Set) or isinstance(right, Set):
        return isinstance(left, Set) and isinstance(right, Set) and left.symmetric_difference(right).is_empty is True
    if isinstance(left, And) or isinstance(right, And):
        # Math-Verify unwraps a chain to its endpoints. Evaluate each link before comparing the whole statement.
        left, right = _reduce_chain(left), _reduce_chain(right)
        if isinstance(left, And) or isinstance(right, And):
            return left == right
        return verify(left, right, float_rounding=precision)
    return True


def _score_box(reference: str, box: str, config: "MathVerifyConfig", sender: Connection) -> None:
    if sys.platform == "linux":
        # A replaced server worker's fork must release its inherited listening socket.
        pr_set_pdeathsig = 1
        prctl = ctypes.CDLL(None).prctl
        prctl.argtypes = (ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong)
        if prctl(pr_set_pdeathsig, signal.SIGKILL, 0, 0, 0) != 0:
            return
        if os.getppid() != mp.parent_process().pid:  # The parent died before prctl ran.
            return
    # The forked verifier must not run uvicorn's shutdown handlers.
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, signal.SIG_DFL)
    with sender, open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
        latex = LatexExtractionConfig()
        latex = replace(latex, normalization_config=replace(latex.normalization_config, units=config.units))
        metric = math_metric(
            gold_extraction_target=(latex,),
            pred_extraction_target=(ExprExtractionConfig(), latex),
            precision=config.precision,
        )
        try:
            score, _ = metric([_boxed_math(reference)], [r"\boxed{" + box + "}"])
            match = score > 0.5
            if match and config.bidirectional and box != reference:
                match = _reverse_match(box, reference, latex, config.precision)
        except (ValueError, TimeoutException):
            match = False
        sender.send(match)


class BoxedMathVerifier:
    def __init__(self, config: "MathVerifyConfig", server_name: str):
        self.config = config
        self.slots = MeteredSemaphore(
            config.max_concurrency,
            component="resources_server",
            limit_name="library_verifier",
            server_name=server_name,
        )

    async def matches(self, reference: str, box: str) -> bool:
        async with resources_operation(self.slots, operation="verify", executor="subprocess") as operation:
            ctx = mp.get_context("fork")
            receiver, sender = ctx.Pipe(duplex=False)
            with receiver, sender:
                process = ctx.Process(target=_score_box, args=(reference, box, self.config, sender))
                loop = asyncio.get_running_loop()
                exited = asyncio.Event()
                process.start()
                try:
                    sender.close()
                    loop.add_reader(process.sentinel, exited.set)
                    try:
                        await asyncio.wait_for(exited.wait(), self.config.timeout_seconds)
                    finally:
                        loop.remove_reader(process.sentinel)
                    if process.exitcode is None:
                        await asyncio.to_thread(process.join)
                    if process.exitcode == 0:
                        return receiver.recv()
                    operation.outcome = "crashed"
                except TimeoutError:
                    operation.outcome = "timeout"
                except EOFError:
                    operation.outcome = "crashed"
                finally:
                    if process.is_alive():
                        # Even a wedged symbolic computation must release its slot on cancellation or timeout.
                        process.kill()
                        await asyncio.to_thread(process.join)
                    process.close()
                return False
