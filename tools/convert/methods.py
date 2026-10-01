"""Method requests/results and the built-in adapters to quantization or encoded import.

Methods own input traversal, chunking, and auxiliary values. Numerical algorithms
return codes/scales; artifact.tensor_output owns their physical byte placement.
User methods accept the same PrepareRequest and return a PreparedMethod.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from math import prod
import struct
from typing import Callable, Mapping

import torch

from tools.artifact.formats import (
    DirectFormat,
    QuantFormat,
    get_format,
    valid_positive_fp32_word,
)
from tools.artifact.schema import TensorSpec
from tools.artifact.tensor_output import TensorOutput

from .quantization.fp8_row import quantize_bf16_rows
from .quantization.groupwise import quantize_matrix
from .sources.logical import EncodedRows, LogicalSource

UseKey = tuple[str, str]
AuxiliaryKey = tuple[str, str, str]


@dataclass(frozen=True, slots=True)
class MethodInput:
    parameter: str
    source: LogicalSource
    uses: tuple[UseKey, ...]


@dataclass(frozen=True, slots=True)
class AuxiliaryValue:
    format: str
    shape: tuple[int, ...]
    data: bytes

    @classmethod
    def activation_divisor(cls, value: bytes | float) -> AuxiliaryValue:
        raw = value if isinstance(value, bytes) else struct.pack("<f", value)
        if len(raw) != 4 or not valid_positive_fp32_word(struct.unpack("<I", raw)[0]):
            raise ValueError("activation input divisor must be positive finite FP32")
        return cls("fp32", (), raw)


@dataclass(frozen=True, slots=True)
class PreparedMethod:
    produce: Callable[[TensorOutput], None]
    auxiliaries: Mapping[AuxiliaryKey, AuxiliaryValue] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class PrepareRequest:
    target: TensorSpec
    inputs: tuple[MethodInput, ...]
    policies: Mapping[UseKey, str]
    parameters: Mapping[str, object]
    device: str = "cuda"
    rows_per_chunk: int = 512
    auxiliary_overrides: Mapping[AuxiliaryKey, AuxiliaryValue] = field(
        default_factory=dict
    )
    source_offsets: tuple[int, ...] = field(init=False)

    def __post_init__(self) -> None:
        values = [0]
        for item in self.inputs:
            values.append(values[-1] + prod(item.source.shape))
        object.__setattr__(self, "source_offsets", tuple(values))

    @property
    def source(self) -> LogicalSource:
        if len(self.inputs) != 1:
            raise ValueError("this method has multiple inputs; use request.inputs")
        return self.inputs[0].source

    def require_input_shape(self, shape: tuple[int, ...]) -> None:
        if self.source.shape != tuple(shape):
            raise ValueError(f"source shape {self.source.shape} differs from {shape}")

    def job(self, *, produce, auxiliaries=None) -> PreparedMethod:
        return PreparedMethod(produce, {} if auxiliaries is None else dict(auxiliaries))

    def values(self, begin: int, end: int) -> torch.Tensor:
        """Read the ordered logical inputs as one C-order parent element sequence."""
        pieces = []
        if not 0 <= begin <= end <= self.source_offsets[-1]:
            raise ValueError(f"{self.target.id}: requested values exceed method inputs")
        index = bisect_right(self.source_offsets, begin) - 1
        while begin < end:
            high = min(end, self.source_offsets[index + 1])
            pieces.append(
                self.inputs[index].source.values(
                    begin - self.source_offsets[index],
                    high - self.source_offsets[index],
                )
            )
            begin = high
            index += 1
        if not pieces:
            return torch.empty(0)
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def encoded_rows(self, begin: int, end: int) -> EncodedRows:
        pieces = []
        cursor = 0
        for item in self.inputs:
            source = item.source
            if len(source.shape) != 2 or source.shape[1] != self.target.shape[1]:
                raise ValueError(
                    f"{item.parameter}: encoded grouping requires complete rows"
                )
            low, high = max(begin, cursor), min(end, cursor + source.shape[0])
            if low < high:
                if source.read_encoded is None:
                    raise ValueError(f"{item.parameter}: encoded rows are unavailable")
                pieces.append(source.read_encoded(low - cursor, high - cursor))
            cursor += source.shape[0]
        if not pieces or end > cursor:
            raise ValueError("encoded method input range is invalid")
        first = pieces[0]
        if any(
            (p.format, p.weight_divisor) != (first.format, first.weight_divisor)
            for p in pieces
        ):
            raise ValueError("encoded inputs cannot share one parent format/divisor")
        if len(pieces) == 1:
            return first
        return EncodedRows(
            first.format,
            torch.cat([p.codes for p in pieces]),
            torch.cat([p.scales for p in pieces]),
            first.weight_divisor,
        )


Method = Callable[[PrepareRequest], PreparedMethod]
_DIRECT_DTYPES = {"bf16": torch.bfloat16, "fp32": torch.float32, "int32": torch.int32}


def _preflight(request: PrepareRequest, *, values: bool = True) -> None:
    if request.parameters:
        raise ValueError(
            f"{request.target.id}: this method accepts no numerical parameters"
        )
    if prod(request.target.shape) != sum(
        prod(item.source.shape) for item in request.inputs
    ):
        raise ValueError(
            f"{request.target.id}: parent and logical source counts differ"
        )
    if type(request.rows_per_chunk) is not int or request.rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be positive")
    if values:
        for item in request.inputs:
            item.source.values(0, 1)


def cast_direct(request: PrepareRequest) -> PreparedMethod:
    """Convert values at the explicit target BF16/FP32/INT32 boundary."""
    if not isinstance(get_format(request.target.format), DirectFormat):
        raise ValueError("cast_direct requires a direct target format")
    _preflight(request)
    dtype = _DIRECT_DTYPES[request.target.format]
    chunk = request.rows_per_chunk * (
        request.target.shape[-1] if len(request.target.shape) > 1 else 1
    )
    elements = prod(request.target.shape)

    def produce(output):
        for begin in range(0, elements, chunk):
            values = request.values(begin, min(elements, begin + chunk))
            if dtype == torch.int32 and (
                bool((values < -(1 << 31)).any())
                or bool((values > (1 << 31) - 1).any())
            ):
                raise ValueError(
                    "int32 conversion source is outside the representable range"
                )
            values = values.to(dtype=dtype)
            output.write_values(begin, values)

    return request.job(produce=produce)


def grouped_absmax(request: PrepareRequest) -> PreparedMethod:
    """Use the existing grouped max-abs, FP16-scale and code-rounding algorithm."""
    if (
        not isinstance(get_format(request.target.format), QuantFormat)
        or len(request.target.shape) != 2
    ):
        raise ValueError("grouped_absmax requires a grouped-integer matrix target")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = request.values(begin * k, end * k).reshape(end - begin, k)
            if not values.dtype.is_floating_point:
                raise TypeError(
                    "grouped_absmax source must provide floating-point values"
                )
            encoded = quantize_matrix(
                values, request.target.format, device=request.device
            )
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def fp8_row_maxabs(request: PrepareRequest) -> PreparedMethod:
    """Round inputs to BF16, then quantize to FP8 codes with BF16 row scales."""
    if request.target.format != "fp8_e4m3fn_row_bf16" or len(request.target.shape) != 2:
        raise ValueError("fp8_row_maxabs requires the row-scaled FP8 matrix format")
    _preflight(request)
    n, k = request.target.shape

    def produce(output):
        for begin in range(0, n, request.rows_per_chunk):
            end = min(n, begin + request.rows_per_chunk)
            values = (
                request.values(begin * k, end * k)
                .reshape(end - begin, k)
                .to(torch.bfloat16)
            )
            encoded = quantize_bf16_rows(values)
            output.write_codes(begin, encoded.codes, encoded.scales)

    return request.job(produce=produce)


def import_encoded(request: PrepareRequest) -> PreparedMethod:
    """Preserve the current FP8/NVFP4 source codes, scales and weight divisor."""
    if (
        request.target.format not in ("nvfp4", "fp8_e4m3fn_row_bf16")
        or len(request.target.shape) != 2
    ):
        raise ValueError("import_encoded requires a known encoded matrix target")
    _preflight(request, values=False)
    auxiliaries = {}
    weight_divisor = None
    for item in request.inputs:
        source = item.source
        if source.read_encoded is None:
            raise ValueError(f"{item.parameter}: encoded rows are unavailable")
        first = source.read_encoded(0, 1)
        if first.format != request.target.format:
            raise ValueError(
                f"{item.parameter}: source {first.format} differs from target {request.target.format}"
            )
        if first.format == "nvfp4":
            if weight_divisor is None:
                weight_divisor = first.weight_divisor
            elif first.weight_divisor != weight_divisor:
                raise ValueError(
                    "NVFP4 weight divisors differ; choose separate parents or a conversion method"
                )
            for parameter, input_name in item.uses:
                key = (parameter, input_name, "activation_input_divisor")
                if key in request.auxiliary_overrides:
                    auxiliaries[key] = request.auxiliary_overrides[key]
                elif request.policies[(parameter, input_name)] == "AllowA4":
                    if source.input_divisor is None:
                        raise ValueError(
                            f"{parameter}: supply an activation divisor for AllowA4"
                        )
                    auxiliaries[key] = AuxiliaryValue.activation_divisor(
                        source.input_divisor()
                    )
    n = request.target.shape[0]
    chunk = (
        max(128, request.rows_per_chunk // 128 * 128)
        if request.target.format == "nvfp4"
        else request.rows_per_chunk
    )

    def produce(output):
        for begin in range(0, n, chunk):
            words = request.encoded_rows(begin, min(n, begin + chunk))
            output.write_codes(begin, words.codes, words.scales, words.weight_divisor)

    return request.job(produce=produce, auxiliaries=auxiliaries)


# NVFP4 decoding contract (docs/maintainer/tensor-formats.md §3.3):
#   W[n,k] = decode_e2m1(c[n,k]) * decode_e4m3fn(s[n,g]) / d_w,  g = k // 16
# so the encoder picks one matrix divisor d_w that places the largest block scale at the
# E4M3FN maximum, then rounds every 16-wide block's scale to E4M3FN and its codes to E2M1.
_E2M1_MIDPOINTS = (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0)
_E2M1_MAX = 6.0
_E4M3FN_MAX = 448.0
_NVFP4_GROUP = 16


def _e2m1_words(scaled: torch.Tensor) -> torch.Tensor:
    """Round finite values to the nearest E2M1 word, ties to the even code."""
    magnitude = scaled.abs().clamp_(max=_E2M1_MAX)
    midpoints = torch.tensor(_E2M1_MIDPOINTS, dtype=magnitude.dtype, device=magnitude.device)
    down = torch.bucketize(magnitude, midpoints, right=False)
    up = torch.bucketize(magnitude, midpoints, right=True)
    codes = torch.where((down != up) & (down % 2 != 0), up, down).to(torch.uint8)
    return codes | (scaled < 0).to(torch.uint8) * 8


def quantize_nvfp4_matrix(
    values: torch.Tensor, weight_divisor: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return packed E2M1 codes ``[rows, K/2]`` and natural E4M3FN scales ``[rows, K/16]``."""
    rows, k = values.shape
    if k % _NVFP4_GROUP:
        raise ValueError("NVFP4 requires K to be a multiple of 16")
    blocks = values.to(torch.float32).reshape(rows, k // _NVFP4_GROUP, _NVFP4_GROUP)
    ideal = blocks.abs().amax(dim=-1) * (weight_divisor / _E2M1_MAX)
    scale_words = ideal.clamp_(0.0, _E4M3FN_MAX).to(torch.float8_e4m3fn)
    scales = scale_words.float()
    safe = torch.where(scales > 0, scales, torch.ones_like(scales))
    scaled = blocks * (weight_divisor / safe)[:, :, None]
    scaled = torch.where((scales > 0)[:, :, None], scaled, torch.zeros_like(scaled))
    words = _e2m1_words(scaled).reshape(rows, k)
    packed = words[:, 0::2] | (words[:, 1::2] << 4)
    return packed.contiguous(), scale_words.view(torch.uint8).contiguous()


def nvfp4_blockwise(request: PrepareRequest) -> PreparedMethod:
    """Quantize floating-point inputs to NVFP4 with a matrix divisor at the E4M3FN maximum.

    Pass one finds the matrix max-abs; pass two rounds each 16-wide block. Every input row
    lands in the same parent under one divisor, so the parent stays importable as a single
    NVFP4 weight. This is weight-only quantization: no activation divisor is produced, so the
    consuming sites must keep the default ``A16Only`` activation policy.
    """
    if request.target.format != "nvfp4" or len(request.target.shape) != 2:
        raise ValueError("nvfp4_blockwise requires the nvfp4 matrix format")
    _preflight(request)
    n, k = request.target.shape
    if n % 128 or k % _NVFP4_GROUP:
        raise ValueError("nvfp4_blockwise needs N % 128 == 0 and K % 16 == 0")
    chunk = max(128, request.rows_per_chunk // 128 * 128)

    def produce(output):
        amax = torch.zeros((), dtype=torch.float32, device=request.device)
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).to(request.device)
            if not values.dtype.is_floating_point:
                raise TypeError("nvfp4_blockwise source must provide floating-point values")
            if not bool(torch.isfinite(values).all()):
                raise ValueError(f"{request.target.id}: source values are not finite")
            amax = torch.maximum(amax, values.abs().amax().float())
        if float(amax) == 0.0:
            raise ValueError(f"{request.target.id}: an all-zero matrix has no NVFP4 divisor")
        # Largest possible block scale (amax / 6) * d_w == 448 exactly at the matrix max-abs.
        weight_divisor = _E4M3FN_MAX * _E2M1_MAX / float(amax)
        divisor_word = struct.pack("<f", weight_divisor)
        divisor = struct.unpack("<f", divisor_word)[0]
        for begin in range(0, n, chunk):
            end = min(n, begin + chunk)
            values = request.values(begin * k, end * k).to(request.device).reshape(end - begin, k)
            codes, scales = quantize_nvfp4_matrix(values, divisor)
            output.write_codes(begin, codes.cpu(), scales.cpu(), divisor_word)

    return request.job(produce=produce)


METHODS: dict[str, Method] = {
    "cast_direct": cast_direct,
    "grouped_absmax": grouped_absmax,
    "fp8_row_maxabs": fp8_row_maxabs,
    "import_encoded": import_encoded,
    "nvfp4_blockwise": nvfp4_blockwise,
}
