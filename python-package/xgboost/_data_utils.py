"""Helpers for interfacing array like objects."""

import copy
import ctypes
import functools
import json
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Protocol,
    Tuple,
    Type,
    TypedDict,
    TypeGuard,
    Union,
    cast,
    overload,
)

import numpy as np

from ._typing import CNumericPtr, DataType, NumpyDType, NumpyOrCupy
from .compat import import_cupy, lazy_isinstance

if TYPE_CHECKING:
    import pyarrow as pa


# Used for accepting inputs for numpy and cupy arrays
class _ArrayLikeArg(Protocol):
    @property
    def __array_interface__(self) -> "ArrayInf": ...


class _CupyArrayLikeArg(Protocol):
    @property
    def __cuda_array_interface__(self) -> dict: ...


ArrayInf = TypedDict(
    "ArrayInf",
    {
        "data": Tuple[int, bool],
        "typestr": str,
        "version": Literal[3],
        "strides": Optional[Tuple[int, ...]],
        "shape": Tuple[int, ...],
        "mask": Union["ArrayInf", None, _ArrayLikeArg],
    },
)

StringArray = TypedDict("StringArray", {"offsets": ArrayInf, "values": ArrayInf})


def array_hasobject(data: DataType) -> bool:
    """Whether the numpy array has object dtype."""
    return hasattr(data.dtype, "hasobject") and data.dtype.hasobject


def cuda_array_interface(data: DataType) -> bytes:
    """Make cuda array interface str."""
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    interface = data.__cuda_array_interface__
    if "mask" in interface:
        interface["mask"] = interface["mask"].__cuda_array_interface__
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str


def from_array_interface(interface: ArrayInf, zero_copy: bool = False) -> NumpyOrCupy:
    """Convert array interface to numpy or cupy array"""

    class Array:
        """Wrapper type for communicating with numpy and cupy."""

        _interface: Optional[ArrayInf] = None

        @property
        def __array_interface__(self) -> Optional[ArrayInf]:
            return self._interface

        @__array_interface__.setter
        def __array_interface__(self, interface: ArrayInf) -> None:
            self._interface = copy.copy(interface)
            # Convert some fields to tuple as required by numpy
            self._interface["shape"] = tuple(self._interface["shape"])
            self._interface["data"] = (
                self._interface["data"][0],
                self._interface["data"][1],
            )
            strides = self._interface.get("strides", None)
            if strides is not None:
                self._interface["strides"] = tuple(strides)

        @property
        def __cuda_array_interface__(self) -> Optional[ArrayInf]:
            return self.__array_interface__

        @__cuda_array_interface__.setter
        def __cuda_array_interface__(self, interface: ArrayInf) -> None:
            self.__array_interface__ = interface

    arr = Array()

    if "stream" in interface:
        # CUDA stream is presented, this is a __cuda_array_interface__.
        arr.__cuda_array_interface__ = interface
        out = import_cupy().array(arr, copy=not zero_copy)
    else:
        arr.__array_interface__ = interface
        out = np.array(arr, copy=not zero_copy)

    return out


def make_array_interface(
    ptr: CNumericPtr, shape: Tuple[int, ...], dtype: Type[np.number], is_cuda: bool
) -> ArrayInf:
    """Make an __(cuda)_array_interface__ from a pointer."""
    # Use an empty array to handle typestr and descr
    if is_cuda:
        empty = import_cupy().empty(shape=(0,), dtype=dtype)
        array = empty.__cuda_array_interface__  # pylint: disable=no-member
    else:
        empty = np.empty(shape=(0,), dtype=dtype)
        array = empty.__array_interface__  # pylint: disable=no-member

    addr = ctypes.cast(ptr, ctypes.c_void_p).value
    length = int(np.prod(shape))
    # Handle empty dataset.
    assert addr is not None or length == 0

    if addr is None:
        return array

    array["data"] = (addr, True)
    if is_cuda:
        array["stream"] = 2
    array["shape"] = shape
    array["strides"] = None
    return array


def is_arrow_dict(data: Any) -> TypeGuard["pa.DictionaryArray"]:
    return lazy_isinstance(data, "pyarrow.lib", "DictionaryArray")


class CatAccessor(Protocol):
    """Protocol for pandas cat accessor."""

    @property
    def categories(self) -> np.ndarray: ...

    @property
    def codes(self) -> np.ndarray: ...


def _is_pd_cat(data: Any) -> TypeGuard[CatAccessor]:
    return hasattr(data, "categories") and hasattr(data, "codes")


@functools.cache
def _arrow_typestr() -> Dict["pa.DataType", str]:
    import pyarrow as pa

    mapping = {
        pa.int8(): "<i1",
        pa.int16(): "<i2",
        pa.int32(): "<i4",
        pa.int64(): "<i8",
        pa.uint8(): "<u1",
        pa.uint16(): "<u2",
        pa.uint32(): "<u4",
        pa.uint64(): "<u8",
    }

    return mapping


def _arrow_cat_inf(
    cats: "pa.StringArray",
    codes: Union[_ArrayLikeArg, _CupyArrayLikeArg, "pa.IntegerArray"],
) -> Tuple[StringArray, ArrayInf, Tuple]:
    import pyarrow as pa

    # fixme: account for offset
    assert cats.offset == 0
    # fixme: assert arrow's ordering is the same as cudf.
    buffers: List[pa.Buffer] = cats.buffers()
    mask, offset, data = buffers
    assert offset.is_cpu

    off_len = len(cats) + 1
    assert offset.size == off_len * (np.iinfo(np.int32).bits / 8)

    joffset: ArrayInf = {
        "data": (int(offset.address), True),
        "typestr": "<i4",
        "version": 3,
        "strides": None,
        "shape": (off_len,),
        "mask": None,
    }

    def make_buf_inf(buf: pa.Buffer, typestr: str) -> ArrayInf:
        return {
            "data": (int(buf.address), True),
            "typestr": typestr,
            "version": 3,
            "strides": None,
            "shape": (buf.size,),
            "mask": None,
        }

    jdata = make_buf_inf(data, "<i1")

    if mask is not None:
        # fixme: test cudf mask
        jdata["mask"] = make_buf_inf(mask, "<i1")

    jnames: StringArray = {"offsets": joffset, "values": jdata}

    def make_array_inf(
        array: Any,
    ) -> Tuple[ArrayInf, Optional[Tuple[pa.Buffer, pa.Buffer]]]:
        if hasattr(codes, "__cuda_array_interface__"):
            inf = array.__cuda_array_interface__
            if "mask" in inf:
                inf["mask"] = inf["mask"].__array_cuda_interface__
            return inf, None
        elif hasattr(codes, "__array_interface__"):
            inf = array.__array_interface__
            if "mask" in inf:
                inf["mask"] = inf["mask"].__array_interface__
            return inf, None

        if not isinstance(array, pa.IntegerArray):
            raise TypeError("The encoding for categorical data must be integer.")
        buffers: List[pa.Buffer] = array.buffers()
        mask, data = buffers

        jdata = make_buf_inf(data, _arrow_typestr()[array.type])
        if mask is not None:
            # fixme: test cudf mask
            jdata["mask"] = make_buf_inf(mask, "<i1")

        inf = cast(ArrayInf, jdata)
        return inf, (mask, data)

    cats_tmp = (mask, offset, data)
    jcodes, codes_tmp = make_array_inf(codes)

    return jnames, jcodes, (cats_tmp, codes_tmp)


def _npstr_to_arrow_strarr(strarr: np.ndarray) -> Tuple[np.ndarray, str]:
    lenarr = np.vectorize(len)
    offsets = np.cumsum(np.concatenate([np.array([0], dtype=np.int64), lenarr(strarr)]))
    values = strarr.sum()
    # fixme: assert not null-terminated
    return offsets.astype(np.int32), values


def _ensure_np_dtype(
    data: DataType, dtype: Optional[NumpyDType]
) -> Tuple[np.ndarray, Optional[NumpyDType]]:
    if array_hasobject(data) or data.dtype in [np.float16, np.bool_]:
        dtype = np.float32
        data = data.astype(dtype, copy=False)
    if not data.flags.aligned:
        data = np.require(data, requirements="A")
    return data, dtype


@overload
def array_interface_dict(data: np.ndarray) -> ArrayInf: ...


@overload
def array_interface_dict(data: CatAccessor) -> Tuple[StringArray, ArrayInf, Tuple]: ...


@overload
def array_interface_dict(
    data: "pa.DictionaryArray",
) -> Tuple[StringArray, ArrayInf, Tuple]: ...


def array_interface_dict(
    data: Union[np.ndarray, CatAccessor, "pa.DictionaryType"]
) -> Union[ArrayInf, Tuple[StringArray, ArrayInf, Tuple]]:
    if is_arrow_dict(data):
        cats = data.dictionary
        codes = data.indices
        jnames, jcodes, buf = _arrow_cat_inf(cats, codes)
        return jnames, jcodes, buf
    if _is_pd_cat(data):
        cats = data.categories
        codes = data.codes

        offsets, values = _npstr_to_arrow_strarr(cats.values)
        offsets, _ = _ensure_np_dtype(offsets, np.int32)
        joffsets = array_interface_dict(offsets)
        bvalues = values.encode("utf-8")
        ptr = ctypes.c_void_p.from_buffer(ctypes.c_char_p(bvalues)).value
        assert ptr is not None

        jvalues: ArrayInf = {
            "data": (ptr, True),
            "typestr": "|i1",
            "shape": (len(values),),
            "strides": None,
            "version": 3,
            "mask": None,
        }
        jnames = {"offsets": joffsets, "values": jvalues}

        jcodes = array_interface_dict(codes.values)

        buf = (offsets, values, bvalues)
        return jnames, jcodes, buf

    assert isinstance(data, np.ndarray)
    if array_hasobject(data):
        raise ValueError("Input data contains `object` dtype.  Expecting numeric data.")
    ainf = data.__array_interface__
    if "mask" in ainf:
        ainf["mask"] = ainf["mask"].__array_interface__
    return cast(ArrayInf, ainf)


def array_interface(data: np.ndarray) -> bytes:
    """Make array interface str."""
    interface = array_interface_dict(data)
    interface_str = bytes(json.dumps(interface), "utf-8")
    return interface_str
