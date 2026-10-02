"""Rebuild src/cambobulator/models/selfie_segmenter_landscape.onnx from MediaPipe's .tflite.

Only needed to refresh the vendored model. Run it in a throwaway Python 3.11 venv:

    py -3.11 -m venv .convert
    .convert\\Scripts\\pip install tensorflow==2.15.1 tf2onnx==1.16.1 onnx
    .convert\\Scripts\\python scripts\\convert_segmenter.py

tf2onnx converts everything except MediaPipe's custom ``Convolution2DTransposeBias``
op (the model's last layer). That op is a transposed convolution plus bias, and
its custom options are TfLiteTransposeConvParams (padding, stride_w, stride_h).
This script swaps it for a standard ONNX ConvTranspose.

The result matches MediaPipe's own output to ~1e-5 with OpenCV >= 4.9 (4.8's
dnn module gets it wrong). Input: RGB, NCHW, 1x3x144x256, values in [0, 1].
Output: 1x1x144x256 person probability.
"""

from __future__ import annotations

import struct
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import numpy as np
import onnx
from onnx import checker, helper, numpy_helper, shape_inference

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/image_segmenter/"
    "selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite"
)
OUT = Path(__file__).resolve().parents[1] / "src" / "cambobulator" / "models" / "selfie_segmenter_landscape.onnx"


def transpose_conv_options(tflite: Path) -> tuple[int, int, int]:
    from tensorflow.lite.python import schema_py_generated as schema

    model = schema.Model.GetRootAsModel(tflite.read_bytes(), 0)
    graph = model.Subgraphs(0)
    for i in range(graph.OperatorsLength()):
        op = graph.Operators(i)
        if model.OperatorCodes(op.OpcodeIndex()).CustomCode() == b"Convolution2DTransposeBias":
            return struct.unpack("<3i", bytes(op.CustomOptionsAsNumpy()))
    raise SystemExit("Convolution2DTransposeBias not found: the model changed, revisit this script")


def replace_custom_op(model: onnx.ModelProto, stride_w: int, stride_h: int) -> onnx.ModelProto:
    g = model.graph
    inits = {t.name: t for t in g.initializer}
    idx = next(i for i, n in enumerate(g.node) if n.op_type == "TFL_Convolution2DTransposeBias")
    node = g.node[idx]
    x_nhwc, w_name, b_name = node.input
    # tf2onnx feeds the op through an NCHW -> NHWC transpose; bypass it.
    to_nhwc = next(n for n in g.node if x_nhwc in n.output)
    assert to_nhwc.op_type == "Transpose" and list(to_nhwc.attribute[0].ints) == [0, 2, 3, 1]
    assert sum(x_nhwc in n.input for n in g.node) == 1
    w = numpy_helper.to_array(inits[w_name])  # TFLite OHWI
    assert w.shape[0] == 1, "the NHWC output is only layout-compatible with NCHW for one channel"
    kh, kw = w.shape[1:3]
    assert (kh, kw) == (stride_h, stride_w), "SAME padding is only zero padding when kernel == stride"
    g.initializer.append(numpy_helper.from_array(np.ascontiguousarray(w.transpose(3, 0, 1, 2)), "segment_head_W"))
    head = helper.make_node("ConvTranspose", [to_nhwc.input[0], "segment_head_W", b_name], [node.output[0]],
                            name="segment_head", kernel_shape=[kh, kw], strides=[stride_h, stride_w],
                            pads=[0, 0, 0, 0])
    g.node.remove(node)
    g.node.insert(idx, head)
    g.node.remove(to_nhwc)
    used = {i for n in g.node for i in n.input}
    for t in list(g.initializer):
        if t.name not in used:
            g.initializer.remove(t)
    del g.value_info[:]
    model = shape_inference.infer_shapes(model)
    checker.check_model(model, full_check=True)
    return model


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tflite = Path(tmp) / "model.tflite"
        raw = Path(tmp) / "raw.onnx"
        urllib.request.urlretrieve(MODEL_URL, tflite)
        padding, stride_w, stride_h = transpose_conv_options(tflite)
        # padding 1 = SAME; with kernel == stride that needs no padding, which the asserts below rely on.
        assert padding == 1 and (stride_w, stride_h) == (2, 2), (padding, stride_w, stride_h)
        subprocess.run([sys.executable, "-m", "tf2onnx.convert", "--tflite", str(tflite), "--output", str(raw),
                        "--opset", "17", "--inputs-as-nchw", "input_1", "--outputs-as-nchw", "segment_back"],
                       check=True)
        model = replace_custom_op(onnx.load(raw), stride_w, stride_h)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
