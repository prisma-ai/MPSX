import onnx
from onnx import numpy_helper
import numpy as np
import argparse

DATA_TYPES = {np.float32: 1, np.float16: 10}


def convert_tensor_infos_to_types(tensor_infos, data_type):
    return [
        onnx.helper.make_tensor_value_info(
            ti.name,
            DATA_TYPES[data_type],
            [dim.dim_value for dim in ti.type.tensor_type.shape.dim]
        ) for ti in tensor_infos
    ]


def convert_tensors_to_type(tensors, data_type):
    return [
        numpy_helper.from_array(
            numpy_helper.to_array(t).astype(data_type),
            name=t.name
        ) for t in tensors
    ]


def find_tensors_for_transposition(nodes):
    swizzle_candidates = {}
    for node in nodes:
        if node.op_type == 'Conv':
            for attr in node.attribute:
                if attr.name == 'group' and attr.i != 1:
                    swizzle_candidates[node.input[1]] = attr.i
    return swizzle_candidates


def transpose_depthwise_conv_weights(tensors, names):
    return [
        numpy_helper.from_array(
            np.transpose(numpy_helper.to_array(t), (1, 0, 2, 3)),
            name=t.name
        ) if t.name in names and t.dims[0] == names[t.name] else t for t in tensors
    ]


def convert_onnx_to_mpsx(model, data_type):
    swizzle_candidates = find_tensors_for_transposition(model.graph.node)
    swizzled_initializer = transpose_depthwise_conv_weights(
        model.graph.initializer, swizzle_candidates)
    new_model = onnx.helper.make_model(
        onnx.helper.make_graph(
            nodes=model.graph.node,
            name=model.graph.name,
            inputs=convert_tensor_infos_to_types(model.graph.input, data_type),
            outputs=convert_tensor_infos_to_types(
                model.graph.output, data_type),
            initializer=convert_tensors_to_type(
                swizzled_initializer, data_type)
        ),
        producer_name='MPSX'
    )
    return new_model


def main():
    parser = argparse.ArgumentParser(
        description='Convert ONNX model to MPSX format')
    parser.add_argument('--half', required=False,
                        help='Use FP16 weights', action='store_true')
    parser.add_argument('--input', required=True, help='Path to ONNX model')
    parser.add_argument('--output', required=True, help='Path to MPSX model')
    args = parser.parse_args()

    onnx_model = onnx.load(args.input)

    if onnx_model.producer_name == 'MPSX':
        return

    onnx.helper.strip_doc_string(onnx_model)
    data_type = np.float16 if args.half else np.float32
    mpsx_model = convert_onnx_to_mpsx(onnx_model, data_type)
    onnx.save(mpsx_model, args.output)

    print('Done!')


if __name__ == "__main__":
    main()
