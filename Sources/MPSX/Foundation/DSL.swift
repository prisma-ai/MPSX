import MetalPerformanceShadersGraph

public extension MPSGraph {
    @inlinable
    func const(_ vector: [Float], shape: [Int]? = nil) -> MPSGraphTensor {
        constant(vector.rawData, shape: (shape ?? [vector.count]).nsnumbers, dataType: .float32)
    }

    #if arch(arm64)
    @inlinable
    func const(_ vector: [Swift.Float16], shape: [Int]? = nil) -> MPSGraphTensor {
        constant(vector.rawData, shape: (shape ?? [vector.count]).nsnumbers, dataType: .float16)
    }
    #endif
}

public extension MPSGraphTensor {
    @inlinable
    func const(_ scalar: Float) -> MPSGraphTensor {
        operation.graph.constant(Double(scalar), dataType: dataType)
    }
}

public extension MPSGraphTensor {
    @inlinable
    func cast(to dataType: MPSDataType) -> MPSGraphTensor {
        dataType != self.dataType ? operation.graph.cast(self, to: dataType, name: UUID().uuidString) : self
    }

    @inlinable
    var shapeTensor: MPSGraphTensor {
        operation.graph.shapeOf(self, name: nil)
    }

    @inlinable
    func reshape(_ shape: [Int]) -> MPSGraphTensor {
        operation.graph.reshape(self, shape: shape.nsnumbers, name: nil)
    }

    @inlinable
    func reshape(_ shapeTensor: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.reshape(self, shapeTensor: shapeTensor, name: nil)
    }

    @inlinable
    func transpose(_ dim1: Int, _ dim2: Int) -> MPSGraphTensor {
        dim1 != dim2 ? operation.graph.transposeTensor(self, dimension: dim1, withDimension: dim2, name: nil) : self
    }
}

public extension MPSGraphTensor {
    func squeeze(_ axes: [Int] = []) -> MPSGraphTensor {
        if #available(iOS 15.4, macOS 12.3, *) {
            if axes.isEmpty {
                return operation.graph.squeeze(self, name: nil)
            }
            return operation.graph.squeeze(self, axes: axes.nsnumbers, name: nil)
        }

        guard let shape else {
            return self
        }

        return reshape(shape.map(\.intValue).squeeze(axes: axes))
    }

    func unsqueeze(_ axes: [Int]) -> MPSGraphTensor {
        guard !axes.isEmpty else {
            return self
        }

        if #available(iOS 15.4, macOS 12.3, *) {
            return operation.graph.expandDims(self, axes: axes.nsnumbers, name: nil)
        }

        guard let shape else {
            return self
        }

        return reshape(shape.map(\.intValue).unsqueeze(axes: axes))
    }

    func transpose(_ permutation: [Int]) -> MPSGraphTensor {
        if #available(iOS 16.1, macOS 13.1, *) {
            return operation.graph.transpose(self, permutation: permutation.nsnumbers, name: nil)
        }

        // https://github.com/pytorch/pytorch/blob/af3964a8725236c78ce969b827fdeee1c5c54110/torch/tensor.py#L200

        var output = self
        var permutation = permutation

        for i in permutation.indices {
            let p = permutation[i]

            if p != i, p != -1 {
                var j = i

                while true {
                    let k = permutation[j]

                    output = output.transpose(j, k)

                    permutation[j] = -1

                    j = k

                    if permutation[j] == i {
                        break
                    }
                }

                permutation[j] = -1
            }
        }

        return output
    }

    @inlinable
    func toNHWC() -> MPSGraphTensor {
        assert(shape?.count == 4, "Operation \(#function) should be performed on 4 dimensional tensors")
        return transpose([0, 2, 3, 1])
    }

    @inlinable
    func toNCHW() -> MPSGraphTensor {
        assert(shape?.count == 4, "Operation \(#function) should be performed on 4 dimensional tensors")
        return transpose([0, 3, 1, 2])
    }
}

public extension MPSGraphTensor {
    @inlinable
    func mean(axes: [Int]) -> MPSGraphTensor {
        operation.graph.mean(of: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    func variance(axes: [Int], mean: MPSGraphTensor? = nil) -> MPSGraphTensor {
        if let mean {
            return operation.graph.variance(of: self, mean: mean, axes: axes.nsnumbers, name: nil)
        }
        return operation.graph.variance(of: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    func meanAndVariance(axes: [Int]) -> (mean: MPSGraphTensor, variance: MPSGraphTensor) {
        let axes = axes.nsnumbers
        let mean = operation.graph.mean(of: self, axes: axes, name: nil)
        let variance = operation.graph.variance(of: self, mean: mean, axes: axes, name: nil)
        return (mean, variance)
    }
}

public extension MPSGraphTensor {
    @inlinable
    func pow(_ x: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.power(self, x, name: nil)
    }

    @inlinable
    func pow(_ x: Float) -> MPSGraphTensor {
        operation.graph.power(self, const(x), name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func + (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.addition(x, y, name: nil)
    }

    @inlinable
    static func + (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 0 ? x.operation.graph.addition(x, x.const(y), name: nil) : x
    }

    @inlinable
    static func + (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        x != 0 ? y.operation.graph.addition(y.const(x), y, name: nil) : y
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func - (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.subtraction(x, y, name: nil)
    }

    @inlinable
    static func - (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 0 ? x.operation.graph.subtraction(x, x.const(y), name: nil) : x
    }

    @inlinable
    static func - (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        if x != 0 {
            return y.operation.graph.subtraction(y.const(x), y, name: nil)
        }
        return y.operation.graph.negative(with: y, name: nil)
    }

    @inlinable
    static prefix func - (x: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.negative(with: x, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func * (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.multiplication(x, y, name: nil)
    }

    @inlinable
    static func * (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 1 ? x.operation.graph.multiplication(x, x.const(y), name: nil) : x
    }

    @inlinable
    static func * (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        x != 1 ? y.operation.graph.multiplication(y.const(x), y, name: nil) : y
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func / (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.division(x, y, name: nil)
    }

    @inlinable
    static func / (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 1 ? x.operation.graph.division(x, x.const(y), name: nil) : x
    }

    @inlinable
    static func / (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.division(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func == (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.equal(x, y, name: nil)
    }

    @inlinable
    static func == (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.equal(x, x.const(y), name: nil)
    }

    @inlinable
    static func == (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.equal(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func != (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.notEqual(x, y, name: nil)
    }

    @inlinable
    static func != (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.notEqual(x, x.const(y), name: nil)
    }

    @inlinable
    static func != (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.notEqual(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func < (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.lessThan(x, y, name: nil)
    }

    @inlinable
    static func < (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.lessThan(x, x.const(y), name: nil)
    }

    @inlinable
    static func < (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.lessThan(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func <= (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.lessThanOrEqualTo(x, y, name: nil)
    }

    @inlinable
    static func <= (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.lessThanOrEqualTo(x, x.const(y), name: nil)
    }

    @inlinable
    static func <= (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.lessThanOrEqualTo(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func > (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.greaterThan(x, y, name: nil)
    }

    @inlinable
    static func > (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.greaterThan(x, x.const(y), name: nil)
    }

    @inlinable
    static func > (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.greaterThan(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    static func >= (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.greaterThanOrEqualTo(x, y, name: nil)
    }

    @inlinable
    static func >= (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.greaterThanOrEqualTo(x, x.const(y), name: nil)
    }

    @inlinable
    static func >= (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.greaterThanOrEqualTo(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    func sum(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionSum(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    func product(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionProduct(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    func min(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionMinimum(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    func max(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionMaximum(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    func argmax(axis: Int) -> MPSGraphTensor {
        operation.graph.reductionArgMaximum(with: self, axis: axis, name: nil)
    }

    @inlinable
    func argmin(axis: Int) -> MPSGraphTensor {
        operation.graph.reductionArgMinimum(with: self, axis: axis, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    func clamp(min: Float, max: Float) -> MPSGraphTensor {
        operation.graph.clamp(self, min: const(min), max: const(max), name: nil)
    }

    @inlinable
    func clamp(min: Float, max: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.clamp(self, min: const(min), max: max, name: nil)
    }

    @inlinable
    func clamp(min: MPSGraphTensor, max: Float) -> MPSGraphTensor {
        operation.graph.clamp(self, min: min, max: const(max), name: nil)
    }

    @inlinable
    func clamp(min: MPSGraphTensor, max: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.clamp(self, min: min, max: max, name: nil)
    }

    @inlinable
    func abs() -> MPSGraphTensor {
        operation.graph.absolute(with: self, name: nil)
    }

    @inlinable
    func squareRoot() -> MPSGraphTensor {
        operation.graph.squareRoot(with: self, name: nil)
    }

    @inlinable
    func square() -> MPSGraphTensor {
        operation.graph.square(with: self, name: nil)
    }

    @inlinable
    func ceil() -> MPSGraphTensor {
        operation.graph.ceil(with: self, name: nil)
    }

    @inlinable
    func floor() -> MPSGraphTensor {
        operation.graph.floor(with: self, name: nil)
    }

    @inlinable
    func round() -> MPSGraphTensor {
        operation.graph.round(with: self, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    func pad(
        mode: MPSGraphPaddingMode,
        top: Int,
        bottom: Int,
        left: Int,
        right: Int,
        constant: Double = .zero,
        channelsFirst: Bool = true
    ) -> MPSGraphTensor {
        operation.graph.padTensor(
            self,
            with: mode,
            leftPadding: channelsFirst ? [0, 0, top, left].nsnumbers : [0, top, left, 0].nsnumbers,
            rightPadding: channelsFirst ? [0, 0, bottom, right].nsnumbers : [0, bottom, right, 0].nsnumbers,
            constantValue: constant,
            name: nil
        )
    }

    @inlinable
    func slice(
        axis: Int,
        start: Int,
        length: Int
    ) -> MPSGraphTensor {
        operation.graph.sliceTensor(
            self,
            dimension: axis,
            start: start,
            length: length,
            name: nil
        )
    }

    @available(iOS 16.0, macOS 13.0, *)
    func sort(axis: Int, descending: Bool = false) -> MPSGraphTensor {
        if descending {
            return operation.graph.sort(self, axis: axis, descending: true, name: nil)
        }
        return operation.graph.sort(self, axis: axis, name: nil)
    }

    @inlinable
    func resize(
        mode: MPSGraphResizeMode,
        layout: MPSGraphTensorNamedDataLayout,
        height: Int,
        width: Int
    ) -> MPSGraphTensor {
        operation.graph.resize(
            self,
            size: [height, width].nsnumbers,
            mode: mode,
            centerResult: true,
            alignCorners: false,
            layout: layout,
            name: nil
        )
    }
}

public extension MPSGraph {
    @inlinable
    func placeholder(ishape: [Int], dataType: MPSDataType, name: String) -> MPSGraphTensor {
        placeholder(shape: ishape.nsnumbers, dataType: dataType, name: name)
    }

    @inlinable
    func imagePlaceholder(
        dataType: MPSDataType,
        channels: Int,
        height: Int,
        width: Int,
        name: String
    ) -> MPSGraphTensor {
        placeholder(shape: [1, channels, height, width].nsnumbers, dataType: dataType, name: name)
    }

    @inlinable
    func imagePlaceholder(
        dataType: MPSDataType,
        height: Int,
        width: Int,
        channels: Int,
        name: String
    ) -> MPSGraphTensor {
        placeholder(shape: [1, height, width, channels].nsnumbers, dataType: dataType, name: name)
    }
}

public extension MPSGraph {
    @inlinable
    func min(_ x: MPSGraphTensor, _ y: MPSGraphTensor) -> MPSGraphTensor {
        minimum(x, y, name: nil)
    }

    @inlinable
    func min(_ x: MPSGraphTensor, _ y: Float) -> MPSGraphTensor {
        minimum(x, x.const(y), name: nil)
    }

    @inlinable
    func min(_ x: Float, _ y: MPSGraphTensor) -> MPSGraphTensor {
        minimum(y.const(x), y, name: nil)
    }
}

public extension MPSGraph {
    @inlinable
    func max(_ x: MPSGraphTensor, _ y: MPSGraphTensor) -> MPSGraphTensor {
        maximum(x, y, name: nil)
    }

    @inlinable
    func max(_ x: MPSGraphTensor, _ y: Float) -> MPSGraphTensor {
        maximum(x, x.const(y), name: nil)
    }

    @inlinable
    func max(_ x: Float, _ y: MPSGraphTensor) -> MPSGraphTensor {
        maximum(y.const(x), y, name: nil)
    }
}

public extension MPSGraph {
    @inlinable
    func matmul(_ x: MPSGraphTensor, _ y: MPSGraphTensor) -> MPSGraphTensor {
        matrixMultiplication(primary: x, secondary: y, name: nil)
    }
}
