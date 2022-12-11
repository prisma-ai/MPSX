import MetalPerformanceShadersGraph

public extension MPSGraph {
    func const(_ vector: [Float]) -> MPSGraphTensor {
        assert(!vector.isEmpty)

        return constant(vector.rawData, shape: [vector.count].nsnumbers, dataType: .float32)
    }

    func const(_ vector: [Float], shape: [Int]) -> MPSGraphTensor {
        assert(!vector.isEmpty && shape.reduce(1, *) == vector.count)

        return constant(vector.rawData, shape: shape.nsnumbers, dataType: .float32)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    func const(_ scalar: Float) -> MPSGraphTensor {
        operation.graph.constant(Double(scalar), dataType: dataType)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    func cast(to dataType: MPSDataType) -> MPSGraphTensor {
        operation.graph.cast(self, to: dataType, name: UUID().uuidString)
    }

    @inlinable
    @inline(__always)
    func tshape() -> MPSGraphTensor {
        operation.graph.shapeOf(self, name: nil)
    }

    @inlinable
    @inline(__always)
    func reshape(_ shape: [NSNumber]) -> MPSGraphTensor {
        operation.graph.reshape(self, shape: shape, name: nil)
    }

    @inlinable
    @inline(__always)
    func reshape(_ shapeTensor: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.reshape(self, shapeTensor: shapeTensor, name: nil)
    }

    @inlinable
    @inline(__always)
    func transpose(_ dim1: Int, _ dim2: Int) -> MPSGraphTensor {
        operation.graph.transposeTensor(self, dimension: dim1, withDimension: dim2, name: nil)
    }

    func transpose(_ permutation: [Int]) -> MPSGraphTensor {
        if #available(iOS 16.0, macOS 13.0, *) {
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

        return reshape(shape.map(\.intValue).squeeze(axes: axes).nsnumbers)
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

        return reshape(shape.map(\.intValue).unsqueeze(axes: axes).nsnumbers)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    func pow(_ x: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.power(self, x, name: nil)
    }

    @inlinable
    @inline(__always)
    func pow(_ x: Float) -> MPSGraphTensor {
        operation.graph.power(self, const(x), name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func + (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.addition(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func + (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 0 ? x.operation.graph.addition(x, x.const(y), name: nil) : x
    }

    @inlinable
    @inline(__always)
    static func + (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        x != 0 ? y.operation.graph.addition(y.const(x), y, name: nil) : y
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func - (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.subtraction(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func - (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 0 ? x.operation.graph.subtraction(x, x.const(y), name: nil) : x
    }

    @inlinable
    @inline(__always)
    static func - (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        if x != 0 {
            return y.operation.graph.subtraction(y.const(x), y, name: nil)
        }
        return y.operation.graph.negative(with: y, name: nil)
    }

    @inlinable
    @inline(__always)
    static prefix func - (x: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.negative(with: x, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func * (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.multiplication(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func * (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 1 ? x.operation.graph.multiplication(x, x.const(y), name: nil) : x
    }

    @inlinable
    @inline(__always)
    static func * (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        x != 1 ? y.operation.graph.multiplication(y.const(x), y, name: nil) : y
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func / (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.division(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func / (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        y != 1 ? x.operation.graph.division(x, x.const(y), name: nil) : x
    }

    @inlinable
    @inline(__always)
    static func / (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.division(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func == (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.equal(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func == (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.equal(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    static func == (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.equal(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func != (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.notEqual(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func != (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.notEqual(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    static func != (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.notEqual(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func < (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.lessThan(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func < (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.lessThan(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    static func < (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.lessThan(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func <= (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.lessThanOrEqualTo(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func <= (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.lessThanOrEqualTo(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    static func <= (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.lessThanOrEqualTo(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func > (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.greaterThan(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func > (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.greaterThan(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    static func > (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.greaterThan(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    static func >= (x: MPSGraphTensor, y: MPSGraphTensor) -> MPSGraphTensor {
        x.operation.graph.greaterThanOrEqualTo(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    static func >= (x: MPSGraphTensor, y: Float) -> MPSGraphTensor {
        x.operation.graph.greaterThanOrEqualTo(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    static func >= (x: Float, y: MPSGraphTensor) -> MPSGraphTensor {
        y.operation.graph.greaterThanOrEqualTo(y.const(x), y, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    func sum(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionSum(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    @inline(__always)
    func product(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionProduct(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    @inline(__always)
    func min(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionMinimum(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    @inline(__always)
    func max(axes: [Int]) -> MPSGraphTensor {
        operation.graph.reductionMaximum(with: self, axes: axes.nsnumbers, name: nil)
    }

    @inlinable
    @inline(__always)
    func mean(axes: [Int]) -> MPSGraphTensor {
        operation.graph.mean(of: self, axes: axes.nsnumbers, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
    func clamp(min: Float, max: Float) -> MPSGraphTensor {
        operation.graph.clamp(self, min: const(min), max: const(max), name: nil)
    }

    @inlinable
    @inline(__always)
    func clamp(min: Float, max: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.clamp(self, min: const(min), max: max, name: nil)
    }

    @inlinable
    @inline(__always)
    func clamp(min: MPSGraphTensor, max: Float) -> MPSGraphTensor {
        operation.graph.clamp(self, min: min, max: const(max), name: nil)
    }

    @inlinable
    @inline(__always)
    func clamp(min: MPSGraphTensor, max: MPSGraphTensor) -> MPSGraphTensor {
        operation.graph.clamp(self, min: min, max: max, name: nil)
    }

    @inlinable
    @inline(__always)
    func abs() -> MPSGraphTensor {
        operation.graph.absolute(with: self, name: nil)
    }

    @inlinable
    @inline(__always)
    func squareRoot() -> MPSGraphTensor {
        operation.graph.squareRoot(with: self, name: nil)
    }

    @inlinable
    @inline(__always)
    func ceil() -> MPSGraphTensor {
        operation.graph.ceil(with: self, name: nil)
    }

    @inlinable
    @inline(__always)
    func floor() -> MPSGraphTensor {
        operation.graph.floor(with: self, name: nil)
    }

    @inlinable
    @inline(__always)
    func round() -> MPSGraphTensor {
        operation.graph.round(with: self, name: nil)
    }
}

public extension MPSGraphTensor {
    @inlinable
    @inline(__always)
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
    @inline(__always)
    func imagePlaceholder(
        dataType: MPSDataType,
        channels: Int,
        height: Int,
        width: Int,
        name: String? = nil
    ) -> MPSGraphTensor {
        placeholder(shape: [1, channels, height, width].nsnumbers, dataType: dataType, name: name)
    }

    @inlinable
    @inline(__always)
    func imagePlaceholder(
        dataType: MPSDataType,
        height: Int,
        width: Int,
        channels: Int,
        name: String? = nil
    ) -> MPSGraphTensor {
        placeholder(shape: [1, height, width, channels].nsnumbers, dataType: dataType, name: name)
    }
}

public extension MPSGraph {
    @inlinable
    @inline(__always)
    func min(_ x: MPSGraphTensor, _ y: MPSGraphTensor) -> MPSGraphTensor {
        minimum(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    func min(_ x: MPSGraphTensor, _ y: Float) -> MPSGraphTensor {
        minimum(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    func min(_ x: Float, _ y: MPSGraphTensor) -> MPSGraphTensor {
        minimum(y.const(x), y, name: nil)
    }
}

public extension MPSGraph {
    @inlinable
    @inline(__always)
    func max(_ x: MPSGraphTensor, _ y: MPSGraphTensor) -> MPSGraphTensor {
        maximum(x, y, name: nil)
    }

    @inlinable
    @inline(__always)
    func max(_ x: MPSGraphTensor, _ y: Float) -> MPSGraphTensor {
        maximum(x, x.const(y), name: nil)
    }

    @inlinable
    @inline(__always)
    func max(_ x: Float, _ y: MPSGraphTensor) -> MPSGraphTensor {
        maximum(y.const(x), y, name: nil)
    }
}

public extension MPSGraph {
    @inlinable
    @inline(__always)
    func matmul(_ x: MPSGraphTensor, _ y: MPSGraphTensor) -> MPSGraphTensor {
        matrixMultiplication(primary: x, secondary: y, name: nil)
    }
}
