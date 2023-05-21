extension OnnxModel {
    /// Split the graph into subgraphs.
    /// - Parameter splitOps: The names of the operators in the graph by which you want to split
    /// - Returns: A map where key is the name of the node (node opType exists in splitOps) and value is an array of input tensors that are "alive" at the time of the split.
    func split(by splitOps: Set<String>) -> [String: Set<String>] {
        struct NodeCounter {
            /// How many times the output of a node will be used as an input
            var readCount: Int
            /// Node lifetime: lower bound == first occurrence in the graph, upper bound == last occurrence as input in the graph.
            var range: ClosedRange<Int>
        }

        let graph = proto.graph

        var counters: [String: NodeCounter] = [:]
        counters.reserveCapacity(graph.node.count)

        var nodeIndices: [String: Int] = [:]
        nodeIndices.reserveCapacity(graph.node.count)

        // STEP 1: calculate counters for every node from start to end

        for (index, node) in graph.node.enumerated() {
            for outputName in node.output {
                nodeIndices[outputName] = index
            }

            for inputName in node.input {
                // only runtime tensors

                guard initializer[inputName] == nil else {
                    continue
                }

                let inputIndex = nodeIndices[inputName, default: 0]

                let counter = counters[inputName, default: .init(readCount: 0, range: inputIndex ... inputIndex)]

                counters[inputName] = .init(
                    readCount: counter.readCount + 1,
                    range: counter.range.lowerBound ... index
                )
            }
        }

        // STEP 2: now we have counters for every node and can split graph into subgraphs using this info.

        var subgraphs: [String: Set<String>] = [:]

        for (index, node) in graph.node.enumerated() {
            for inputName in node.input {
                guard initializer[inputName] == nil,
                      var counter = counters[inputName]
                else {
                    continue
                }

                if counter.readCount > 0 {
                    counter.readCount -= 1

                    counters[inputName] = counter
                } else {
                    counters[inputName] = nil
                }
            }

            // split

            if splitOps.contains(node.opType) {
                // TODO: optimize using some tree data structure/sorted collection

                let inputs = counters.filter { $0.value.range.contains(index) }

                subgraphs[node.name] = Set(inputs.map(\.key)).subtracting(Set(node.output))
            }
        }

        return subgraphs
    }
}
