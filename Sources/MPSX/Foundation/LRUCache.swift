final class LRUCache<Key: Hashable, Value> {
    // MARK: Lifecycle

    init(_ capacity: Int) {
        self.capacity = max(0, capacity)
        nodesDict = [Key: Node]()
    }

    // MARK: Internal

    var count: Int {
        nodesDict.count
    }

    func get(_ key: Key) -> Value? {
        guard let node = nodesDict[key] else {
            return nil
        }

        remove(node)
        add(node)
        return node.value
    }

    func put(_ key: Key, _ value: Value) {
        if let node = nodesDict[key] {
            node.value = value
            remove(node)
            add(node)
        } else {
            let node = Node(key, value)
            if nodesDict.count == capacity, let tail {
                nodesDict.removeValue(forKey: tail.key)
                remove(tail)
            }
            add(node)
            nodesDict[key] = node
        }
    }

    // MARK: Private

    private final class Node {
        // MARK: Lifecycle

        init(_ key: Key, _ value: Value) {
            self.key = key
            self.value = value
        }

        // MARK: Internal

        var key: Key
        var value: Value
        var prev: Node?
        var next: Node?
    }

    private var nodesDict: [Key: Node]
    private var capacity: Int
    private var head: Node?
    private var tail: Node?

    private func add(_ node: Node) {
        node.next = head
        node.prev = nil

        if let head {
            head.prev = node
        }

        head = node

        if tail == nil {
            tail = head
        }
    }

    private func remove(_ node: Node) {
        let prev = node.prev
        let next = node.next

        if let prev {
            prev.next = next
        } else {
            head = next
        }

        if let next {
            next.prev = prev
        } else {
            tail = prev
        }
    }
}
