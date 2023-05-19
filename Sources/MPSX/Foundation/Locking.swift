import Foundation
import os

protocol Lock {
    func lock()
    func unlock()
}

extension NSLock: Lock {}

@available(iOS 16.0, macOS 13.0, *)
extension OSAllocatedUnfairLock: Lock where State == Void {}

func newLock() -> Lock {
    if #available(iOS 16.0, macOS 13.0, *) {
        return OSAllocatedUnfairLock<Void>()
    }
    return NSLock()
}

extension Lock {
    @discardableResult
    @inlinable
    func execute<T>(_ action: () -> T) -> T {
        lock(); defer { unlock() }
        return action()
    }
}
