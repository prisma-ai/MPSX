@testable import MPSX
import XCTest

final class LRUCacheTests: XCTestCase {
    func testLRUCacheInitialization() {
        let cache = LRUCache<String, Int>(5)
        XCTAssertEqual(cache.count, 0)
    }

    func testPutAndGet() {
        let cache = LRUCache<String, Int>(5)
        cache.put("TestKey", 10)
        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.get("TestKey"), 10)
    }

    func testLRUCacheCapacityOverflow() {
        let cache = LRUCache<String, Int>(2)
        cache.put("Key1", 1)
        cache.put("Key2", 2)
        cache.put("Key3", 3)
        XCTAssertEqual(cache.count, 2)
        XCTAssertNil(cache.get("Key1"))
        XCTAssertEqual(cache.get("Key2"), 2)
        XCTAssertEqual(cache.get("Key3"), 3)
    }

    func testLRUCacheRetrievalUpdates() {
        let cache = LRUCache<String, Int>(2)
        cache.put("Key1", 1)
        cache.put("Key2", 2)
        XCTAssertEqual(cache.get("Key1"), 1) // Access Key1
        cache.put("Key3", 3) // This should remove Key2
        XCTAssertEqual(cache.count, 2)
        XCTAssertNil(cache.get("Key2"))
        XCTAssertEqual(cache.get("Key1"), 1)
        XCTAssertEqual(cache.get("Key3"), 3)
    }

    func testLRUCacheReplacement() {
        let cache = LRUCache<String, Int>(2)
        cache.put("Key1", 1)
        cache.put("Key1", 2)
        XCTAssertEqual(cache.count, 1)
        XCTAssertEqual(cache.get("Key1"), 2)
    }
}
