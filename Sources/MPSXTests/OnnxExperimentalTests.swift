import MetalKit
import MetalPerformanceShaders
import MetalPerformanceShadersGraph
@testable import MPSX
import XCTest

final class OnnxExperimentalTests: XCTestCase {
    func testModelSplit() throws {
        let model = try OnnxModel(data: data(bundlePath: "\(testResourcesPath)/shufflenet-v2-12.onnx"))

        let transposeSubgraphs: [String: Set<String>] = [
            "Transpose_20": ["358"],
            "Transpose_35": ["374"],
            "Transpose_50": ["390"],
            "Transpose_65": ["406"],
            "Transpose_84": ["425"],
            "Transpose_99": ["441"],
            "Transpose_114": ["457"],
            "Transpose_129": ["473"],
            "Transpose_144": ["489"],
            "Transpose_159": ["505"],
            "Transpose_174": ["521"],
            "Transpose_189": ["537"],
            "Transpose_208": ["556"],
            "Transpose_223": ["572"],
            "Transpose_238": ["588"],
            "Transpose_253": ["604"],
        ]

        let concatSubgraphs: [String: Set<String>] = [
            "Concat_17": ["347", "355"],
            "Concat_32": ["362", "371"],
            "Concat_47": ["378", "387"],
            "Concat_62": ["394", "403"],
            "Concat_81": ["414", "422"],
            "Concat_96": ["429", "438"],
            "Concat_111": ["445", "454"],
            "Concat_126": ["461", "470"],
            "Concat_141": ["477", "486"],
            "Concat_156": ["493", "502"],
            "Concat_171": ["509", "518"],
            "Concat_186": ["525", "534"],
            "Concat_205": ["545", "553"],
            "Concat_220": ["560", "569"],
            "Concat_235": ["576", "585"],
            "Concat_250": ["592", "601"],
        ]

        XCTAssertEqual(model.split(by: ["Transpose"]), transposeSubgraphs)
        XCTAssertEqual(model.split(by: ["Concat"]), concatSubgraphs)
        XCTAssertEqual(model.split(by: ["Concat", "Transpose"]), concatSubgraphs.merging(transposeSubgraphs, uniquingKeysWith: { x, _ in x }))
    }
}
