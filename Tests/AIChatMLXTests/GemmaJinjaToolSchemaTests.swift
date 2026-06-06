import XCTest
@testable import AIChatMLX

final class GemmaJinjaToolSchemaTests: XCTestCase {

    func test_enumOnlyProperty_gainsTypeString() {
        let encoded: [String: Any] = [
            "type": "object",
            "properties": [
                "action": [
                    "enum": ["memory_recall", "memory_store"],
                    "description": "Action to perform",
                ] as [String: Any],
            ],
            "required": ["action"],
        ]

        let safe = GemmaJinjaToolSchema.sanitizeParameters(encoded)
        let action = (safe["properties"] as? [String: Any])?["action"] as? [String: Any]

        XCTAssertEqual(action?["type"] as? String, "string")
        XCTAssertNil(action?["enum"])
        XCTAssertTrue((action?["description"] as? String)?.contains("memory_recall") == true)
    }

    func test_integerProperty_passesThrough() {
        let encoded: [String: Any] = [
            "type": "object",
            "properties": [
                "limit": ["type": "integer", "description": "Max results"] as [String: Any],
            ],
        ]

        let safe = GemmaJinjaToolSchema.sanitizeParameters(encoded)
        let limit = (safe["properties"] as? [String: Any])?["limit"] as? [String: Any]

        XCTAssertEqual(limit?["type"] as? String, "integer")
    }

    func test_arrayProperty_preservesItems() {
        let encoded: [String: Any] = [
            "type": "object",
            "properties": [
                "breadcrumbs": [
                    "type": "array",
                    "description": "Labels",
                    "items": ["type": "string"],
                ] as [String: Any],
            ],
        ]

        let safe = GemmaJinjaToolSchema.sanitizeParameters(encoded)
        let breadcrumbs = (safe["properties"] as? [String: Any])?["breadcrumbs"] as? [String: Any]
        let items = breadcrumbs?["items"] as? [String: Any]

        XCTAssertEqual(breadcrumbs?["type"] as? String, "array")
        XCTAssertEqual(items?["type"] as? String, "string")
    }
}
