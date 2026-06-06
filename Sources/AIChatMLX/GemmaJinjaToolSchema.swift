import Foundation

/// Normalizes JSON Schema tool parameter dictionaries for Gemma 4's Jinja chat template.
///
/// Gemma's template applies `value['type'] | upper` on every property and branches on
/// STRING, ARRAY, OBJECT, etc. It fails when `type` is missing.
///
/// The Swift `JSONSchema` package encodes properties that were parsed with a top-level
/// `enum` key as `SchemaType.enum`, emitting only `"enum": [...]` without `"type"`.
/// This helper repairs that at the MLX boundary so all `ChatRequestOptions.ToolDefinition`
/// consumers are safe regardless of how schemas were constructed upstream.
enum GemmaJinjaToolSchema {

    /// Sanitize a full tool parameters object (`type`, `properties`, `required`).
    static func sanitizeParameters(_ raw: [String: Any]) -> [String: Any] {
        var result = raw
        if var properties = result["properties"] as? [String: Any] {
            for (key, value) in properties {
                guard let prop = value as? [String: Any] else { continue }
                properties[key] = sanitizeProperty(prop)
            }
            result["properties"] = properties
        }
        if result["type"] == nil {
            result["type"] = "object"
        }
        return result
    }

    /// Sanitize a single property schema.
    static func sanitizeProperty(_ raw: [String: Any]) -> [String: Any] {
        var prop = raw

        if let enumValues = prop.removeValue(forKey: "enum") as? [String] {
            let hint = enumValues.joined(separator: ", ")
            let existing = prop["description"] as? String ?? ""
            prop["description"] = existing.isEmpty
                ? "Allowed values: \(hint)."
                : "\(existing) Allowed values: \(hint)."
        } else if let enumValues = prop.removeValue(forKey: "enum") as? [Any] {
            let strings = enumValues.compactMap { $0 as? String }
            if !strings.isEmpty {
                let hint = strings.joined(separator: ", ")
                let existing = prop["description"] as? String ?? ""
                prop["description"] = existing.isEmpty
                    ? "Allowed values: \(hint)."
                    : "\(existing) Allowed values: \(hint)."
            }
        }

        if let types = prop["type"] as? [Any], let first = types.compactMap({ $0 as? String }).first {
            prop["type"] = first
        }

        if var items = prop["items"] as? [String: Any] {
            prop["items"] = sanitizeProperty(items)
        }

        if prop["type"] == nil {
            prop["type"] = "string"
        }

        return prop
    }
}
