import Foundation

/// Converts tool schemas to the strict subset providers accept for JSON-schema constrained
/// sampling: every property required (optional ones become nullable) and no additional
/// properties. Ports pi-ai's `makeStrictJsonSchema`.
enum StrictSchema {
    struct Unsupported: Error {
        let reason: String
    }

    private static let unsupportedKeys = [
        "$ref", "$defs", "definitions", "allOf", "oneOf", "patternProperties", "dependentSchemas", "dependencies",
        "unevaluatedProperties", "propertyNames", "contains", "prefixItems", "not", "if", "then", "else",
    ]

    /// Provider-specific keywords its strict mode rejects.
    typealias KeywordCheck = (String, JSONValue) -> Bool

    /// Anthropic strict tool use rejects these for the whole request.
    static let anthropicKeywordCheck: KeywordCheck = { key, value in
        switch key {
        case "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "maxItems", "uniqueItems",
            "minContains", "maxContains", "minProperties", "maxProperties":
            true
        case "minItems":
            value != 0 && value != 1
        case "format":
            !["date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"]
                .contains(value.stringValue ?? "")
        default:
            false
        }
    }

    /// Whether to send `tool` with strict sampling: `true` when requested and convertible,
    /// `nil` when strict mode is unavailable or the schema does not fit and the tool only prefers it.
    static func resolve(_ tool: Tool, supportsStrictMode: Bool, keywordCheck: KeywordCheck? = nil) throws -> Bool? {
        guard let mode = tool.strict else { return nil }
        guard supportsStrictMode else {
            if mode == .require {
                throw LLMError.invalidRequest("Tool \"\(tool.name)\" requires JSON-schema constrained sampling, but strict tools are unsupported.")
            }
            return nil
        }
        do {
            _ = try strict(tool.parameters, keywordCheck: keywordCheck)
            return true
        } catch let error as Unsupported {
            if mode == .require {
                throw LLMError.invalidRequest("Tool \"\(tool.name)\" requires JSON-schema constrained sampling, but \(error.reason).")
            }
            return nil
        }
    }

    static func parameters(of tool: Tool, strict isStrict: Bool?) -> JSONValue {
        guard isStrict == true, let converted = try? strict(tool.parameters, keywordCheck: nil) else {
            return tool.parameters
        }
        return converted
    }

    static func strict(_ schema: JSONValue, keywordCheck: KeywordCheck?) throws -> JSONValue {
        guard case .object(let root) = schema, root["type"] == "object" else {
            throw Unsupported(reason: "root schema must have type object")
        }
        return try strictNode(.object(root), keywordCheck: keywordCheck)
    }

    private static func strictNode(_ node: JSONValue, keywordCheck: KeywordCheck?) throws -> JSONValue {
        guard case .object(var schema) = node else { throw Unsupported(reason: "boolean schemas are unsupported") }
        if let key = unsupportedKeys.first(where: { schema[$0] != nil }) {
            throw Unsupported(reason: "\(key) schemas are unsupported")
        }
        if let keywordCheck, let entry = schema.first(where: { keywordCheck($0.key, $0.value) }) {
            throw Unsupported(reason: "\(entry.key): \(entry.value.jsonString()) is unsupported")
        }

        if let anyOf = schema["anyOf"] {
            guard case .array(let variants) = anyOf, !variants.isEmpty else {
                throw Unsupported(reason: "anyOf must contain at least one schema")
            }
            schema["anyOf"] = .array(try variants.map { variant in
                if case .object(let object) = variant {
                    let types = object["type"].flatMap { $0.stringValue.map { [$0] } ?? $0.arrayValue?.compactMap(\.stringValue) } ?? []
                    if types.contains("object") || types.contains("array") || object["properties"] != nil || object["items"] != nil {
                        throw Unsupported(reason: "object and array unions are unsupported")
                    }
                }
                return try strictNode(variant, keywordCheck: keywordCheck)
            })
        }

        if let items = schema["items"] {
            if case .array = items { throw Unsupported(reason: "tuple schemas are unsupported") }
            schema["items"] = try strictNode(items, keywordCheck: keywordCheck)
        }

        let isObject = schema["type"] == "object"
        if schema["properties"] != nil, !isObject { throw Unsupported(reason: "properties require type object") }
        guard isObject else { return .object(schema) }
        if let additional = schema["additionalProperties"], additional != false {
            throw Unsupported(reason: "schema-valued or true additionalProperties is unsupported")
        }
        let properties: JSONObject
        switch schema["properties"] {
        case nil: properties = [:]
        case .object(let value)?: properties = value
        default: throw Unsupported(reason: "object properties must be a schema map")
        }
        let required: Set<String>
        switch schema["required"] {
        case nil: required = []
        case .array(let names)? where names.allSatisfy({ $0.stringValue != nil }): required = Set(names.compactMap(\.stringValue))
        default: throw Unsupported(reason: "object required must be a string array")
        }
        guard required.isSubset(of: properties.keys) else { throw Unsupported(reason: "required contains an unknown property") }

        var strictProperties: JSONObject = [:]
        for (key, property) in properties {
            let converted = try strictNode(property, keywordCheck: keywordCheck)
            strictProperties[key] = required.contains(key) || allowsNull(converted) ? converted : ["anyOf": [converted, ["type": "null"]]]
        }
        schema["properties"] = .object(strictProperties)
        schema["required"] = .array(properties.keys.sorted().map(JSONValue.string))
        schema["additionalProperties"] = false
        return .object(schema)
    }

    private static func allowsNull(_ schema: JSONValue) -> Bool {
        guard case .object(let object) = schema else { return false }
        if object["type"] == "null" || object["type"]?.arrayValue?.contains("null") == true { return true }
        if object["const"] == .null || object["enum"]?.arrayValue?.contains(.null) == true { return true }
        return object["anyOf"]?.arrayValue?.contains(where: allowsNull) == true
    }
}
