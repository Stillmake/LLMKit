import Foundation

/// A JSON document. Request bodies, tool schemas, tool arguments and provider-specific
/// extras all travel as `JSONValue`, so adapters build wire payloads without mirroring SDK types.
public enum JSONValue: Sendable, Hashable {
    case null
    case bool(Bool)
    case number(Double)
    case string(String)
    case array([JSONValue])
    case object([String: JSONValue])
}

public typealias JSONObject = [String: JSONValue]

// MARK: - Literals

// No ExpressibleByNilLiteral: `nil` in a `JSONValue?` context must mean "absent", not `.null`.
extension JSONValue: ExpressibleByBooleanLiteral, ExpressibleByIntegerLiteral, ExpressibleByFloatLiteral,
    ExpressibleByStringLiteral, ExpressibleByArrayLiteral, ExpressibleByDictionaryLiteral
{
    public init(booleanLiteral value: Bool) { self = .bool(value) }
    public init(integerLiteral value: Int) { self = .number(Double(value)) }
    public init(floatLiteral value: Double) { self = .number(value) }
    public init(stringLiteral value: String) { self = .string(value) }
    public init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
    public init(dictionaryLiteral elements: (String, JSONValue)...) {
        self = .object(Dictionary(elements, uniquingKeysWith: { _, last in last }))
    }
}

// MARK: - Access

extension JSONValue {
    public subscript(key: String) -> JSONValue? {
        guard case .object(let object) = self else { return nil }
        return object[key]
    }

    public subscript(index: Int) -> JSONValue? {
        guard case .array(let array) = self, array.indices.contains(index) else { return nil }
        return array[index]
    }

    public var stringValue: String? {
        guard case .string(let value) = self else { return nil }
        return value
    }

    public var doubleValue: Double? {
        guard case .number(let value) = self else { return nil }
        return value
    }

    public var intValue: Int? {
        guard case .number(let value) = self, value.rounded() == value, abs(value) < 0x1p53 else { return nil }
        return Int(value)
    }

    public var boolValue: Bool? {
        guard case .bool(let value) = self else { return nil }
        return value
    }

    public var arrayValue: [JSONValue]? {
        guard case .array(let value) = self else { return nil }
        return value
    }

    public var objectValue: JSONObject? {
        guard case .object(let value) = self else { return nil }
        return value
    }

    public var isNull: Bool { self == .null }

    /// Recursively merges `other` into `self`: objects merge key by key, everything else is replaced.
    public mutating func deepMerge(_ other: JSONValue) {
        guard case .object(var base) = self, case .object(let overlay) = other else {
            self = other
            return
        }
        self = .null
        for (key, value) in overlay {
            if var existing = base[key] {
                existing.deepMerge(value)
                base[key] = existing
            } else {
                base[key] = value
            }
        }
        self = .object(base)
    }
}

// MARK: - Serialization

extension JSONValue {
    /// Parses complete JSON. Throws on malformed input; use `PartialJSON` for streamed fragments.
    public init(data: Data) throws {
        self.init(foundation: try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed]))
    }

    public init(parsing text: String) throws {
        try self.init(data: Data(text.utf8))
    }

    /// Serializes with sorted keys so identical payloads produce identical bytes, which keeps
    /// provider prompt caches warm across requests.
    public func data() -> Data {
        do {
            return try JSONSerialization.data(
                withJSONObject: foundationObject,
                options: [.fragmentsAllowed, .sortedKeys, .withoutEscapingSlashes]
            )
        } catch {
            assertionFailure("JSONValue is always serializable: \(error)")
            return Data("null".utf8)
        }
    }

    public func jsonString() -> String {
        String(decoding: data(), as: UTF8.self)
    }

    init(foundation value: Any) {
        switch value {
        case let number as NSNumber where CFGetTypeID(number) == CFBooleanGetTypeID():
            self = .bool(number.boolValue)
        case let number as NSNumber:
            self = .number(number.doubleValue)
        case let string as String:
            self = .string(string)
        case let array as [Any]:
            self = .array(array.map(JSONValue.init(foundation:)))
        case let object as [String: Any]:
            self = .object(object.mapValues(JSONValue.init(foundation:)))
        default:
            self = .null
        }
    }

    var foundationObject: Any {
        switch self {
        case .null: NSNull()
        case .bool(let value): NSNumber(value: value)
        case .number(let value):
            value.rounded() == value && abs(value) < 0x1p53 ? NSNumber(value: Int64(value)) : NSNumber(value: value)
        case .string(let value): value
        case .array(let value): value.map(\.foundationObject)
        case .object(let value): value.mapValues(\.foundationObject)
        }
    }
}

// MARK: - Codable

extension JSONValue: Codable {
    public init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            self = .object(try container.decode(JSONObject.self))
        }
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .null: try container.encodeNil()
        case .bool(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .string(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        }
    }
}
