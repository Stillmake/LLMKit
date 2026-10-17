import Foundation

/// Best-effort parsing of tool-call arguments while they stream.
///
/// Mirrors pi-ai's `parseStreamingJson`: complete JSON is parsed strictly; otherwise a lenient
/// recursive-descent pass keeps every value that is complete so far, closes open containers,
/// keeps the prefix of an unterminated string, tolerates raw control characters and invalid
/// escapes inside strings (pi's `repairJson`), and drops a trailing key with no value.
public enum PartialJSON {
    public static func parseObject(_ text: String) -> JSONObject {
        guard !text.allSatisfy(\.isWhitespace) else { return [:] }
        if let value = try? JSONValue(parsing: text), case .object(let object) = value {
            return object
        }
        var parser = Parser(bytes: Array(text.utf8))
        return parser.parseValue()?.objectValue ?? [:]
    }

    /// Parses the complete arguments text of a finished tool call, as pi-ai's `parseToolCallArguments`.
    ///
    /// Unlike `parseObject`, this never keeps a partially parsed prefix: text that is not a valid
    /// JSON object (after `repairJson`) yields empty arguments and an error that includes the text.
    public static func parseToolCallArguments(_ text: String) -> (arguments: JSONObject, error: String?) {
        guard !text.allSatisfy(\.isWhitespace) else { return ([:], nil) }
        let message: String
        do {
            let value = try (try? JSONValue(parsing: text)) ?? JSONValue(parsing: repair(text))
            if case .object(let object) = value { return (object, nil) }
            message = "Tool arguments must be a JSON object"
        } catch {
            message = error.localizedDescription
        }
        return ([:], "\(message)\n\nReceived arguments:\n\(text)")
    }

    /// Escapes raw control characters inside strings and doubles backslashes before invalid
    /// escapes, as pi-ai's `repairJson`.
    static func repair(_ json: String) -> String {
        let scalars = Array(json.unicodeScalars)
        var repaired = String.UnicodeScalarView()
        var inString = false
        var index = 0
        while index < scalars.count {
            let scalar = scalars[index]
            index += 1
            guard inString else {
                repaired.append(scalar)
                inString = scalar == "\""
                continue
            }
            switch scalar {
            case "\"":
                repaired.append(scalar)
                inString = false
            case "\\":
                let next = index < scalars.count ? scalars[index] : nil
                if next == "u", index + 5 <= scalars.count, scalars[index + 1..<index + 5].allSatisfy(\.properties.isASCIIHexDigit) {
                    repaired.append(contentsOf: scalars[index - 1..<index + 5])
                    index += 5
                } else if let next, "\"\\/bfnrt".unicodeScalars.contains(next) {
                    repaired.append(contentsOf: ["\\", next])
                    index += 1
                } else {
                    repaired.append(contentsOf: "\\\\".unicodeScalars)
                }
            case "\u{08}": repaired.append(contentsOf: "\\b".unicodeScalars)
            case "\u{0C}": repaired.append(contentsOf: "\\f".unicodeScalars)
            case "\n": repaired.append(contentsOf: "\\n".unicodeScalars)
            case "\r": repaired.append(contentsOf: "\\r".unicodeScalars)
            case "\t": repaired.append(contentsOf: "\\t".unicodeScalars)
            case "\u{00}"..."\u{1F}": repaired.append(contentsOf: String(format: "\\u%04x", scalar.value).unicodeScalars)
            default: repaired.append(scalar)
            }
        }
        return String(repaired)
    }

    private struct Parser {
        let bytes: [UInt8]
        var position = 0

        var isAtEnd: Bool { position >= bytes.count }

        mutating func skipWhitespace() {
            while !isAtEnd, [0x20, 0x09, 0x0A, 0x0D].contains(bytes[position]) {
                position += 1
            }
        }

        mutating func parseValue() -> JSONValue? {
            skipWhitespace()
            guard !isAtEnd else { return nil }
            switch bytes[position] {
            case UInt8(ascii: "{"):
                position += 1
                var object: JSONObject = [:]
                while true {
                    skipWhitespace()
                    guard !isAtEnd else { return .object(object) }
                    switch bytes[position] {
                    case UInt8(ascii: "}"):
                        position += 1
                        return .object(object)
                    case UInt8(ascii: ","):
                        position += 1
                    case UInt8(ascii: "\""):
                        guard let key = parseString(requireTerminated: true) else { return .object(object) }
                        skipWhitespace()
                        guard !isAtEnd, bytes[position] == UInt8(ascii: ":") else { return .object(object) }
                        position += 1
                        guard let value = parseValue() else { return .object(object) }
                        object[key] = value
                    default:
                        position = bytes.count
                        return .object(object)
                    }
                }

            case UInt8(ascii: "["):
                position += 1
                var array: [JSONValue] = []
                while true {
                    skipWhitespace()
                    guard !isAtEnd else { return .array(array) }
                    switch bytes[position] {
                    case UInt8(ascii: "]"):
                        position += 1
                        return .array(array)
                    case UInt8(ascii: ","):
                        position += 1
                    default:
                        guard let value = parseValue() else { return .array(array) }
                        array.append(value)
                    }
                }

            case UInt8(ascii: "\""):
                return parseString().map(JSONValue.string)
            case UInt8(ascii: "t"):
                return parseLiteral("true", .bool(true))
            case UInt8(ascii: "f"):
                return parseLiteral("false", .bool(false))
            case UInt8(ascii: "n"):
                return parseLiteral("null", .null)

            default:
                let start = position
                while !isAtEnd, "+-0123456789.eE".utf8.contains(bytes[position]) {
                    position += 1
                }
                guard position > start else {
                    position = bytes.count
                    return nil
                }
                // A number cut off at the end of input may still grow; only keep it once terminated.
                guard !isAtEnd else { return nil }
                return Double(String(decoding: bytes[start..<position], as: UTF8.self)).map(JSONValue.number)
            }
        }

        /// Returns the decoded string; an unterminated string yields its prefix unless `requireTerminated`.
        mutating func parseString(requireTerminated: Bool = false) -> String? {
            position += 1
            var utf8: [UInt8] = []
            while !isAtEnd {
                let byte = bytes[position]
                position += 1
                guard byte != UInt8(ascii: "\"") else { return String(decoding: utf8, as: UTF8.self) }
                guard byte == UInt8(ascii: "\\") else {
                    utf8.append(byte)
                    continue
                }
                guard !isAtEnd else { break }
                let escape = bytes[position]
                position += 1
                switch escape {
                case UInt8(ascii: "\""), UInt8(ascii: "\\"), UInt8(ascii: "/"): utf8.append(escape)
                case UInt8(ascii: "b"): utf8.append(0x08)
                case UInt8(ascii: "f"): utf8.append(0x0C)
                case UInt8(ascii: "n"): utf8.append(0x0A)
                case UInt8(ascii: "r"): utf8.append(0x0D)
                case UInt8(ascii: "t"): utf8.append(0x09)
                case UInt8(ascii: "u"):
                    guard let high = parseHex4() else {
                        // Truncated or malformed \u escape: keep it literally, as repairJson does.
                        utf8 += Array("\\u".utf8)
                        continue
                    }
                    var scalar = Unicode.Scalar(high)
                    if (0xD800..<0xDC00).contains(high) {
                        let checkpoint = position
                        if position + 1 < bytes.count, bytes[position] == UInt8(ascii: "\\"), bytes[position + 1] == UInt8(ascii: "u") {
                            position += 2
                            if let low = parseHex4(), (0xDC00..<0xE000).contains(low) {
                                scalar = Unicode.Scalar(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))
                            } else {
                                position = checkpoint
                            }
                        }
                    }
                    utf8 += Array(String(Character(scalar ?? "\u{FFFD}")).utf8)
                default:
                    // Invalid escape: keep the backslash literally.
                    utf8 += [UInt8(ascii: "\\"), escape]
                }
            }
            return requireTerminated ? nil : String(decoding: utf8, as: UTF8.self)
        }

        mutating func parseHex4() -> UInt32? {
            guard position + 4 <= bytes.count,
                bytes[position..<position + 4].allSatisfy({ ($0 >= 0x30 && $0 <= 0x39) || ($0 | 0x20 >= 0x61 && $0 | 0x20 <= 0x66) }),
                let value = UInt32(String(decoding: bytes[position..<position + 4], as: UTF8.self), radix: 16)
            else { return nil }
            position += 4
            return value
        }

        mutating func parseLiteral(_ literal: String, _ value: JSONValue) -> JSONValue? {
            let expected = Array(literal.utf8)
            let available = bytes[position..<min(bytes.count, position + expected.count)]
            guard expected.starts(with: available) else {
                position = bytes.count
                return nil
            }
            position += available.count
            return available.count == expected.count ? value : nil
        }
    }
}
