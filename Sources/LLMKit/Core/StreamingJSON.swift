import Foundation

/// Incremental partial JSON parsing for streamed tool-call arguments, ported from pi-ai's
/// `utils/streaming-json.ts`. Open containers are copied only when producing a snapshot.
final class StreamingJSON {
    private enum Slot { case root, member(String), element(Int) }
    private final class Frame {
        var object: JSONObject?
        var array: [JSONValue]?
        let slot: Slot?
        var key: String?
        var phase: Phase
        enum Phase { case key, colon, value, comma }
        init(object: Bool, slot: Slot?) {
            self.object = object ? [:] : nil
            self.array = object ? nil : []
            self.slot = slot
            self.phase = object ? .key : .value
        }
        var size: Int { object?.count ?? array?.count ?? 0 }
    }
    private struct OpenString {
        var isKey: Bool
        var slot: Slot?
        var text = ""
        var escape = ""
        var highSurrogate: UInt16?
    }
    private var root: JSONValue?
    private var frames: [Frame] = []
    private var openString: OpenString?
    private var scalar = ""
    private var scalarSlot: Slot?
    private var done = false
    private var dirty = false
    private var credit = 0
    private var lastSnapshot: JSONValue?

    func append(_ text: String) {
        credit += 256 + 8 * text.utf8.count
        for char in text.unicodeScalars {
            if done { break }
            consume(char)
        }
    }

    private func consume(_ char: Unicode.Scalar) {
        if var string = openString {
            openString = nil
            if !string.escape.isEmpty {
                if string.escape == "\\" {
                    let escapes: [Unicode.Scalar: String] = ["\"": "\"", "\\": "\\", "/": "/", "b": "\u{8}", "f": "\u{c}", "n": "\n", "r": "\r", "t": "\t"]
                    if let decoded = escapes[char] { string.escape = ""; appendString(decoded, to: &string); openString = string; return }
                    if char == "u" { string.escape = "\\u"; openString = string; return }
                    string.escape = ""
                    appendString("\\", to: &string)
                } else if char.properties.isASCIIHexDigit {
                    string.escape.unicodeScalars.append(char)
                    if string.escape.count == 6 {
                        let unit = UInt16(string.escape.suffix(4), radix: 16)!
                        string.escape = ""
                        if (0xD800...0xDBFF).contains(unit) {
                            if string.highSurrogate != nil { appendString("�", to: &string) }
                            string.highSurrogate = unit
                        } else if (0xDC00...0xDFFF).contains(unit), let high = string.highSurrogate {
                            string.highSurrogate = nil
                            let code = 0x10000 + (UInt32(high) - 0xD800) * 0x400 + UInt32(unit) - 0xDC00
                            appendString(String(Unicode.Scalar(code)!), to: &string)
                        } else {
                            appendString(String(Unicode.Scalar(UInt32(unit)) ?? "�"), to: &string)
                        }
                    }
                    openString = string
                    return
                } else {
                    let literal = string.escape
                    string.escape = ""
                    appendString(literal, to: &string)
                }
            }
            switch char {
            case "\"":
                if string.highSurrogate != nil { string.highSurrogate = nil; appendString("�", to: &string) }
                openString = nil
                if string.isKey { frames.last?.key = string.text; frames.last?.phase = .colon }
                else { place(string.slot, value: .string(string.text)); valueEnded() }
            case "\\": string.escape = "\\"; openString = string
            default: appendString(String(char), to: &string); openString = string
            }
            return
        }
        if !scalar.isEmpty {
            if !Self.isDelimiter(char) { scalar.unicodeScalars.append(char); dirty = true; return }
            endScalar()
            if done { return }
        }
        switch char {
        case " ", "\n", "\r", "\t": return
        case "\"":
            let isKey = frames.last.map { $0.object != nil && ($0.phase == .key || $0.phase == .comma) } ?? false
            let slot = isKey ? nil : valueSlot()
            openString = OpenString(isKey: isKey, slot: slot)
            if !isKey { place(slot, value: .string("")) }
        case "{", "[":
            let slot = valueSlot()
            let object = char == "{"
            place(slot, value: object ? .object([:]) : .array([]))
            frames.append(Frame(object: object, slot: slot)); dirty = true
        case "}", "]":
            guard let frame = frames.popLast() else { return }
            let value: JSONValue = frame.object.map(JSONValue.object) ?? .array(frame.array ?? [])
            place(frame.slot, value: value)
            dirty = true; valueEnded()
        case ":": if frames.last?.phase == .colon { frames.last?.phase = .value }
        case ",": if let frame = frames.last { frame.phase = frame.object == nil ? .value : .key }
        default: scalar = String(char); scalarSlot = valueSlot(); dirty = true
        }
    }

    private func appendString(_ text: String, to string: inout OpenString) {
        if string.highSurrogate != nil { string.highSurrogate = nil; string.text += "�" }
        string.text += text
        if !string.isKey { dirty = true }
    }
    private func valueSlot() -> Slot? {
        guard let frame = frames.last else { return root == nil ? .root : nil }
        if let array = frame.array { return .element(array.count) }
        if (frame.phase == .value || frame.phase == .colon), let key = frame.key { return .member(key) }
        return nil
    }
    private func place(_ slot: Slot?, value: JSONValue) {
        guard let slot else { return }
        dirty = true
        switch slot {
        case .root: root = value
        case .member(let key): frames.last?.object?[key] = value
        case .element(let index):
            guard let frame = frames.last, let count = frame.array?.count else { return }
            if index == count { frame.array?.append(value) }
            else if index >= 0, index < count { frame.array?[index] = value }
        }
    }
    private func valueEnded() {
        if let frame = frames.last { frame.phase = .comma }
        else if root != nil { done = true }
    }
    private func endScalar() {
        let token = scalar; let slot = scalarSlot
        scalar = ""; scalarSlot = nil
        if let value = Self.scalarValue(token, pending: false) { place(slot, value: value) }
        valueEnded()
    }
    private static func isDelimiter(_ char: Unicode.Scalar) -> Bool {
        " \t\r\n,:{}[]\"".unicodeScalars.contains(char)
    }
    private static func scalarValue(_ token: String, pending: Bool) -> JSONValue? {
        for (literal, value): (String, JSONValue) in [("true", true), ("false", false), ("null", .null)] {
            if token == literal || (pending && literal.hasPrefix(token)) { return value }
        }
        let pattern = #"^-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?$"#
        if token.range(of: pattern, options: .regularExpression) != nil, let value = Double(token), value.isFinite { return .number(value) }
        if pending, let range = token.range(of: #"[eE][+-]?$"#, options: .regularExpression) {
            return scalarValue(String(token[..<range.lowerBound]), pending: false)
        }
        return nil
    }
    func snapshot() -> JSONObject {
        guard dirty else { return lastSnapshot?.objectValue ?? [:] }
        let cost = frames.reduce(0) { $0 + $1.size + 1 } + (openString?.text.utf8.count ?? 0)
        guard cost <= credit else { return lastSnapshot?.objectValue ?? [:] }
        credit -= cost; dirty = false
        var pending: (Slot, JSONValue)?
        if scalar.count <= 64, let slot = scalarSlot, let value = Self.scalarValue(scalar, pending: true) { pending = (slot, value) }
        if let string = openString, !string.isKey, let slot = string.slot { pending = (slot, .string(string.text)) }
        if frames.isEmpty {
            if let (slot, value) = pending, case .root = slot { lastSnapshot = value }
            else { lastSnapshot = root }
        } else {
            var child: JSONValue?
            var childSlot: Slot?
            for frame in frames.reversed() {
                var object = frame.object
                var array = frame.array
                func set(_ slot: Slot, _ value: JSONValue) {
                    switch slot {
                    case .member(let key): object?[key] = value
                    case .element(let index):
                        if index == array?.count { array?.append(value) }
                        else if array?.indices.contains(index) == true { array?[index] = value }
                    case .root: break
                    }
                }
                if child == nil, let (slot, value) = pending { set(slot, value) }
                if let child, let childSlot { set(childSlot, child) }
                child = object.map(JSONValue.object) ?? .array(array ?? [])
                childSlot = frame.slot
            }
            if case .root? = childSlot { lastSnapshot = child } else { lastSnapshot = root }
        }
        return lastSnapshot?.objectValue ?? [:]
    }
}
