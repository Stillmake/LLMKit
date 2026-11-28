/// Replay formats stored in a Completions thinking signature.
enum OpenAICompletionsReasoning {
    case field(String)
    case details([JSONValue])
    case opaque(field: String?, value: String)

    /// Stream fields in priority order. Some providers duplicate the same text under multiple names.
    static let fields = ["reasoning_content", "reasoning", "reasoning_text"]

    static func isReasoningDetail(_ detail: JSONObject) -> Bool {
        let commonFieldsValid = (detail["id"].map { $0 == .null || $0.stringValue != nil } ?? true)
            && (detail["format"].map { $0.stringValue != nil } ?? true)
            && (detail["index"].map { $0.doubleValue != nil } ?? true)
        guard commonFieldsValid else { return false }
        switch detail["type"]?.stringValue {
        case "reasoning.summary": return detail["summary"]?.stringValue != nil
        case "reasoning.encrypted": return detail["data"]?.stringValue != nil
        case "reasoning.text":
            return detail["text"]?.stringValue != nil && (detail["signature"].map { $0 == .null || $0.stringValue != nil } ?? true)
        default: return false
        }
    }
}
