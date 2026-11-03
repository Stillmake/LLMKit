import Foundation

/// Ports pi-ai's `utils/openrouter-cost.ts`.
enum OpenRouterCost {
    static func apply(_ reported: JSONValue, to usage: inout Usage) {
        func valid(_ value: JSONValue?) -> Double? {
            guard let number = value?.doubleValue, number.isFinite, number >= 0 else { return nil }
            return number
        }
        guard let cost = valid(reported["cost"]), let byok = reported["is_byok"]?.boolValue else { return }
        var total = cost
        if byok {
            guard let details = reported["cost_details"], !details.isNull else { return }
            let upstream: Double
            if details["upstream_inference_cost"]?.isNull == true {
                guard let prompt = valid(details["upstream_inference_prompt_cost"]),
                      let completion = valid(details["upstream_inference_completions_cost"]) else { return }
                upstream = prompt + completion
            } else {
                guard let value = valid(details["upstream_inference_cost"]) else { return }
                upstream = value
            }
            guard upstream.isFinite else { return }
            total += upstream
        }
        usage.cost.total = total
        usage.cost.totalFromProvider = true
    }
}
