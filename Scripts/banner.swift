#!/usr/bin/env swift
// Renders the README banner with SwiftUI's ImageRenderer. From the repo root:
//     swift Scripts/banner.swift
// Writes assets/banner.png at 2560×800 (1280×400 points at 2×).

import AppKit
import SwiftUI

/// Five wire formats folding into one call, drawn in a 1024-point icon space.
struct Lanes: Shape {
    func path(in rect: CGRect) -> Path {
        var path = Path()
        for y in [320.0, 416, 608, 704] {
            let pull = abs(y - 512) > 150 ? 430.0 : 400
            path.move(to: CGPoint(x: 250, y: y))
            path.addCurve(to: CGPoint(x: 640, y: 512), control1: CGPoint(x: pull, y: y), control2: CGPoint(x: 470, y: 512))
        }
        path.move(to: CGPoint(x: 250, y: 512))
        path.addLine(to: CGPoint(x: 780, y: 512))
        return path.applying(CGAffineTransform(scaleX: rect.width / 1024, y: rect.height / 1024))
    }
}

struct Banner: View {
    var body: some View {
        HStack(spacing: 56) {
            ZStack {
                // Apple's continuous corners at the icon proportion: 22.5% of the side.
                RoundedRectangle(cornerRadius: 200 * 0.225, style: .continuous)
                    .fill(LinearGradient(colors: [Color(hex: 0xFF8D28), Color(hex: 0xFF383C)], startPoint: .top, endPoint: .bottom))
                Lanes()
                    .stroke(.white, style: StrokeStyle(lineWidth: 200 * 46 / 1024, lineCap: .round))
            }
            .frame(width: 200, height: 200)

            VStack(alignment: .leading, spacing: 14) {
                Text("LLMKit")
                    .font(.system(size: 104, weight: .bold))
                    .tracking(-104 * 0.045)
                Text("One Swift API for 40 LLM providers.")
                    .font(.system(size: 34, weight: .medium))
                    .tracking(-34 * 0.02)
                    .foregroundStyle(Color(hex: 0xA1A1A6))
            }
            .foregroundStyle(Color(hex: 0xF5F5F7))
        }
        .frame(width: 1280, height: 400)
        .background(Color(hex: 0x050505))
    }
}

extension Color {
    init(hex: UInt32) {
        self.init(.sRGB, red: Double(hex >> 16 & 0xFF) / 255, green: Double(hex >> 8 & 0xFF) / 255, blue: Double(hex & 0xFF) / 255)
    }
}

try MainActor.assumeIsolated {
    let renderer = ImageRenderer(content: Banner())
    renderer.scale = 2
    let output = URL(fileURLWithPath: "assets/banner.png")
    guard let image = renderer.cgImage,
        let png = NSBitmapImageRep(cgImage: image).representation(using: .png, properties: [:])
    else { fatalError("ImageRenderer produced no image") }
    try png.write(to: output)
    print("Wrote \(output.path) (\(image.width)×\(image.height))")
}
