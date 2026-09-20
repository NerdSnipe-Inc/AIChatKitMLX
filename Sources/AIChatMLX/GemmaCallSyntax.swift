import Foundation

/// Parser for Gemma's textual function-call syntax: `call:NAME{key:value,...}`.
///
/// Values use Gemma's `<|"|>string<|"|>` quoting (also tolerated: JSON `"..."` strings and the
/// `<escape>` delimiter), bare numbers / booleans / null, nested objects and arrays. String
/// contents are opaque — braces, commas and quotes inside them never affect structure.
enum GemmaCallSyntax {

    static let quote = #"<|"|>"#
    static let escape = "<escape>"

    struct Call {
        let name: String
        let argumentsJSON: String
    }

    /// Result of looking for a call at a given `call:` position.
    enum Scan {
        /// Not a call (e.g. prose like "give me a call: 555-1234").
        case notACall
        /// Looks like the start of a call but the text ends before it is complete.
        case incomplete
        /// A complete call spanning `range` (from `call:` through the closing `}`).
        case complete(Call, range: Range<String.Index>)
        /// Structurally complete (`call:name{...}`) but the arguments could not be parsed.
        case malformed(range: Range<String.Index>)
    }

    // MARK: - Public entry points

    /// Examines `text` at `start` (which must point at `call:`).
    static func scan(_ text: String, at start: String.Index) -> Scan {
        var i = text.index(start, offsetBy: 5)  // past "call:"
        let nameStart = i
        while i < text.endIndex, isNameChar(text[i]) { i = text.index(after: i) }
        if i == text.endIndex { return .incomplete }
        guard i > nameStart, text[i] == "{" else { return .notACall }
        let name = String(text[nameStart..<i])
        guard let close = balancedEnd(in: text, openingBrace: i) else { return .incomplete }
        let bodyRange = text.index(after: i)..<close
        let range = start..<text.index(after: close)
        guard let json = argumentsJSON(from: String(text[bodyRange])) else { return .malformed(range: range) }
        return .complete(Call(name: name, argumentsJSON: json), range: range)
    }

    /// Whether `text[idx]` may begin a call: start of text or a preceding non-word character.
    static func isCallBoundary(_ text: String, _ idx: String.Index) -> Bool {
        guard idx > text.startIndex else { return true }
        let prev = text[text.index(before: idx)]
        return !(prev.isLetter || prev.isNumber || prev == "_")
    }

    /// Finds the first *boundary-respecting* `call:` at or after `from`.
    static func nextCallCandidate(in text: String, from: String.Index? = nil, before limit: String.Index? = nil) -> String.Index? {
        var search = from ?? text.startIndex
        let end = limit ?? text.endIndex
        while search < end, let r = text.range(of: "call:", range: search..<text.endIndex) {
            if r.lowerBound >= end { return nil }
            if isCallBoundary(text, r.lowerBound) { return r.lowerBound }
            search = r.upperBound
        }
        return nil
    }

    /// Every complete call in `body` (e.g. the inside of `<|tool_call>…<tool_call|>`), in order.
    /// Returns `nil` when nothing parseable was found.
    static func parseCalls(in body: String) -> [Call]? {
        var calls: [Call] = []
        var cursor = body.startIndex
        while let idx = nextCallCandidate(in: body, from: cursor) {
            switch scan(body, at: idx) {
            case .complete(let call, let range):
                calls.append(call)
                cursor = range.upperBound
            case .malformed(let range):
                cursor = range.upperBound
            case .notACall, .incomplete:
                cursor = body.index(idx, offsetBy: 5)
            }
        }
        return calls.isEmpty ? nil : calls
    }

    /// Converts the text between a call's braces into a JSON object string; `nil` if unparseable.
    static func argumentsJSON(from body: String) -> String? {
        var p = ArgParser(text: body)
        guard let obj = p.parseMembers() else { return nil }
        guard let data = try? JSONSerialization.data(withJSONObject: obj, options: [.sortedKeys, .withoutEscapingSlashes]),
              let s = String(data: data, encoding: .utf8) else { return nil }
        return s
    }

    // MARK: - Scanning helpers

    static func isNameChar(_ c: Character) -> Bool {
        c.isLetter || c.isNumber || c == "_" || c == "-" || c == "."
    }

    /// Index of the `}` matching the `{` at `open`, skipping string literals; `nil` if not yet closed.
    static func balancedEnd(in text: String, openingBrace open: String.Index) -> String.Index? {
        var depth = 0
        var i = open
        while i < text.endIndex {
            if let end = skipString(in: text, at: i) {
                switch end {
                case .some(let next): i = next; continue
                case .none: return nil            // unterminated string: still streaming
                }
            }
            let ch = text[i]
            if ch == "{" { depth += 1 }
            else if ch == "}" {
                depth -= 1
                if depth == 0 { return i }
            }
            i = text.index(after: i)
        }
        return nil
    }

    /// If a string literal starts at `i`, returns `.some(indexAfterIt)` or `.some(nil)` when it is
    /// unterminated. Returns `nil` when no string starts at `i`.
    private static func skipString(in text: String, at i: String.Index) -> String.Index??  {
        let rest = text[i...]
        for delim in [quote, escape] where rest.hasPrefix(delim) {
            let contentStart = text.index(i, offsetBy: delim.count)
            if let close = text.range(of: delim, range: contentStart..<text.endIndex) { return .some(close.upperBound) }
            return .some(nil)
        }
        if text[i] == "\"" {
            var j = text.index(after: i)
            while j < text.endIndex {
                if text[j] == "\\" {
                    let n = text.index(after: j)
                    if n >= text.endIndex { return .some(nil) }
                    j = text.index(after: n); continue
                }
                if text[j] == "\"" { return .some(text.index(after: j)) }
                j = text.index(after: j)
            }
            return .some(nil)
        }
        return nil
    }

    // MARK: - Argument value parser

    private struct ArgParser {
        let text: String
        var i: String.Index
        init(text: String) { self.text = text; self.i = text.startIndex }

        private mutating func skipSpace() {
            while i < text.endIndex, text[i].isWhitespace { i = text.index(after: i) }
        }
        private var atEnd: Bool { i >= text.endIndex }

        /// `key:value,key:value` (object body without braces).
        mutating func parseMembers(until close: Character? = nil) -> [String: Any]? {
            var obj: [String: Any] = [:]
            skipSpace()
            while true {
                if atEnd { return close == nil ? obj : nil }
                if let close, text[i] == close { i = text.index(after: i); return obj }
                guard let key = parseKey() else { return nil }
                skipSpace()
                guard !atEnd, text[i] == ":" else { return nil }
                i = text.index(after: i)
                guard let value = parseValue() else { return nil }
                obj[key] = value
                skipSpace()
                if atEnd { return close == nil ? obj : nil }
                if text[i] == "," { i = text.index(after: i); skipSpace(); continue }
                if let close, text[i] == close { i = text.index(after: i); return obj }
                return nil
            }
        }

        private mutating func parseKey() -> String? {
            skipSpace()
            if let s = parseStringLiteral() { return s }
            let start = i
            while i < text.endIndex, GemmaCallSyntax.isNameChar(text[i]) { i = text.index(after: i) }
            return i > start ? String(text[start..<i]) : nil
        }

        private mutating func parseStringLiteral() -> String? {
            guard !atEnd else { return nil }
            let rest = text[i...]
            for delim in [GemmaCallSyntax.quote, GemmaCallSyntax.escape] where rest.hasPrefix(delim) {
                let cs = text.index(i, offsetBy: delim.count)
                guard let close = text.range(of: delim, range: cs..<text.endIndex) else { return nil }
                i = close.upperBound
                return String(text[cs..<close.lowerBound])
            }
            if text[i] == "\"" {
                var out = ""
                var j = text.index(after: i)
                while j < text.endIndex {
                    let c = text[j]
                    if c == "\\" {
                        j = text.index(after: j)
                        guard j < text.endIndex else { return nil }
                        switch text[j] {
                        case "n": out.append("\n")
                        case "t": out.append("\t")
                        case "r": out.append("\r")
                        case "u":
                            let hexStart = text.index(after: j)
                            if let hexEnd = text.index(hexStart, offsetBy: 4, limitedBy: text.endIndex),
                               let code = UInt32(text[hexStart..<hexEnd], radix: 16), let sc = Unicode.Scalar(code) {
                                out.unicodeScalars.append(sc); j = text.index(before: hexEnd)
                            }
                        default: out.append(text[j])
                        }
                    } else if c == "\"" {
                        i = text.index(after: j)
                        return out
                    } else { out.append(c) }
                    j = text.index(after: j)
                }
                return nil
            }
            return nil
        }

        private mutating func parseValue() -> Any? {
            skipSpace()
            guard !atEnd else { return nil }
            if let s = parseStringLiteral() { return s }
            let c = text[i]
            if c == "{" { i = text.index(after: i); return parseMembers(until: "}") }
            if c == "[" { i = text.index(after: i); return parseArray() }
            // bare token up to , } ]
            let start = i
            while i < text.endIndex, !",}]".contains(text[i]) { i = text.index(after: i) }
            let raw = text[start..<i].trimmingCharacters(in: .whitespacesAndNewlines)
            if raw.isEmpty { return nil }
            switch raw {
            case "true": return true
            case "false": return false
            case "null": return NSNull()
            default: break
            }
            if let n = Int(raw) { return n }
            if let d = Double(raw), d.isFinite { return d }
            return raw
        }

        private mutating func parseArray() -> [Any]? {
            var arr: [Any] = []
            skipSpace()
            while true {
                if atEnd { return nil }
                if text[i] == "]" { i = text.index(after: i); return arr }
                guard let v = parseValue() else { return nil }
                arr.append(v)
                skipSpace()
                if atEnd { return nil }
                if text[i] == "," { i = text.index(after: i); skipSpace(); continue }
                if text[i] == "]" { i = text.index(after: i); return arr }
                return nil
            }
        }
    }
}
