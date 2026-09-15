import Foundation

/// Diagnostic-only subset. Original quantized pieces are stored consecutively
/// without expansion; the owning store decodes them into its ordinary outputs.
package final class SourceNativeExpertSample {
    package let layout: PackedExpertLayout
    package let layer: Int
    package let experts: [Int]
    package let owner: ObjectIdentifier

    package init(layout: PackedExpertLayout, layer: Int, experts: [Int], owner: ExpertStore) {
        self.layout = layout
        self.layer = layer
        self.experts = experts
        self.owner = ObjectIdentifier(owner)
    }
}
