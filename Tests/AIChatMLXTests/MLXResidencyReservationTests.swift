import XCTest
import MLX
@testable import AIChatMLX

// Tests for MLXModelRuntime's wired-memory reservation bookkeeping — the mechanism that bounds
// and coordinates the memory footprint of the .primary and .auxiliary residency slots.
//
// These drive `reserve(weightBytes:for:)` / `releaseReservation(for:)` directly with synthetic
// weight sizes and a recording coordinator, so no model download and no Metal device is required.
// (A command-line `swift test` run has no compiled MLX metallib, so touching the real
// WiredMemoryManager — which probes `Device.defaultDevice()` — aborts the process.)
//
// Assertions cover both the runtime's own bookkeeping (which slot holds which ticket) and the
// ticket lifecycle handed to the coordinator (start/end, size, kind, policy grouping).

/// Records every ticket start/end the runtime performs.
private actor RecordingCoordinator: WiredMemoryCoordinating {
    struct Entry: Sendable, Equatable {
        let id: UUID
        let size: Int
        let policyID: AnyHashable
        let isReservation: Bool
    }

    private(set) var started: [Entry] = []
    private(set) var ended: [Entry] = []

    /// Tickets started and not yet ended, in start order.
    var live: [Entry] {
        started.filter { entry in !ended.contains(where: { $0.id == entry.id }) }
    }

    func start(_ ticket: WiredMemoryTicket) async {
        started.append(Self.entry(ticket))
    }

    func end(_ ticket: WiredMemoryTicket) async {
        ended.append(Self.entry(ticket))
    }

    private static func entry(_ ticket: WiredMemoryTicket) -> Entry {
        let isReservation: Bool
        switch ticket.kind {
        case .reservation: isReservation = true
        case .active: isReservation = false
        }
        return Entry(
            id: ticket.id,
            size: ticket.size,
            policyID: ticket.policy.id,
            isReservation: isReservation
        )
    }
}

final class MLXResidencyReservationTests: XCTestCase {

    private let primaryBytes = 4_000_000
    private let auxiliaryBytes = 1_500_000
    private let replacementBytes = 9_000_000

    private func makeRuntime() -> (MLXModelRuntime, RecordingCoordinator) {
        let coordinator = RecordingCoordinator()
        return (MLXModelRuntime(coordinator: coordinator), coordinator)
    }

    // MARK: - Load registers a reservation

    func test_reservation_startedWhenModelLoadsIntoSlot() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: primaryBytes, for: .primary)

        let ticketID = await runtime.reservationTicketID(for: .primary)
        XCTAssertNotNil(ticketID, "Loading a model into a slot must create a reservation ticket")
        let size = await runtime.reservationSizeBytes(for: .primary)
        XCTAssertEqual(size, primaryBytes, "Reservation must be sized to the model's weight bytes")

        let started = await coordinator.started
        XCTAssertEqual(started.count, 1)
        XCTAssertEqual(started.first?.id, ticketID, "The stored ticket must be the started one")
        XCTAssertEqual(started.first?.size, primaryBytes)
        XCTAssertEqual(
            started.first?.isReservation,
            true,
            "Loaded weights must use a .reservation ticket so they never elevate the wired limit while idle"
        )
        XCTAssertEqual(
            started.first?.policyID,
            MLXModelRuntime.residencyPolicy.id,
            "Reservations must be registered on the shared residency policy"
        )
        let ended = await coordinator.ended
        XCTAssertTrue(ended.isEmpty)
    }

    func test_zeroWeightBytes_doesNotCreateReservation() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: 0, for: .primary)

        let reserved = await runtime.reservedResidencies
        XCTAssertTrue(reserved.isEmpty, "A zero-byte measurement must not register a ticket")
        let started = await coordinator.started
        XCTAssertTrue(started.isEmpty)
    }

    // MARK: - Eviction / replacement ends the reservation

    func test_reservation_endedWhenSlotIsReleased() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: primaryBytes, for: .primary)
        guard let ticketID = await runtime.reservationTicketID(for: .primary) else {
            return XCTFail("Expected a reservation ticket for .primary")
        }

        let didRelease = await runtime.releaseReservation(for: .primary)
        XCTAssertTrue(didRelease)

        let stillHeld = await runtime.reservationTicketID(for: .primary)
        XCTAssertNil(stillHeld, "Releasing a slot must clear its stored reservation reference")

        let ended = await coordinator.ended
        XCTAssertEqual(ended.map(\.id), [ticketID], "Exactly the released slot's ticket must be ended")
        let live = await coordinator.live
        XCTAssertTrue(live.isEmpty, "No reservation may outlive its slot's model")
    }

    func test_releasingEmptySlot_isNoOp() async {
        let (runtime, coordinator) = makeRuntime()

        let didRelease = await runtime.releaseReservation(for: .auxiliary)

        XCTAssertFalse(didRelease, "Releasing a slot with no reservation must not report a release")
        let ended = await coordinator.ended
        XCTAssertTrue(ended.isEmpty, "No ticket may be ended when the slot holds none")
    }

    func test_reservation_replacedWhenSlotLoadsDifferentModel() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: primaryBytes, for: .primary)
        guard let firstID = await runtime.reservationTicketID(for: .primary) else {
            return XCTFail("Expected a reservation ticket for .primary")
        }

        // Loading a different model into the same slot evicts the previous one.
        await runtime.reserve(weightBytes: replacementBytes, for: .primary)

        let secondID = await runtime.reservationTicketID(for: .primary)
        XCTAssertNotNil(secondID)
        XCTAssertNotEqual(secondID, firstID, "Replacement must install a new reservation ticket")
        let size = await runtime.reservationSizeBytes(for: .primary)
        XCTAssertEqual(size, replacementBytes)

        let ended = await coordinator.ended
        XCTAssertEqual(
            ended.map(\.id),
            [firstID],
            "The superseded model's reservation must be ended exactly once"
        )
        let live = await coordinator.live
        XCTAssertEqual(live.map(\.id), [secondID])
        XCTAssertEqual(live.map(\.size), [replacementBytes])
    }

    // MARK: - Both slots share one policy

    func test_bothSlots_registerReservationsUnderSameSharedPolicy() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: primaryBytes, for: .primary)
        await runtime.reserve(weightBytes: auxiliaryBytes, for: .auxiliary)

        let primaryID = await runtime.reservationTicketID(for: .primary)
        let auxiliaryID = await runtime.reservationTicketID(for: .auxiliary)
        XCTAssertNotNil(primaryID)
        XCTAssertNotNil(auxiliaryID)
        XCTAssertNotEqual(primaryID, auxiliaryID, "Each slot must own a distinct reservation ticket")

        let reserved = await runtime.reservedResidencies
        XCTAssertEqual(reserved, [.primary, .auxiliary])
        let primarySize = await runtime.reservationSizeBytes(for: .primary)
        let auxiliarySize = await runtime.reservationSizeBytes(for: .auxiliary)
        XCTAssertEqual(primarySize, primaryBytes)
        XCTAssertEqual(auxiliarySize, auxiliaryBytes)

        let live = await coordinator.live
        XCTAssertEqual(live.count, 2, "Two occupied slots must hold two live reservations")
        XCTAssertEqual(Set(live.map(\.size)), [primaryBytes, auxiliaryBytes])

        // One policy identity across both slots => the manager groups them together and sums their
        // sizes into a single budget instead of accounting each slot independently.
        let policies = Set(live.map(\.policyID))
        XCTAssertEqual(policies, [MLXModelRuntime.residencyPolicy.id],
                       "Both slots' reservations must group under the one shared residency policy")

        // Releasing one slot must leave the other slot's reservation intact.
        await runtime.releaseReservation(for: .primary)
        let remaining = await runtime.reservedResidencies
        XCTAssertEqual(remaining, [.auxiliary])
        let stillLive = await coordinator.live
        XCTAssertEqual(stillLive.map(\.id), [auxiliaryID])
    }

    // MARK: - Generation tickets share the same policy

    func test_generationTicket_isActiveKindOnSharedPolicy() {
        let ticket = MLXModelRuntime.makeGenerationTicket()

        XCTAssertEqual(ticket.size, MLXModelRuntime.generationWorkspaceBytes)
        XCTAssertGreaterThan(ticket.size, 0)
        if case .active = ticket.kind {} else {
            XCTFail("Generation tickets must be .active so they drive the wired limit while running")
        }
        XCTAssertEqual(
            ticket.policy.id,
            MLXModelRuntime.residencyPolicy.id,
            "Generation tickets must group with the residency reservations under one policy"
        )
    }
}
