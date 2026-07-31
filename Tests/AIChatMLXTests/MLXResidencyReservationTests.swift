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

/// Coordinator that denies admission for any reservation whose size, summed with the sizes already
/// resident in the other slots, exceeds `budget` — the same shape as `WiredSumPolicy`'s clamp to
/// `GPU.maxRecommendedWorkingSetBytes()`, without touching Metal.
///
/// `start(_:)` deliberately *hangs forever* on a denied ticket, mirroring the real
/// `WiredMemoryTicket.start()` (documented: "If admission is denied, this suspends until capacity
/// is available or the task is cancelled") and its unbounded `while !policy.canAdmit(...)` waiter
/// loop in `WiredMemoryManager`. Any test that completes against this fake therefore proves the
/// runtime never started a ticket that would have been denied.
private final class DenyingCoordinator: WiredMemoryCoordinating, @unchecked Sendable {
    private let lock = NSLock()
    private let budget: Int
    private var _started: [UUID] = []
    private var _ended: [UUID] = []
    private var _admissionQueries: [(size: Int, existing: [Int])] = []

    init(budget: Int) { self.budget = budget }

    var started: [UUID] { lock.withLock { _started } }
    var ended: [UUID] { lock.withLock { _ended } }
    var admissionQueries: [(size: Int, existing: [Int])] { lock.withLock { _admissionQueries } }

    func canAdmitReservation(size: Int, alongside existingSizes: [Int]) -> Bool {
        lock.withLock { _admissionQueries.append((size, existingSizes)) }
        return existingSizes.reduce(0, +) + size <= budget
    }

    func start(_ ticket: WiredMemoryTicket) async {
        if !canAdmitReservation(size: ticket.size, alongside: []) {
            // The real manager parks here indefinitely. Never return.
            await withCheckedContinuation { (_: CheckedContinuation<Void, Never>) in }
        }
        lock.withLock { _started.append(ticket.id) }
    }

    func end(_ ticket: WiredMemoryTicket) async {
        lock.withLock { _ended.append(ticket.id) }
    }
}

final class MLXResidencyAdmissionTests: XCTestCase {

    private let budget = 10_000_000

    /// The core regression: a second slot whose weights would push the shared budget over the
    /// policy's limit must not block `reserve(...)`. If the runtime started the ticket anyway,
    /// `DenyingCoordinator.start` would never return and this test would time out.
    func test_deniedAdmission_skipsReservationInsteadOfHanging() async {
        let coordinator = DenyingCoordinator(budget: budget)
        let runtime = MLXModelRuntime(coordinator: coordinator)

        // .primary already holds a large model (as a download-triggered .auxiliary load would find).
        await runtime.reserve(weightBytes: 8_000_000, for: .primary)
        await runtime.test_markWeightBytes(8_000_000, for: .primary)

        // 8 MB + 6 MB > the 10 MB budget => the policy would deny this reservation.
        await runtime.reserve(weightBytes: 6_000_000, for: .auxiliary)

        let auxTicket = await runtime.reservationTicketID(for: .auxiliary)
        XCTAssertNil(
            auxTicket,
            "A reservation the policy would deny must be skipped, never started — starting it " +
            "would suspend container(for:residency:) indefinitely in the manager's waiter loop"
        )
        XCTAssertEqual(
            coordinator.started.count, 1,
            "Only the admissible .primary reservation may be started"
        )
        let queries = coordinator.admissionQueries
        XCTAssertEqual(
            queries.last?.existing, [8_000_000],
            "The pre-check must account for the weights already resident in the other slot"
        )
    }

    /// A skipped reservation must leave no residue: nothing for the runtime to end, and no ticket
    /// registered in the shared manager that the runtime has no record of.
    func test_skippedReservation_leavesNoTicketToLeak() async {
        let coordinator = DenyingCoordinator(budget: budget)
        let runtime = MLXModelRuntime(coordinator: coordinator)

        await runtime.test_markWeightBytes(9_000_000, for: .primary)
        await runtime.reserve(weightBytes: 9_000_000, for: .auxiliary)

        let reserved = await runtime.reservedResidencies
        XCTAssertTrue(reserved.isEmpty, "A skipped reservation must not be recorded")
        XCTAssertTrue(
            coordinator.started.isEmpty,
            "No ticket may reach the shared manager without a matching MLXModelRuntime entry"
        )

        // Releasing the slot afterwards must be a clean no-op — never an end() for a ticket that
        // was never started (which trips WiredMemoryManager's DEBUG "Ticket not active" assert).
        let didRelease = await runtime.releaseReservation(for: .auxiliary)
        XCTAssertFalse(didRelease)
        XCTAssertTrue(coordinator.ended.isEmpty)

        // And releaseSlot (the MLXProvider.releaseResidency path) is equally clean.
        await runtime.releaseSlot(.auxiliary)
        XCTAssertTrue(coordinator.ended.isEmpty, "Nothing was started, so nothing may be ended")
    }

    /// The gate must not be over-eager: a reservation that fits is still taken normally.
    func test_admissibleReservation_isStillStarted() async {
        let coordinator = DenyingCoordinator(budget: budget)
        let runtime = MLXModelRuntime(coordinator: coordinator)

        await runtime.test_markWeightBytes(3_000_000, for: .primary)
        await runtime.reserve(weightBytes: 4_000_000, for: .auxiliary)

        let ticketID = await runtime.reservationTicketID(for: .auxiliary)
        XCTAssertNotNil(ticketID, "A reservation that fits the budget must still be taken")
        XCTAssertEqual(coordinator.started, [ticketID].compactMap { $0 })
    }

    /// Releasing the big model in the other slot must make the previously-denied size admissible
    /// again — i.e. the gate reads live state, it isn't a permanent opt-out.
    func test_reservationBecomesAdmissible_afterOtherSlotReleases() async {
        let coordinator = DenyingCoordinator(budget: budget)
        let runtime = MLXModelRuntime(coordinator: coordinator)

        await runtime.test_markWeightBytes(8_000_000, for: .primary)
        await runtime.reserve(weightBytes: 6_000_000, for: .auxiliary)
        var auxTicket = await runtime.reservationTicketID(for: .auxiliary)
        XCTAssertNil(auxTicket)

        await runtime.releaseSlot(.primary)
        await runtime.reserve(weightBytes: 6_000_000, for: .auxiliary)
        auxTicket = await runtime.reservationTicketID(for: .auxiliary)
        XCTAssertNotNil(auxTicket, "Once .primary is released, .auxiliary's reservation fits again")
    }

    /// `residentWeightBytes(in:)` must report the slot's real footprint even when its reservation
    /// was skipped — that's what the host app's download gate reads.
    func test_residentWeightBytes_reportedIndependentlyOfReservation() async {
        let coordinator = DenyingCoordinator(budget: budget)
        let runtime = MLXModelRuntime(coordinator: coordinator)

        await runtime.test_markWeightBytes(8_000_000, for: .primary)
        let bytes = await runtime.residentWeightBytes(in: .primary)
        XCTAssertEqual(bytes, 8_000_000)

        await runtime.releaseSlot(.primary)
        let cleared = await runtime.residentWeightBytes(in: .primary)
        XCTAssertNil(cleared, "Releasing a slot must clear its recorded footprint")
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

    // MARK: - Force-release without replacement (releaseSlot / MLXProvider.releaseResidency)

    func test_releaseSlot_clearsResidentModelNameAndReservation() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: primaryBytes, for: .primary)
        await runtime.test_markLoaded(name: "mlx-community/gemma-4-e4b-it-4bit", for: .primary)
        guard let ticketID = await runtime.reservationTicketID(for: .primary) else {
            return XCTFail("Expected a reservation ticket for .primary")
        }
        let stillLoadedBeforeRelease = await runtime.isModelNameLoaded(
            "mlx-community/gemma-4-e4b-it-4bit", for: .primary
        )
        XCTAssertTrue(stillLoadedBeforeRelease)

        await runtime.releaseSlot(.primary)

        let loadedAfterRelease = await runtime.isModelNameLoaded(
            "mlx-community/gemma-4-e4b-it-4bit", for: .primary
        )
        XCTAssertFalse(
            loadedAfterRelease,
            "releaseSlot must clear the resident model name so a subsequent " +
            "container(for:residency:) call for the same name re-triggers loadModelContainer " +
            "rather than reusing the (now-dropped) container"
        )
        let stillHeld = await runtime.reservationTicketID(for: .primary)
        XCTAssertNil(stillHeld, "releaseSlot must end and clear the slot's reservation ticket")

        let ended = await coordinator.ended
        XCTAssertEqual(
            ended.map(\.id), [ticketID],
            "Exactly the released slot's reservation ticket must be ended"
        )
    }

    func test_releaseSlot_leavesOtherSlotUntouched() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.reserve(weightBytes: primaryBytes, for: .primary)
        await runtime.reserve(weightBytes: auxiliaryBytes, for: .auxiliary)
        await runtime.test_markLoaded(name: "primary-model", for: .primary)
        await runtime.test_markLoaded(name: "aux-model", for: .auxiliary)

        await runtime.releaseSlot(.primary)

        let primaryStillLoaded = await runtime.isModelNameLoaded("primary-model", for: .primary)
        let auxStillLoaded = await runtime.isModelNameLoaded("aux-model", for: .auxiliary)
        XCTAssertFalse(primaryStillLoaded)
        XCTAssertTrue(
            auxStillLoaded,
            "Releasing .primary must not affect .auxiliary's resident model"
        )
        let auxTicket = await runtime.reservationTicketID(for: .auxiliary)
        XCTAssertNotNil(auxTicket, "Releasing .primary must not end .auxiliary's reservation")

        let ended = await coordinator.ended
        XCTAssertEqual(ended.count, 1, "Only the released slot's reservation may be ended")
    }

    func test_releaseSlot_onEmptySlot_isNoOp() async {
        let (runtime, coordinator) = makeRuntime()

        await runtime.releaseSlot(.primary)

        let loaded = await runtime.isModelNameLoaded("anything", for: .primary)
        XCTAssertFalse(loaded)
        let ended = await coordinator.ended
        XCTAssertTrue(ended.isEmpty, "Releasing an already-empty slot must not end any ticket")
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
