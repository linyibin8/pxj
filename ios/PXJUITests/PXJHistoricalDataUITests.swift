import XCTest
import UIKit

final class PXJHistoricalDataUITests: XCTestCase {
    private let marker = ProcessInfo.processInfo.environment["PXJ_HISTORY_TEST_MARKER"] ?? "PXJ-HISTORY-SIM-20260629"
    private let sessionId = ProcessInfo.processInfo.environment["PXJ_HISTORY_SESSION_ID"] ?? ""
    private let mistakeId = ProcessInfo.processInfo.environment["PXJ_HISTORY_MISTAKE_ID"] ?? ""

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    private func requiredEnv(_ name: String) throws -> String {
        guard let value = ProcessInfo.processInfo.environment[name], !value.isEmpty else {
            throw XCTSkip("\(name) is required for historical-photo simulator UI tests.")
        }
        return value
    }

    private func makeApp(section: String) throws -> XCUIApplication {
        let app = XCUIApplication()
        app.launchEnvironment["PXJ_AUTOLOGIN_EMAIL"] = try requiredEnv("PXJ_UI_TEST_EMAIL")
        app.launchEnvironment["PXJ_AUTOLOGIN_PW"] = try requiredEnv("PXJ_UI_TEST_PASSWORD")
        app.launchEnvironment["PXJ_IPAD_SECTION"] = section
        app.launchEnvironment["PXJ_TEXT_ONLY"] = "1"
        return app
    }

    private func element(_ app: XCUIApplication, _ identifier: String) -> XCUIElement {
        app.descendants(matching: .any)[identifier].firstMatch
    }

    private func textContaining(_ app: XCUIApplication, _ value: String) -> XCUIElement {
        app.staticTexts.containing(NSPredicate(format: "label CONTAINS %@", value)).firstMatch
    }

    private func findInList(_ target: XCUIElement, list: XCUIElement, timeout: TimeInterval) -> Bool {
        if target.waitForExistence(timeout: 4) {
            return true
        }
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            list.swipeUp()
            if target.waitForExistence(timeout: 1) {
                return true
            }
        }
        return false
    }

    private func snap(_ name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    func testIPad_01_historyListShowsSeededHistoricalPhotoSession() throws {
        try XCTSkipUnless(UIDevice.current.userInterfaceIdiom == .pad, "iPad history test requires an iPad simulator.")

        let app = try makeApp(section: "history")
        app.launch()

        let list = element(app, "ipad-history-list")
        XCTAssertTrue(
            list.waitForExistence(timeout: 45),
            "History list did not load for marker \(marker)."
        )

        let seededSession = sessionId.isEmpty
            ? textContaining(app, marker)
            : element(app, "history-session-row-\(sessionId)")
        XCTAssertTrue(
            findInList(seededSession, list: list, timeout: 30),
            "Seeded historical-photo session \(marker) is not visible."
        )
        snap("pxj-history-list-\(marker)")

        seededSession.tap()
        XCTAssertTrue(
            element(app, "history-continue-button").waitForExistence(timeout: 20),
            "History detail did not open for seeded session \(marker)."
        )
        snap("pxj-history-detail-\(marker)")
    }

    func testIPad_02_mistakeBookShowsSeededHistoricalMistake() throws {
        try XCTSkipUnless(UIDevice.current.userInterfaceIdiom == .pad, "iPad mistake test requires an iPad simulator.")

        let app = try makeApp(section: "mistakes")
        app.launch()

        let list = element(app, "ipad-mistake-list")
        XCTAssertTrue(
            list.waitForExistence(timeout: 45),
            "Mistake list did not load for marker \(marker)."
        )

        let seededMistake = mistakeId.isEmpty
            ? textContaining(app, marker)
            : element(app, "mistake-row-\(mistakeId)")
        XCTAssertTrue(
            findInList(seededMistake, list: list, timeout: 30),
            "Seeded historical-photo mistake \(marker) is not visible."
        )
        snap("pxj-mistake-list-\(marker)")

        seededMistake.tap()
        XCTAssertTrue(
            element(app, "mistake-review-button").waitForExistence(timeout: 20),
            "Mistake detail did not open for seeded mistake \(marker)."
        )
        snap("pxj-mistake-detail-\(marker)")
    }
}

final class PXJHistoricalDataIPhoneUITests: XCTestCase {
    private let marker = ProcessInfo.processInfo.environment["PXJ_HISTORY_TEST_MARKER"] ?? "PXJ-HISTORY-SIM-20260629"

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    private func requiredEnv(_ name: String) throws -> String {
        guard let value = ProcessInfo.processInfo.environment[name], !value.isEmpty else {
            throw XCTSkip("\(name) is required for historical-photo simulator UI tests.")
        }
        return value
    }

    private func makeApp() throws -> XCUIApplication {
        let app = XCUIApplication()
        app.launchEnvironment["PXJ_AUTOLOGIN_EMAIL"] = try requiredEnv("PXJ_UI_TEST_EMAIL")
        app.launchEnvironment["PXJ_AUTOLOGIN_PW"] = try requiredEnv("PXJ_UI_TEST_PASSWORD")
        app.launchEnvironment["PXJ_TEXT_ONLY"] = "1"
        return app
    }

    private func element(_ app: XCUIApplication, _ identifier: String) -> XCUIElement {
        app.descendants(matching: .any)[identifier].firstMatch
    }

    private func snap(_ name: String) {
        let attachment = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        attachment.name = name
        attachment.lifetime = .keepAlways
        add(attachment)
    }

    func testIPhone_01_workbenchLaunchesWithSeededHistoricalAccount() throws {
        try XCTSkipUnless(UIDevice.current.userInterfaceIdiom == .phone, "iPhone workbench test requires an iPhone simulator.")

        XCUIDevice.shared.orientation = .landscapeLeft

        let app = try makeApp()
        app.launch()

        XCTAssertTrue(
            element(app, "immersive-workbench").waitForExistence(timeout: 45),
            "iPhone workbench did not appear for seeded account \(marker)."
        )
        snap("pxj-iphone-workbench-\(marker)")

        let toolMenu = element(app, "tool-menu-toggle")
        XCTAssertTrue(toolMenu.waitForExistence(timeout: 10), "Tool menu is missing.")
        toolMenu.tap()

        XCTAssertTrue(
            element(app, "burst-action").waitForExistence(timeout: 8),
            "Burst action is missing from the iPhone tool menu."
        )
        XCTAssertTrue(
            element(app, "history-action").waitForExistence(timeout: 8),
            "History action is missing from the iPhone tool menu."
        )
        snap("pxj-iphone-tool-menu-\(marker)")
    }
}
