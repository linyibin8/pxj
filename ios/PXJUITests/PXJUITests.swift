import XCTest

final class PXJUITests: XCTestCase {
    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    func testLaunchShowsChatComposerAndPrimaryActions() throws {
        let app = XCUIApplication()
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()

        XCTAssertTrue(app.descendants(matching: .any)["immersive-workbench"].waitForExistence(timeout: 8))
        XCTAssertTrue(app.descendants(matching: .any)["bottom-chat-dock"].waitForExistence(timeout: 5))
        XCTAssertGreaterThan(app.windows.element(boundBy: 0).frame.width, app.windows.element(boundBy: 0).frame.height)
        XCTAssertTrue(app.descendants(matching: .any)["composer-mode-toggle"].waitForExistence(timeout: 3))
        let expandedVoice = app.descendants(matching: .any)["voice-action"]
        let collapsedVoice = app.descendants(matching: .any)["voice-collapsed-action"]
        XCTAssertTrue(
            expandedVoice.waitForExistence(timeout: 2) || collapsedVoice.waitForExistence(timeout: 2)
        )
        XCTAssertTrue(app.descendants(matching: .any)["tool-menu-toggle"].waitForExistence(timeout: 3))
        XCTAssertFalse(app.staticTexts["Pai 学习教练"].exists)
        XCTAssertFalse(app.staticTexts["学习现场"].exists)
    }

    func testFloatingToolMenuCanBeExpanded() throws {
        let app = XCUIApplication()
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()

        let toolToggle = app.descendants(matching: .any)["tool-menu-toggle"]
        XCTAssertTrue(toolToggle.waitForExistence(timeout: 8))
        toolToggle.tap()

        let burstAction = app.descendants(matching: .any)["burst-action"]
        XCTAssertTrue(burstAction.waitForExistence(timeout: 3))
        XCTAssertTrue(["智能观察", "停止观察"].contains(burstAction.label))
        XCTAssertTrue(app.descendants(matching: .any)["new-conversation-action"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["context-action"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["activity-action"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["history-action"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["tool-menu-close"].exists)
        XCTAssertFalse(app.descendants(matching: .any)["stage-new-conversation"].exists)
        XCTAssertFalse(app.descendants(matching: .any)["capture-action"].exists)
        XCTAssertFalse(app.descendants(matching: .any)["chat-capture-action"].exists)

        XCTAssertEqual(app.descendants(matching: .any)["tool-menu-close"].label, "收起工具")
    }

    func testTextComposerCanBeOpened() throws {
        let app = XCUIApplication()
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()

        let composerToggle = app.descendants(matching: .any)["composer-mode-toggle"]
        XCTAssertTrue(composerToggle.waitForExistence(timeout: 8))
        composerToggle.tap()

        XCTAssertTrue(app.descendants(matching: .any)["bottom-chat-dock"].exists)
        XCTAssertTrue(app.descendants(matching: .any)["composer-mode-toggle"].exists)
    }

    func testTextComposerCanSwitchBackToVoice() throws {
        let app = XCUIApplication()
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()

        let composerToggle = app.descendants(matching: .any)["composer-mode-toggle"]
        XCTAssertTrue(composerToggle.waitForExistence(timeout: 8))
        composerToggle.tap()
        XCTAssertTrue(app.descendants(matching: .any)["chat-composer"].waitForExistence(timeout: 3))

        let keyboardVoiceToggle = app.descendants(matching: .any)["keyboard-voice-toggle"].firstMatch
        if keyboardVoiceToggle.waitForExistence(timeout: 2) {
            keyboardVoiceToggle.tap()
        } else {
            composerToggle.tap()
        }

        let voiceAction = app.descendants(matching: .any)["voice-action"].firstMatch
        XCTAssertTrue(voiceAction.waitForExistence(timeout: 4))
        XCTAssertTrue(["按住说话", "松开发送当前画面", "识别中"].contains(voiceAction.label))
    }
}
