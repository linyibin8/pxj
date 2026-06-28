import XCTest

// iPhone 回归套件（横屏沉浸工作台）。autologin 进入；验证工作台、工具菜单、真问答、#7 动作收敛。
final class iPhoneUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = true }

    private func makeApp(textOnly: Bool = false) -> XCUIApplication {
        let app = XCUIApplication()
        let env = ProcessInfo.processInfo.environment
        if let email = env["PXJ_UI_TEST_EMAIL"], let password = env["PXJ_UI_TEST_PASSWORD"] {
            app.launchEnvironment["PXJ_AUTOLOGIN_EMAIL"] = email
            app.launchEnvironment["PXJ_AUTOLOGIN_PW"] = password
        }
        if textOnly { app.launchEnvironment["PXJ_TEXT_ONLY"] = "1" }
        return app
    }
    private func snap(_ app: XCUIApplication, _ name: String) {
        let s = XCTAttachment(screenshot: XCUIScreen.main.screenshot()); s.name = name; s.lifetime = .keepAlways; add(s)
    }
    private func el(_ app: XCUIApplication, _ id: String) -> XCUIElement {
        app.descendants(matching: .any)[id].firstMatch
    }

    // 1) 工作台加载（回归）
    func testIPhone_01_workbench() {
        let app = makeApp()
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()
        XCTAssertTrue(el(app, "immersive-workbench").waitForExistence(timeout: 25), "未进入工作台")
        sleep(2); snap(app, "20-iphone-workbench")
        XCTAssertTrue(el(app, "composer-mode-toggle").waitForExistence(timeout: 5), "缺 composer 切换")
        XCTAssertTrue(el(app, "tool-menu-toggle").waitForExistence(timeout: 5), "缺工具菜单")
        XCTAssertTrue(el(app, "voice-action").exists || el(app, "voice-collapsed-action").exists, "缺语音按钮")
    }

    // 2) 工具菜单展开
    func testIPhone_02_toolMenu() {
        let app = makeApp()
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()
        let toggle = el(app, "tool-menu-toggle")
        XCTAssertTrue(toggle.waitForExistence(timeout: 25))
        toggle.tap(); sleep(1); snap(app, "21-iphone-toolmenu")
        XCTAssertTrue(el(app, "burst-action").waitForExistence(timeout: 4), "工具菜单缺观察")
        XCTAssertTrue(el(app, "history-action").exists, "工具菜单缺历史")
        XCTAssertTrue(el(app, "context-action").exists, "工具菜单缺上下文")
    }

    // 3) 文字问答 + #7 实质回答出动作
    func testIPhone_03_textQA() {
        let app = makeApp(textOnly: true)
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()
        XCTAssertTrue(el(app, "immersive-workbench").waitForExistence(timeout: 25))
        let toggle = el(app, "composer-mode-toggle")
        if toggle.waitForExistence(timeout: 6) { toggle.tap(); sleep(1) }
        let field = el(app, "chat-composer")
        XCTAssertTrue(field.waitForExistence(timeout: 6), "文字输入框未出现")
        field.tap(); field.typeText("请讲解：7乘以8等于多少，并说明怎么记忆乘法口诀")
        el(app, "send-chat").tap()
        let answered = app.staticTexts["AI 回答"].waitForExistence(timeout: 45)
        sleep(1); snap(app, "22-iphone-answer")
        XCTAssertTrue(answered, "iPhone 45s 内未返回回答")
        XCTAssertTrue(el(app, "assistant-answer-actions").waitForExistence(timeout: 5), "iPhone 实质回答未出现动作行")
    }

    // 4) 闲聊不出动作（#7 反向）
    func testIPhone_04_trivialChat() {
        let app = makeApp(textOnly: true)
        XCUIDevice.shared.orientation = .landscapeLeft
        app.launch()
        XCTAssertTrue(el(app, "immersive-workbench").waitForExistence(timeout: 25))
        let toggle = el(app, "composer-mode-toggle")
        if toggle.waitForExistence(timeout: 6) { toggle.tap(); sleep(1) }
        let field = el(app, "chat-composer")
        XCTAssertTrue(field.waitForExistence(timeout: 6))
        field.tap(); field.typeText("嗨")
        el(app, "send-chat").tap()
        _ = app.staticTexts["AI 回答"].waitForExistence(timeout: 45)
        sleep(1); snap(app, "23-iphone-trivial")
        XCTAssertFalse(el(app, "assistant-answer-actions").waitForExistence(timeout: 4), "iPhone 闲聊仍显示动作行")
    }
}
