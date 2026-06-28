import XCTest

// iPad 回归套件。launchEnvironment 自动登录（DEBUG 钩子）。截图用 XCUIScreen（app 繁忙也能抓）。
final class iPadUITests: XCTestCase {
    override func setUpWithError() throws { continueAfterFailure = true }

    private func makeApp(section: String? = nil, textOnly: Bool = false) -> XCUIApplication {
        let app = XCUIApplication()
        let env = ProcessInfo.processInfo.environment
        if let email = env["PXJ_UI_TEST_EMAIL"], let password = env["PXJ_UI_TEST_PASSWORD"] {
            app.launchEnvironment["PXJ_AUTOLOGIN_EMAIL"] = email
            app.launchEnvironment["PXJ_AUTOLOGIN_PW"] = password
        }
        if let section { app.launchEnvironment["PXJ_IPAD_SECTION"] = section }
        if textOnly { app.launchEnvironment["PXJ_TEXT_ONLY"] = "1" }  // 纯文字模式：模拟器无相机也能跑通问答
        return app
    }
    private func snap(_ name: String) {
        let s = XCTAttachment(screenshot: XCUIScreen.main.screenshot())
        s.name = name; s.lifetime = .keepAlways; add(s)
    }
    private func el(_ app: XCUIApplication, _ id: String) -> XCUIElement {
        app.descendants(matching: .any)[id].firstMatch
    }
    private func waitLearn(_ app: XCUIApplication, _ t: TimeInterval = 25) -> Bool {
        el(app, "ipad-composer-send").waitForExistence(timeout: t)
    }

    // 1) 分区视觉巡检（按 section 逐个重启，稳定）+ 竖横屏
    func testIPad_01_sectionsVisual() {
        for (sec, name) in [("learn","01-learn"),("history","02-history"),("mistakes","03-mistakes"),("memory","04-memory"),("settings","05-settings")] {
            let app = makeApp(section: sec)
            app.launch()
            _ = el(app, "ipad-section-\(sec == "learn" ? "history" : "learn")").waitForExistence(timeout: 20) // 侧栏出现
            sleep(3); snap(name)
            if sec == "learn" {
                XCUIDevice.shared.orientation = .portrait; sleep(2); snap("01b-learn-portrait")
                XCUIDevice.shared.orientation = .landscapeLeft; sleep(1)
            }
            app.terminate()
        }
    }

    // 2) 文字问答（诊断键盘/发送 + 思考器#5 + 动作#7）
    func testIPad_02_textQA() {
        let app = makeApp(textOnly: true)
        app.launch()
        XCTAssertTrue(waitLearn(app), "未进入学习页")
        XCUIDevice.shared.orientation = .landscapeLeft; sleep(1)
        let field = el(app, "ipad-composer-field")
        XCTAssertTrue(field.waitForExistence(timeout: 8), "输入框未出现")
        field.tap()
        field.typeText("请讲解：一个长方形长8宽5，面积和周长各是多少？")
        snap("06a-after-typing")  // 看键盘是否挡住发送
        let send = el(app, "ipad-composer-send")
        // 若发送被键盘挡住不可点，先收起键盘
        if !send.isHittable { app.typeText("\n"); sleep(1) }
        send.tap()
        snap("06b-after-send")
        let thinking = app.staticTexts["AI 正在思考你的问题…"]
        if thinking.waitForExistence(timeout: 5) { snap("06c-thinking") }
        let answered = app.staticTexts["AI 回答"].waitForExistence(timeout: 60)
        sleep(1); snap("07-answer")
        XCTAssertTrue(answered, "60s 内未返回 AI 回答")
        XCTAssertTrue(el(app, "assistant-answer-actions").waitForExistence(timeout: 5), "实质回答未出现动作行(#7)")
    }

    // 3) 闲聊不出动作(#7 反向)
    func testIPad_03_trivialChat() {
        let app = makeApp(textOnly: true)
        app.launch()
        XCTAssertTrue(waitLearn(app), "未进入学习页")
        XCUIDevice.shared.orientation = .landscapeLeft; sleep(1)
        let field = el(app, "ipad-composer-field")
        XCTAssertTrue(field.waitForExistence(timeout: 8))
        field.tap(); field.typeText("你好")
        let send = el(app, "ipad-composer-send")
        if !send.isHittable { app.typeText("\n"); sleep(1) }
        send.tap()
        let answered = app.staticTexts["AI 回答"].waitForExistence(timeout: 60)
        sleep(1); snap("08-trivial-answer")
        XCTAssertTrue(answered, "闲聊 60s 内未返回回答")
        XCTAssertFalse(el(app, "assistant-answer-actions").waitForExistence(timeout: 3), "闲聊仍显示动作行(#7)——可能 AI 回答超阈值")
    }

    // 4) 输入切换 语音⇄文字(#4)
    func testIPad_04_inputToggle() {
        let app = makeApp()
        app.launch()
        XCTAssertTrue(waitLearn(app), "未进入学习页")
        snap("09-input-text")
        el(app, "ipad-input-toggle").tap(); sleep(1); snap("10-input-voice")
        let voice = el(app, "voice-action").waitForExistence(timeout: 4) || app.staticTexts["按住说话"].waitForExistence(timeout: 2)
        XCTAssertTrue(voice, "切到语音模式未见按住说话")
        el(app, "ipad-input-toggle").tap(); sleep(1); snap("11-input-text-again")
        XCTAssertTrue(el(app, "ipad-composer-field").waitForExistence(timeout: 3), "切回文字后输入框未出现")
    }

    // 5) 历史→进入对话(#1)
    func testIPad_05_historyContinue() {
        let app = makeApp(section: "history")
        app.launch()
        let firstRow = app.staticTexts.containing(NSPredicate(format: "label CONTAINS '回合'")).firstMatch
        if firstRow.waitForExistence(timeout: 15) {
            snap("12-history-list")
            firstRow.tap(); sleep(3); snap("13-history-report")
            let cont = app.buttons.containing(NSPredicate(format: "label CONTAINS '进入对话'")).firstMatch
            if cont.waitForExistence(timeout: 6) {
                cont.tap()
                XCTAssertTrue(el(app, "ipad-composer-send").waitForExistence(timeout: 10), "进入对话后未切到学习页")
                sleep(2); snap("14-after-continue")
            } else { snap("13b-no-continue"); XCTFail("未找到进入对话按钮") }
        } else { snap("12-history-empty") }
    }

    // 7) 错题本详情（#3 卡片化）— 需账号已有错题
    func testIPad_07_mistakeDetail() {
        let app = makeApp(section: "mistakes")
        app.launch()
        // 直接点错题标题（避开侧栏 cell）
        let row = app.staticTexts.containing(NSPredicate(format: "label CONTAINS '长方形周长'")).firstMatch
        if row.waitForExistence(timeout: 15) {
            snap("17-mistakes-list")
            row.tap(); sleep(2); snap("18-mistake-detail")
            let ok = app.staticTexts["题目"].waitForExistence(timeout: 5)
                || app.staticTexts["订正建议"].exists
                || app.staticTexts["正确答案"].exists
            XCTAssertTrue(ok, "错题详情未渲染")
        } else { snap("17-mistakes-empty"); XCTFail("错题列表为空") }
    }

    // 6) 设置交互
    func testIPad_06_settings() {
        let app = makeApp(section: "settings")
        app.launch()
        sleep(4); snap("15-settings")
        let firstSwitch = app.switches.firstMatch
        if firstSwitch.waitForExistence(timeout: 6) {
            firstSwitch.tap(); sleep(1); snap("16-settings-toggled"); firstSwitch.tap()
        } else { XCTFail("设置开关未出现") }
    }
}
