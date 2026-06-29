# Simulator UI Test Plan - Historical Photo Regression

Marker: `PXJ-HISTORY-SIM-20260629`

Purpose: verify the pxj iOS simulator experience with real historical photos copied from the original `xue` project, instead of empty or synthetic UI state.

## Test Data

- Source host: `ydz@100.64.0.13`
- Source directory: `/home/ydz/services/xue/backend/data/images`
- Target backend: `https://pxj.evowit.com`
- Target account: `pxj-sim-history-20260629@example.com`
- Data is intentionally retained after the run.

The seeded session must include:

- at least three historical `.jpg` files uploaded as a pxj learning session;
- session goal/title containing `PXJ-HISTORY-SIM-20260629`;
- one manual mistake item containing `PXJ-HISTORY-SIM-20260629`;
- enough metadata for the history list and mistake book to render without empty states.

## Generated XCUITest Cases

1. `PXJHistoricalDataUITests/testIPad_01_historyListShowsSeededHistoricalPhotoSession`
   - Launches the iPad app directly into History.
   - Confirms the history list loads.
   - Finds the seeded historical-photo session by marker.
   - Opens the session and confirms the continue button appears in the report pane.
   - Keeps screenshots in the `.xcresult`.

2. `PXJHistoricalDataUITests/testIPad_02_mistakeBookShowsSeededHistoricalMistake`
   - Launches the iPad app directly into Mistakes.
   - Confirms the mistake list loads.
   - Finds the seeded manual mistake by marker.
   - Opens the mistake detail and confirms the review action appears.
   - Keeps screenshots in the `.xcresult`.

3. `PXJHistoricalDataIPhoneUITests/testIPhone_01_workbenchLaunchesWithSeededHistoricalAccount`
   - Launches the iPhone app with the same seeded account.
   - Confirms the workbench appears.
   - Opens the tool menu and confirms core actions remain available.
   - Keeps screenshots in the `.xcresult`.

## Run Targets

- iPad simulator: iPad Pro 13-inch (M5), iOS 26.5
- iPhone simulator: iPhone 17 Pro, iOS 26.5

## Pass Criteria

- All generated tests pass on the Mac simulator host.
- Test result bundles remain under `/Users/macstar/Code/PXJ/TestResults`.
- Seeded backend data and uploaded historical photos are not deleted.
