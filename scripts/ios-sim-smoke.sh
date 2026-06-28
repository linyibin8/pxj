#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IOS_DIR="$ROOT_DIR/ios"
PROJECT="$IOS_DIR/PXJ.xcodeproj"
SCHEME="${SCHEME:-PXJ}"
SIMULATOR_NAME="${SIMULATOR_NAME:-iPhone 17 Pro}"
SIMULATOR_OS="${SIMULATOR_OS:-latest}"
DESTINATION="${DESTINATION:-platform=iOS Simulator,name=$SIMULATOR_NAME,OS=$SIMULATOR_OS}"
DERIVED_DATA="${DERIVED_DATA:-$ROOT_DIR/tmp/ios-smoke/DerivedData}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/tmp/ios-smoke}"
SCREENSHOT_PATH="${SCREENSHOT_PATH:-$RESULT_DIR/pxj-launch.png}"
APP_BUNDLE_ID="${APP_BUNDLE_ID:-com.linyibin8.pxj}"
APP_PATH="$DERIVED_DATA/Build/Products/Debug-iphonesimulator/PXJ.app"

mkdir -p "$RESULT_DIR"

echo "== PXJ simulator smoke =="
echo "Project: $PROJECT"
echo "Scheme: $SCHEME"
echo "Destination: $DESTINATION"

echo "== Build and UI tests =="
xcodebuild \
  -project "$PROJECT" \
  -scheme "$SCHEME" \
  -configuration Debug \
  -destination "$DESTINATION" \
  -derivedDataPath "$DERIVED_DATA" \
  -resultBundlePath "$RESULT_DIR/PXJTest.xcresult" \
  CODE_SIGNING_ALLOWED=NO \
  test

echo "== Launch screenshot =="
xcrun simctl boot "$SIMULATOR_NAME" >/dev/null 2>&1 || true
xcrun simctl install booted "$APP_PATH"
xcrun simctl terminate booted "$APP_BUNDLE_ID" >/dev/null 2>&1 || true
xcrun simctl launch booted "$APP_BUNDLE_ID"
sleep "${SCREENSHOT_DELAY:-5}"
xcrun simctl io booted screenshot "$SCREENSHOT_PATH"

echo "== Recent app logs =="
APP_LOGS="$(xcrun simctl spawn booted log show --last 20s --style compact 2>/dev/null \
  | grep -Ei "$APP_BUNDLE_ID|pxj" \
  | tail -120 || true)"
if [[ -n "$APP_LOGS" ]]; then
  echo "$APP_LOGS"
fi

if echo "$APP_LOGS" | grep -Eiq "crash|exception|fatal error"; then
  echo "App crash-like log entry found." >&2
  exit 1
fi

echo "Screenshot: $SCREENSHOT_PATH"
echo "Smoke test passed."
