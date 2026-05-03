#!/usr/bin/env bash
# demo.sh — launch doc-layout on Android, open dvurog.djvu, go to page 4,
# detect layout, and save a screen recording to /tmp/demo.mp4
set -euo pipefail

PACKAGE="com.flet.doc_layout"
ACTIVITY=".MainActivity"
DJVU_DEVICE="/sdcard/Download/dvurog.djvu"
RECORD_DEVICE="/sdcard/demo_record.mp4"
OUTPUT="/tmp/demo.mp4"

# ── helpers ────────────────────────────────────────────────────────────────
tap()  { adb shell input tap "$1" "$2"; }
wait_for_text() {
    local text="$1" timeout="${2:-20}"
    echo "  waiting for: $text"
    for i in $(seq 1 "$timeout"); do
        if adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1 \
           && adb shell cat /sdcard/ui.xml | grep -q "$text"; then
            return 0
        fi
        sleep 1
    done
    echo "TIMEOUT waiting for: $text" >&2
    return 1
}
find_center() {
    # find_center "text" → prints "x y"
    local text="$1"
    adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
    local bounds
    bounds=$(adb shell cat /sdcard/ui.xml \
        | grep -o "text=\"${text}\"[^>]*bounds=\"\[[0-9,]*\]\[[0-9,]*\]\"" \
        | grep -o 'bounds="\[[0-9,]*\]\[[0-9,]*\]"' \
        | head -1)
    if [ -z "$bounds" ]; then
        echo "ERROR: could not find element with text='$text'" >&2
        return 1
    fi
    # bounds="[x1,y1][x2,y2]"
    local x1 y1 x2 y2
    x1=$(echo "$bounds" | grep -o '\[[0-9]*,[0-9]*\]' | head -1 | tr -d '[]' | cut -d, -f1)
    y1=$(echo "$bounds" | grep -o '\[[0-9]*,[0-9]*\]' | head -1 | tr -d '[]' | cut -d, -f2)
    x2=$(echo "$bounds" | grep -o '\[[0-9]*,[0-9]*\]' | tail -1 | tr -d '[]' | cut -d, -f1)
    y2=$(echo "$bounds" | grep -o '\[[0-9]*,[0-9]*\]' | tail -1 | tr -d '[]' | cut -d, -f2)
    echo $(( (x1 + x2) / 2 )) $(( (y1 + y2) / 2 ))
}
tap_text() {
    local coords
    coords=$(find_center "$1")
    local x y
    x=$(echo "$coords" | awk '{print $1}')
    y=$(echo "$coords" | awk '{print $2}')
    echo "  tap '$1' at $x $y"
    tap "$x" "$y"
}

# ── push DjVu if not already there ─────────────────────────────────────────
echo "==> Pushing DjVu file to device..."
adb push /home/sergey/Downloads/books/dvurog.djvu "$DJVU_DEVICE" 2>&1 | tail -1

# ── start screen recording in background ───────────────────────────────────
echo "==> Starting screen recording..."
adb shell screenrecord --bit-rate 4000000 "$RECORD_DEVICE" &
RECORD_PID=$!
sleep 1   # give recorder a moment to start

# ── launch app ─────────────────────────────────────────────────────────────
echo "==> Launching app..."
adb shell am force-stop "$PACKAGE"
sleep 1
adb shell am start -n "${PACKAGE}/${ACTIVITY}"
sleep 4

# ── wait for "Open DjVu" button ────────────────────────────────────────────
echo "==> Waiting for app to load..."
wait_for_text "Open DjVu" 30

# ── tap Open DjVu ──────────────────────────────────────────────────────────
echo "==> Tapping Open DjVu..."
tap_text "Open DjVu"
sleep 3

# ── navigate file picker to /sdcard/Download/dvurog.djvu ───────────────────
# The Android file picker (ACTION_OPEN_DOCUMENT) shows recent files first.
# We use the search / browse path to reach Downloads.
echo "==> Navigating file picker..."

# Try tapping the hamburger / "Show roots" button (top-left, ~3 finger widths in)
tap 72 120
sleep 1

# Tap "Downloads" in the drawer
wait_for_text "Downloads" 10
tap_text "Downloads"
sleep 2

# Tap dvurog.djvu
wait_for_text "dvurog" 10
tap_text "dvurog"
sleep 6   # wait for page 1 to render

# ── navigate to page 4 (tap Next 3 times) ──────────────────────────────────
echo "==> Navigating to page 4..."
for i in 1 2 3; do
    echo "  next ($i/3)"
    # The next button has content-desc arrow_forward_ios; find by page label region
    # Safer: find the ">" icon button — it's to the right of the page label
    adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
    NEXT_BOUNDS=$(adb shell cat /sdcard/ui.xml \
        | grep -o 'content-desc="[^"]*"[^>]*bounds="\[[0-9,]*\]\[[0-9,]*\]"' \
        | grep -i "forward\|next\|arrow_forward" \
        | grep -o 'bounds="\[[0-9,]*\]\[[0-9,]*\]"' | head -1)
    if [ -n "$NEXT_BOUNDS" ]; then
        NX1=$(echo "$NEXT_BOUNDS" | grep -o '\[[0-9]*,[0-9]*\]' | head -1 | tr -d '[]' | cut -d, -f1)
        NY1=$(echo "$NEXT_BOUNDS" | grep -o '\[[0-9]*,[0-9]*\]' | head -1 | tr -d '[]' | cut -d, -f2)
        NX2=$(echo "$NEXT_BOUNDS" | grep -o '\[[0-9]*,[0-9]*\]' | tail -1 | tr -d '[]' | cut -d, -f1)
        NY2=$(echo "$NEXT_BOUNDS" | grep -o '\[[0-9]*,[0-9]*\]' | tail -1 | tr -d '[]' | cut -d, -f2)
        NX=$(( (NX1 + NX2) / 2 ))
        NY=$(( (NY1 + NY2) / 2 ))
        echo "  tapping next at $NX $NY"
        tap "$NX" "$NY"
    else
        echo "  WARNING: next button not found by content-desc, trying tap_text"
        tap_text ">"
    fi
    sleep 4   # wait for page to render
done

# ── verify we are on page 4 ────────────────────────────────────────────────
echo "==> Checking page label..."
adb shell uiautomator dump /sdcard/ui.xml >/dev/null 2>&1
adb shell cat /sdcard/ui.xml | grep -o 'Page [0-9]* / [0-9]*' | head -1 || true

# ── tap Detect Layout ──────────────────────────────────────────────────────
echo "==> Tapping Detect Layout..."
wait_for_text "Detect Layout" 10
tap_text "Detect Layout"

# ── wait for detection to finish ───────────────────────────────────────────
echo "==> Waiting for detection to complete..."
sleep 12

# ── stop recording ─────────────────────────────────────────────────────────
echo "==> Stopping screen recording..."
kill "$RECORD_PID" 2>/dev/null || true
adb shell pkill -2 screenrecord 2>/dev/null || true
sleep 2

# ── pull recording ─────────────────────────────────────────────────────────
echo "==> Pulling recording to $OUTPUT..."
adb pull "$RECORD_DEVICE" "$OUTPUT"
echo ""
echo "Done! Recording saved to: $OUTPUT"
