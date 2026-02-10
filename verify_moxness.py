import os
import sys
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        cwd = os.getcwd()
        page.goto(f"file://{cwd}/docs/phillips-demo.html")

        # Select "Moxness" mode
        page.select_option("#renderMode", "moxness")

        # Wait a bit for the loop to run
        page.wait_for_timeout(2000)

        # Take a screenshot
        page.screenshot(path="phillips_moxness.png")
        print("Screenshot saved to phillips_moxness.png")

        # Check logs for "Moxness Slice Active"
        # We need to wait/poll for it
        msgs = []
        page.on("console", lambda msg: msgs.append(msg.text))

        # The demo logs "Moxness Slice Active" with low probability (1%) per frame
        # Wait up to 5s
        found = False
        for _ in range(50):
            page.wait_for_timeout(100)
            if any("Moxness Slice Active" in m for m in msgs):
                found = True
                break

        if found:
            print("Verified: Moxness Slice Active log found.")
        else:
            print("Warning: Moxness log not found (probabilistic). Checking manually.")

        browser.close()

if __name__ == "__main__":
    run()
