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
        page.select_option("#renderMode", "moxness_gpu")

        # Enable Post Process
        page.check("#postProcess")

        # Wait a bit for the loop to run
        page.wait_for_timeout(2000)

        # Take a screenshot
        page.screenshot(path="phillips_holographic.png")
        print("Screenshot saved to phillips_holographic.png")

        # Check logs for "Holographic Kernel v4.0 Active"
        content = page.content()
        if "Holographic Kernel v4.0 Active" in content:
             print("Verified: Holographic Kernel v4.0 Active.")
        else:
             print("Warning: Kernel Log not found.")

        browser.close()

if __name__ == "__main__":
    run()
