import os
import sys
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()

        cwd = os.getcwd()
        page.goto(f"file://{cwd}/docs/phillips-demo.html")

        # Select "Moxness" mode (GPU version)
        page.select_option("#renderMode", "moxness_gpu")

        # Wait a bit for the loop to run
        page.wait_for_timeout(2000)

        # Take a screenshot
        page.screenshot(path="phillips_moxness.png")
        print("Screenshot saved to phillips_moxness.png")

        # Check logs for "GPU Kernel Active"
        # We need to wait/poll for it
        content = page.content()
        if "GPU Kernel Active" in content:
             print("Verified: GPU Kernel Active.")
        else:
             print("Warning: GPU Kernel Log not found in DOM.")

        browser.close()

if __name__ == "__main__":
    run()
