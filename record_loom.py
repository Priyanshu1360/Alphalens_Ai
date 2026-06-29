import time
import os
import glob
from playwright.sync_api import sync_playwright

def run():
    with sync_playwright() as p:
        print("Launching browser...")
        browser = p.chromium.launch(headless=True)
        video_dir = r"C:\Users\parih\.gemini\antigravity-ide\brain\717b22dd-8968-47fb-80cf-163bb99b4fd0"
        context = browser.new_context(
            viewport={'width': 1280, 'height': 800},
            record_video_dir=video_dir,
            record_video_size={'width': 1280, 'height': 800}
        )
        
        page = context.new_page()
        
        print("Navigating to localhost:3000")
        page.goto("http://localhost:3000/")
        page.wait_for_selector("textarea", timeout=10000)
        time.sleep(3)
        
        # Query 1
        query1 = "Hi, how fast can you answer?"
        print(f"Typing: {query1}")
        page.type("textarea", query1, delay=70)
        time.sleep(1)
        page.keyboard.press("Enter")
        time.sleep(6)  # wait for streaming to finish
        
        # Query 2
        query2 = "What were the primary drivers of Amazon's AWS growth in Q2 2024?"
        print(f"Typing: {query2}")
        page.type("textarea", query2, delay=50)
        time.sleep(1)
        page.keyboard.press("Enter")
        time.sleep(18)  # wait for retrieval + generation
        
        # Scroll a bit
        page.mouse.wheel(0, 300)
        time.sleep(2)
        
        # Query 3
        query3 = "Plot a bar chart of Amazon AWS revenue growth showing Q2 2024 (19) and Six Months 2024 (18)."
        print(f"Typing: {query3}")
        page.type("textarea", query3, delay=50)
        time.sleep(1)
        page.keyboard.press("Enter")
        time.sleep(22)  # wait for chart generation
        
        # Scroll to see chart
        page.mouse.wheel(0, 500)
        time.sleep(5)
        
        # Navigate to Audit Logs
        print("Navigating to Audit Logs")
        page.click("a[href='/audit']")
        time.sleep(3)
        
        # Click a row to expand
        print("Expanding audit log row")
        page.locator("table tbody tr").first.click()
        time.sleep(5)
        
        print("Closing context to save video...")
        context.close()
        browser.close()
        
        # Rename video
        videos = glob.glob(os.path.join(video_dir, "*.webm"))
        for v in videos:
            if "final_demo_video" not in v:
                os.rename(v, os.path.join(video_dir, "final_demo_video.webm"))
                print(f"Saved as final_demo_video.webm")
                break

if __name__ == "__main__":
    run()
