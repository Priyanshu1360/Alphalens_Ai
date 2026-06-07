import time
import re
import matplotlib.pyplot as plt
import io
import base64

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(["Q1", "Q2", "Q3", "Q4"], [1, 2, 3, 4], color='#4da6ff')
plt.tight_layout()
buf = io.BytesIO()
plt.savefig(buf, format='png', transparent=True, dpi=120)
plt.close(fig)
buf.seek(0)
img_base64 = base64.b64encode(buf.read()).decode('utf-8')

answer = f"Here is the chart:\n\n![Matplotlib Chart](data:image/png;base64,{img_base64})\n\n"

t0 = time.time()
clean_answer = re.sub(r"!\[.*?\]\(data:image/.*?;base64,[A-Za-z0-9+/=]+\)", "[Chart Image omitted for grading]", answer)
t1 = time.time()

with open('regex_test.txt', 'w') as f:
    f.write(f"Match success: {clean_answer != answer}\n")
    f.write(f"Time: {t1 - t0}\n")
    f.write(f"Length base64: {len(img_base64)}\n")
