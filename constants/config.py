import matplotlib.pyplot as plt

ACCENT  = "#2563eb"
ACCENT2 = "#dc2626"
ACCENT3 = "#059669"
CARD_BG = "#f0f4ff"
TEXT    = "#1e293b"

plt.style.use("default")

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
.metric-card {
    background: linear-gradient(135deg, #f0f4ff, #e8eeff);
    border: 1px solid #c8d0f0; border-radius: 16px;
    padding: 20px; text-align: center; transition: transform 0.2s;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2rem; font-weight: 700; font-family: 'JetBrains Mono'; }
.metric-label { font-size: 0.75rem; color: #555e7a; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }
.section-header {
    font-size: 1.3rem; font-weight: 600; border-left: 4px solid #2563eb;
    padding-left: 12px; margin: 24px 0 16px 0; color: #1e293b;
}
.tag { display: inline-block; background: #dbeafe; color: #1d4ed8;
    border-radius: 20px; padding: 2px 12px; font-size: 0.75rem; font-weight: 600; margin: 2px; }
.predict-box {
    background: linear-gradient(135deg, #eff6ff, #dbeafe);
    border: 2px solid #2563eb; border-radius: 20px;
    padding: 32px; text-align: center; margin-top: 20px;
}
</style>
"""
