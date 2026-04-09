"""
GlassBox Cybersecurity Report Generator
Enterprise Edition – Customer-Facing Overview (Expanded)
"""

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak
)
from reportlab.platypus.flowables import Flowable
import os

# ── Output path ────────────────────────────────────────────────────────────────
OUT = os.path.join(os.path.dirname(__file__), "GlassBox_Enterprise_Security_Overview.pdf")

# ── Colour palette ─────────────────────────────────────────────────────────────
NAVY       = colors.HexColor("#0D1B2A")
TEAL       = colors.HexColor("#1B998B")
TEAL_LIGHT = colors.HexColor("#D6F0ED")
SLATE      = colors.HexColor("#2E4057")
SILVER     = colors.HexColor("#F4F6F8")
MUTED      = colors.HexColor("#6B7280")
WHITE      = colors.white
AMBER      = colors.HexColor("#FEF3C7")

PW, PH = A4
ML = MR = 18*mm

# ── Custom flowables ───────────────────────────────────────────────────────────
class SectionBanner(Flowable):
    def __init__(self, text, bg=NAVY, fg=WHITE, width=None, height=10*mm):
        super().__init__()
        self.text   = text
        self.bg     = bg
        self.fg     = fg
        self.bwidth = width or (PW - ML - MR)
        self.bheight= height

    def wrap(self, *_):
        return self.bwidth, self.bheight + 2*mm

    def draw(self):
        c = self.canv
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.bwidth, self.bheight, 3, fill=1, stroke=0)
        c.setFillColor(self.fg)
        c.setFont("Helvetica-Bold", 11)
        c.drawString(4*mm, 3*mm, self.text)


class CalloutBox(Flowable):
    def __init__(self, text, bg=TEAL_LIGHT, bar=TEAL, width=None, font_size=9):
        super().__init__()
        self.text      = text
        self.bg        = bg
        self.bar       = bar
        self.bwidth    = width or (PW - ML - MR)
        self.font_size = font_size
        self._lines    = None

    def wrap(self, avail_w, avail_h):
        self.bwidth = avail_w
        chars_per_line = int(self.bwidth / (self.font_size * 0.52))
        words = self.text.split()
        lines, cur = [], ""
        for w in words:
            if len(cur) + len(w) + 1 <= chars_per_line:
                cur = (cur + " " + w).strip()
            else:
                lines.append(cur); cur = w
        if cur: lines.append(cur)
        self._lines = lines
        h = len(lines) * (self.font_size + 3) + 6*mm
        return self.bwidth, h

    def draw(self):
        c = self.canv
        h = len(self._lines) * (self.font_size + 3) + 6*mm
        c.setFillColor(self.bg)
        c.roundRect(0, 0, self.bwidth, h, 3, fill=1, stroke=0)
        c.setFillColor(self.bar)
        c.rect(0, 0, 2.5*mm, h, fill=1, stroke=0)
        c.setFillColor(SLATE)
        c.setFont("Helvetica", self.font_size)
        y = h - self.font_size - 4*mm
        for line in self._lines:
            c.drawString(5*mm, y, line)
            y -= self.font_size + 3


# ── Style sheet ────────────────────────────────────────────────────────────────
base = getSampleStyleSheet()

def S(name, **kw):
    return ParagraphStyle(name, **kw)

TITLE  = S("Title",  fontName="Helvetica-Bold", fontSize=26, textColor=NAVY, spaceAfter=2*mm, leading=32)
SUB    = S("Sub",    fontName="Helvetica",      fontSize=13, textColor=TEAL, spaceAfter=4*mm, leading=18)
H1     = S("H1",     fontName="Helvetica-Bold", fontSize=13, textColor=WHITE, spaceAfter=0, leading=16)
H2     = S("H2",     fontName="Helvetica-Bold", fontSize=11, textColor=SLATE, spaceAfter=2*mm, spaceBefore=3*mm, leading=14)
H3     = S("H3",     fontName="Helvetica-Bold", fontSize=10, textColor=TEAL, spaceAfter=1*mm, spaceBefore=2*mm, leading=13)
BODY   = S("Body",   fontName="Helvetica",      fontSize=9,  textColor=SLATE, spaceAfter=2*mm, leading=14, alignment=TA_JUSTIFY)
BULLET = S("Bullet", fontName="Helvetica",      fontSize=9,  textColor=SLATE, spaceAfter=1*mm, leading=13, leftIndent=8*mm, bulletIndent=3*mm)
MONO   = S("Mono",   fontName="Courier",        fontSize=8,  textColor=SLATE, backColor=SILVER, spaceAfter=2*mm, leading=12, leftIndent=4*mm)
META   = S("Meta",   fontName="Helvetica",      fontSize=8,  textColor=MUTED, spaceAfter=0, leading=11)
FOOT   = S("Foot",   fontName="Helvetica",      fontSize=7,  textColor=MUTED, alignment=TA_CENTER)

TH = S("TH", fontName="Helvetica-Bold", fontSize=8.5, textColor=WHITE, alignment=TA_CENTER)
TD = S("TD", fontName="Helvetica",      fontSize=8.5, textColor=SLATE, leading=12, alignment=TA_LEFT)

def bp(text, bullet="•"):
    return Paragraph(f"{bullet}  {text}", BULLET)

def sp(n=3):
    return Spacer(1, n*mm)

def hr():
    return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#E5E7EB"), spaceAfter=3*mm, spaceBefore=1*mm)


# ── Header / footer ───────────────────────────────────────────────────────────
def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PH - 10*mm, PW, 10*mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.setFont("Helvetica-Bold", 8)
    canvas.drawString(ML, PH - 6*mm, "GlassBox")
    canvas.setFillColor(WHITE)
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(PW - MR, PH - 6*mm, "Enterprise Security Overview · Confidential")
    canvas.setFillColor(SILVER)
    canvas.rect(0, 0, PW, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(PW/2, 3*mm, f"Page {doc.page}")
    canvas.restoreState()

def on_first_page(canvas, doc):
    canvas.saveState()
    canvas.setFillColor(NAVY)
    canvas.rect(0, PH - 45*mm, PW, 45*mm, fill=1, stroke=0)
    canvas.setFillColor(TEAL)
    canvas.rect(0, PH - 47*mm, PW, 2*mm, fill=1, stroke=0)
    canvas.setFillColor(SILVER)
    canvas.rect(0, 0, PW, 8*mm, fill=1, stroke=0)
    canvas.setFillColor(MUTED)
    canvas.setFont("Helvetica", 7)
    canvas.drawCentredString(PW/2, 3*mm, "Page 1")
    canvas.restoreState()


# ── Helper: data table ────────────────────────────────────────────────────────
def data_table(headers, rows, col_widths=None):
    data = [[Paragraph(f"<b>{h}</b>", TH) for h in headers]]
    for row in rows:
        data.append([Paragraph(str(c), TD) for c in row])
    cw = col_widths or [(PW - ML - MR) / len(headers)] * len(headers)
    t = Table(data, colWidths=cw, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND",  (0,0), (-1,0),  NAVY),
        ("ROWBACKGROUNDS",(0,1),(-1,-1),[WHITE, SILVER]),
        ("GRID",        (0,0), (-1,-1), 0.4, colors.HexColor("#D1D5DB")),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 5),
        ("RIGHTPADDING",(0,0), (-1,-1), 5),
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
    ]))
    return t


# ── Document content – EXPANDED CUSTOMER-FACING VERSION ───────────────────────
def build():
    doc = SimpleDocTemplate(
        OUT, pagesize=A4,
        leftMargin=ML, rightMargin=MR,
        topMargin=50*mm, bottomMargin=18*mm,
    )
    story = []
    W = PW - ML - MR

    # ── Cover ──────────────────────────────────────────────────────────────────
    story += [
        sp(8),
        Paragraph("GlassBox", TITLE),
        Paragraph("Transparent AI Security Platform", SUB),
        hr(),
        Paragraph(
            "Your security platforms already make millions of critical decisions every day. "
            "GlassBox gives you complete visibility into every single one — so your SOC analysts "
            "understand exactly why an action was taken, your compliance team has perfect audit trails, "
            "and your models adapt automatically when threats change. No more black-box guesswork.",
            BODY),
        sp(6),
        Paragraph("Enterprise Security Overview · March 2026", META),
        Paragraph("Designed for Okta, Cloudflare, CrowdStrike, Palo Alto Networks, SentinelOne and beyond", META),
        PageBreak(),
    ]

    # ── 1. The Challenge ───────────────────────────────────────────────────────
    story.append(SectionBanner("1.  The Challenge: Why Traditional Security AI Creates Risk"))
    story += [sp(3),
        Paragraph(
            "Modern cybersecurity platforms rely on AI to block logins, classify traffic, "
            "and isolate endpoints. These systems are fast and accurate on normal activity — "
            "but they hide their reasoning. When something goes wrong, your team is left asking “why?”.",
            BODY),
        sp(2),
        Paragraph("Three real-world problems follow:", BODY)]

    story.append(data_table(
        ["Problem", "What It Means for Your Team", "Business Impact"],
        [
            ["Unexplained decisions", "SOC analysts see only a risk score", "Alert fatigue, delayed triage, and lost trust in the system"],
            ["Slow adaptation to new attacks", "Attackers evolve faster than your models can retrain", "Detection gaps that last days or weeks"],
            ["Compliance & audit pressure", "You must prove every automated decision", "Manual evidence gathering, long audit cycles, and regulatory risk"],
        ],
        col_widths=[50*mm, 70*mm, 55*mm]
    ))
    story += [sp(3),
        Paragraph(
            "GlassBox was built to solve these exact problems — giving you transparent, adaptive, "
            "and compliance-ready AI that works inside your existing tools.",
            BODY), sp(3), hr()]

    # ── 2. How GlassBox Works ──────────────────────────────────────────────────
    story.append(SectionBanner("2.  How GlassBox Delivers Transparent Security"))
    story += [sp(3),
        Paragraph(
            "GlassBox breaks every decision into clear signals you define yourself. "
            "It then shows you exactly which signals drove the outcome and automatically learns "
            "from real incidents to stay ahead of new threats. The result is security you can trust, explain, "
            "and prove.",
            BODY), sp(2)]

    mechs = [
        ("Precise Signal Attribution", 
         "Every decision is decomposed into the exact signals that mattered — so your analysts instantly "
         "see why a login was blocked or why traffic was flagged. No more guessing."),
        ("Cross-Signal Intelligence", 
         "GlassBox automatically detects dangerous combinations of signals that individually look normal. "
         "This catches sophisticated attacks that single-signal tools miss."),
        ("Automatic Adaptation", 
         "When your environment changes or new attack patterns appear, GlassBox learns directly from the "
         "incidents and strengthens your models — without manual retraining or extra labels."),
        ("Flexible Signal Design", 
         "You choose how many signals to use. Keep it simple for fast triage or go deeper for forensic detail. "
         "GlassBox adapts to whatever level your team needs."),
    ]
    for title, body in mechs:
        story += [Paragraph(title, H3), Paragraph(body, BODY)]

    story += [sp(2), CalloutBox(
        "Your SOC gets clear explanations. Your compliance team gets perfect audit evidence. "
        "Your security improves automatically. All without changing how you work today.",
        bg=TEAL_LIGHT, bar=TEAL), sp(3), hr()]

    # ── 3. Okta – Identity & Access Management ─────────────────────────────────
    story.append(SectionBanner("3.  Okta – Stronger Identity & Access Protection"))
    story += [sp(3),
        Paragraph(
            "Okta handles billions of authentication events every day. The biggest challenge is that "
            "risk scores alone don’t tell you why a login was blocked or allowed. Analysts waste time "
            "investigating, and compliance teams struggle to prove decisions.",
            BODY), sp(2)]

    story.append(Paragraph("How GlassBox helps", H2))
    story.append(data_table(
        ["Signal Group", "What It Detects"],
        [
            ["Geospatial", "Impossible travel, malicious networks, proxy/VPN use"],
            ["Device", "Spoofed devices, malware fingerprints, headless browsers"],
            ["Behavioural", "Credential stuffing, session hijacking, unusual typing patterns"],
            ["Temporal", "Off-hours access, burst attempts, abnormal timing"],
            ["Contextual", "Privilege escalation, access to sensitive resources"],
        ],
        col_widths=[45*mm, 130*mm]
    ))

    story += [sp(3),
        Paragraph(
            "When a login is blocked, your analyst receives an exact, easy-to-read breakdown showing "
            "which signals drove the decision. The dominant signal is highlighted so you can act immediately. "
            "Every explanation is 100% reproducible — perfect for audits.",
            BODY), sp(2),
        Paragraph(
            "If attackers evolve their tactics, GlassBox automatically learns from the missed or false-positive "
            "events and updates your model — keeping your identity protection current without extra work.",
            BODY),
        sp(3), hr()]

    # ── 4. Cloudflare – Network Security & Zero Trust ──────────────────────────
    story.append(SectionBanner("4.  Cloudflare – Network Security & Zero Trust"))
    story += [sp(3),
        Paragraph(
            "Cloudflare protects 20% of global web traffic. The challenge is separating legitimate traffic "
            "from sophisticated bots and DDoS attacks — especially when individual signals look normal.",
            BODY), sp(2)]

    story.append(Paragraph("How GlassBox helps", H2))
    story.append(data_table(
        ["Signal Group", "What It Detects"],
        [
            ["TLS Fingerprint", "JA3/JA4 patterns and cipher behaviour"],
            ["HTTP Behaviour", "Header ordering and User-Agent consistency"],
            ["Request Timing", "Burst patterns and think-time distribution"],
            ["JS Challenge", "Canvas and WebGL anomalies"],
            ["IP Reputation", "ASN class and abuse history"],
            ["Session Graph", "Page visit sequences and referrer chains"],
            ["Payload Analysis", "Form fill timing and parameter patterns"],
        ],
        col_widths=[50*mm, 125*mm]
    ))

    story += [sp(2),
        Paragraph(
            "GlassBox automatically detects when two signals together are suspicious even if each one alone "
            "appears benign — catching advanced bots and new DDoS variants.",
            BODY),
        sp(2),
        Paragraph(
            "When new attack patterns appear, GlassBox learns directly from the traffic and strengthens "
            "your rules — keeping your Zero Trust and WAF protection current.",
            BODY),
        sp(3), hr()]

    # ── 5. CrowdStrike – Endpoint Detection & Response ─────────────────────────
    story.append(SectionBanner("5.  CrowdStrike – Endpoint Detection & Response"))
    story += [sp(3),
        Paragraph(
            "CrowdStrike Falcon processes endpoint telemetry from thousands of organisations. "
            "The highest-stakes problem is that a wrong block can stop production systems, "
            "while a missed detection lets ransomware through.",
            BODY), sp(2)]

    story.append(Paragraph("How GlassBox helps", H2))
    story.append(data_table(
        ["Signal Group", "What It Detects"],
        [
            ["Process Lineage", "Parent-child chains and injection attempts"],
            ["File System", "Ransomware encryption patterns"],
            ["Network Egress", "C2 communication and exfiltration"],
            ["Registry", "Persistence mechanisms"],
            ["Memory", "In-memory execution and shellcode"],
            ["User Context", "Privilege escalation and lateral movement"],
        ],
        col_widths=[45*mm, 130*mm]
    ))

    story += [sp(2),
        Paragraph(
            "Your analysts see exactly which endpoint signals caused an isolation decision. "
            "Cross-signal detection catches injection techniques that look normal in one area but anomalous in another.",
            BODY),
        sp(2),
        Paragraph(
            "When a tenant sees repeated false positives (e.g., in CI/CD pipelines), GlassBox learns "
            "from those specific incidents and adapts the model for that environment — without affecting other customers.",
            BODY),
        sp(6), hr()]

    # ── 6. Palo Alto & SentinelOne – SOC & Threat Intelligence ─────────────────
    story.append(SectionBanner("6.  Palo Alto Networks & SentinelOne – SOC & Threat Intelligence"))
    story += [sp(3),
        Paragraph(
            "In XSIAM and Singularity platforms, the challenge is turning thousands of alerts into "
            "actionable incidents and keeping threat intelligence current.",
            BODY), sp(2)]

    story.append(Paragraph("How GlassBox helps", H2))
    story += [
        Paragraph("For Palo Alto Cortex XSIAM", H3),
        Paragraph(
            "GlassBox adds clear attribution to every correlated incident. Your L1 analysts see "
            "which signal groups drove the classification — so they can contain threats faster "
            "without escalating every case.", BODY),
        sp(2),
        Paragraph("For SentinelOne Singularity", H3),
        Paragraph(
            "GlassBox automatically discovers behavioural sub-patterns and builds a living taxonomy "
            "of threats. Your threat intelligence team gets fresh, accurate insights without manual curation.",
            BODY),
        sp(3), hr()]

    # ── 7. Compliance-Grade Explainability ─────────────────────────────────────
    story.append(SectionBanner("7.  Compliance-Grade Explainability"))
    story += [sp(3),
        Paragraph(
            "GlassBox meets the strictest regulatory requirements because every decision is "
            "exactly reproducible and fully documented.",
            BODY), sp(2)]

    story.append(data_table(
        ["Regulation", "Requirement", "How GlassBox Helps"],
        [
            ["GDPR Art. 22", "Explain automated decisions", "Exact signal attribution per decision"],
            ["SOC2 Type II", "Documented rationale for controls", "Complete audit trail with every explanation"],
            ["FedRAMP / HIPAA", "Auditable access decisions", "Reproducible evidence from model weights"],
            ["NIS2", "Causal analysis of incidents", "Clear signal-level breakdown for every incident"],
        ],
        col_widths=[40*mm, 65*mm, 70*mm]
    ))
    story += [sp(3), hr()]

    # ── 8. Flexible Signal Design ──────────────────────────────────────────────
    story.append(SectionBanner("8.  Flexible Signal Design – Choose Your Level of Detail"))
    story += [sp(3),
        Paragraph(
            "You decide how detailed the explanations should be. GlassBox works at any level you choose.",
            BODY), sp(2)]

    patterns = [
        ("Simple Mode", "Use 3–4 signals for fast, high-volume triage (ideal for edge filtering or basic bot detection)."),
        ("Standard Mode", "Use 5–7 signals for balanced SOC triage — the sweet spot for most enterprise deployments."),
        ("Detailed Mode", "Use 8+ signals for deep forensic analysis and compliance reporting."),
    ]
    for title, body in patterns:
        story += [Paragraph(title, H3), Paragraph(body, BODY)]

    story += [sp(2), CalloutBox(
        "The same core platform adapts to all three modes. Your team chooses the right level of detail "
        "for each use case — no code changes required.",
        bg=TEAL_LIGHT, bar=TEAL), sp(3), hr()]

    # ── 9. Automatic Adaptation & Synthetic Data ───────────────────────────────
    story.append(SectionBanner("9.  Automatic Adaptation & Targeted Synthetic Data"))
    story += [sp(3),
        Paragraph(
            "When your models encounter new threats or environment changes, GlassBox learns directly "
            "from real incidents and strengthens protection automatically.",
            BODY), sp(2)]

    story.append(data_table(
        ["Traditional Methods", "GlassBox Approach"],
        [
            ["Manual retraining or blind augmentation", "Learns only from actual failures in your environment"],
            ["Generic synthetic data", "Targeted generation focused on the exact signals that failed"],
            ["Long cycles involving security teams", "Fast, automated improvement with no extra labelling"],
        ],
        col_widths=[90*mm, 85*mm]
    ))
    story += [sp(3), hr()]

    # ── 10. Competitive Positioning ────────────────────────────────────────────
    story.append(SectionBanner("10.  Why GlassBox Stands Apart"))
    story += [sp(3)]

    story.append(data_table(
        ["Capability", "Traditional AI", "Post-hoc Tools", "GlassBox"],
        [
            ["Per-decision explanation", "No", "Approximate", "Exact & reproducible"],
            ["Cross-signal attack detection", "No", "No", "Built-in"],
            ["Automatic adaptation from incidents", "No", "No", "Yes"],
            ["Compliance-ready audit trails", "No", "No", "Yes"],
            ["Flexible signal design", "No", "No", "Yes"],
        ],
        col_widths=[65*mm, 40*mm, 40*mm, 40*mm]
    ))
    story += [sp(3), hr()]

    # ── 11. Conclusion & Next Step ─────────────────────────────────────────────
    story.append(SectionBanner("11.  Ready to See GlassBox in Your Environment?"))
    story += [sp(4),
        Paragraph(
            "GlassBox transforms your security AI from a black box into a trusted partner. "
            "Your analysts get clarity, your compliance team gets proof, and your models stay ahead "
            "of evolving threats — all while working seamlessly with your existing platforms.",
            BODY),
        sp(3),
      
     
        sp(6),
        Paragraph("GlassBox Enterprise Security Platform · Confidential Overview", FOOT),
    ]

    # ── Build PDF ──────────────────────────────────────────────────────────────
    doc.build(story, onFirstPage=on_first_page, onLaterPages=on_page)
    print(f"✅ Expanded customer-facing report written → {OUT}")

if __name__ == "__main__":
    build()