"""
Realistic email datasets for the three tasks.
Ground truth labels determined by clear, deterministic rules.
"""

# ─────────────────────────────────────────────
# TASK 1 (Easy) — Single email triage, 5 emails
# ─────────────────────────────────────────────
TASK1_EMAILS = [
    {
        "id": "e1_001",
        "subject": "URGENT: Production server down - customer impact",
        "sender": "alerts@monitoring.company.com",
        "body": "Critical alert: The production API server has been unresponsive for 10 minutes. "
                "Approximately 2,400 customers are currently affected. Error rate is at 98%. "
                "Immediate action required. Incident #INC-8821.",
        "timestamp": "2024-01-15T09:03:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e1_002",
        "subject": "Team lunch this Friday?",
        "sender": "colleague@company.com",
        "body": "Hey! Thinking of organizing a team lunch this Friday at noon. "
                "Let me know if you're in. Was thinking the new Italian place on 5th.",
        "timestamp": "2024-01-15T09:15:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e1_003",
        "subject": "Invoice #4421 attached - Payment due Jan 30",
        "sender": "billing@supplier.com",
        "body": "Please find attached Invoice #4421 for services rendered in December 2023. "
                "Total amount: $12,450. Payment due by January 30, 2024. "
                "Please process per your standard procedure.",
        "timestamp": "2024-01-15T09:22:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
    {
        "id": "e1_004",
        "subject": "You've won a FREE iPhone 15! Click now!!!",
        "sender": "prize-notify@win-rewards123.biz",
        "body": "Congratulations!! You have been selected as our lucky winner!! "
                "Click the link below to claim your FREE iPhone 15 Pro Max!!! "
                "Offer expires in 24 hours! Don't miss out!!!",
        "timestamp": "2024-01-15T09:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e1_005",
        "subject": "Q4 report draft for your review",
        "sender": "manager@company.com",
        "body": "Hi, please review the attached Q4 performance report draft when you get a chance. "
                "No rush — board presentation isn't until next week. "
                "Let me know if you have any suggested changes.",
        "timestamp": "2024-01-15T09:45:00Z",
        "has_attachment": True,
        "thread_length": 2,
    },
]

TASK1_GROUND_TRUTH = {
    "e1_001": {"label": "urgent", "priority": 1},
    "e1_002": {"label": "low",    "priority": 4},
    "e1_003": {"label": "normal", "priority": 2},
    "e1_004": {"label": "spam",   "priority": 5},
    "e1_005": {"label": "normal", "priority": 3},
}

# ──────────────────────────────────────────────────────────────
# TASK 2 (Medium) — Batch inbox triage, 10 emails, time pressure
# ──────────────────────────────────────────────────────────────
TASK2_EMAILS = [
    {
        "id": "e2_001",
        "subject": "Security breach detected — immediate response needed",
        "sender": "security@company.com",
        "body": "Our intrusion detection system flagged unauthorized access to the HR database "
                "at 08:47 UTC. Approx. 3,200 employee records may be exposed. "
                "Security team is assembling. Need exec sign-off to begin incident response protocol.",
        "timestamp": "2024-01-15T08:50:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e2_002",
        "subject": "Re: Re: Re: Project timeline update",
        "sender": "contractor@external.com",
        "body": "Following up on our earlier thread. We've revised the timeline to accommodate "
                "the new requirements. The new delivery date is Feb 14. Please confirm by EOD.",
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": True,
        "thread_length": 4,
    },
    {
        "id": "e2_003",
        "subject": "Newsletter: January industry roundup",
        "sender": "newsletter@industrynews.com",
        "body": "Welcome to the January edition of the Industry Insider! This month: "
                "AI trends, market analysis, top movers, upcoming conferences, and more. "
                "Read the full issue below.",
        "timestamp": "2024-01-15T09:05:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e2_004",
        "subject": "CEO meeting prep — 2pm today",
        "sender": "ceo-assistant@company.com",
        "body": "Reminder: You're presenting to the CEO at 2pm today in Conference Room A. "
                "Please bring the updated product roadmap and last quarter's metrics. "
                "The CEO has 30 minutes allocated.",
        "timestamp": "2024-01-15T09:10:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e2_005",
        "subject": "Your Amazon order has shipped",
        "sender": "shipment-tracking@amazon.com",
        "body": "Your order #112-3847291-4821039 has shipped! Estimated delivery: Jan 17. "
                "Track your package: [tracking link]. Thank you for shopping with Amazon.",
        "timestamp": "2024-01-15T09:20:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e2_006",
        "subject": "Contract renewal — action required by Jan 20",
        "sender": "legal@partner-corp.com",
        "body": "The master service agreement between our companies expires January 31. "
                "Please review the attached renewal terms and return signed copies by Jan 20 "
                "to avoid service interruption. Changes are highlighted in red.",
        "timestamp": "2024-01-15T09:25:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
    {
        "id": "e2_007",
        "subject": "Fwd: FW: Fwd: Funny cat video 😂",
        "sender": "friend@personal.com",
        "body": "Haha you have to see this! [forwarded chain of a cat video link] "
                "Too funny, reminded me of your cat. Hope you're doing well!",
        "timestamp": "2024-01-15T09:35:00Z",
        "has_attachment": False,
        "thread_length": 5,
    },
    {
        "id": "e2_008",
        "subject": "Employee satisfaction survey — 5 mins",
        "sender": "hr@company.com",
        "body": "Hi team! We're running our quarterly engagement survey. "
                "It takes about 5 minutes and your responses are anonymous. "
                "Survey closes Friday. Link: [survey link]. Thank you!",
        "timestamp": "2024-01-15T09:40:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e2_009",
        "subject": "FINAL NOTICE: Your account will be suspended",
        "sender": "support@paypa1-secure.net",
        "body": "Your PayPal account has been compromised! Verify your details immediately "
                "or your account will be permanently suspended. Click here to verify: [link]",
        "timestamp": "2024-01-15T09:50:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e2_010",
        "subject": "Bug report: checkout flow crashes on mobile",
        "sender": "qa@company.com",
        "body": "Reproducible bug found in production: the checkout flow crashes when users "
                "tap 'confirm order' on iOS 17+ Safari. Affects ~15% of mobile orders. "
                "Steps to reproduce attached. Needs dev team attention.",
        "timestamp": "2024-01-15T10:00:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
]

TASK2_GROUND_TRUTH = {
    "e2_001": {"label": "urgent",  "priority": 1},
    "e2_002": {"label": "normal",  "priority": 2},
    "e2_003": {"label": "archive", "priority": 5},
    "e2_004": {"label": "urgent",  "priority": 1},
    "e2_005": {"label": "archive", "priority": 5},
    "e2_006": {"label": "normal",  "priority": 2},
    "e2_007": {"label": "low",     "priority": 4},
    "e2_008": {"label": "low",     "priority": 4},
    "e2_009": {"label": "spam",    "priority": 5},
    "e2_010": {"label": "urgent",  "priority": 1},
}

# ─────────────────────────────────────────────────────────────────────────────
# TASK 3 (Hard) — Multi-account inbox + routing rules + reply drafts, 12 emails
# ─────────────────────────────────────────────────────────────────────────────
TASK3_RULES = """
ROUTING RULES:
1. Emails from @vip-client.com or mentioning "enterprise deal" → assign_to: "sales"
2. Security alerts, breach mentions, or suspicious senders → assign_to: "security"
3. Bug reports, production issues, system errors → assign_to: "engineering"
4. HR, benefits, payroll, compliance emails → assign_to: "hr"
5. Legal contracts, NDAs, renewals → assign_to: "legal"
6. All other business emails → assign_to: "general"
7. Spam/phishing → assign_to: "security" (for review and blocking)

PRIORITY RULES:
- Any email affecting >1000 customers or >$100k revenue: priority 1
- C-level sender or recipient: priority 1
- Legal deadline within 7 days: priority 1
- Normal business matters: priority 2-3
- FYI/newsletters/low-urgency: priority 4-5
"""

TASK3_EMAILS = [
    {
        "id": "e3_001",
        "subject": "Enterprise deal — $2.4M contract ready to sign",
        "sender": "cto@vip-client.com",
        "body": "We've finalized internal approvals for the enterprise deal. "
                "The $2.4M contract is ready for signature. Can we schedule a call this week? "
                "Our board is excited to move forward.",
        "timestamp": "2024-01-15T08:00:00Z",
        "has_attachment": True,
        "thread_length": 6,
    },
    {
        "id": "e3_002",
        "subject": "Ransomware detected on engineering workstations",
        "sender": "soc@security-vendor.com",
        "body": "CRITICAL: Our endpoint detection tool found ransomware signatures on 4 engineering "
                "workstations. The affected machines have been isolated. Immediate containment "
                "and IR engagement recommended. Do not delay.",
        "timestamp": "2024-01-15T08:05:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_003",
        "subject": "NDA renewal — expires Jan 22",
        "sender": "counsel@lawfirm.com",
        "body": "The mutual NDA between your company and TechPartner Inc. expires January 22, "
                "one week from today. Please review and sign the renewal attached. "
                "Failure to renew will expose confidential project discussions.",
        "timestamp": "2024-01-15T08:15:00Z",
        "has_attachment": True,
        "thread_length": 1,
    },
    {
        "id": "e3_004",
        "subject": "Payment system down — all transactions failing",
        "sender": "alerts@payments.company.com",
        "body": "ALERT: The payment processing service is returning 500 errors for 100% of "
                "transactions since 08:03 UTC. Estimated revenue impact: $18,000/minute. "
                "On-call engineer has been paged. Executive awareness needed.",
        "timestamp": "2024-01-15T08:20:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_005",
        "subject": "Open enrollment closes Friday — action required",
        "sender": "benefits@hr.company.com",
        "body": "Reminder: Benefits open enrollment closes this Friday at 5pm. "
                "If you do not make selections, you will be defaulted to last year's plan. "
                "Login to the HR portal to review your options.",
        "timestamp": "2024-01-15T08:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_006",
        "subject": "Re: API rate limits causing partner integration failures",
        "sender": "partnerships@bigtech.com",
        "body": "Our integration has been hitting your API rate limits since yesterday's deployment. "
                "We're seeing 429 errors on 40% of requests. This is affecting our 50,000 end users. "
                "Please increase our limits or we'll need to pause the integration.",
        "timestamp": "2024-01-15T08:40:00Z",
        "has_attachment": True,
        "thread_length": 3,
    },
    {
        "id": "e3_007",
        "subject": "Unsubscribe from SaaS Weekly Digest",
        "sender": "digest@saasweekly.io",
        "body": "This week in SaaS: funding rounds, product launches, and the top hires. "
                "Read the full digest below. To unsubscribe, click here.",
        "timestamp": "2024-01-15T08:50:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_008",
        "subject": "Urgent: Board wants updated financials before tomorrow's call",
        "sender": "cfo@company.com",
        "body": "The board chair just called me — they want updated Q4 financials before the "
                "10am call tomorrow. Can you get the revised numbers to me by tonight? "
                "This is high priority.",
        "timestamp": "2024-01-15T09:00:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_009",
        "subject": "Verify your email — click the link below",
        "sender": "noreply@dropbox-secure-verify.net",
        "body": "Someone tried to access your Dropbox account. Verify your identity immediately "
                "to prevent account suspension. Click: http://dropbox-secure-verify.net/verify",
        "timestamp": "2024-01-15T09:10:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_010",
        "subject": "Data pipeline failing — ETL jobs not completing",
        "sender": "data-engineering@company.com",
        "body": "The nightly ETL pipeline has failed for 3 consecutive nights. "
                "Analytics dashboards are now showing stale data from 3 days ago. "
                "The sales team's daily reports are affected. Need engineering review.",
        "timestamp": "2024-01-15T09:20:00Z",
        "has_attachment": True,
        "thread_length": 2,
    },
    {
        "id": "e3_011",
        "subject": "New hire onboarding — start date Feb 1",
        "sender": "recruiting@company.com",
        "body": "We've confirmed the offer acceptance for Sarah Chen, Senior Engineer. "
                "Start date: February 1. Please coordinate equipment provisioning, "
                "system access, and day-1 orientation schedule.",
        "timestamp": "2024-01-15T09:30:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
    {
        "id": "e3_012",
        "subject": "Congrats!! You've been pre-approved for a $50,000 business loan!!!",
        "sender": "loans@fast-biz-capital-now.com",
        "body": "CONGRATULATIONS!!! Your business has been PRE-APPROVED for up to $50,000!! "
                "No credit check needed! Apply in 5 minutes! Click NOW before this offer expires!!!",
        "timestamp": "2024-01-15T09:40:00Z",
        "has_attachment": False,
        "thread_length": 1,
    },
]

TASK3_GROUND_TRUTH = {
    "e3_001": {"label": "urgent",  "priority": 1, "assign_to": "sales"},
    "e3_002": {"label": "urgent",  "priority": 1, "assign_to": "security"},
    "e3_003": {"label": "urgent",  "priority": 1, "assign_to": "legal"},
    "e3_004": {"label": "urgent",  "priority": 1, "assign_to": "engineering"},
    "e3_005": {"label": "normal",  "priority": 3, "assign_to": "hr"},
    "e3_006": {"label": "urgent",  "priority": 1, "assign_to": "engineering"},
    "e3_007": {"label": "archive", "priority": 5, "assign_to": "general"},
    "e3_008": {"label": "urgent",  "priority": 1, "assign_to": "general"},
    "e3_009": {"label": "spam",    "priority": 5, "assign_to": "security"},
    "e3_010": {"label": "normal",  "priority": 2, "assign_to": "engineering"},
    "e3_011": {"label": "normal",  "priority": 3, "assign_to": "hr"},
    "e3_012": {"label": "spam",    "priority": 5, "assign_to": "security"},
}

TASK_SPECS = {
    "task1": {
        "task_id": "task1",
        "name": "Single Inbox Triage",
        "description": (
            "Triage 5 emails from a professional inbox. Assign the correct label "
            "(urgent/normal/low/spam/archive) and priority (1-5) to each email. "
            "Labels and priorities must match clear, unambiguous signals in the email content."
        ),
        "difficulty": "easy",
        "max_steps": 10,
        "num_emails": 5,
        "success_threshold": 0.7,
        "emails": TASK1_EMAILS,
        "ground_truth": TASK1_GROUND_TRUTH,
        "rules_context": None,
    },
    "task2": {
        "task_id": "task2",
        "name": "Batch Inbox Triage Under Time Pressure",
        "description": (
            "Triage 10 emails from a busy executive inbox with mixed urgency levels, spam, "
            "newsletters, and time-sensitive items. Must correctly identify the critical items "
            "(security breach, CEO prep, production bug) vs noise (shipping notification, "
            "forwarded chain, survey). Each email gets label + priority."
        ),
        "difficulty": "medium",
        "max_steps": 15,
        "num_emails": 10,
        "success_threshold": 0.7,
        "emails": TASK2_EMAILS,
        "ground_truth": TASK2_GROUND_TRUTH,
        "rules_context": None,
    },
    "task3": {
        "task_id": "task3",
        "name": "Multi-Account Triage with Routing Rules",
        "description": (
            "Triage 12 emails across a complex enterprise inbox. Must apply routing rules to "
            "assign each email to the correct team (sales/security/engineering/hr/legal/general), "
            "set correct label and priority. Includes a $2.4M deal, ransomware alert, NDA deadline, "
            "payment outage, and phishing attempts. Tests rule-following under ambiguity."
        ),
        "difficulty": "hard",
        "max_steps": 20,
        "num_emails": 12,
        "success_threshold": 0.7,
        "emails": TASK3_EMAILS,
        "ground_truth": TASK3_GROUND_TRUTH,
        "rules_context": TASK3_RULES,
    },
}
