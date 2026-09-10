"""
Script to reproducibly create the official CLAUSE demonstration policy corpus.
Generates 6 substantial DOCX policy documents with realistic metadata, structured sections,
and consistent policy rules for demonstration and benchmark evaluation.
"""

from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def create_policy_document(
    filename: str,
    title: str,
    policy_id: str,
    version: str,
    effective_date: str,
    owner: str,
    sections: list[tuple[str, str]],
):
    doc = docx.Document()

    # Document Title
    heading = doc.add_heading(title, level=0)
    heading.alignment = WD_ALIGN_PARAGRAPH.LEFT

    # Metadata Table
    table = doc.add_table(rows=5, cols=2)
    table.style = 'Table Grid'
    
    meta_pairs = [
        ("Policy Identifier:", policy_id),
        ("Version:", version),
        ("Effective Date:", effective_date),
        ("Policy Owner:", owner),
        ("Classification:", "Internal Company Policy"),
    ]
    
    for idx, (label, val) in enumerate(meta_pairs):
        row_cells = table.rows[idx].cells
        row_cells[0].text = label
        row_cells[1].text = val
        # Bold labels
        row_cells[0].paragraphs[0].runs[0].bold = True

    doc.add_paragraph()  # Spacing

    # Notice
    notice_p = doc.add_paragraph()
    notice_run = notice_p.add_run("DEMONSTRATION POLICY DISCLOSURE: This document is an original synthetic policy generated for the CLAUSE Enterprise RAG demonstration and benchmark evaluation.")
    notice_run.italic = True
    notice_run.font.size = Pt(9)
    notice_run.font.color.rgb = RGBColor(110, 113, 109)

    doc.add_paragraph()  # Spacing

    # Sections
    for sec_title, sec_content in sections:
        doc.add_heading(sec_title, level=1)
        doc.add_paragraph(sec_content)

    output_path = OUTPUT_DIR / filename
    doc.save(str(output_path))
    print(f"Created policy document: {output_path}")


def build_corpus():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Remote Work Policy (HR-RW-017)
    create_policy_document(
        filename="Remote Work Policy.docx",
        title="Remote Work & Hybrid Workplace Policy",
        policy_id="HR-RW-017",
        version="3.2",
        effective_date="January 15, 2025",
        owner="Human Resources Department",
        sections=[
            ("1. Purpose & Scope",
             "This policy governs all telecommuting, work-from-home (WFH), and hybrid workplace arrangements for eligible company employees. "
             "The objective is to enable work flexibility while maintaining security, operational collaboration, and compliance with HR-RW-017 standards."),
            ("2. Probationary Employee Eligibility",
             "Under HR-RW-017 guidelines, employees during their initial probationary period (the first 90 days of employment) may work remotely or from home "
             "only upon receiving explicit written manager approval and HR endorsement. Probationary employees are required to participate in structured onboarding "
             "and must demonstrate satisfactory core competencies before full-time hybrid flexibility is authorized."),
            ("3. Workspace & Equipment Standards",
             "All remote employees are provided with a standard enterprise laptop and peripherals. Employees must maintain a quiet, secure workspace "
             "with high-speed internet (minimum 50 Mbps download). Company equipment must not be shared with household members or third parties."),
            ("4. Communication & Responsiveness",
             "Remote workers must remain reachable during core business hours via enterprise chat, email, and video conference. "
             "Status indicators on communication tools must accurately reflect availability."),
            ("5. Security & Compliance",
             "Remote connections must use authorized VPN tunnels and Multi-Factor Authentication. Printing confidential company documents at home "
             "is strictly prohibited without prior Information Security officer authorization."),
        ]
    )

    # 2. Annual Leave & PTO Policy (HR-LV-004)
    create_policy_document(
        filename="Annual Leave & PTO Policy.docx",
        title="Annual Leave, Paid Time Off (PTO) & Holiday Policy",
        policy_id="HR-LV-004",
        version="2.1",
        effective_date="January 1, 2025",
        owner="Human Resources Department",
        sections=[
            ("1. Purpose & Scope",
             "This policy outlines time-off entitlements, paid time off (PTO) accrual schedules, sick leave, statutory holidays, and leave request procedures under policy HR-LV-004."),
            ("2. Annual Accrual Rates",
             "Full-time regular employees accrue PTO at a rate of 1.66 days per completed calendar month of service, totaling 20 days per calendar year. "
             "Accrual begins on the first day of full-time employment."),
            ("3. Carry-Over & Unused PTO Limits",
             "A maximum of 5 accrued unused PTO days may be carried over into the following calendar year. Any unused balance exceeding 5 days is forfeited on December 31st unless exception approval is granted by HR."),
            ("4. Request & Approval Workflow",
             "Leave requests exceeding 3 consecutive business days must be submitted through the HR Portal at least 14 calendar days in advance. "
             "Requests are reviewed and approved based on team operational coverage and manager discretion."),
            ("5. Statutory Holidays & Sick Days",
             "The company observes 11 paid statutory holidays annually. Employees receive 10 paid sick days per year, which do not roll over annually."),
        ]
    )

    # 3. Expense Reimbursement Policy (FIN-EXP-011)
    create_policy_document(
        filename="Expense Reimbursement Policy.docx",
        title="Business Travel & Expense Reimbursement Policy",
        policy_id="FIN-EXP-011",
        version="4.0",
        effective_date="February 1, 2025",
        owner="Finance & Accounting",
        sections=[
            ("1. Purpose & Scope",
             "This policy sets rules and expense limits for business travel, client entertainment, training, and operational purchases under policy FIN-EXP-011."),
            ("2. Receipt & Documentation Requirements",
             "Itemized receipts are strictly required for all reimbursable expenses exceeding $25. Summary credit card slips without itemized line items are not acceptable receipts for reimbursement. "
             "All claims must be submitted via the expense portal within 30 calendar days of incurring the expense."),
            ("3. Meals & Daily Per Diem",
             "The standard per diem limit for business meals is $75 per day per employee. Alcohol purchases are non-reimbursable unless part of an authorized client entertainment event with director pre-approval."),
            ("4. Travel & Lodging",
             "Domestic economy air travel must be booked at least 14 days in advance. Hotel accommodations should not exceed standard corporate partner rates ($250 per night standard limit)."),
            ("5. Non-Reimbursable Items",
             "Personal entertainment, parking fines, traffic violations, personal grooming, and unauthorized software subscriptions are strictly non-reimbursable under FIN-EXP-011."),
        ]
    )

    # 4. Information Security Policy (SEC-IS-002)
    create_policy_document(
        filename="Information Security Policy.docx",
        title="Enterprise Information Security & Acceptable Use Policy",
        policy_id="SEC-IS-002",
        version="5.1",
        effective_date="January 10, 2025",
        owner="Information Security & IT Risk",
        sections=[
            ("1. Purpose & Scope",
             "Establishes mandatory technical and operational security controls to protect company assets, customer data, and system integrity under policy SEC-IS-002."),
            ("2. Multi-Factor Authentication (MFA) Requirements",
             "Multi-Factor Authentication (MFA) is strictly mandatory for all employees accessing corporate systems, email, VPN tunnels, and cloud repositories under SEC-IS-002. "
             "MFA authentication tokens must be approved using company-registered mobile authenticators or hardware tokens."),
            ("3. Password & Credential Standards",
             "Passwords must be at least 16 characters long, incorporating uppercase, lowercase, numbers, and special symbols. Passwords must be changed annually and must not be reused."),
            ("4. Device Security & Encryption",
             "All enterprise laptops, mobile devices, and storage media must have Full Disk Encryption (FDE) enabled. Unapproved USB flash drives or external hard drives are prohibited."),
            ("5. Incident Reporting & Escalation",
             "Any suspected security breach, lost device, or phishing attempt must be reported to the IT Security Helpdesk within 1 hour of discovery."),
        ]
    )

    # 5. Attendance & Flexible Work Hours Policy (HR-AT-009)
    create_policy_document(
        filename="Attendance & Flexible Work Hours Policy.docx",
        title="Working Hours, Attendance & Core Hours Policy",
        policy_id="HR-AT-009",
        version="1.4",
        effective_date="March 1, 2025",
        owner="Human Resources Department",
        sections=[
            ("1. Purpose & Scope",
             "Establishes standard working hours, core collaboration periods, flexible work schedule options, and attendance reporting obligations under policy HR-AT-009."),
            ("2. Standard Workweek & Core Collaboration Hours",
             "The standard workweek comprises 40 hours, Monday through Friday. All full-time team members must be active and available during core collaboration hours from 10:00 AM to 3:00 PM local time."),
            ("3. Flexible Scheduling Options",
             "Employees may adjust start and end times between 7:00 AM and 6:00 PM, provided they complete 8 daily hours and maintain presence during mandatory core hours."),
            ("4. Unexcused Absence & Tardiness",
             "Unexcused absences or habitual tardiness exceeding 3 occurrences in a quarter are subject to HR review and progressive performance management."),
            ("5. Overtime Authorization",
             "Non-exempt employees must obtain written supervisor authorization prior to working any hours in excess of 40 hours in a single workweek."),
        ]
    )

    # 6. Employee Benefits & Wellness Policy (HR-BEN-006)
    create_policy_document(
        filename="Employee Benefits & Wellness Policy.docx",
        title="Health Benefits, Wellness Stipend & Employee Care Policy",
        policy_id="HR-BEN-006",
        version="2.0",
        effective_date="January 1, 2025",
        owner="Benefits & Payroll",
        sections=[
            ("1. Purpose & Scope",
             "Outlines healthcare coverage, retirement matching, wellness allowances, and employee care benefits provided under policy HR-BEN-006."),
            ("2. Healthcare Insurance Plan Coverage",
             "The company pays 85% of monthly healthcare insurance premiums for full-time employees and 50% for enrolled dependents. Coverage takes effect on the 1st of the month following start date."),
            ("3. Annual Wellness Stipend",
             "Full-time employees receive a $500 annual wellness stipend, which may be submitted for gym memberships, fitness trackers, ergonomics, or wellness apps under HR-BEN-006."),
            ("4. 401(k) Retirement Savings Match",
             "The company matches 100% of employee 401(k) contributions up to 4% of eligible annual salary. Matching contributions vest immediately upon enrollment."),
            ("5. Commuter & Transit Stipend",
             "Public transit commuting stipends of $100 per month are available for eligible office commuters under policy HR-BEN-006."),
        ]
    )


if __name__ == "__main__":
    build_corpus()
