from tkinter import INSERT
import gradio as gr
import numpy as np
import cv2
import tensorflow as tf
import plotly.graph_objects as go
from datetime import datetime
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch, cm
from reportlab.lib.pagesizes import letter
import os
import qrcode
import mysql.connector
import pandas as pd



# ================= MODEL =================
model = tf.keras.models.load_model("brain_tumor_model2.h5")
class_labels = ['glioma', 'meningioma', 'notumor', 'pituitary']


# ================= MYSQL CONNECTION =================
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="logan",  # <-- change if needed
    database="bsd_database"
)
cursor = db.cursor()

# ================= PAGINATED HISTORY =================
current_page = 0
page_size = 5

def fetch_history():
    global current_page
    offset = current_page * page_size

    query = """
    SELECT id,
           patient_name,
           age,
           gender,
           blood_group,
           disease,
           confidence,
           scan_datetime
    FROM patient_reports
    WHERE is_deleted = 0
    ORDER BY id DESC
    LIMIT %s OFFSET %s
    """

    cursor.execute(query, (page_size, offset))
    return cursor.fetchall()



def next_page():
    global current_page
    current_page += 1
    return fetch_history()


def previous_page():
    global current_page
    if current_page > 0:
        current_page -= 1
    return fetch_history()




# ================= AI EXPLANATION =================
disease_info = {
    "glioma": "Glioma is a tumor that originates in the glial cells of the brain. It can affect brain function depending on its size and location. Immediate clinical evaluation is recommended.",
    "meningioma": "Meningioma is a tumor arising from the meninges, the protective layers of the brain. It is typically slow-growing but may cause pressure-related neurological symptoms.",
    "pituitary": "Pituitary tumors develop in the pituitary gland and may impact hormonal balance. Further endocrinological assessment may be required.",
    "notumor": "No abnormal tumor patterns were detected in the MRI scan. The brain structure appears normal based on AI analysis."
}

if not os.path.exists("reports"):
    os.makedirs("reports")



# ================= PDF GENERATION =================
def generate_pdf(patient_name, age, gender, contact,
                 scan_id, blood_group,
                 disease, confidence, image):


    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    report_id = f"BSD-{timestamp}"

    filename = f"reports/{patient_name}_{timestamp}.pdf"

    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    elements = []

    def add_design(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica-Bold", 90)
        canvas.setFillColorRGB(0.92, 0.92, 0.92)
        canvas.translate(letter[0]/2, letter[1]/2)
        canvas.rotate(45)
        canvas.drawCentredString(0, 0, "KARUNYA")
        canvas.restoreState()

        canvas.setStrokeColor(colors.darkblue)
        canvas.setLineWidth(2)
        canvas.rect(20, 20, letter[0]-40, letter[1]-40)

    section_style = ParagraphStyle(
        name='Section',
        fontSize=14,
        textColor=colors.darkblue,
        spaceBefore=10,
        spaceAfter=6
    )

    # ===== HEADER =====
    logo = Image("assets/logo.png", width=1*inch, height=1*inch)
    karunya_img = Image("assets/karunya_text.png",
                        width=3.2*inch, height=0.9*inch)

    header_table = Table([[logo, karunya_img]],
                         colWidths=[80, 370])

    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('LEFTPADDING', (1,0), (1,0), 2*cm)
    ]))

    elements.append(header_table)
    elements.append(Spacer(1, 0.3 * inch))
    # ===== HOSPITAL TITLE =====
    elements.append(Paragraph(
        "<para align='center'><b>HOSPITALS</b></para>",
        ParagraphStyle(
            name='Hosp',
            fontSize=24,
            textColor=colors.darkblue,
            spaceAfter=15
        )
    ))

    elements.append(Spacer(1, 0.08 * inch))

    # ===== ADDRESS =====
    elements.append(Paragraph(
        "<para align='center'>Karunya Nagar, Siruvani Road,</para>",
        ParagraphStyle(
            name='Addr1',
            fontSize=12,
            spaceAfter=4
        )
    ))

    elements.append(Paragraph(
        "<para align='center'>Nallurvayal, Coimbatore - 641114</para>",
        ParagraphStyle(
            name='Addr2',
            fontSize=12,
            spaceAfter=18
        )
    ))

    # ===== PATIENT DETAILS =====
    elements.append(Paragraph("Patient Details", section_style))

    patient_table = Table([
    ["Patient Name", patient_name],
    ["Age", age],
    ["Gender", gender],
    ["Contact", contact],
    ["Scan ID", scan_id],
    ["Blood Group", blood_group],
    ["Scan Date & Time", current_time],
    ["Report ID", report_id],
], colWidths=[160, 290])


    patient_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(patient_table)
    elements.append(Spacer(1, 15))

    # ===== DIAGNOSIS =====
    elements.append(Paragraph("Diagnosis Details", section_style))

    explanation_style = ParagraphStyle(
        name='ExplainStyle',
        fontSize=10,
        leading=14
    )

    wrapped_explanation = Paragraph(
        disease_info[disease],
        explanation_style
    )

    diagnosis_table = Table([
        ["Detected Condition", disease],
        ["Confidence Level", f"{confidence:.2f}%"],
        ["AI Explanation", wrapped_explanation]
    ], colWidths=[160, 290])

    diagnosis_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(diagnosis_table)
    elements.append(Spacer(1, 15))

    # ================= PAGE BREAK =================
    elements.append(PageBreak())

    # ===== PAGE 2 START =====
    elements.append(Paragraph("MRI Scan Image", section_style))

    temp_img = "reports/temp_scan.png"
    cv2.imwrite(temp_img,
                cv2.cvtColor(np.array(image),
                             cv2.COLOR_RGB2BGR))

    elements.append(Image(temp_img,
                          width=4.5*inch,
                          height=3.1*inch))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph("Doctor Approval", section_style))

    sig_path = "assets/signature.png"
    signature_img = Image(sig_path,
                          width=2.5*inch,
                          height=1*inch)

    doctor_block = Table([
        ["Doctor Name", "Dr. DALLAS"],
        ["Designation", "Chief Neurologist"],
        ["Signature", signature_img]
    ], colWidths=[160, 290])

    doctor_block.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black)
    ]))

    elements.append(doctor_block)
    elements.append(Spacer(1, 25))

    elements.append(Paragraph("Scan QR for Full Report Verification", section_style))

    full_report_data = f"""
Report ID      : {report_id}
Patient Name   : {patient_name}
Scan Date      : {current_time}
Disease        : {disease}
Confidence     : {confidence:.2f}%

Doctor         : Dr. DALLAS
"""

    qr = qrcode.QRCode(
        version=None,
        error_correction=qrcode.constants.ERROR_CORRECT_Q,
        box_size=8,
        border=2
    )

    qr.add_data(full_report_data)
    qr.make(fit=True)

    qr_img = qr.make_image(fill_color="black", back_color="white")
    qr_path = "reports/temp_qr.png"
    qr_img.save(qr_path)

    elements.append(Image(qr_path,
                          width=2.3*inch,
                          height=2.3*inch))

    doc.build(elements, onFirstPage=add_design,
              onLaterPages=add_design)
    
    

    # ===== SAVE TO DATABASE ===== 
    insert_query = """
    INSERT INTO patient_reports
    (patient_name, age, gender, contact, scan_id, blood_group,
    disease, confidence, scan_datetime, doctor_name)
    VALUES (%s, %s, %s, %s, %s, %s,
        %s, %s, %s, %s)
"""

    values = (
        patient_name,
        age,
        gender,
        contact,
        scan_id,
        blood_group,
        disease,
        confidence,
        current_time,
        "Dr. DALLAS"
        )


    cursor.execute(insert_query, values)
    db.commit()

    return filename


# ================= PREDICTION =================
def predict(patient_name, image):

    # 🔄 Loading message first
    yield "🔄 Analyzing MRI Scan...", None

    if not patient_name.strip():
        yield "⚠ Please enter patient name.", None
        return

    if image is None:
        yield "⚠ Please upload MRI image.", None
        return

    img = np.array(image)

    # Convert grayscale to RGB if needed
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img = cv2.resize(img, (150, 150))
    img = np.expand_dims(img, axis=0)
    img = img / 255.0

    prediction = model.predict(img)[0]
    index = np.argmax(prediction)
    disease = class_labels[index]
    confidence = float(prediction[index]) * 100
    explanation = disease_info[disease]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=class_labels,
        y=prediction,
        marker=dict(color=prediction, colorscale='Turbo')
    ))

    summary = f"""
Patient Name : {patient_name}
Disease      : {disease}
Confidence   : {confidence:.2f}%

AI Explanation:
{explanation}
"""

    yield summary, fig


# ================= DASHBOARD ANALYTICS =================
def load_dashboard():

    # Total Patients
    cursor.execute("SELECT COUNT(*) FROM patient_reports WHERE is_deleted = 0")
    total_patients = cursor.fetchone()[0]

    # Average Confidence
    cursor.execute("SELECT AVG(confidence) FROM patient_reports")
    avg_conf = cursor.fetchone()[0]
    if avg_conf is None:
        avg_conf = 0

    # Disease Distribution
    cursor.execute("SELECT disease, COUNT(*) FROM patient_reports GROUP BY disease")
    data = cursor.fetchall()

    diseases = []
    counts = []

    for row in data:
        diseases.append(row[0])
        counts.append(row[1])

    # Create Graph
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=diseases,
        y=counts
    ))

    fig.update_layout(
        title="Disease Distribution",
        template="plotly_dark"
    )

    summary = f"""
Total Scans : {total_patients}
Average AI Confidence : {avg_conf:.2f}%
"""

    return summary, fig

# ================= EXPORT TO EXCEL =================
def export_to_excel():
    query = """
    SELECT 
        id,
        patient_name,
        age,
        gender,
        contact,
        blood_group,
        disease,
        confidence,
        scan_datetime
    FROM patient_reports
    WHERE is_deleted = 0
    ORDER BY id DESC
    """

    cursor.execute(query)
    records = cursor.fetchall()

    df = pd.DataFrame(records, columns=[
        "ID",
        "Patient Name",
        "Age",
        "Gender",
        "Mobile Number",
        "Blood Group",
        "Disease",
        "Confidence (%)",
        "Scan Date & Time"
    ])

    # Format datetime properly
    df["Scan Date & Time"] = pd.to_datetime(df["Scan Date & Time"])
    df["Scan Date & Time"] = df["Scan Date & Time"].dt.strftime("%Y-%m-%d %H:%M:%S")

    file_path = "reports/patient_history_full.xlsx"

    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Patient History")

        worksheet = writer.sheets["Patient History"]

        # Auto column width adjust
        for col in worksheet.columns:
            max_length = 0
            column = col[0].column_letter

            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass

            worksheet.column_dimensions[column].width = max_length + 3

    return file_path
    
# ================= DOWNLOAD REPORT =================
def download_report(patient_name, age, gender, contact,
                    scan_id, blood_group,
                    summary, image):

    if image is None or not summary.strip():
        return None

    lines = summary.strip().split("\n")
    disease = lines[1].split(":")[1].strip()
    confidence = float(lines[2].split(":")[1].replace("%","").strip())

    file = generate_pdf(
        patient_name,
        age,
        gender,
        contact,
        scan_id,
        blood_group,
        disease,
        confidence,
        image
    )

    return file

# ================= LOGIN =================
users = {
    "admin": {"password": "admin123", "role": "Admin"},
    "doctor": {"password": "doc1", "role": "Doctor"}
}
def login(username, password):
    if username in users and users[username]["password"] == password:
        role = users[username]["role"]

        if role == "Admin":
            return (
                "Login Successful ✅ (Admin)",
                gr.update(visible=True),    # show main panel
                gr.update(visible=False),   # hide login panel
                gr.update(visible=True),    # delete button
                gr.update(visible=True),    # recycle button
                gr.update(visible=True),    # restore id input
                gr.update(visible=True)     # restore button
            )
        else:
            return (
                "Login Successful ✅ (Doctor)",
                gr.update(visible=True),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False)
            )
    else:
        return (
            "Invalid Credentials ❌",
            gr.update(visible=False),
            gr.update(visible=True),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False),
            gr.update(visible=False)
        )

def delete_record(record_id):
    try:
        query = """
        UPDATE patient_reports
        SET is_deleted = 1,
            deleted_at = NOW()
        WHERE id = %s
        """
        cursor.execute(query, (record_id,))
        db.commit()

        return fetch_history()   # auto refresh table

    except Exception as e:
        print("Delete Error:", e)
        return fetch_history()



def fetch_deleted_records():
    query = """
    SELECT id,
           patient_name,
           age,
           gender,
           blood_group,
           disease,
           confidence,
           scan_datetime
    FROM patient_reports
    WHERE is_deleted = 1
    ORDER BY deleted_at DESC
    """
    cursor.execute(query)
    return cursor.fetchall()

def view_deleted_records():
    query = """
   SELECT id, patient_name, age, gender, blood_group,
       disease, confidence, scan_datetime

    FROM patient_reports
    WHERE is_deleted = 1
    ORDER BY id DESC
    LIMIT 5
    """
    cursor.execute(query)
    return cursor.fetchall()

def restore_record(record_id):
    try:
        query = """
        UPDATE patient_reports
        SET is_deleted = 0
        WHERE id = %s
        """
        cursor.execute(query, (record_id,))
        db.commit()

        return fetch_deleted_records()

    except Exception as e:
        print("Restore Error:", e)
        return fetch_deleted_records()

import smtplib
from email.message import EmailMessage

def send_email_report(receiver_email, pdf_path, patient_name):

    try:
        sender_email = "daldarmanmar.mmdd@gmail.com"
        app_password = "fmolnhgzefbttuml"   # no spaces

        if pdf_path is None:
            return "❌ Please generate report first."

        msg = EmailMessage()
        msg["Subject"] = f"Brain Storm Detection Report - {patient_name}"
        msg["From"] = sender_email
        msg["To"] = receiver_email

        msg.set_content(f"""
Dear {patient_name},

Your MRI scan report is attached.

Regards,
Brain Storm Detection System
""")

        with open(pdf_path, "rb") as f:
            file_data = f.read()
            file_name = os.path.basename(pdf_path)

        msg.add_attachment(
            file_data,
            maintype="application",
            subtype="pdf",
            filename=file_name
        )

        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=10) as smtp:
            smtp.login(sender_email, app_password)
            smtp.send_message(msg)

        return "✅ Email Sent Successfully!"

    except Exception as e:
        return f"❌ Error: {str(e)}"


# ================= GLASSMORPHISM UI =================
with gr.Blocks(title="Brain Tumor Detection") as app:

# ===== CSS =====
    gr.HTML("""
<style>
body {
    background: linear-gradient(135deg, #0f172a, #1e293b);
}

.glass {
    background: rgba(255, 255, 255, 0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    border: 1px solid rgba(255,255,255,0.1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.37);
}

.gr-dropdown select {
    background-color: #1e293b !important;
    color: white !important;
}

.gr-dropdown-menu {
    z-index: 9999 !important;
}

.gr-button-primary {
    background-color: #2563eb !important;
    border-radius: 8px !important;
}
</style>
""")
    # ================= HEADER =================
    with gr.Row():

        with gr.Column(scale=1):
            gr.Image(
                value="assets/logo.png",
                show_label=False,
                height=90,
                container=False
            )

        with gr.Column(scale=3):
            gr.Markdown("""
            <h1 style='
                text-align:center;
                font-weight:900;
                font-size:38px;
                letter-spacing:2px;
                margin-top:15px;'>
            🧠 BRAIN STORM DETECTION
            </h1>
            """)

    # ================= LOGIN =================
    with gr.Column(visible=True) as login_panel:
        gr.Markdown("### 🔐 Secure Login")
        username = gr.Textbox(label="Username")
        password = gr.Textbox(label="Password", type="password")
        login_btn = gr.Button("Login", variant="primary")
        login_status = gr.Textbox(label="Status", interactive=False)

    # ================= MAIN PANEL =================
    with gr.Column(visible=False) as main_panel:

        with gr.Row():

            # LEFT SIDE
            with gr.Column(scale=1, elem_classes="glass"):

                gr.Markdown("### 🧾 Patient Details")
                patient_input = gr.Textbox(label="Patient Name")
                age_input = gr.Number(label="Age")
                gender_input = gr.Textbox(
                    label="Gender (Male / Female / Other)"
                    )
                contact_input = gr.Textbox(label="Contact Number")
                scan_id_input = gr.Textbox(label="Scan ID")
                blood_input = gr.Textbox(
                    label="Blood Group (A+, O-, etc)"
                    )


                img_input = gr.Image(type="pil", height=280)
                predict_btn = gr.Button("Analyze Scan", variant="primary")

                gr.Markdown("### 📄 Report")
                download_btn = gr.Button("Generate Report")
                file_output = gr.File(file_count="single")

                # 📧 Email Section
                gr.Markdown("### 📧 Send Report via Email")
                email_input = gr.Textbox(label="Receiver Email")
                email_btn = gr.Button("Send Email")
                email_status = gr.Textbox(label="Email Status", interactive=False)

                gr.Markdown("### 📊 Export")
                export_btn = gr.Button("Export to Excel")
                excel_file = gr.File(file_count="single")

             

                gr.Markdown("### ⚙ Admin Controls")
                delete_id_input = gr.Number(label="Delete ID")
                delete_btn = gr.Button("Delete", visible=False)
                recycle_btn = gr.Button("♻ Deleted", visible=False)
                restore_id_input = gr.Number(label="Restore ID", visible=False)
                restore_btn = gr.Button("Restore", visible=False)

                       # RIGHT SIDE
            with gr.Column(scale=1, elem_classes="glass"):

                gr.Markdown("### 🧠 AI Diagnosis")
                result_box = gr.Textbox(lines=8)
                graph_output = gr.Plot()

                gr.Markdown("### 📈 Dashboard")
                dashboard_btn = gr.Button("Load Dashboard", variant="primary")
                dashboard_summary = gr.Textbox()
                dashboard_graph = gr.Plot()

                # ================= HISTORY SECTION =================
                gr.Markdown("### 🗂 History")

                history_btn = gr.Button("Load History")

                history_table = gr.Dataframe(
                    headers=[
                        "ID",
                        "Patient",
                        "Age",
                        "Gender",
                        "Blood Group",
                        "Disease",
                        "Confidence",
                        "Date"
                    ],
                    interactive=False,
                    wrap=True
                )

                with gr.Row():
                    prev_btn = gr.Button("⬅ Previous")
                    next_btn = gr.Button("Next ➡")




    # ================= BUTTON CONNECTIONS =================

    login_btn.click(
        login,
        inputs=[username, password],
        outputs=[
            login_status,
            main_panel,
            login_panel,
            delete_btn,
            recycle_btn,
            restore_id_input,
            restore_btn
        ]
    )

    predict_btn.click(
        predict,
        inputs=[patient_input, img_input],
        outputs=[result_box, graph_output]
    )

    download_btn.click(
    download_report,
    inputs=[
        patient_input,
        age_input,
        gender_input,
        contact_input,
        scan_id_input,
        blood_input,
        result_box,
        img_input
    ],
    outputs=[file_output]
)


    history_btn.click(
    fetch_history, outputs=[history_table])
    next_btn.click(next_page, outputs=[history_table])
    prev_btn.click(previous_page, outputs=[history_table])

    dashboard_btn.click(
        load_dashboard,
        outputs=[dashboard_summary, dashboard_graph]
    )

    delete_btn.click(
        delete_record,
        inputs=[delete_id_input],
        outputs=[history_table]
    )

    recycle_btn.click(
        fetch_deleted_records,
        outputs=[history_table]
    )

    restore_btn.click(
        restore_record,
        inputs=[restore_id_input],
        outputs=[history_table]
    )

    email_btn.click(
        send_email_report,
        inputs=[email_input, file_output, patient_input],
        outputs=[email_status]
    )

    export_btn.click(
        export_to_excel,
        outputs=[excel_file]
    )

app.launch()