import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
import matplotlib.pyplot as plt
import pandas as pd

def send_email(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password):
    msg = MIMEText(body)
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())

def send_email_with_attachments(subject, body, to_email, from_email, smtp_server, smtp_port, smtp_user, smtp_password, attachments=None):
    msg = MIMEMultipart()
    msg.attach(MIMEText(body, "plain"))
    msg['Subject'] = subject
    msg['From'] = from_email
    msg['To'] = to_email

    # Attach files
    if attachments:
        for file_path in attachments:
            part = MIMEBase('application', "octet-stream")
            with open(file_path, 'rb') as file:
                part.set_payload(file.read())
            encoders.encode_base64(part)
            part.add_header('Content-Disposition', f'attachment; filename={os.path.basename(file_path)}')
            msg.attach(part)

    with smtplib.SMTP_SSL(smtp_server, smtp_port) as server:
        server.login(smtp_user, smtp_password)
        server.sendmail(from_email, [to_email], msg.as_string())

def save_avg_cycle_time_chart(avg_cycle_time_dict, filename="avg_cycle_time_chart.png"):
    if not avg_cycle_time_dict or "error" in avg_cycle_time_dict:
        return None
    types = list(avg_cycle_time_dict.keys())
    times = list(avg_cycle_time_dict.values())
    plt.figure(figsize=(8, 5))
    plt.bar(types, times, color='skyblue')
    plt.xlabel("Change Type")
    plt.ylabel("Average Cycle Time (days)")
    plt.title("Average Cycle Time per Change Type")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def save_df_to_csv(df, filename="change_control_analysis.csv"):
    df.to_csv(filename, index=False)
    return filename

def save_forecast_chart(forecast_list, filename="forecast_chart.png"):
    if not forecast_list or "error" in forecast_list:
        return None
    df = pd.DataFrame(forecast_list)
    if df.empty or 'date' not in df or 'forecast_on_time_percentage' not in df:
        return None
    plt.figure(figsize=(8, 5))
    plt.plot(df['date'], df['forecast_on_time_percentage'], marker='o', color='orange')
    plt.xlabel("Date")
    plt.ylabel("Forecasted On-Time %")
    plt.title("Forecasted On-Time Percentage (Next 7 Days)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def save_delays_by_department_chart(root_cause_dict, filename="delays_by_department_chart.png"):
    if not root_cause_dict or "error" in root_cause_dict:
        return None
    departments = list(root_cause_dict.keys())
    counts = list(root_cause_dict.values())
    plt.figure(figsize=(8, 5))
    plt.bar(departments, counts, color='salmon')
    plt.xlabel("Department")
    plt.ylabel("Number of Delays")
    plt.title("Delays by Department")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    return filename

def save_kpi_summary_csv(kpi_summary, filename="kpi_summary.csv"):
    if not kpi_summary or "error" in kpi_summary:
        return None
    df = pd.DataFrame([kpi_summary])
    df.to_csv(filename, index=False)
    return filename

def save_forecast_csv(forecast_list, filename="forecast.csv"):
    if not forecast_list or "error" in forecast_list:
        return None
    df = pd.DataFrame(forecast_list)
    df.to_csv(filename, index=False)
    return filename 