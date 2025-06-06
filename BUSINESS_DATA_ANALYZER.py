import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import google.generativeai as genai
from io import BytesIO
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import datetime
from reportlab.lib.utils import ImageReader



# --- Analyzer class ---
class DataAnalyzer:
    def __init__(self, data_file):
        if hasattr(data_file, 'read'):
            self.data = self.load_data(data_file)
        elif isinstance(data_file, str) and os.path.exists(data_file):
            self.data = self.load_data(data_file)
        else:
            raise FileNotFoundError("Invalid file or path")

    def load_data(self, file):
        if isinstance(file, str):
            if file.endswith(".csv"):
                return pd.read_csv(file)
            elif file.endswith((".xls", ".xlsx")):
                return pd.read_excel(file)
        else:
            if file.name.endswith(".csv"):
                return pd.read_csv(file)
            elif file.name.endswith((".xls", ".xlsx")):
                return pd.read_excel(file)
        raise ValueError("Unsupported file format")

    def get_summary(self):
        return self.data.describe()

    def get_head(self):
        return self.data.head()

    def get_columns(self):
        return self.data.columns.tolist()

    def get_data(self):
        return self.data

# --- Visualization and UI ---
class BusinessDataAnalyzerApp:
    def __init__(self, analyzer):
        self.analyzer = analyzer

    def run(self):
        st.subheader("🔍 Data Preview")
        st.dataframe(self.analyzer.get_head())

        st.subheader("📈 Summary Statistics")
        st.dataframe(self.analyzer.get_summary())

        st.subheader("📊 Visualizations")
        data = self.analyzer.get_data()
        numeric_columns = data.select_dtypes(include='number').columns.tolist()

        if numeric_columns:
            col1, col2 = st.columns(2)

            with col1:
                x_col = st.selectbox("Select column for Bar/Line Chart", numeric_columns, key="x1")
                st.bar_chart(data[x_col])

            with col2:
                y_col = st.selectbox("Select column for Pie Chart", numeric_columns, key="y1")
                pie_data = data[y_col].value_counts().head(5)
                fig, ax = plt.subplots()
                ax.pie(pie_data, labels=pie_data.index, autopct="%1.1f%%")
                ax.axis("equal")
                st.pyplot(fig)
        else:
            st.warning("No numeric columns available for charting.")

# --- Ask Gemini with Data Context ---
def ask_gemini(prompt, dataframe):
    if not prompt or dataframe is None:
        return "Please provide a prompt and upload a dataset."

    try:
        genai.configure(api_key="AIzaSyDyHSktxPtcOk8w2h16phzpKwkjvgDBncU")  # Replace with your Gemini API key
        model = genai.GenerativeModel("gemini-1.5-flash")

        df_preview = dataframe.head(10).to_csv(index=False)
        full_prompt = f"""
The user uploaded the following dataset (preview shown):

{df_preview}

Now answer this question based on the data:
{prompt}
"""

        response = model.generate_content(full_prompt)
        return response.text

    except Exception as e:
        return f"Error while contacting Gemini: {e}"

# --- Generate PDF Report ---


def generate_pdf_report(df, file_name="data_report.pdf"):
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    margin = 50
    y = height - margin

    # Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(margin, y, "📊 Data Analysis Report")
    y -= 30

    # Timestamp
    c.setFont("Helvetica", 10)
    c.drawString(margin, y, f"Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    y -= 30

    # Columns
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "🔸 Columns:")
    y -= 20
    c.setFont("Helvetica", 10)
    for col in df.columns:
        c.drawString(margin + 10, y, f"- {col}")
        y -= 15
        if y < 80:
            c.showPage()
            y = height - margin

    # Summary statistics
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "📈 Summary Statistics:")
    y -= 20
    summary = df.describe().round(2)
    for index, row in summary.iterrows():
        line = f"{index}: " + ", ".join([f"{col}={val}" for col, val in row.items()])
        c.setFont("Helvetica", 10)
        c.drawString(margin + 10, y, line[:100])
        y -= 15
        if y < 80:
            c.showPage()
            y = height - margin

    # Sample data
    c.setFont("Helvetica-Bold", 12)
    c.drawString(margin, y, "📝 Sample Data (first 5 rows):")
    y -= 20
    sample_data = df.head()
    for i, row in sample_data.iterrows():
        line = ", ".join([str(val) for val in row])
        c.setFont("Helvetica", 10)
        c.drawString(margin + 10, y, line[:100])
        y -= 15
        if y < 80:
            c.showPage()
            y = height - margin

    # Visualizations
    numeric_cols = df.select_dtypes(include='number').columns.tolist()
    if numeric_cols:
        # Bar chart
        plt.figure(figsize=(4, 3))
        df[numeric_cols[0]].value_counts().head(5).plot(kind='bar')
        plt.title(f"Bar Chart: {numeric_cols[0]}")
        plt.tight_layout()
        bar_img = BytesIO()
        plt.savefig(bar_img, format='PNG')
        plt.close()
        bar_img.seek(0)
        bar_image = ImageReader(bar_img)
        c.showPage()
        c.drawImage(bar_image, margin, height / 2, width=400, preserveAspectRatio=True)

        # Pie chart
        plt.figure(figsize=(4, 4))
        df[numeric_cols[0]].value_counts().head(5).plot.pie(autopct='%1.1f%%')
        plt.title(f"Pie Chart: {numeric_cols[0]}")
        plt.ylabel("")
        pie_img = BytesIO()
        plt.savefig(pie_img, format='PNG')
        plt.close()
        pie_img.seek(0)
        pie_image = ImageReader(pie_img)
        c.drawImage(pie_image, margin + 20, height / 2 - 300, width=350, preserveAspectRatio=True)

    c.save()
    buffer.seek(0)
    return buffer



# --- Download Excel File ---
def generate_excel_download(df):
    output = BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=False, sheet_name='Data')
    output.seek(0)
    return output

# --- Main App ---
def main():
    st.set_page_config(page_title="📈 Business Data Analyzer", layout="wide")
    st.title("📈 Business Data Analyzer")
    st.markdown("Upload your CSV or Excel file for analysis.")

    uploaded_file = st.file_uploader("Upload File", type=["csv", "xlsx"])
    analyzer = None

    try:
        if uploaded_file is not None:
            analyzer = DataAnalyzer(uploaded_file)
            app = BusinessDataAnalyzerApp(analyzer)
            app.run()

            # PDF Report Download
            st.markdown("### 🧾 Download Reports")
            pdf_buffer = generate_pdf_report(analyzer.get_data())
            st.download_button(
                label="📄 Download PDF Report",
                data=pdf_buffer,
                file_name="data_analysis_report.pdf",
                mime="application/pdf"
            )

            # Excel Report Download
            excel_buffer = generate_excel_download(analyzer.get_data())
            st.download_button(
                label="📥 Download Data as Excel",
                data=excel_buffer,
                file_name="analyzed_data.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

        else:
            st.info("No file uploaded. Using default `sales_data.csv` if available.")
            default_path = "sales_data.csv"
            if os.path.exists(default_path):
                analyzer = DataAnalyzer(default_path)
                app = BusinessDataAnalyzerApp(analyzer)
                app.run()
            else:
                st.error("Please upload a file or place `sales_data.csv` in the same folder.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

    # --- Chatbot ---
    st.markdown("---")
    st.markdown("## 🤖 Ask AI about the uploaded data")

    if "gemini_chat" not in st.session_state:
        st.session_state.gemini_chat = []

    user_question = st.chat_input("Ask a question like: 'Which product sells most?'")
    if user_question and analyzer:
        st.session_state.gemini_chat.append(("user", user_question))
        reply = ask_gemini(user_question, analyzer.get_data())
        st.session_state.gemini_chat.append(("bot", reply))

    for role, msg in st.session_state.gemini_chat:
        if role == "user":
            st.chat_message("user").write(msg)
        else:
            st.chat_message("assistant").write(msg)

if __name__ == "__main__":
    main()
