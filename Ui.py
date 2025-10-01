import streamlit as st
import pandas as pd
import plotly.express as px
import io
import matplotlib.pyplot as plt
import shap
import requests
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from io import BytesIO

st.set_page_config(page_title="SIA - Smart Institutional Approval", layout="wide")
st.title("SIA (Smart Institutional Approval)")
st.markdown(
    """
    **AI-Assisted Institutional Approval Dashboard**  
    Evaluate and visualize AI Readiness, Compliance, and Institutional Quality.
    Upload your dataset to get scores, explanations, and actionable insights.
    """
)

uploaded_file = st.file_uploader("Upload Institutional Data (.xlsx)", type=["xlsx"])
if uploaded_file:
    with st.spinner("..."):
        try:
            files = {"file": uploaded_file}
            response = requests.post("http://127.0.0.1:5000/predict", files=files)
            
            if response.status_code == 200:
                result = response.json()
                metrics = result['metrics']
                df = pd.DataFrame(result['data'])

                st.subheader("Summary Metrics")
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Average Doc Sufficiency %", f"{df['Doc_Sufficiency_%'].mean():.2f}%")
                col2.metric("Average Compliance Score", f"{df['Compliance_Score'].mean():.2f}")
                col3.metric("Average AI Readiness", f"{df['AI_Readiness_Score'].mean():.2f}%")
                col4.metric("Average Quality Index", f"{df['Quality_Index'].mean():.2f}")
                col5.metric("MAE", f"{metrics['MAE']:.3f}")
                col6.metric("RMSE", f"{metrics['RMSE']:.3f}")
                st.write(f"Precision on flagged institutions: {metrics['Precision']:.3f}")

                st.subheader("SIA - Institution Dataset")
                st.dataframe(df[['Institution', 'Compliance_Score', 'Doc_Sufficiency_%', 'AI_Readiness_Score']])

                st.subheader("AI Readiness Clusters")
                def color_map(score):
                    if score < 50:
                        return 'red'
                    elif score < 88:
                        return 'yellow'
                    else:
                        return 'green'
                
                df['Cluster_Color'] = df['AI_Readiness_Score'].apply(color_map)
                fig = px.bar(
                    df,
                    x='Institution',
                    y='AI_Readiness_Score',
                    color='Cluster_Color',
                    color_discrete_map={'red':'red','yellow':'yellow','green':'green'},
                    hover_data=['Compliance_Score','Doc_Sufficiency_%','Quality_Index'],
                    height=500
                )
                fig.update_layout(xaxis={'categoryorder':'array','categoryarray':df['Institution'].tolist()})
                st.plotly_chart(fig, use_container_width=True)

                st.subheader("Global Feature Importance (SHAP)")
                shap_values = df['SHAP_Values'].apply(lambda x: x if isinstance(x,list) else [0]*10).tolist()
                features = ['Faculty_Count','Infra_Score','NAAC_Score','Doc_Sufficiency_%','Compliance_Score',
                            'Quality_Index','Financial_Health_Score','Research_Productivity','Outcome_Indicator','Trust_Risk']
                shap_values = pd.DataFrame(shap_values, columns=features).values
                fig_shap, ax = plt.subplots(figsize=(8,5))
                shap.summary_plot(shap_values, df[features], plot_type='bar', show=False)
                for spine in ax.spines.values():
                    spine.set_visible(False)
                st.pyplot(fig_shap)

                st.subheader("Institution-Specific Explanation")
                selected_inst = st.selectbox("Select Institution", df['Institution'])
                inst_index = df[df['Institution']==selected_inst].index[0]
                shap_vals_inst = shap_values[inst_index]
                top_features = sorted(zip(features, shap_vals_inst), key=lambda x: abs(x[1]), reverse=True)[:5]
                st.write("Top factors affecting AI Readiness:")
                for feat, val in top_features:
                    if val > 0:
                        st.write(f"✅ {feat}: increases AI Readiness by {val:.2f}")
                    elif val < 0:
                        st.write(f"❌ {feat}: decreases AI Readiness by {abs(val):.2f}")
                    else:
                        st.write(f"⚪ {feat}: neutral impact")

                st.subheader("Low-Score Highlights & Reasons")
                low_df = df[df["AI_Readiness_Score"] < 70].copy()
                low_df['Reasons_Flagged'] = low_df['Reasons_Flagged'].apply(lambda x: ", ".join(x) if isinstance(x, list) else "Meets minimum standards")
                low_df = low_df.sort_values("AI_Readiness_Score").head(50)
                st.dataframe(low_df[['Institution', 'AI_Readiness_Score', 'Reasons_Flagged']])

                st.subheader("Download Excel Report")
                export_cols = [
                    'Institution','Faculty_Count','Infra_Score','NAAC_Grade','NAAC_Score',
                    'Doc_Sufficiency_%','Compliance_Score','Quality_Index','Financial_Health_Score',
                    'Research_Productivity','Outcome_Indicator','Trust_Risk','AI_Readiness_Score'
                ]
                df_export = df[export_cols]
                output = io.BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    workbook = writer.book
                    worksheet = workbook.add_worksheet('AI_Readiness_Report')
                    writer.sheets['AI_Readiness_Report'] = worksheet
                    worksheet.merge_range('A1:M1', 'SIA (Smart Institutional Approval) - AI Readiness Report')
                    df_export.to_excel(writer, index=False, startrow=1, sheet_name='AI_Readiness_Report')
                processed_data = output.getvalue()
                st.download_button(
                    "Download Excel Report",
                    data=processed_data,
                    file_name="SIA_AI_Readiness_Report.xlsx",
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

                st.subheader("Download Individual College PDF Report")
                if st.button(f"Generate PDF for {selected_inst}"):
                    pdf_buffer = BytesIO()
                    doc = SimpleDocTemplate(pdf_buffer, pagesize=A4)
                    styles = getSampleStyleSheet()
                    story = []

                    story.append(Paragraph("SIA - Smart Institutional Approval", styles['Title']))
                    story.append(Spacer(1,12))
                    story.append(Paragraph(f"College: {selected_inst}", styles['Heading2']))
                    story.append(Paragraph(f"AI Readiness Score: {df.loc[inst_index, 'AI_Readiness_Score']:.2f}", styles['Normal']))
                    rank = int(df['AI_Readiness_Score'].rank(ascending=False)[inst_index])
                    story.append(Paragraph(f"Rank among all colleges: {rank}", styles['Normal']))
                    story.append(Spacer(1,12))

                    story.append(Paragraph("Summary Metrics (All Colleges)", styles['Heading3']))
                    summary_table_data = [
                        ['Metric','Average'],
                        ['Doc Sufficiency %', f"{df['Doc_Sufficiency_%'].mean():.2f}"],
                        ['Compliance Score', f"{df['Compliance_Score'].mean():.2f}"],
                        ['AI Readiness Score', f"{df['AI_Readiness_Score'].mean():.2f}"],
                        ['Quality Index', f"{df['Quality_Index'].mean():.2f}"]
                    ]
                    story.append(Table(summary_table_data))
                    story.append(Spacer(1,12))

                    # Top 5 SHAP Features
                    story.append(Paragraph("Top 5 Features Affecting AI Readiness", styles['Heading3']))
                    for feat, val in top_features:
                        impact = "increases" if val>0 else "decreases" if val<0 else "neutral"
                        icon = "✅" if val>0 else "❌" if val<0 else "⚪"
                        story.append(Paragraph(f"{icon} {feat}: {impact} AI Readiness by {abs(val):.2f}", styles['Normal']))
                    story.append(Spacer(1,12))

                    fig, ax = plt.subplots(figsize=(5,3))
                    feat_names, feat_vals = zip(*top_features)
                    colors = ['green' if v>0 else 'red' for v in feat_vals]
                    ax.barh(feat_names, feat_vals, color=colors)
                    ax.set_xlabel("SHAP Value")
                    ax.set_title("Top 5 Features")
                    plt.tight_layout()
                    plot_buffer = BytesIO()
                    plt.savefig(plot_buffer, format='PNG')
                    plt.close(fig)
                    plot_buffer.seek(0)
                    story.append(Image(plot_buffer, width=400, height=150))
                    story.append(Spacer(1,12))

                    # Suggestions Section (fixed major points)
                    story.append(Paragraph("Suggestions to Improve AI Readiness", styles['Heading3']))
                    
                    # List of major improvement points
                    improvement_points = [
                        "Increase faculty strength and fill vacant positions",
                        "Enhance research output per faculty and fund more research projects",
                        "Ensure all required documents are uploaded and kept up-to-date",
                        "Improve compliance with NAAC and other regulatory standards",
                        "Boost graduate placement rates and median CTC to improve productivity and outcomes",
                        "Increase revenue per student and overall financial health"
                    ]
                    
                    for point in improvement_points:
                        story.append(Paragraph(f"- {point}", styles['Normal']))

                    doc.build(story)
                    pdf_buffer.seek(0)
                    st.download_button(
                        "Download PDF Report",
                        data=pdf_buffer,
                        file_name=f"SIA_{selected_inst}_AI_Readiness_Report.pdf",
                        mime='application/pdf'
                    )
            else:
                st.error(f"Backend returned error: {response.status_code} - {response.text}")
        except Exception as e:
            st.error(f"Error communicating with backend: {e}")
else:
    st.warning("Please upload an institutional dataset to view the dashboard.")
